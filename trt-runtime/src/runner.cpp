#include "edge/trt/runner.hpp"

#include <limits>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#include "edge/trt/tensor_io.hpp"

namespace edge::trt {
class Logger : public nvinfer1::ILogger {
public:
    void log(
        Severity severity,
        const char* msg
    ) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << '\n';
        }
    }
};

static Logger gLogger;

Runner::Runner(
    const std::string& enginePath
) {
    try {
        const auto data = loadEngineFile(enginePath);

        runtime_.reset(nvinfer1::createInferRuntime(gLogger));

        if (!runtime_)
            throw std::runtime_error("Failed to create TensorRT runtime");

        engine_.reset(runtime_->deserializeCudaEngine(
            data.data(),
            data.size()
        ));

        if (!engine_)
            throw std::runtime_error("Failed to deserialize engine");

        if (engine_->getNbIOTensors() != 2) {
            throw std::runtime_error("Expected exactly one input and one output");
        }

        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);

            const auto mode = engine_->getTensorIOMode(name);

            if (mode == nvinfer1::TensorIOMode::kINPUT)
                inputName_ = name;

            if (mode == nvinfer1::TensorIOMode::kOUTPUT)
                outputName_ = name;

            if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
                engine_->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR ||
                engine_->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
                throw std::runtime_error("Expected FP32 linear device I/O");
            }
        }

        if (!inputName_ || !outputName_)
            throw std::runtime_error("Missing input/output tensor");

        inputInfo_ = {inputName_, engine_->getTensorShape(inputName_), engine_->getTensorDataType(inputName_)};
        outputInfo_ = {outputName_, engine_->getTensorShape(outputName_), engine_->getTensorDataType(outputName_)};

        inputElements_ = getNumElements(engine_->getTensorShape(inputName_));

        outputElements_ = getNumElements(engine_->getTensorShape(outputName_));

        if (inputElements_ > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            outputElements_ > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error("Tensor byte count overflow");
        }

        inputBytes_ = inputElements_ * sizeof(float);

        outputBytes_ = outputElements_ * sizeof(float);

        context_.reset(engine_->createExecutionContext());

        if (!context_)
            throw std::runtime_error("Failed to create execution context");

        checkCuda(
            cudaMalloc(
                &deviceInput_,
                inputBytes_
            ),
            "cudaMalloc input"
        );

        checkCuda(
            cudaMalloc(
                &deviceOutput_,
                outputBytes_
            ),
            "cudaMalloc output"
        );

        if (!context_->setTensorAddress(
                inputName_,
                deviceInput_
            ) ||
            !context_->setTensorAddress(
                outputName_,
                deviceOutput_
            )) {
            throw std::runtime_error("Failed to set tensor addresses");
        }

        checkCuda(
            cudaStreamCreate(&stream_),
            "cudaStreamCreate"
        );

        for (auto& event : events_)
            checkCuda(
                cudaEventCreate(&event),
                "cudaEventCreate"
            );
    } catch (...) {
        // The destructor is not called if construction throws.
        release();

        throw;
    }
}

Runner::~Runner() {
    release();
}

void Runner::release() noexcept {
    if (stream_)
        cudaStreamSynchronize(stream_);

    if (graphExec_)
        cudaGraphExecDestroy(graphExec_);

    if (graph_)
        cudaGraphDestroy(graph_);

    graphExec_ = nullptr;
    graph_ = nullptr;

    for (auto& event : events_) {
        if (event)
            cudaEventDestroy(event);

        event = nullptr;
    }

    if (stream_)
        cudaStreamDestroy(stream_);

    stream_ = nullptr;

    if (deviceInput_)
        cudaFree(deviceInput_);

    if (deviceOutput_)
        cudaFree(deviceOutput_);

    deviceInput_ = deviceOutput_ = nullptr;
    // Members then destroy context -> engine -> runtime, in that order.
}

std::vector<float> Runner::infer(
    const std::vector<float>& input,
    TrtTimings* timings
) {
    if (input.size() != inputElements_)
        throw std::runtime_error("Input size mismatch");

    std::vector<float> output(outputElements_);

    try {
        if (timings)
            checkCuda(
                cudaEventRecord(
                    events_[0],
                    stream_
                ),
                "record H2D start"
            );

        checkCuda(
            cudaMemcpyAsync(
                deviceInput_,
                input.data(),
                inputBytes_,
                cudaMemcpyHostToDevice,
                stream_
            ),
            "H2D"
        );

        if (timings)
            checkCuda(
                cudaEventRecord(
                    events_[1],
                    stream_
                ),
                "record H2D end"
            );

        if (!context_->enqueueV3(stream_))
            throw std::runtime_error("enqueueV3 failed");

        if (timings)
            checkCuda(
                cudaEventRecord(
                    events_[2],
                    stream_
                ),
                "record inference end"
            );

        checkCuda(
            cudaMemcpyAsync(
                output.data(),
                deviceOutput_,
                outputBytes_,
                cudaMemcpyDeviceToHost,
                stream_
            ),
            "D2H"
        );

        if (timings)
            checkCuda(
                cudaEventRecord(
                    events_[3],
                    stream_
                ),
                "record D2H end"
            );

        checkCuda(
            cudaStreamSynchronize(stream_),
            "cudaStreamSynchronize"
        );

        if (timings) {
            checkCuda(
                cudaEventElapsedTime(
                    &timings->h2dMs,
                    events_[0],
                    events_[1]
                ),
                "H2D elapsed"
            );

            checkCuda(
                cudaEventElapsedTime(
                    &timings->inferenceMs,
                    events_[1],
                    events_[2]
                ),
                "inference elapsed"
            );

            checkCuda(
                cudaEventElapsedTime(
                    &timings->d2hMs,
                    events_[2],
                    events_[3]
                ),
                "D2H elapsed"
            );
        }
    } catch (...) {
        // Keep host buffers alive until any already queued copies have finished.
        cudaStreamSynchronize(stream_);

        throw;
    }

    return output;
}

void Runner::enqueueDevice(
    bool useGraph
) {
    if (useGraph) {
        if (!graphExec_)
            throw std::runtime_error("CUDA Graph has not been prepared");

        checkCuda(
            cudaGraphLaunch(
                graphExec_,
                stream_
            ),
            "cudaGraphLaunch"
        );
    } else if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 failed");
    }
}

void Runner::prepareGraph() {
    if (graphExec_)
        return;
    // Initialize input and flush TensorRT's lazy work before capture.
    checkCuda(
        cudaMemsetAsync(
            deviceInput_,
            0,
            inputBytes_,
            stream_
        ),
        "graph warmup input"
    );

    enqueueDevice(false);

    checkCuda(
        cudaStreamSynchronize(stream_),
        "graph warmup sync"
    );

    checkCuda(
        cudaStreamBeginCapture(
            stream_,
            cudaStreamCaptureModeThreadLocal
        ),
        "begin capture"
    );

    if (!context_->enqueueV3(stream_)) {
        cudaGraph_t aborted = nullptr;
        cudaStreamEndCapture(
            stream_,
            &aborted
        );

        if (aborted)
            cudaGraphDestroy(aborted);

        throw std::runtime_error("TensorRT graph capture enqueue failed");
    }

    checkCuda(
        cudaStreamEndCapture(
            stream_,
            &graph_
        ),
        "end capture"
    );

    try {
        checkCuda(
            cudaGraphInstantiate(
                &graphExec_,
                graph_,
                nullptr,
                nullptr,
                0
            ),
            "instantiate graph"
        );
    } catch (...) {
        cudaGraphDestroy(graph_);

        graph_ = nullptr;

        throw;
    }
}

}
