#include "include/trt_runner.hpp"

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

#include "include/infer_raw.hpp"

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

void checkCuda(
    cudaError_t error,
    const char* operation
) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

std::vector<char> loadEngineFile(
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error("Failed to open engine: " + path);
    }

    file.seekg(
        0,
        std::ios::end
    );

    const auto end = file.tellg();

    if (end <= 0)
        throw std::runtime_error("Invalid engine size: " + path);

    const std::size_t size = static_cast<std::size_t>(end);

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<char> data(size);

    file.read(
        data.data(),
        size
    );

    if (!file)
        throw std::runtime_error("Failed to read engine: " + path);

    return data;
}

std::vector<float> loadFloatFile(
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    file.seekg(
        0,
        std::ios::end
    );

    const auto end = file.tellg();

    if (end <= 0)
        throw std::runtime_error("Invalid float file size: " + path);

    const std::size_t bytes = static_cast<std::size_t>(end);

    if (bytes % sizeof(float) != 0) {
        throw std::runtime_error("Invalid float binary file: " + path);
    }

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<float> data(bytes / sizeof(float));

    file.read(
        reinterpret_cast<char*>(data.data()),
        bytes
    );

    if (!file)
        throw std::runtime_error("Failed to read float file: " + path);

    return data;
}

std::string dimsToString(
    const nvinfer1::Dims& dims
) {
    std::string s = "[";

    for (int i = 0; i < dims.nbDims; ++i) {
        s += std::to_string(dims.d[i]);

        if (i != dims.nbDims - 1) {
            s += "x";
        }
    }

    s += "]";

    return s;
}

std::size_t getNumElements(
    const nvinfer1::Dims& dims
) {
    if (dims.nbDims < 0)
        throw std::runtime_error("Invalid tensor rank");

    std::size_t elements = 1;

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Invalid tensor dimension");
        }

        if (elements > std::numeric_limits<std::size_t>::max() /
                           static_cast<std::size_t>(dims.d[i])) {
            throw std::runtime_error("Tensor element count overflow");
        }

        elements *= static_cast<std::size_t>(dims.d[i]);
    }

    return elements;
}

void saveFloatBinary(
    const std::string& path,
    const std::vector<float>& data
) {
    std::ofstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    file.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(float))
    );

    if (!file) {
        throw std::runtime_error("Failed to write output file: " + path);
    }
}

TrtRunner::TrtRunner(
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

TrtRunner::~TrtRunner() {
    release();
}

void TrtRunner::release() noexcept {
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

std::vector<float> TrtRunner::infer(
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

bool TrtRunner::isFixedYolo() const {
    const auto in = engine_->getTensorShape(inputName_);

    const auto out = engine_->getTensorShape(outputName_);

    return in.nbDims == 4 && in.d[0] == 1 && in.d[1] == 3 && in.d[2] == 640 &&
           in.d[3] == 640 && out.nbDims == 3 && out.d[0] == 1 && out.d[1] == 84 &&
           out.d[2] == 8400;
}

void TrtRunner::enqueueDevice(
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

void TrtRunner::prepareGraph() {
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
