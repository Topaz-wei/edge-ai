// 本文件实现通用 TensorRT 执行器；形状变化不重新分配设备缓冲，不包含模型语义。
#include "edge/trt/runner.hpp"
#include "edge/trt/tensor_io.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace edge::trt {
namespace {
class Logger : public nvinfer1::ILogger {
public:
    void log(
        Severity severity,
        const char* message
    ) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};
Logger logger;

std::size_t tensorBytes(const nvinfer1::Dims& shape) {
    const auto count = getNumElements(shape);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::runtime_error("Tensor byte count overflow");
    }
    return count * sizeof(float);
}
} // namespace

Runner::Runner(
    const std::string& enginePath,
    int profileIndex
) {
    try {
        const auto data = loadEngineFile(enginePath);
        runtime_.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime_) {
            throw std::runtime_error("Failed to create runtime");
        }
        engine_.reset(runtime_->deserializeCudaEngine(
            data.data(),
            data.size()
        ));
        if (!engine_ || engine_->hasImplicitBatchDimension()) {
            throw std::runtime_error("Expected an explicit-batch engine");
        }
        int inputCount = 0;
        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* name = engine_->getIOTensorName(index);
            TensorInfo info;
            info.name = name;
            info.engineShape = engine_->getTensorShape(name);
            info.dtype = engine_->getTensorDataType(name);
            info.mode = engine_->getTensorIOMode(name);
            if (info.dtype != nvinfer1::DataType::kFLOAT ||
                engine_->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR ||
                engine_->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
                engine_->isShapeInferenceIO(name)) {
                throw std::runtime_error("Expected execution-only FP32 linear device I/O: " + info.name);
            }
            if (info.mode == nvinfer1::TensorIOMode::kINPUT) {
                ++inputCount;
                inputIndex_ = tensors_.size();
                for (int axis = 0; axis < info.engineShape.nbDims; ++axis) {
                    dynamic_ = dynamic_ || info.engineShape.d[axis] == -1;
                }
            } else if (info.mode == nvinfer1::TensorIOMode::kOUTPUT) {
                outputIndices_.push_back(tensors_.size());
            } else {
                throw std::runtime_error("Unknown tensor mode");
            }
            tensors_.push_back(info);
        }
        if (inputCount != 1 || outputIndices_.empty()) {
            throw std::runtime_error("Expected one input and at least one output");
        }
        if (profileIndex < 0 || profileIndex >= profileCount()) {
            throw std::runtime_error("Invalid optimization profile");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create context");
        }
        checkCuda(
            cudaStreamCreate(&stream_),
            "create stream"
        );
        for (auto& event : events_) {
            checkCuda(
                cudaEventCreate(&event),
                "create event"
            );
        }
        // 对所有 profile 的 MAX 输入解析输出，取最大容量，后续切换不移动地址。
        // 数据依赖输出或 MAX 不能界定容量的网络不在本版支持范围。
        for (int profile = 0; profile < profileCount(); ++profile) {
            if (!context_->setOptimizationProfileAsync(
                    profile,
                    stream_
                )) {
                throw std::runtime_error("Cannot select profile");
            }
            auto maximum = tensors_[inputIndex_].engineShape;
            if (dynamic_) {
                maximum = engine_->getProfileShape(
                    tensors_[inputIndex_].name.c_str(),
                    profile,
                    nvinfer1::OptProfileSelector::kMAX
                );
            }
            if (!context_->setInputShape(
                    tensors_[inputIndex_].name.c_str(),
                    maximum
                ) || context_->inferShapes(
                    0,
                    nullptr
                ) != 0) {
                throw std::runtime_error("Cannot resolve profile MAX shapes");
            }
            for (auto& info : tensors_) {
                const auto shape = context_->getTensorShape(info.name.c_str());
                info.capacityBytes = std::max(
                    info.capacityBytes,
                    tensorBytes(shape)
                );
            }
            checkCuda(
                cudaStreamSynchronize(stream_),
                "profile initialization sync"
            );
        }
        buffers_.resize(tensors_.size(), nullptr);
        for (std::size_t index = 0; index < tensors_.size(); ++index) {
            checkCuda(
                cudaMalloc(
                    &buffers_[index],
                    tensors_[index].capacityBytes
                ),
                "allocate tensor capacity"
            );
            if (!context_->setTensorAddress(
                    tensors_[index].name.c_str(),
                    buffers_[index]
                )) {
                throw std::runtime_error("Cannot bind tensor address");
            }
        }
        selectProfile(profileIndex);
    } catch (...) {
        release();
        throw;
    }
}

Runner::~Runner() { release(); }

int Runner::profileCount() const { return engine_->getNbOptimizationProfiles(); }

void Runner::destroyGraph() noexcept {
    if (graphExec_) {
        cudaGraphExecDestroy(graphExec_);
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
    }
    graphExec_ = nullptr;
    graph_ = nullptr;
}

void Runner::release() noexcept {
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }
    destroyGraph();
    // 先销毁 context，确保它不再引用设备缓冲。
    context_.reset();
    for (auto pointer : buffers_) {
        if (pointer) {
            cudaFree(pointer);
        }
    }
    buffers_.clear();
    for (auto& event : events_) {
        if (event) {
            cudaEventDestroy(event);
        }
        event = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    stream_ = nullptr;
}

void Runner::requireReady() const {
    if (!ready_) {
        throw std::runtime_error("Set a valid input shape before using tensors");
    }
}

std::size_t Runner::findTensor(const std::string& name) const {
    for (std::size_t index = 0; index < tensors_.size(); ++index) {
        if (tensors_[index].name == name) {
            return index;
        }
    }
    throw std::runtime_error("Unknown tensor: " + name);
}

const TensorInfo& Runner::tensorInfo(const std::string& name) const {
    requireReady();
    return tensors_[findTensor(name)];
}

void* Runner::deviceTensor(const std::string& name) const {
    requireReady();
    return buffers_[findTensor(name)];
}

const TensorInfo& Runner::inputInfo() const {
    requireReady();
    return tensors_[inputIndex_];
}

const TensorInfo& Runner::outputInfo() const {
    requireReady();
    if (outputIndices_.size() != 1) {
        throw std::runtime_error("Single-output API cannot be used with multiple outputs");
    }
    return tensors_[outputIndices_[0]];
}

std::size_t Runner::inputElements() const { return inputInfo().activeBytes / sizeof(float); }
std::size_t Runner::outputElements() const { return outputInfo().activeBytes / sizeof(float); }
float* Runner::deviceInput() const { return static_cast<float*>(deviceTensor(inputInfo().name)); }
const float* Runner::deviceOutput() const { return static_cast<const float*>(deviceTensor(outputInfo().name)); }

void Runner::resolveShapes() {
    if (context_->inferShapes(
            0,
            nullptr
        ) != 0) {
        throw std::runtime_error("Input shape inference failed");
    }
    auto resolved = tensors_;
    for (auto& info : resolved) {
        info.shape = context_->getTensorShape(info.name.c_str());
        info.activeBytes = tensorBytes(info.shape);
        if (info.activeBytes > info.capacityBytes) {
            throw std::runtime_error("Runtime tensor exceeds allocated MAX capacity: " + info.name);
        }
    }
    // 保持 TensorInfo 对象地址稳定；旧调用方可能持有其引用。
    for (std::size_t index = 0; index < resolved.size(); ++index) {
        tensors_[index] = resolved[index];
    }
    ready_ = true;
}

void Runner::selectProfile(int profileIndex) {
    if (profileIndex < 0 || profileIndex >= profileCount()) {
        throw std::runtime_error("Invalid optimization profile");
    }
    checkCuda(
        cudaStreamSynchronize(stream_),
        "sync before profile change"
    );
    destroyGraph();
    ready_ = false;
    if (!context_->setOptimizationProfileAsync(
            profileIndex,
            stream_
        )) {
        throw std::runtime_error("Cannot select optimization profile");
    }
    profileIndex_ = profileIndex;
    auto shape = tensors_[inputIndex_].engineShape;
    if (dynamic_) {
        shape = engine_->getProfileShape(
            tensors_[inputIndex_].name.c_str(),
            profileIndex_,
            nvinfer1::OptProfileSelector::kOPT
        );
    }
    setInputShape(shape);
}

void Runner::setInputShape(const nvinfer1::Dims& shape) {
    const auto& input = tensors_[inputIndex_];
    if (shape.nbDims != input.engineShape.nbDims) {
        throw std::runtime_error("Input rank mismatch");
    }
    tensorBytes(shape);
    const auto minimum = engine_->getProfileShape(
        input.name.c_str(),
        profileIndex_,
        nvinfer1::OptProfileSelector::kMIN
    );
    const auto maximum = engine_->getProfileShape(
        input.name.c_str(),
        profileIndex_,
        nvinfer1::OptProfileSelector::kMAX
    );
    for (int axis = 0; axis < shape.nbDims; ++axis) {
        if (input.engineShape.d[axis] != -1 && shape.d[axis] != input.engineShape.d[axis]) {
            throw std::runtime_error("Static input dimension mismatch");
        }
        if (dynamic_ && (shape.d[axis] < minimum.d[axis] || shape.d[axis] > maximum.d[axis])) {
            throw std::runtime_error("Input shape outside profile range");
        }
    }
    checkCuda(
        cudaStreamSynchronize(stream_),
        "sync before shape change"
    );
    destroyGraph();
    ready_ = false;
    if (!context_->setInputShape(
            input.name.c_str(),
            shape
        )) {
        throw std::runtime_error("setInputShape failed");
    }
    resolveShapes();
}

NamedOutputs Runner::inferNamed(
    const std::vector<float>& input,
    TrtTimings* timings
) {
    requireReady();
    if (input.size() != inputElements()) {
        throw std::runtime_error("Input size mismatch");
    }
    NamedOutputs outputs;
    for (auto index : outputIndices_) {
        outputs[tensors_[index].name].resize(tensors_[index].activeBytes / sizeof(float));
    }
    const auto record = [&](int index) {
        if (timings) {
            checkCuda(
                cudaEventRecord(
                    events_[index],
                    stream_
                ),
                "record transfer/inference event"
            );
        }
    };
    try {
        record(0);
        checkCuda(
            cudaMemcpyAsync(
                deviceInput(),
                input.data(),
                inputInfo().activeBytes,
                cudaMemcpyHostToDevice,
                stream_
            ),
            "H2D"
        );
        record(1);
        enqueueDevice();
        record(2);
        for (auto index : outputIndices_) {
            const auto& info = tensors_[index];
            checkCuda(
                cudaMemcpyAsync(
                    outputs.at(info.name).data(),
                    buffers_[index],
                    info.activeBytes,
                    cudaMemcpyDeviceToHost,
                    stream_
                ),
                "D2H"
            );
        }
        record(3);
        checkCuda(
            cudaStreamSynchronize(stream_),
            "inference sync"
        );
        if (timings) {
            std::array<float*, 3> values{&timings->h2dMs, &timings->inferenceMs, &timings->d2hMs};
            for (int index = 0; index < 3; ++index) {
                checkCuda(
                    cudaEventElapsedTime(
                        values[index],
                        events_[index],
                        events_[index + 1]
                    ),
                    "elapsed event time"
                );
            }
        }
    } catch (...) {
        cudaStreamSynchronize(stream_);
        throw;
    }
    return outputs;
}

std::vector<float> Runner::infer(
    const std::vector<float>& input,
    TrtTimings* timings
) {
    const auto name = outputInfo().name;
    auto outputs = inferNamed(
        input,
        timings
    );
    return std::move(outputs.at(name));
}

void Runner::enqueueDevice(bool useGraph) {
    requireReady();
    if (useGraph) {
        if (!graphExec_) {
            throw std::runtime_error("CUDA Graph has not been prepared");
        }
        checkCuda(
            cudaGraphLaunch(
                graphExec_,
                stream_
            ),
            "graph launch"
        );
    } else if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 failed");
    }
}

void Runner::prepareGraph() {
    requireReady();
    if (dynamic_) {
        throw std::runtime_error("CUDA Graph is supported only for static engines");
    }
    if (graphExec_) {
        return;
    }
    checkCuda(
        cudaMemsetAsync(
            deviceInput(),
            0,
            inputInfo().activeBytes,
            stream_
        ),
        "graph warmup input"
    );
    enqueueDevice();
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
        if (aborted) {
            cudaGraphDestroy(aborted);
        }
        throw std::runtime_error("Graph capture enqueue failed");
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
        destroyGraph();
        throw;
    }
}
} // namespace edge::trt
