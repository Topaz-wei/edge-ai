// 本文件定义通用 TensorRT 执行器；管理具名 I/O、动态形状和持久设备缓冲。
#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace edge::trt {
struct TensorInfo {
    std::string name;
    nvinfer1::Dims shape{}; // 当前 context 的实际形状。
    nvinfer1::DataType dtype{};
    nvinfer1::Dims engineShape{};
    nvinfer1::TensorIOMode mode{};
    std::size_t activeBytes{0};
    std::size_t capacityBytes{0};
};
struct TrtTimings {
    float h2dMs{0};
    float inferenceMs{0};
    float d2hMs{0};
};
using NamedOutputs = std::map<std::string, std::vector<float>>;

// 一个实例拥有一个 context/stream，不可并发调用；支持单输入、多输出 FP32 线性设备 I/O。
class Runner {
public:
    explicit Runner(
        const std::string& enginePath,
        int profileIndex = 0
    );
    ~Runner();
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    void setInputShape(const nvinfer1::Dims& shape);
    void selectProfile(int profileIndex);
    int profileIndex() const { return profileIndex_; }
    int profileCount() const;
    const std::vector<TensorInfo>& tensors() const {
        requireReady();
        return tensors_;
    }
    const TensorInfo& tensorInfo(const std::string& name) const;
    void* deviceTensor(const std::string& name) const;

    NamedOutputs inferNamed(
        const std::vector<float>& input,
        TrtTimings* timings = nullptr
    );
    // 兼容旧调用方；多输出时明确拒绝，禁止静默选择第一个输出。
    std::vector<float> infer(
        const std::vector<float>& input,
        TrtTimings* timings = nullptr
    );
    const TensorInfo& inputInfo() const;
    const TensorInfo& outputInfo() const;
    std::size_t inputElements() const;
    std::size_t outputElements() const;
    float* deviceInput() const;
    const float* deviceOutput() const;
    cudaStream_t stream() const { return stream_; }
    void prepareGraph();
    void enqueueDevice(bool useGraph = false);

private:
    std::size_t findTensor(const std::string& name) const;
    void resolveShapes();
    void requireReady() const;
    void release() noexcept;
    void destroyGraph() noexcept;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<TensorInfo> tensors_;
    std::vector<void*> buffers_;
    std::size_t inputIndex_{0};
    std::vector<std::size_t> outputIndices_;
    int profileIndex_{0};
    bool dynamic_{false};
    bool ready_{false};
    cudaStream_t stream_{nullptr};
    std::array<cudaEvent_t, 4> events_{};
    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t graphExec_{nullptr};
};
} // namespace edge::trt
