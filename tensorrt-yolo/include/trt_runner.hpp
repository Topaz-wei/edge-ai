#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

struct TrtTimings {
    float h2dMs{0};
    float inferenceMs{0};
    float d2hMs{0};
};

// One synchronous, non-thread-safe inference stream per runner.
// Supports fixed-shape, single-input/output engines with FP32 linear device I/O.
class TrtRunner {
public:
    explicit TrtRunner(const std::string& enginePath);

    ~TrtRunner();

    TrtRunner(const TrtRunner&) = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;

    // Optional timings use reusable CUDA events on the inference stream.
    std::vector<float> infer(
        const std::vector<float>& input,
        TrtTimings* timings = nullptr
    );

    std::size_t inputElements() const {
        return inputElements_;
    }

    std::size_t outputElements() const {
        return outputElements_;
    }

    // Device API: caller supplies preprocessed input on this stream, and may consume
    // output on the same stream. No copies/synchronization in enqueueDevice().
    float* deviceInput() const {
        return static_cast<float*>(deviceInput_);
    }

    const float* deviceOutput() const {
        return static_cast<const float*>(deviceOutput_);
    }

    cudaStream_t stream() const {
        return stream_;
    }

    void prepareGraph();

    void enqueueDevice(bool useGraph = false);

    bool isFixedYolo() const;

private:
    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t graphExec_{nullptr};
    void release() noexcept;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    const char* inputName_{nullptr};
    const char* outputName_{nullptr};
    std::size_t inputElements_{0}, outputElements_{0};
    std::size_t inputBytes_{0}, outputBytes_{0};
    void* deviceInput_{nullptr};
    void* deviceOutput_{nullptr};
    cudaStream_t stream_{nullptr};
    std::array<cudaEvent_t, 4> events_{};
};
