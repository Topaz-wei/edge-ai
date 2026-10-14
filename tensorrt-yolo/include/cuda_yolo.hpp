#pragma once

#include "include/postprocess_raw.hpp"
#include "include/trt_runner.hpp"

#include <opencv2/core/mat.hpp>

#include <memory>

struct CudaPipelineTimings {
    float uploadMs{}, preprocessMs{}, inferenceMs{}, postprocessMs{}, downloadMs{};
};

// Owns persistent GPU scratch and pinned staging. Calls enqueue onto one external
// stream, whose lifetime must exceed this object. Not thread-safe.
class CudaYoloOps {
public:
    explicit CudaYoloOps(cudaStream_t stream);

    ~CudaYoloOps();

    CudaYoloOps(const CudaYoloOps&) = delete;
    CudaYoloOps& operator=(const CudaYoloOps&) = delete;
    void upload(const cv::Mat& bgr);

    void preprocess(float* deviceInput);

    void postprocess(const float* deviceOutput);

    std::vector<Detection> download();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class CudaYoloPipeline {
public:
    explicit CudaYoloPipeline(
        const std::string& enginePath,
        bool graph = true
    );

    std::vector<Detection> infer(
        const cv::Mat& bgr,
        CudaPipelineTimings* timings = nullptr
    );

private:
    TrtRunner runner_;
    CudaYoloOps ops_;
    bool graph_;
    struct Events;
    std::shared_ptr<Events> events_;
};
