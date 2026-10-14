#include "include/cuda_yolo.hpp"
#include "include/infer_raw.hpp"
#include "include/nvtx_range.hpp"

#include <stdexcept>

struct CudaYoloPipeline::Events {
    cudaEvent_t event[6]{};

    Events() {
        try {
            for (auto& e : event)
                checkCuda(
                    cudaEventCreate(&e),
                    "pipeline event"
                );
        } catch (...) {
            for (auto e : event)
                if (e)
                    cudaEventDestroy(e);

            throw;
        }
    }

    ~Events() {
        for (auto e : event)
            cudaEventDestroy(e);
    }
};

CudaYoloPipeline::CudaYoloPipeline(
    const std::string& enginePath,
    bool graph
)
    : runner_(enginePath),
      ops_(runner_.stream()),
      graph_(graph),
      events_(new Events) {
    if (!runner_.isFixedYolo())
        throw std::runtime_error("CUDA pipeline requires [1,3,640,640] -> [1,84,8400]");

    if (graph_)
        runner_.prepareGraph();
}

std::vector<Detection> CudaYoloPipeline::infer(
    const cv::Mat& bgr,
    CudaPipelineTimings* timings
) {
    const NvtxRange frameRange("YOLO Frame");

    const auto stream = runner_.stream();

    auto record = [&](int i) {
        if (timings)
            checkCuda(
                cudaEventRecord(
                    events_->event[i],
                    stream
                ),
                "record pipeline event"
            );
    };

    try {
        record(0);

        {
            const NvtxRange range("BGR Upload");

            ops_.upload(bgr);
        }

        record(1);

        {
            const NvtxRange range("CUDA Preprocess");

            ops_.preprocess(runner_.deviceInput());
        }

        record(2);

        {
            const NvtxRange range("TensorRT Inference");

            runner_.enqueueDevice(graph_);
        }

        record(3);

        {
            const NvtxRange range("CUDA Postprocess");

            ops_.postprocess(runner_.deviceOutput());
        }

        record(4);

        std::vector<Detection> detections;

        {
            const NvtxRange range("Results Download");

            detections = ops_.download();
        }

        record(5);

        if (timings) {
            checkCuda(
                cudaEventSynchronize(events_->event[5]),
                "pipeline event sync"
            );

            float* values[] = {
                &timings->uploadMs,
                &timings->preprocessMs,
                &timings->inferenceMs,
                &timings->postprocessMs,
                &timings->downloadMs
            };

            for (int i = 0; i < 5; ++i)
                checkCuda(
                    cudaEventElapsedTime(
                        values[i],
                        events_->event[i],
                        events_->event[i + 1]
                    ),
                    "pipeline elapsed"
                );
        }

        return detections;
    } catch (...) {
        cudaStreamSynchronize(stream);

        throw;
    }
}
