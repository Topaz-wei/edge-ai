#include "include/trt_runner.hpp"
#include "include/yolo_preprocess.hpp"
#include "include/postprocess_raw.hpp"

#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int warmup = 50;
constexpr int iterations = 200;

double elapsed(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct Samples {
    std::vector<double> values;
    Samples() { values.reserve(iterations); }
    double mean() const {
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }
    void print(const char* name) const {
        auto sorted = values;
        std::sort(sorted.begin(), sorted.end());
        // Nearest-rank percentile: ceil(p * N), using one-based ranks.
        const auto percentile = [&](double p) {
            return sorted[static_cast<std::size_t>(std::ceil(p * sorted.size())) - 1];
        };
        std::cout << std::left << std::setw(22) << name << std::right
                  << std::setw(12) << mean() << std::setw(12) << percentile(0.50)
                  << std::setw(12) << percentile(0.95) << '\n';
    }
};
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine.plan> <business_image>\n";
        return 1;
    }
    try {
        // Decode and initialize once, outside warmup and measurement.
        const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        if (image.empty()) throw std::runtime_error("Cannot decode business image");
        // Validate geometry before initializing GPU resources.
        const auto referenceInput = preprocess(image);
        TrtRunner runner(argv[1]);
        if (runner.inputElements() != referenceInput.size() || runner.outputElements() != 84 * 8400) {
            throw std::runtime_error("Expected fixed640 YOLO input/output element counts");
        }
        const auto referenceOutput = runner.infer(referenceInput);
        const auto referenceDetections = postprocess(referenceOutput, false);
        if (std::any_of(referenceOutput.begin(), referenceOutput.end(),
                        [](float value) { return !std::isfinite(value); })) {
            throw std::runtime_error("Non-finite inference output");
        }
        for (int i = 0; i < warmup; ++i) {
            auto input = preprocess(image);
            TrtTimings timing;
            auto output = runner.infer(input, &timing);
            auto detections = postprocess(output, false);
        }

        Samples preprocessing, h2d, inference, d2h, postprocessing, e2e;
        Samples hostInfer;
        std::vector<float> lastOutput;
        std::vector<Detection> lastDetections;
        for (int i = 0; i < iterations; ++i) {
            const auto start = Clock::now();
            auto input = preprocess(image);
            const auto afterPreprocess = Clock::now();
            TrtTimings timing;
            auto output = runner.infer(input, &timing);
            const auto afterInfer = Clock::now();
            auto detections = postprocess(output, false);
            const auto end = Clock::now();

            preprocessing.values.push_back(elapsed(start, afterPreprocess));
            hostInfer.values.push_back(elapsed(afterPreprocess, afterInfer));
            h2d.values.push_back(timing.h2dMs);
            inference.values.push_back(timing.inferenceMs);
            d2h.values.push_back(timing.d2hMs);
            postprocessing.values.push_back(elapsed(afterInfer, end));
            e2e.values.push_back(elapsed(start, end));
            // Keep the final results observable; no logging/comparison inside timers.
            if (i == iterations - 1) {
                lastOutput = std::move(output);
                lastDetections = std::move(detections);
            }
        }
        if (lastOutput != referenceOutput || lastDetections.size() != referenceDetections.size()) {
            throw std::runtime_error("Benchmark output changed from reference inference");
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "YOLO TensorRT synchronous pipeline benchmark\n"
                  << "Engine     : " << argv[1] << '\n'
                  << "Image      : " << argv[2] << '\n'
                  << "Warmup     : " << warmup << '\n'
                  << "Iterations : " << iterations << '\n'
                  << "I/O        : FP32, pageable host vectors\n"
                  << "CPU stages: steady_clock; GPU stages: same-stream CUDA events\n\n"
                  << std::left << std::setw(22) << "Stage (ms)" << std::right
                  << std::setw(12) << "Mean" << std::setw(12) << "P50" << std::setw(12) << "P95" << '\n';
        preprocessing.print("Preprocess CPU");
        h2d.print("H2D event");
        inference.print("Inference event");
        d2h.print("D2H event");
        postprocessing.print("Postprocess CPU");
        hostInfer.print("Runner CPU wall");
        e2e.print("E2E CPU wall");
        std::cout << "\nFPS (1000/mean E2E): " << 1000.0 / e2e.mean() << '\n'
                  << "E2E excludes decode/init, logging, result comparison and final vector destruction.\n"
                  << "It includes host allocations, enqueue/sync and event instrumentation.\n"
                  << "Event spans may include stream idle/host submission gaps; they need not sum to E2E.\n"
                  << "Output unchanged after warmup/measurement.\n";
        std::cout << std::setprecision(6);
        printDetections(lastDetections);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
