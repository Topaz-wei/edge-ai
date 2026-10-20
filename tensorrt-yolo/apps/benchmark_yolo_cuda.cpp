#include "include/yolo_engine.hpp"
#include "include/cuda_yolo.hpp"
#include "include/yolo_preprocess.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int warmup = 50, iterations = 200;

double ms(
    Clock::time_point a,
    Clock::time_point b
) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void requireEqual(
    const std::vector<Detection>& a,
    const std::vector<Detection>& b
) {
    if (a.size() != b.size())
        throw std::runtime_error("Detection count changed");

    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto& x = a[i];
        const auto& y = b[i];

        if (x.candidate != y.candidate || x.classId != y.classId ||
            x.confidence != y.confidence || x.modelBox.x1 != y.modelBox.x1 ||
            x.modelBox.y1 != y.modelBox.y1 || x.modelBox.x2 != y.modelBox.x2 ||
            x.modelBox.y2 != y.modelBox.y2 || x.businessBox.x1 != y.businessBox.x1 ||
            x.businessBox.y1 != y.businessBox.y1 ||
            x.businessBox.x2 != y.businessBox.x2 ||
            x.businessBox.y2 != y.businessBox.y2)
            throw std::runtime_error("Detection differs from CPU reference");
    }
}
}

int main(
    int argc,
    char** argv
) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine.plan> <business_image> <cpu|cuda|graph>\n";

        return 1;
    }

    try {
        const std::string mode = argv[3];

        if (mode != "cpu" && mode != "cuda" && mode != "graph")
            throw std::runtime_error("Invalid mode");

        auto image = cv::imread(
            argv[2],
            cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION
        );

        const auto tensor = preprocess(image);

        std::vector<Detection> reference;
        { // Independent CPU reference; its engine is destroyed before measurement.
            TrtRunner referenceRunner(argv[1]);

            if (!isFixedYolo(referenceRunner))
                throw std::runtime_error("Expected fixed640 YOLO engine");

            reference = postprocess(
                referenceRunner.infer(tensor),
                false
            );
        }

        std::unique_ptr<TrtRunner> cpu;
        std::unique_ptr<CudaYoloPipeline> gpu;

        if (mode == "cpu")
            cpu.reset(new TrtRunner(argv[1]));
        else
            gpu.reset(new CudaYoloPipeline(
                argv[1],
                mode == "graph"
            ));

        std::array<std::vector<double>, 6> samples;

        for (auto& s : samples)
            s.reserve(iterations);

        auto run = [&](std::array<double, 6>& times) {
            if (cpu) {
                auto start = Clock::now();

                auto input = preprocess(image);

                auto afterPre = Clock::now();

                TrtTimings t;
                auto output = cpu->infer(
                    input,
                    &t
                );

                auto beforePost = Clock::now();

                auto detections = postprocess(
                    output,
                    false
                );

                auto end = Clock::now();

                times = {
                    ms(start,
                        afterPre),
                    t.h2dMs,
                    t.inferenceMs,
                    t.d2hMs,
                    ms(beforePost,
                        end),
                    0
                };

                return detections;
            }

            CudaPipelineTimings t;
            auto detections = gpu->infer(
                image,
                &t
            );

            times = {
                t.uploadMs,
                t.preprocessMs,
                t.inferenceMs,
                t.postprocessMs,
                t.downloadMs,
                0
            };

            return detections;
        };
        std::vector<Detection> last;

        for (int i = 0; i < warmup; ++i) {
            std::array<double, 6> t{};
            last = run(t);
        }

        requireEqual(
            reference,
            last
        );

        for (int i = 0; i < iterations; ++i) {
            std::array<double, 6> t{};
            auto start = Clock::now();

            auto detections = run(t);

            t[5] =
                ms(start,
                    Clock::now());

            for (int j = 0; j < 6; ++j)
                samples[j].push_back(t[j]);

            if (i == iterations - 1)
                last = std::move(detections);
        }

        requireEqual(
            reference,
            last
        );

        const std::array<const char*, 6> cpuNames = {
            "Preprocess CPU",
            "H2D event",
            "Inference event",
            "D2H event",
            "Postprocess CPU",
            "E2E wall"
        };
        const std::array<const char*, 6> gpuNames = {
            "BGR upload span",
            "Preprocess event",
            "Inference event",
            "Postprocess event",
            "Results download span",
            "E2E wall"
        };
        const auto& names = cpu ? cpuNames : gpuNames;
        std::cout
            << "Engine: " << argv[1] << "\nMode: " << mode
            << "\nWarmup: 50\nIterations: 200\n"
            << "GPU name/stats depend on current clocks and load; no frequency changes applied.\n"
            << std::fixed << std::setprecision(3) << std::left << std::setw(25)
            << "Stage (ms)" << std::right << std::setw(12) << "Mean" << std::setw(12)
            << "P50" << std::setw(12) << "P95" << '\n';
        double meanE2E = 0;

        for (int j = 0; j < 6; ++j) {
            auto& s = samples[j];
            const double mean = std::accumulate(
                                    s.begin(),
                                    s.end(),
                                    0.0
                                ) /
                                s.size();

            std::sort(
                s.begin(),
                s.end()
            );

            std::cout << std::left << std::setw(25) << names[j] << std::right
                      << std::setw(12) << mean << std::setw(12) << s[99]
                      << std::setw(12) << s[189] << '\n';

            if (j == 5)
                meanE2E = mean;
        }

        std::cout
            << "FPS (1000/mean E2E): " << 1000.0 / meanE2E << '\n'
            << "E2E includes staging, copies, GPU work, waits and per-frame allocations.\n"
            << "Excludes decode, initialization, warmup, validation, logging and drawing.\n"
            << "Upload/download event spans include host staging or count round-trip gaps.\n"
            << "Graph covers TensorRT only; GPU transfers BGR + selected detections, not raw tensors.\n"
            << "Final detections match CPU reference exactly.\n";
        std::cout << std::setprecision(6);

        printDetections(last);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';

        return 1;
    }
}
