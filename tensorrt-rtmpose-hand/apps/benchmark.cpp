// 本程序对两个独立 engine 进行交替性能测量；读取与初始化不计时，保留全部原始样本。
// GPU 计时复用 Runner 的 CUDA events；CPU 阶段与端到端时间使用 steady_clock。
#include "edge/trt/runner.hpp"
#include "rtmpose/reference.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

using Clock = std::chrono::steady_clock;

struct Sample {
    cv::Mat image;
    rtmpose::Region region;
    std::vector<float> tensor;
};

struct Batch {
    std::vector<std::size_t> indices;
    std::vector<rtmpose::Region> regions;
    std::vector<float> tensor;
};

struct Measurement {
    double preprocessMs = 0;
    double decodeMs = 0;
    double hostMs = 0;
    edge::trt::TrtTimings device;
};

double elapsed(
    Clock::time_point start,
    Clock::time_point end
) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Measurement measure(
    edge::trt::Runner& runner,
    const std::vector<Sample>& samples,
    const Batch& batch,
    bool imageMode
) {
    Measurement result;
    std::vector<float> prepared;
    const auto start = Clock::now();
    if (imageMode) {
        prepared.reserve(batch.tensor.size());
        for (auto index : batch.indices) {
            const auto tensor = rtmpose::preprocess(
                samples[index].image,
                samples[index].region
            );
            prepared.insert(
                prepared.end(),
                tensor.begin(),
                tensor.end()
            );
        }
    }
    const auto inputReady = Clock::now();
    const auto outputs = runner.inferNamed(
        imageMode ? prepared : batch.tensor,
        &result.device
    );
    const auto inferenceDone = Clock::now();
    const auto decoded = rtmpose::decode(
        outputs.at("simcc_x"),
        outputs.at("simcc_y"),
        batch.regions
    );
    const auto end = Clock::now();
    result.preprocessMs = imageMode ? elapsed(start, inputReady) : 0;
    result.decodeMs = elapsed(
        inferenceDone,
        end
    );
    result.hostMs = elapsed(
        start,
        end
    );
    // 结果检查置于计时之后；全量计算与解码均真实执行。
    for (double value : decoded) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Nonfinite decoded result");
        }
    }
    return result;
}

int main(
    int argc,
    char** argv
) {
    try {
        if (argc != 5 && argc != 6) {
            throw std::runtime_error("Usage: rtmpose_benchmark fp32.engine fp16.engine cases.json output.csv [--smoke]");
        }
        const bool smoke = argc == 6 && std::string(argv[5]) == "--smoke";
        if (argc == 6 && !smoke) {
            throw std::runtime_error("Unknown option");
        }
        const int warmup = smoke ? 1 : 100;
        const int iterations = smoke ? 3 : 1000;
        const int rounds = smoke ? 1 : 3;
        if (std::filesystem::exists(argv[4])) {
            throw std::runtime_error("Output exists; choose a new path");
        }
        cv::setNumThreads(1);
        cv::FileStorage manifest(
            argv[3],
            cv::FileStorage::READ
        );
        if (!manifest.isOpened()) {
            throw std::runtime_error("Cannot open cases");
        }
        std::vector<Sample> samples;
        for (const auto& node : manifest["samples"]) {
            std::vector<double> bbox;
            node["bbox"] >> bbox;
            if (bbox.size() != 4) {
                throw std::runtime_error("Invalid bbox");
            }
            Sample sample;
            sample.image = cv::imread(static_cast<std::string>(node["image"]));
            sample.region = rtmpose::makeRegion({bbox[0], bbox[1], bbox[2], bbox[3]});
            sample.tensor = rtmpose::loadNpy(static_cast<std::string>(node["tensor"]));
            if (sample.image.empty()) {
                throw std::runtime_error("Cannot load image");
            }
            samples.push_back(std::move(sample));
        }
        if (samples.size() != 55) {
            throw std::runtime_error("Expected frozen 55 development hands");
        }
        std::array<std::unique_ptr<edge::trt::Runner>, 2> runners{
            std::make_unique<edge::trt::Runner>(argv[1]),
            std::make_unique<edge::trt::Runner>(argv[2])
        };
        const std::array<std::string, 2> precisions{"fp32", "fp16"};
        std::ofstream file(argv[4]);
        if (!file) {
            throw std::runtime_error("Cannot create benchmark output");
        }
        file << "round,batch,mode,precision,order,iteration,start_sample,preprocess_ms,h2d_ms,inference_ms,d2h_ms,decode_ms,e2e_ms\n";
        file << std::setprecision(9);
        for (int round = 0; round < rounds; ++round) {
            for (int count : {1, 2, 3, 4}) {
                // 固定 batch 内循环同一组确定性窗口，尾部按样本清单循环，不丢样本。
                std::vector<Batch> batches;
                for (std::size_t start = 0; start < samples.size(); ++start) {
                    Batch batch;
                    for (int offset = 0; offset < count; ++offset) {
                        const auto index = (start + offset) % samples.size();
                        batch.indices.push_back(index);
                        batch.regions.push_back(samples[index].region);
                        batch.tensor.insert(
                            batch.tensor.end(),
                            samples[index].tensor.begin(),
                            samples[index].tensor.end()
                        );
                    }
                    batches.push_back(std::move(batch));
                }
                for (const std::string mode : {"npy", "image"}) {
                    for (int order = 0; order < 2; ++order) {
                        const int engine = (round + order) % 2;
                        auto& runner = *runners[engine];
                        runner.setInputShape(nvinfer1::Dims4{count, 3, 256, 256});
                        // 预热与采样走同一计时和解码路径，避免事件开销口径不一致。
                        for (int iteration = 0; iteration < warmup; ++iteration) {
                            measure(
                                runner,
                                samples,
                                batches[iteration % batches.size()],
                                mode == "image"
                            );
                        }
                        std::vector<Measurement> measurements;
                        measurements.reserve(iterations);
                        for (int iteration = 0; iteration < iterations; ++iteration) {
                            measurements.push_back(measure(
                                runner,
                                samples,
                                batches[iteration % batches.size()],
                                mode == "image"
                            ));
                        }
                        // 写盘放在整组采样之后，避免每次 I/O 扰动下一次测量。
                        for (int iteration = 0; iteration < iterations; ++iteration) {
                            const auto& value = measurements[iteration];
                            file << round << ',' << count << ',' << mode << ',' << precisions[engine]
                                 << ',' << order << ',' << iteration << ',' << iteration % batches.size()
                                 << ',' << value.preprocessMs << ',' << value.device.h2dMs
                                 << ',' << value.device.inferenceMs << ',' << value.device.d2hMs
                                 << ',' << value.decodeMs << ',' << value.hostMs << '\n';
                        }
                        file.flush();
                        if (!file) {
                            throw std::runtime_error("Cannot write benchmark samples");
                        }
                        std::cout << "round=" << round << " batch=" << count << " mode=" << mode
                                  << " precision=" << precisions[engine] << " samples=" << iterations << std::endl;
                    }
                }
            }
        }
        std::cout << "Complete; warmup=" << warmup << " iterations=" << iterations << " rounds=" << rounds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
