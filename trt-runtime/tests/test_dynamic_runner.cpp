// 本测试构造小型真实 TensorRT 网络，验证动态多输出及旧单输出接口。
#include "edge/trt/runner.hpp"
#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

class TestLogger : public nvinfer1::ILogger {
public:
    void log(
        Severity severity,
        const char* text
    ) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cerr << text << '\n';
        }
    }
};

int main(
    int argc,
    char** argv
) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Expected fixture engine path");
        }
        TestLogger logger;
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(1U));
        auto* input = network->addInput(
            "input",
            nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims2{-1, 3}
        );
        auto* identity = network->addIdentity(*input);
        identity->getOutput(0)->setName("out_z");
        network->markOutput(*identity->getOutput(0));
        auto* negate = network->addUnary(
            *input,
            nvinfer1::UnaryOperation::kNEG
        );
        negate->getOutput(0)->setName("out_a");
        network->markOutput(*negate->getOutput(0));
        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
        for (int index = 0; index < 2; ++index) {
            auto* profile = builder->createOptimizationProfile();
            profile->setDimensions(
                "input",
                nvinfer1::OptProfileSelector::kMIN,
                nvinfer1::Dims2{1 + index, 3}
            );
            profile->setDimensions(
                "input",
                nvinfer1::OptProfileSelector::kOPT,
                nvinfer1::Dims2{2 + index, 3}
            );
            profile->setDimensions(
                "input",
                nvinfer1::OptProfileSelector::kMAX,
                nvinfer1::Dims2{4 + 2 * index, 3}
            );
            config->addOptimizationProfile(profile);
        }
        std::unique_ptr<nvinfer1::IHostMemory> engine(builder->buildSerializedNetwork(
            *network,
            *config
        ));
        if (!engine) {
            throw std::runtime_error("Fixture build failed");
        }
        std::ofstream file(
            argv[1],
            std::ios::binary
        );
        file.write(
            static_cast<const char*>(engine->data()),
            engine->size()
        );
        file.close();
        edge::trt::Runner runner(argv[1]);
        const auto expectRejected = [](auto action) {
            bool rejected = false;
            try {
                action();
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("Invalid operation was accepted");
            }
        };
        auto* address = runner.deviceTensor("out_z");
        const auto capacity = runner.tensorInfo("out_z").capacityBytes;
        for (int profile : {0, 1, 0}) {
            runner.selectProfile(profile);
            for (int batch : {2, 4, 3, 2}) {
                runner.setInputShape(nvinfer1::Dims2{batch, 3});
                std::vector<float> values(batch * 3);
                for (std::size_t index = 0; index < values.size(); ++index) {
                    values[index] = static_cast<float>(index) - 2.0F;
                }
                edge::trt::TrtTimings timings;
                const auto result = runner.inferNamed(
                    values,
                    &timings
                );
                if (result.size() != 2 || result.at("out_z") != values ||
                    runner.deviceTensor("out_z") != address ||
                    runner.tensorInfo("out_z").capacityBytes != capacity ||
                    runner.tensorInfo("out_z").activeBytes != values.size() * sizeof(float)) {
                    throw std::runtime_error("Named output or buffer capacity mismatch");
                }
                for (std::size_t index = 0; index < values.size(); ++index) {
                    if (result.at("out_a")[index] != -values[index]) {
                        throw std::runtime_error("Wrong named output contents");
                    }
                }
            }
        }
        expectRejected([&] { runner.outputInfo(); });
        expectRejected([&] { runner.outputElements(); });
        expectRejected([&] { runner.deviceOutput(); });
        expectRejected([&] { runner.infer({}); });
        expectRejected([&] { runner.inferNamed({}); });
        expectRejected([&] { runner.tensorInfo("missing"); });
        expectRejected([&] { runner.prepareGraph(); });
        expectRejected([&] { runner.selectProfile(2); });
        expectRejected([&] { runner.setInputShape(nvinfer1::Dims2{0, 3}); });
        expectRejected([&] { runner.setInputShape(nvinfer1::Dims2{5, 3}); });
        expectRejected([&] { runner.setInputShape(nvinfer1::Dims2{2, 4}); });
        runner.setInputShape(nvinfer1::Dims2{1, 3});
        runner.inferNamed(std::vector<float>(3, 1));
        runner.selectProfile(1);
        runner.setInputShape(nvinfer1::Dims2{6, 3});
        runner.inferNamed(std::vector<float>(18, 1));
        expectRejected([&] { runner.setInputShape(nvinfer1::Dims2{1, 3}); });
        nvinfer1::Dims badRank{};
        badRank.nbDims = nvinfer1::Dims::MAX_DIMS + 1;
        expectRejected([&] { runner.setInputShape(badRank); });
        for (int repetition = 0; repetition < 2; ++repetition) {
            edge::trt::Runner fresh(argv[1]);
            fresh.setInputShape(nvinfer1::Dims2{1, 3});
            fresh.inferNamed(std::vector<float>(3, 2));
        }
        input->setType(nvinfer1::DataType::kHALF);
        identity->getOutput(0)->setType(nvinfer1::DataType::kHALF);
        negate->getOutput(0)->setType(nvinfer1::DataType::kHALF);
        std::unique_ptr<nvinfer1::IHostMemory> halfEngine(builder->buildSerializedNetwork(
            *network,
            *config
        ));
        if (!halfEngine) {
            throw std::runtime_error("Half-I/O fixture build failed");
        }
        const std::string halfPath = std::string(argv[1]) + ".half";
        std::ofstream halfFile(
            halfPath,
            std::ios::binary
        );
        halfFile.write(
            static_cast<const char*>(halfEngine->data()),
            halfEngine->size()
        );
        halfFile.close();
        expectRejected([&] { edge::trt::Runner rejected(halfPath); });
        std::cout << "PASS: named outputs, batches, profiles, capacity, invalid inputs, legacy guards\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
