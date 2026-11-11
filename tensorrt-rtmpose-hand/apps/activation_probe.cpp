// 独立诊断入口：通过公共 C++ builder 构建增加中间输出的 ONNX，并按样本保存具名激活。
// 不改变冻结 runtime 或模型预处理；原始二进制为 FP32，形状由探针配置记录。
#include "edge/trt/builder.hpp"
#include "edge/trt/runner.hpp"
#include "rtmpose/reference.hpp"
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

class ProbeLogger : public nvinfer1::ILogger {
public:
    void log(
        Severity severity,
        const char* message
    ) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cerr << message << '\n';
        }
    }
};

int main(
    int argc,
    char** argv
) {
    try {
        if (argc != 5) {
            throw std::runtime_error("Usage: rtmpose_probe build-fp32|build-fp16|run config.json engine output_dir");
        }
        const std::string action(argv[1]);
        cv::FileStorage config(
            argv[2],
            cv::FileStorage::READ
        );
        if (!config.isOpened()) {
            throw std::runtime_error("Cannot read probe config");
        }
        if (action == "build-fp32" || action == "build-fp16") {
            edge::trt::BuildOptions options;
            options.onnxPath = static_cast<std::string>(config["onnx"]);
            options.enginePath = argv[3];
            options.precision = action == "build-fp32" ? edge::trt::Precision::FP32 : edge::trt::Precision::FP16;
            options.workspaceBytes = 2ULL * 1024 * 1024 * 1024;
            options.tf32 = false;
            options.inputs = {{"input", {-1, 3, 256, 256}, {1, 3, 256, 256}, {2, 3, 256, 256}, {4, 3, 256, 256}, {}}};
            for (const auto& node : config["outputs"]) {
                edge::trt::BuildOutput output;
                output.name = static_cast<std::string>(node["name"]);
                node["shape"] >> output.shape;
                options.outputs.push_back(output);
            }
            ProbeLogger logger;
            edge::trt::buildEngine(
                options,
                logger
            );
            std::cout << "Probe engine saved: " << argv[3] << '\n';
            return 0;
        }
        if (action != "run") {
            throw std::runtime_error("Unknown action");
        }
        const std::filesystem::path folder(argv[4]);
        if (std::filesystem::exists(folder)) {
            throw std::runtime_error("Refusing to overwrite probe outputs");
        }
        std::filesystem::create_directories(folder);
        edge::trt::Runner runner(argv[3]);
        runner.setInputShape({4, {1, 3, 256, 256}});
        std::size_t count = 0;
        for (const auto& sample : config["samples"]) {
            const auto input = rtmpose::loadNpy(static_cast<std::string>(sample["tensor"]));
            const auto result = runner.inferNamed(input);
            if (result.size() != config["outputs"].size()) {
                throw std::runtime_error("Probe output count mismatch");
            }
            for (const auto& output : config["outputs"]) {
                const std::string name = output["name"];
                std::vector<int> shape;
                output["shape"] >> shape;
                const auto& info = runner.tensorInfo(name);
                if (info.shape.nbDims != static_cast<int>(shape.size())) {
                    throw std::runtime_error("Probe rank mismatch");
                }
                for (std::size_t index = 0; index < shape.size(); ++index) {
                    if (info.shape.d[index] != (shape[index] == -1 ? 1 : shape[index])) {
                        throw std::runtime_error("Probe shape mismatch");
                    }
                }
                const std::string filename = static_cast<std::string>(sample["id"]) + "_" + static_cast<std::string>(output["file"]) + ".bin";
                std::ofstream stream(
                    folder / filename,
                    std::ios::binary
                );
                const auto& values = result.at(name);
                stream.write(
                    reinterpret_cast<const char*>(values.data()),
                    values.size() * sizeof(float)
                );
                stream.close();
                if (!stream) {
                    throw std::runtime_error("Cannot save activation");
                }
            }
            ++count;
        }
        std::ofstream complete(folder / "complete.txt");
        complete << count << '\n';
        complete.close();
        if (!complete) {
            throw std::runtime_error("Cannot save completion marker");
        }
        std::cout << "Saved " << count << " samples\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
