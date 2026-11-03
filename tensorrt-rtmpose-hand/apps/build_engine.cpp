// 本程序读取 RTMPose 构建配置并核验冻结资产，调用公共 C++ builder。
// 固定原始模型、profile 1/2/4 和 FP32 I/O；允许 FP16 fallback，禁止覆盖既有基线。
#include <NvInfer.h>
#include "edge/trt/builder.hpp"
#include <opencv2/core.hpp>
#include <openssl/evp.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

class Logger : public nvinfer1::ILogger {
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

std::string sha256(const fs::path& path) {
    std::ifstream file(
        path,
        std::ios::binary
    );
    if (!file) {
        throw std::runtime_error("Cannot open hash input: " + path.string());
    }
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
        EVP_MD_CTX_new(),
        EVP_MD_CTX_free
    );
    if (!context || EVP_DigestInit_ex(
        context.get(),
        EVP_sha256(),
        nullptr
    ) != 1) {
        throw std::runtime_error("Cannot initialize SHA256");
    }
    std::array<char, 65536> buffer{};
    while (file) {
        file.read(
            buffer.data(),
            buffer.size()
        );
        if (EVP_DigestUpdate(
            context.get(),
            buffer.data(),
            file.gcount()
        ) != 1) {
            throw std::runtime_error("Cannot update SHA256");
        }
    }
    unsigned char digest[EVP_MAX_MD_SIZE]{};
    unsigned int length = 0;
    if (!file.eof() || EVP_DigestFinal_ex(
        context.get(),
        digest,
        &length
    ) != 1) {
        throw std::runtime_error("Cannot finalize SHA256");
    }
    std::ostringstream hex;
    for (unsigned int index = 0; index < length; ++index) {
        hex << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(digest[index]);
    }
    return hex.str();
}

std::vector<int> readShape(const cv::FileNode& node) {
    std::vector<int> shape;
    node >> shape;
    if (shape.size() != 4) {
        throw std::runtime_error("Expected four profile dimensions");
    }
    return shape;
}

int main(
    int argc,
    char** argv
) {
    try {
        if (argc != 4 && argc != 5) {
            throw std::runtime_error("Usage: rtmpose_build module_dir config.json output.engine [--check-assets]");
        }
        const fs::path module(argv[1]);
        const fs::path configPath(argv[2]);
        const bool checkOnly = argc == 5 && std::string(argv[4]) == "--check-assets";
        if (argc == 5 && !checkOnly) {
            throw std::runtime_error("Unknown option");
        }
        cv::FileStorage settings(
            configPath.string(),
            cv::FileStorage::READ
        );
        if (!settings.isOpened()) {
            throw std::runtime_error("Cannot read build config");
        }
        edge::trt::BuildOptions options;
        options.onnxPath = (module / "models/rtmpose_hand.onnx").string();
        options.enginePath = argv[3];
        const std::string precision = settings["selected_precision"];
        if (precision == "fp32") {
            options.precision = edge::trt::Precision::FP32;
        } else if (precision == "fp16") {
            options.precision = edge::trt::Precision::FP16;
        } else if (precision == "int8") {
            options.precision = edge::trt::Precision::INT8;
        } else {
            throw std::runtime_error("Unknown precision");
        }
        const int workspaceMiB = settings["workspace_mib"];
        if (workspaceMiB <= 0 || workspaceMiB > 65536) {
            throw std::runtime_error("Invalid workspace MiB");
        }
        options.workspaceBytes = static_cast<std::size_t>(workspaceMiB) * 1024 * 1024;
        if (settings["tf32"].empty() || static_cast<int>(settings["tf32"]) != 0) {
            throw std::runtime_error("RTMPose baseline requires TF32 off");
        }
        const std::string inputName = settings["input"];
        if (inputName != "input") {
            throw std::runtime_error("Unexpected model input");
        }
        const auto profile = settings["profile"];
        options.inputs = {{inputName, {-1, 3, 256, 256}, readShape(profile["min"]),
                           readShape(profile["opt"]), readShape(profile["max"]), {}}};
        options.outputs = {{"simcc_x", {-1, 21, 512}}, {"simcc_y", {-1, 21, 512}}};
        // INT8 选项已保留，模型校准器尚待 INT8 阶段接入；公共检查会拒绝空校准器。
        edge::trt::validateBuildOptions(options);
        if (sha256(options.onnxPath) != "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29" ||
            sha256(module / "configs/rtmpose_hand_reference.json") !=
            "3036459e1f6bc230c9059f34ea260070928669f54f9b1bfef3e031279aef0aec") {
            throw std::runtime_error("Frozen model/reference hash mismatch");
        }
        std::cout << "Assets verified; precision=" << precision << "; TF32=off; workspace_mib="
                  << workspaceMiB << "; config_sha256=" << sha256(configPath) << std::endl;
        if (checkOnly) {
            return 0;
        }
        Logger logger;
        edge::trt::buildEngine(
            options,
            logger
        );
        std::cout << "Engine saved: " << options.enginePath << "; sha256=" << sha256(options.enginePath) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
