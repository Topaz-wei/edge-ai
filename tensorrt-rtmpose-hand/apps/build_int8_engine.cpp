// RTMPose INT8 独立入口：数据核验、仅供给测试或实际构建；保留旧 FP32/FP16 构建入口。
#include "edge/trt/builder.hpp"
#include "rtmpose/calibrator.hpp"
#include "rtmpose/reference.hpp"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>

class Logger : public nvinfer1::ILogger {
public:
    void log(
        Severity severity,
        const char* text
    ) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cerr << text << '\n';
        }
    }
};

int main(
    int argc,
    char** argv
) {
    try {
        if (argc != 4) {
            throw std::runtime_error("Usage: rtmpose_int8 audit|check-inputs|build module_dir output_file");
        }
        const std::string action(argv[1]);
        if (action != "audit" && action != "check-inputs" && action != "build") {
            throw std::runtime_error("Unknown action");
        }
        const std::filesystem::path module(argv[2]);
        const std::filesystem::path output(argv[3]);
        if (std::filesystem::exists(output)) {
            throw std::runtime_error("Output exists; choose another path");
        }
        auto data = rtmpose::auditCalibration(module);
        if (action == "audit") {
            std::ofstream report(output);
            report << data.provenanceJson;
            report.close();
            if (!report) {
                throw std::runtime_error("Cannot save audit report");
            }
            std::cout << "Verified 352 calibration tensors, disjoint from 55 development entries\n";
            return 0;
        }
        cv::FileStorage settings(
            (module / "configs/rtmpose_hand_builder_int8.json").string(),
            cv::FileStorage::READ
        );
        if (!settings.isOpened() || static_cast<std::string>(settings["selected_precision"]) != "int8" ||
            settings["fp16"].empty() || static_cast<int>(settings["fp16"]) != 0 ||
            settings["tf32"].empty() || static_cast<int>(settings["tf32"]) != 0 ||
            static_cast<int>(settings["workspace_mib"]) != 2048) {
            throw std::runtime_error("Expected INT8 on, FP16/TF32 off, workspace 2048 MiB");
        }
        edge::trt::BuildInput input;
        input.name = "input";
        input.shape = {-1, 3, 256, 256};
        settings["profile"]["min"] >> input.minimum;
        settings["profile"]["opt"] >> input.optimum;
        settings["profile"]["max"] >> input.maximum;
        settings["calibration_shape"] >> input.calibration;
        if (input.minimum != std::vector<int>({1, 3, 256, 256}) ||
            input.optimum != std::vector<int>({2, 3, 256, 256}) ||
            input.maximum != std::vector<int>({4, 3, 256, 256}) || input.calibration != input.minimum) {
            throw std::runtime_error("Expected inference 1/2/4, calibration 1/1/1");
        }
        const std::string cacheName = settings["cache"];
        const std::filesystem::path relativeCache(cacheName);
        if (relativeCache.parent_path() != std::filesystem::path("calibration") ||
            relativeCache.extension() != ".cache" ||
            static_cast<std::string>(settings["cache_meta"]) != cacheName + ".meta.json" ||
            static_cast<std::string>(settings["calibration_order"]) != "calibration/calibration_tensor_order.txt" ||
            static_cast<std::string>(settings["calibration_algorithm"]) != "IInt8EntropyCalibrator2") {
            throw std::runtime_error("Invalid RTMPose calibration/cache configuration");
        }
        // CPU audit不初始化驱动；进入GPU供给/构建阶段才补充实际CUDA runtime版本。
        int runtimeVersion = 0;
        if (cudaRuntimeGetVersion(&runtimeVersion) != cudaSuccess) {
            throw std::runtime_error("Cannot query CUDA runtime; GPU access is required");
        }
        data.provenanceJson.insert(
            data.provenanceJson.rfind('}'),
            ",\n\"cuda_runtime_version\":" + std::to_string(runtimeVersion) + "\n"
        );
        rtmpose::HandCalibrator calibrator(
            data,
            module / relativeCache
        );
        edge::trt::BuildOptions options;
        options.onnxPath = (module / "models/rtmpose_hand.onnx").string();
        options.enginePath = output.string() + ".building";
        options.precision = edge::trt::Precision::INT8;
        options.int8Fp16Fallback = false;
        options.tf32 = false;
        options.workspaceBytes = 2ULL * 1024 * 1024 * 1024;
        options.calibrator = &calibrator;
        options.inputs = {input};
        options.outputs = {{"simcc_x", {-1, 21, 512}}, {"simcc_y", {-1, 21, 512}}};
        edge::trt::validateBuildOptions(options);
        if (action == "check-inputs") {
            void* bindings[]{nullptr};
            const char* names[]{"input"};
            std::size_t count = 0;
            while (calibrator.getBatch(
                bindings,
                names,
                1
            )) {
                const auto expected = rtmpose::loadNpy(data.tensors.at(count).path.string());
                std::vector<float> actual(expected.size());
                if (cudaMemcpy(
                    actual.data(),
                    bindings[0],
                    actual.size() * sizeof(float),
                    cudaMemcpyDeviceToHost
                ) != cudaSuccess || actual != expected) {
                    throw std::runtime_error("Calibrator GPU input differs from frozen NPY");
                }
                ++count;
            }
            calibrator.throwIfFailed();
            if (count != 352 || calibrator.consumed() != 352) {
                throw std::runtime_error("Not all tensors consumed");
            }
            std::ofstream report(output);
            report << "{\"status\":\"passed\",\"tensors\":352,\"max_abs\":0,\"cache_generated\":false,\"engine_built\":false,\"identity\":" << data.provenanceJson << "}\n";
            report.close();
            if (!report) {
                throw std::runtime_error("Cannot write supply verification report");
            }
            std::cout << "352 GPU inputs match frozen NPY exactly; no calibration or engine build executed\n";
            return 0;
        }
        Logger logger;
        try {
            edge::trt::buildEngine(
                options,
                logger
            );
        } catch (...) {
            // builder先失败时也优先暴露校准回调的原始异常，避免只留下泛化构建错误。
            calibrator.throwIfFailed();
            throw;
        }
        // TensorRT 回调不能抛异常；必须在发布 engine 前检查完整消费及 cache。
        calibrator.verifyComplete();
        calibrator.saveCache();
        std::filesystem::rename(
            options.enginePath,
            output
        );
        std::cout << "INT8 engine saved: " << output << "; sha256=" << rtmpose::calibrationFileHash(output) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
