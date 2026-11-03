// 公共 ONNX engine 构建接口；不依赖模型预处理，校准器由调用方提供并持有。
#pragma once
#include <NvInfer.h>
#include <cstddef>
#include <string>
#include <vector>

namespace edge::trt {

enum class Precision { FP32, FP16, INT8 };

struct BuildInput {
    std::string name;
    std::vector<int> shape;
    std::vector<int> minimum;
    std::vector<int> optimum;
    std::vector<int> maximum;
    // INT8 动态输入的校准形状；由模型校准策略明确指定。
    std::vector<int> calibration;
};

struct BuildOutput {
    std::string name;
    std::vector<int> shape;
};

struct BuildOptions {
    std::string onnxPath;
    std::string enginePath;
    Precision precision = Precision::FP32;
    std::size_t workspaceBytes = 2ULL * 1024 * 1024 * 1024;
    bool tf32 = false;
    bool int8Fp16Fallback = true;
    std::vector<BuildInput> inputs;
    std::vector<BuildOutput> outputs;
    // 非拥有指针；对象及其 GPU 输入缓冲必须存活至 buildEngine 返回。
    nvinfer1::IInt8Calibrator* calibrator = nullptr;
};

// CPU 参数校验，便于在创建 GPU builder 前发现配置错误。
void validateBuildOptions(const BuildOptions& options);

// FP32 linear I/O；内部精度由策略决定。拒绝覆盖已有 engine。
void buildEngine(
    const BuildOptions& options,
    nvinfer1::ILogger& logger
);

}
