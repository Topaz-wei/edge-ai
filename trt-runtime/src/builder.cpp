// 本文件实现与模型无关的 TensorRT 构建流程；只接受普通 FP32 I/O 和显式 batch。
#include "edge/trt/builder.hpp"
#include <NvOnnxParser.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>

namespace edge::trt {
namespace {

void require(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

nvinfer1::Dims dimensions(const std::vector<int>& shape) {
    nvinfer1::Dims result{};
    result.nbDims = static_cast<int>(shape.size());
    std::copy(
        shape.begin(),
        shape.end(),
        result.d
    );
    return result;
}

void checkTensor(
    nvinfer1::ITensor& tensor,
    const std::vector<int>& shape
) {
    const auto dims = tensor.getDimensions();
    require(
        dims.nbDims == static_cast<int>(shape.size()),
        "ONNX rank differs from contract"
    );
    for (int index = 0; index < dims.nbDims; ++index) {
        require(
            dims.d[index] == shape[index],
            "ONNX dimensions differ from contract: " + std::string(tensor.getName())
        );
    }
    require(
        tensor.getType() == nvinfer1::DataType::kFLOAT && !tensor.isShapeTensor(),
        "Only FP32 execution I/O supported"
    );
    tensor.setAllowedFormats(1U << static_cast<unsigned>(nvinfer1::TensorFormat::kLINEAR));
}

}

void validateBuildOptions(const BuildOptions& options) {
    require(
        !options.onnxPath.empty() && !options.enginePath.empty() && options.onnxPath != options.enginePath,
        "Distinct ONNX and engine paths are required"
    );
    require(
        options.workspaceBytes > 0 && !options.inputs.empty() && !options.outputs.empty(),
        "Workspace and I/O contracts must be nonempty"
    );
    require(
        options.precision == Precision::FP32 || options.precision == Precision::FP16 ||
        options.precision == Precision::INT8,
        "Unknown precision"
    );
    require(
        options.precision != Precision::INT8 || options.calibrator != nullptr,
        "INT8 requires an explicit calibrator"
    );
    require(
        options.precision == Precision::INT8 || options.calibrator == nullptr,
        "Calibrator supplied for a non-INT8 build"
    );
    std::set<std::string> names;
    for (const auto& input : options.inputs) {
        const auto rank = input.shape.size();
        require(
            !input.name.empty() && names.insert(input.name).second && rank > 0 &&
            rank <= nvinfer1::Dims::MAX_DIMS,
            "Invalid input name or rank"
        );
        require(
            input.minimum.size() == rank && input.optimum.size() == rank && input.maximum.size() == rank,
            "Profile rank mismatch"
        );
        if (options.precision == Precision::INT8) {
            require(
                input.calibration.size() == rank,
                "INT8 requires explicit calibration dimensions"
            );
        }
        for (std::size_t index = 0; index < rank; ++index) {
            const auto fixed = input.shape[index];
            require(
                (fixed == -1 || fixed > 0) && input.minimum[index] > 0 &&
                input.minimum[index] <= input.optimum[index] && input.optimum[index] <= input.maximum[index],
                "Invalid profile dimensions/order"
            );
            require(
                fixed == -1 || (input.minimum[index] == fixed && input.maximum[index] == fixed),
                "Profile changes a static dimension"
            );
            if (options.precision == Precision::INT8) {
                require(
                    input.calibration[index] >= input.minimum[index] &&
                    input.calibration[index] <= input.maximum[index],
                    "Calibration shape outside execution profile"
                );
            }
        }
    }
    for (const auto& output : options.outputs) {
        require(
            !output.name.empty() && names.insert(output.name).second && !output.shape.empty() &&
            output.shape.size() <= nvinfer1::Dims::MAX_DIMS,
            "Invalid output name or rank"
        );
        for (int dim : output.shape) {
            require(
                dim == -1 || dim > 0,
                "Invalid output shape"
            );
        }
    }
}

void buildEngine(
    const BuildOptions& options,
    nvinfer1::ILogger& logger
) {
    validateBuildOptions(options);
    const std::filesystem::path output(options.enginePath);
    const auto temporary = options.enginePath + ".tmp";
    require(
        !std::filesystem::exists(output) && !std::filesystem::exists(temporary),
        "Engine or temporary output exists; choose a new path"
    );
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        throw std::runtime_error("Cannot create TensorRT builder");
    }
    const auto flags = 1U << static_cast<unsigned>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!network || !config) {
        throw std::runtime_error("Cannot create network/config");
    }
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(
        *network,
        logger
    ));
    if (!parser || !parser->parseFromFile(
        options.onnxPath.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)
    )) {
        throw std::runtime_error("ONNX parse failed; see parser log");
    }
    require(
        network->getNbInputs() == static_cast<int>(options.inputs.size()) &&
        network->getNbOutputs() == static_cast<int>(options.outputs.size()),
        "ONNX I/O count differs from contract"
    );
    // 按名字匹配契约；不使用输出顺序解释模型语义。
    for (int index = 0; index < network->getNbInputs(); ++index) {
        auto& tensor = *network->getInput(index);
        const auto found = std::find_if(
            options.inputs.begin(),
            options.inputs.end(),
            [&](const BuildInput& input) { return input.name == tensor.getName(); }
        );
        require(
            found != options.inputs.end(),
            "Unexpected input name"
        );
        checkTensor(
            tensor,
            found->shape
        );
    }
    for (int index = 0; index < network->getNbOutputs(); ++index) {
        auto& tensor = *network->getOutput(index);
        const auto found = std::find_if(
            options.outputs.begin(),
            options.outputs.end(),
            [&](const BuildOutput& value) { return value.name == tensor.getName(); }
        );
        require(
            found != options.outputs.end(),
            "Unexpected output name"
        );
        checkTensor(
            tensor,
            found->shape
        );
    }
    config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE,
        options.workspaceBytes
    );
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    config->clearFlag(nvinfer1::BuilderFlag::kFP16);
    config->clearFlag(nvinfer1::BuilderFlag::kINT8);
    if (options.tf32) {
        config->setFlag(nvinfer1::BuilderFlag::kTF32);
    }
    if (options.precision == Precision::FP16 ||
        (options.precision == Precision::INT8 && options.int8Fp16Fallback)) {
        require(
            builder->platformHasFastFp16(),
            "Fast FP16 unavailable"
        );
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (options.precision == Precision::INT8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(options.calibrator);
    }
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    // profile 由 builder 持有，生命周期覆盖 config 和序列化过程。
    auto* profile = builder->createOptimizationProfile();
    if (!profile) {
        throw std::runtime_error("Cannot create optimization profile");
    }
    for (const auto& input : options.inputs) {
        const std::array<std::vector<int>, 3> shapes{input.minimum, input.optimum, input.maximum};
        for (int index = 0; index < 3; ++index) {
            require(
                profile->setDimensions(
                    input.name.c_str(),
                    static_cast<nvinfer1::OptProfileSelector>(index),
                    dimensions(shapes[index])
                ),
                "Cannot set optimization profile"
            );
        }
    }
    require(
        profile->isValid() && config->addOptimizationProfile(profile) == 0,
        "Cannot add optimization profile"
    );
    if (options.precision == Precision::INT8) {
        auto* calibration = builder->createOptimizationProfile();
        if (!calibration) {
            throw std::runtime_error("Cannot create calibration profile");
        }
        for (const auto& input : options.inputs) {
            for (int index = 0; index < 3; ++index) {
                require(
                    calibration->setDimensions(
                        input.name.c_str(),
                        static_cast<nvinfer1::OptProfileSelector>(index),
                        dimensions(input.calibration)
                    ),
                    "Cannot set calibration profile"
                );
            }
        }
        require(
            config->setCalibrationProfile(calibration),
            "Cannot attach calibration profile"
        );
    }
    std::unique_ptr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(
        *network,
        *config
    ));
    if (!serialized) {
        throw std::runtime_error("TensorRT engine build failed");
    }
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream file(
        temporary,
        std::ios::binary
    );
    file.write(
        static_cast<const char*>(serialized->data()),
        serialized->size()
    );
    file.close();
    if (!file) {
        throw std::runtime_error("Cannot save serialized engine");
    }
    std::filesystem::rename(
        temporary,
        output
    );
}

}
