#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>
#include <filesystem>

#ifdef YOLO_ENABLE_INT8
#include "include/calibrator.hpp"
#endif


class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << '\n';
        }
    }
};


static Logger gLogger;


void printDims(const nvinfer1::Dims& dims) {
    std::cout << "[";

    for (int i = 0; i < dims.nbDims; ++i) {
        std::cout << dims.d[i];

        if (i + 1 != dims.nbDims) {
            std::cout << "x";
        }
    }

    std::cout << "]";
}


bool isFp16ComputeLayer(nvinfer1::ILayer* layer)
{
    using LT = nvinfer1::LayerType;

    switch (layer->getType()) {
        case LT::kCONVOLUTION:
        case LT::kACTIVATION:
        case LT::kELEMENTWISE:
        case LT::kSOFTMAX:
            break;

        default:
            return false;
    }

    // 只处理真正产生浮点执行 tensor 的层。
    bool hasFloatOutput = false;

    for (int j = 0; j < layer->getNbOutputs(); ++j) {
        auto* output = layer->getOutput(j);

        if (!output)
            continue;

        const auto type = output->getType();

        if (type == nvinfer1::DataType::kFLOAT ||
            type == nvinfer1::DataType::kHALF) {
            hasFloatOutput = true;
        }
    }

    return hasFloatOutput;
}


int buildEngine(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <model.onnx>"
            << " <output.plan>"
            << " <fp32|fp16|int8>"
            << " [calibration_directory] [calibration.cache]\n";

        return 1;
    }

    const std::string onnxPath = argv[1];
    const std::string enginePath = argv[2];
    const std::string precision = argv[3];

    if (precision != "fp32" && precision != "fp16" && precision != "int8") {
        std::cerr
            << "Precision must be fp32 or fp16 or int8\n";
        return 1;
    }

    if ((precision == "int8" && argc < 5) || (precision != "int8" && argc != 4)) {
        throw std::runtime_error(
            "INT8 requires calibration_directory and optional cache path; FP32/FP16 require 3 arguments"
        );
    }

    if (std::filesystem::exists(enginePath)) {
        throw std::runtime_error(
            "Engine already exists; choose a new output path: " + enginePath
        );
    }

#ifndef YOLO_ENABLE_INT8
    if (precision == "int8") {
        throw std::runtime_error("INT8 requires YOLO_BUILD_CUDA_PIPELINE=ON");
    }
#endif

    // ------------------------------------------------
    // 1. Create TensorRT builder
    // ------------------------------------------------

    std::unique_ptr<nvinfer1::IBuilder> builder{
        nvinfer1::createInferBuilder(gLogger)
    };

    if (!builder) {
        throw std::runtime_error(
            "Failed to create TensorRT builder"
        );
    }

    // ONNX uses explicit batch mode.
    const uint32_t explicitBatch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
        );

    // ------------------------------------------------
    // 2. Create TensorRT network
    // ------------------------------------------------

    std::unique_ptr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(explicitBatch)
    };

    if (!network) {
        throw std::runtime_error(
            "Failed to create TensorRT network"
        );
    }

    // ------------------------------------------------
    // 3. Builder configuration
    // ------------------------------------------------

    std::unique_ptr<nvinfer1::IBuilderConfig> config{
        builder->createBuilderConfig()
    };

    if (!config) {
        throw std::runtime_error(
            "Failed to create builder config"
        );
    }

    constexpr std::size_t workspaceSize =
        2ULL << 30;  // 2 GiB

    config->setMaxWorkspaceSize(
        workspaceSize
    );

    config->setProfilingVerbosity(
        nvinfer1::ProfilingVerbosity::kDETAILED
    );

    // ------------------------------------------------
    // 4. Create ONNX parser
    // ------------------------------------------------

    std::unique_ptr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(
            *network,
            gLogger
        )
    };

    if (!parser) {
        throw std::runtime_error(
            "Failed to create ONNX parser"
        );
    }

    std::cout
        << "Parsing ONNX: "
        << onnxPath
        << '\n';

    const bool parsed =
        parser->parseFromFile(
            onnxPath.c_str(),
            static_cast<int>(
                nvinfer1::ILogger::Severity::kWARNING
            )
        );

    if (!parsed) {
        std::cerr << "ONNX parsing failed\n";

        for (int i = 0;
             i < parser->getNbErrors();
             ++i) {

            std::cerr
                << parser->getError(i)->desc()
                << '\n';
        }

        return 1;
    }

    // ------------------------------------------------
    // 5. Inspect network
    // ------------------------------------------------

    std::cout << "\n=== Network ===\n";

    std::cout
        << "Layers : "
        << network->getNbLayers()
        << '\n';

    std::cout
        << "Inputs : "
        << network->getNbInputs()
        << '\n';

    std::cout
        << "Outputs: "
        << network->getNbOutputs()
        << '\n';

    for (int i = 0;
         i < network->getNbInputs();
         ++i) {

        auto* tensor =
            network->getInput(i);

        std::cout
            << "Input "
            << i
            << ": "
            << tensor->getName()
            << " ";

        printDims(
            tensor->getDimensions()
        );

        std::cout << '\n';
    }

    for (int i = 0;
         i < network->getNbOutputs();
         ++i) {

        auto* tensor =
            network->getOutput(i);

        std::cout
            << "Output "
            << i
            << ": "
            << tensor->getName()
            << " ";

        printDims(
            tensor->getDimensions()
        );

        std::cout << '\n';
    }

    if (precision == "fp16") {
        if (!builder->platformHasFastFp16()) {
            std::cerr
                << "Fast FP16 is not supported\n";
            return 1;
        }

        config->setFlag(
            nvinfer1::BuilderFlag::kFP16
        );

        std::cout
            << "Precision: FP16 enabled\n";
    } 
    
    if (precision == "fp32") {
        config->clearFlag(
            nvinfer1::BuilderFlag::kTF32
        );

        std::cout
            << "Precision: FP32\n"
            << "TF32 enabled: "
            << (config->getFlag(
                    nvinfer1::BuilderFlag::kTF32
                ) ? "YES" : "NO")
            << "\n";
    }

#ifdef YOLO_ENABLE_INT8
    std::unique_ptr<YOLOCalibrator> calibrator;

    if (precision == "int8") {
        if (!builder->platformHasFastInt8()) {
            std::cerr << "Fast INT8 is not supported\n";
            return 1;
        }

        int forcedFp16Layers = 0;

        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto* layer = network->getLayer(i);

            const std::string name =
                layer->getName() ? layer->getName() : "";

            if (name.find("/model.22/") == std::string::npos)
                continue;

            if (!isFp16ComputeLayer(layer)) {
                std::cout
                    << "[FP16 skip] "
                    << i << " : "
                    << name << '\n';

                continue;
            }

            layer->setPrecision(
                nvinfer1::DataType::kHALF
            );

            std::cout
                << "[FP16 constraint] "
                << i << " : "
                << name << '\n';

            ++forcedFp16Layers;
        }

        if (forcedFp16Layers == 0) {
            throw std::runtime_error(
                "No floating-point model.22 layers matched"
            );
        }

        config->setFlag(
            nvinfer1::BuilderFlag::
                kOBEY_PRECISION_CONSTRAINTS
        );

        if (network->getNbInputs() != 1 || network->getNbOutputs() != 1) {
            throw std::runtime_error("INT8 calibration requires one YOLO input/output");
        }

        const auto* input = network->getInput(0);
        const auto* output = network->getOutput(0);
        const auto in = input->getDimensions();
        const auto out = output->getDimensions();

        if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3 || in.d[2] != 640 ||
            in.d[3] != 640 || out.nbDims != 3 || out.d[0] != 1 || out.d[1] != 84 ||
            out.d[2] != 8400 || input->getType() != nvinfer1::DataType::kFLOAT ||
            output->getType() != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Expected fixed640 YOLO FP32 input/output");
        }

        for (int i = 0; i < network->getNbLayers(); ++i) {
            const auto type = network->getLayer(i)->getType();
            if (type == nvinfer1::LayerType::kQUANTIZE ||
                type == nvinfer1::LayerType::kDEQUANTIZE) {
                throw std::runtime_error(
                    "This calibration path expects ONNX without explicit Q/DQ"
                );
            }
        }

        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);

        const std::string cachePath = argc == 6 ? argv[5] : enginePath + ".cache";
        calibrator = std::make_unique<YOLOCalibrator>(
            argv[4],
            onnxPath,
            input->getName(),
            cachePath
        );
        config->setInt8Calibrator(calibrator.get());

        std::cout << "Precision: INT8 PTQ with FP16 fallback; FP32 I/O\n";
    }
#endif

    // ------------------------------------------------
    // 6. Build serialized TensorRT engine
    // ------------------------------------------------

    std::cout << "\nBuilding engine...\n";

    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine{
        builder->buildSerializedNetwork(
            *network,
            *config
        )
    };

#ifdef YOLO_ENABLE_INT8
    if (calibrator)
        calibrator->throwIfFailed();
#endif

    if (!serializedEngine) {
        throw std::runtime_error(
            "TensorRT engine build failed"
        );
    }

    // ------------------------------------------------
    // 7. Save .plan
    // ------------------------------------------------

#ifdef YOLO_ENABLE_INT8
    if (calibrator) {
        calibrator->verifyComplete();
        calibrator->saveCache();
    }
#endif

    const auto parent = std::filesystem::path(enginePath).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    const auto temporaryPath = enginePath + ".tmp";

    std::ofstream file(
        temporaryPath,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to open output file"
        );
    }

    file.write(
        static_cast<const char*>(
            serializedEngine->data()
        ),
        serializedEngine->size()
    );

    file.close();

    if (!file)
        throw std::runtime_error("Failed to write engine: " + temporaryPath);
    std::filesystem::rename(
        temporaryPath,
        enginePath
    );

    std::cout
        << "\nEngine saved: "
        << enginePath
        << '\n';

    std::cout
        << "Engine size : "
        << serializedEngine->size()
            / (1024.0 * 1024.0)
        << " MiB\n";

    return 0;
}

int main(
    int argc,
    char** argv
) {
    try {
        return buildEngine(
            argc,
            argv
        );
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
