#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>


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


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <model.onnx>"
            << " <output.plan>"
            << " <fp32|fp16>\n";

        return 1;
    }

    const std::string onnxPath = argv[1];
    const std::string enginePath = argv[2];
    const std::string precision = argv[3];

    if (precision != "fp32" && precision != "fp16") {
        std::cerr
            << "Precision must be fp32 or fp16\n";
        return 1;
    }

    // ------------------------------------------------
    // 1. Create TensorRT builder
    // ------------------------------------------------

    nvinfer1::IBuilder* builder =
        nvinfer1::createInferBuilder(gLogger);

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

    nvinfer1::INetworkDefinition* network =
        builder->createNetworkV2(explicitBatch);

    if (!network) {
        throw std::runtime_error(
            "Failed to create TensorRT network"
        );
    }

    // ------------------------------------------------
    // 3. Create ONNX parser
    // ------------------------------------------------

    nvonnxparser::IParser* parser =
        nvonnxparser::createParser(
            *network,
            gLogger
        );

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
    // 4. Inspect network
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

    // ------------------------------------------------
    // 5. Builder configuration
    // ------------------------------------------------

    nvinfer1::IBuilderConfig* config =
        builder->createBuilderConfig();

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
    } else {
        std::cout
            << "Precision: FP32 / TF32 default\n";
    }

    // ------------------------------------------------
    // 6. Build serialized TensorRT engine
    // ------------------------------------------------

    std::cout << "\nBuilding engine...\n";

    nvinfer1::IHostMemory* serializedEngine =
        builder->buildSerializedNetwork(
            *network,
            *config
        );

    if (!serializedEngine) {
        throw std::runtime_error(
            "TensorRT engine build failed"
        );
    }

    // ------------------------------------------------
    // 7. Save .plan
    // ------------------------------------------------

    std::ofstream file(
        enginePath,
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

    std::cout
        << "\nEngine saved: "
        << enginePath
        << '\n';

    std::cout
        << "Engine size : "
        << serializedEngine->size()
            / (1024.0 * 1024.0)
        << " MiB\n";

    // ------------------------------------------------
    // Cleanup
    // ------------------------------------------------

    serializedEngine->destroy();
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

    return 0;
}