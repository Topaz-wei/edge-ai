#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "int8_calibrator.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>


class Logger : public nvinfer1::ILogger {
public:
    void log(
        Severity severity,
        const char* msg
    ) noexcept override {

        if (severity <= Severity::kWARNING) {
            std::cout
                << "[TensorRT] "
                << msg
                << '\n';
        }
    }
};


static Logger gLogger;


int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <model.onnx>"
            << " <calibration.bin>"
            << " <calibration.cache>"
            << " <output.plan>\n";

        return 1;
    }

    try {
        const std::string onnxPath =
            argv[1];

        const std::string calibrationPath =
            argv[2];

        const std::string cachePath =
            argv[3];

        const std::string enginePath =
            argv[4];

        // --------------------------------
        // Builder
        // --------------------------------

        auto* builder =
            nvinfer1::createInferBuilder(
                gLogger
            );

        if (!builder) {
            throw std::runtime_error(
                "Failed to create builder"
            );
        }

        if (!builder->platformHasFastInt8()) {
            throw std::runtime_error(
                "Fast INT8 is not supported"
            );
        }

        const uint32_t explicitBatch =
            1U << static_cast<uint32_t>(
                nvinfer1::
                NetworkDefinitionCreationFlag::
                kEXPLICIT_BATCH
            );

        auto* network =
            builder->createNetworkV2(
                explicitBatch
            );

        auto* parser =
            nvonnxparser::createParser(
                *network,
                gLogger
            );

        if (!parser->parseFromFile(
                onnxPath.c_str(),
                static_cast<int>(
                    nvinfer1::ILogger::
                    Severity::kWARNING
                )
            )) {

            throw std::runtime_error(
                "Failed to parse ONNX"
            );
        }

        // --------------------------------
        // Verify static input
        // --------------------------------

        auto* input =
            network->getInput(0);

        const auto dims =
            input->getDimensions();

        if (
            dims.nbDims != 4 ||
            dims.d[0] != 1 ||
            dims.d[1] != 3 ||
            dims.d[2] != 224 ||
            dims.d[3] != 224
        ) {
            throw std::runtime_error(
                "This INT8 example expects "
                "[1,3,224,224]"
            );
        }

        constexpr std::size_t
            elementsPerSample =
                1ULL * 3 * 224 * 224;

        // --------------------------------
        // Calibrator
        // --------------------------------

        Int8EntropyCalibrator calibrator(
            calibrationPath,
            cachePath,
            elementsPerSample
        );

        // --------------------------------
        // Builder config
        // --------------------------------

        auto* config =
            builder->createBuilderConfig();

        config->setProfilingVerbosity(
            nvinfer1::ProfilingVerbosity::kDETAILED
        );

        constexpr std::size_t workspace =
            2ULL << 30;

        config->setMaxWorkspaceSize(
            workspace
        );

        config->setFlag(
            nvinfer1::BuilderFlag::kINT8
        );

        config->setInt8Calibrator(
            &calibrator
        );

        std::cout
            << "Precision: INT8 enabled\n";

        std::cout
            << "Building INT8 engine...\n";

        // --------------------------------
        // Build
        // --------------------------------

        auto* serialized =
            builder->buildSerializedNetwork(
                *network,
                *config
            );

        if (!serialized) {
            throw std::runtime_error(
                "INT8 build failed"
            );
        }

        // --------------------------------
        // Save
        // --------------------------------

        std::ofstream output(
            enginePath,
            std::ios::binary
        );

        output.write(
            static_cast<const char*>(
                serialized->data()
            ),
            serialized->size()
        );

        output.close();

        std::cout
            << "\nEngine saved: "
            << enginePath
            << '\n';

        std::cout
            << "Engine size : "
            << serialized->size()
                / (1024.0 * 1024.0)
            << " MiB\n";

        serialized->destroy();
        config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();

    } catch (const std::exception& e) {
        std::cerr
            << "ERROR: "
            << e.what()
            << '\n';

        return 1;
    }

    return 0;
}