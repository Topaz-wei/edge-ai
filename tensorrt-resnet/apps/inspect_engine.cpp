#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>


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


std::vector<char> loadEngine(
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to open engine: " + path
        );
    }

    file.seekg(
        0,
        std::ios::end
    );

    const std::size_t size =
        file.tellg();

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<char> data(size);

    file.read(
        data.data(),
        size
    );

    return data;
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <engine.plan>\n";

        return 1;
    }

    try {
        const std::string enginePath =
            argv[1];

        // -----------------------------
        // Load engine bytes
        // -----------------------------

        const auto engineData =
            loadEngine(enginePath);

        // -----------------------------
        // Runtime
        // -----------------------------

        auto* runtime =
            nvinfer1::createInferRuntime(
                gLogger
            );

        if (!runtime) {
            throw std::runtime_error(
                "Failed to create runtime"
            );
        }

        auto* engine =
            runtime->deserializeCudaEngine(
                engineData.data(),
                engineData.size()
            );

        if (!engine) {
            throw std::runtime_error(
                "Failed to deserialize engine"
            );
        }

        // -----------------------------
        // Engine Inspector
        // -----------------------------

        {
            std::unique_ptr<nvinfer1::IEngineInspector> inspector(
                engine->createEngineInspector()
            );

            if (!inspector) {
                throw std::runtime_error(
                    "Failed to create inspector"
                );
            }

            std::cout
            << "\n==============================\n"
            << "TensorRT Engine Inspector\n"
            << "==============================\n\n";

            const char* info =
                inspector->getEngineInformation(
                    nvinfer1::LayerInformationFormat::kJSON
                );

            if (!info) {
                throw std::runtime_error(
                    "Failed to get engine information"
                );
            }

            std::cout << info << '\n';

        } // inspector 在这里自动销毁

        // TensorRT 8.x legacy cleanup
        engine->destroy();
        runtime->destroy();

    } catch (const std::exception& e) {
        std::cerr
            << "ERROR: "
            << e.what()
            << '\n';

        return 1;
    }

    return 0;
}