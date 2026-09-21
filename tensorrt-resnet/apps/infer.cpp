#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << '\n';
        }
    }
};


static Logger gLogger;


void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}


std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "Failed to open engine: " + path
        );
    }

    file.seekg(0, std::ios::end);
    const std::size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    return data;
}


std::vector<float> loadFloatFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "Failed to open file: " + path
        );
    }

    file.seekg(0, std::ios::end);

    const std::size_t bytes = file.tellg();

    if (bytes % sizeof(float) != 0) {
        throw std::runtime_error(
            "Invalid float binary file: " + path
        );
    }

    file.seekg(0, std::ios::beg);

    std::vector<float> data(
        bytes / sizeof(float)
    );

    file.read(
        reinterpret_cast<char*>(data.data()),
        bytes
    );

    return data;
}


std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error(
                "Dynamic or invalid dimension detected"
            );
        }

        result *= dims.d[i];
    }

    return result;
}


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <engine.plan>"
            << " <input.bin>"
            << " <reference.bin>\n";

        return 1;
    }

    try {
        const std::string enginePath = argv[1];
        const std::string inputPath = argv[2];
        const std::string referencePath = argv[3];

        // -------------------------
        // Load TensorRT engine
        // -------------------------

        auto engineData =
            loadEngineFile(enginePath);

        nvinfer1::IRuntime* runtime =
            nvinfer1::createInferRuntime(gLogger);

        if (!runtime) {
            throw std::runtime_error(
                "Failed to create TensorRT runtime"
            );
        }

        nvinfer1::ICudaEngine* engine =
            runtime->deserializeCudaEngine(
                engineData.data(),
                engineData.size()
            );

        if (!engine) {
            throw std::runtime_error(
                "Failed to deserialize engine"
            );
        }

        nvinfer1::IExecutionContext* context =
            engine->createExecutionContext();

        if (!context) {
            throw std::runtime_error(
                "Failed to create execution context"
            );
        }

        // -------------------------
        // Find input/output
        // -------------------------

        const int nbBindings =
            engine->getNbBindings();

        int inputIndex = -1;
        int outputIndex = -1;

        for (int i = 0; i < nbBindings; ++i) {
            if (engine->bindingIsInput(i)) {
                inputIndex = i;
            } else {
                outputIndex = i;
            }
        }

        if (inputIndex < 0 || outputIndex < 0) {
            throw std::runtime_error(
                "Input/output bindings not found"
            );
        }

        const auto inputDims =
            engine->getBindingDimensions(inputIndex);

        const auto outputDims =
            engine->getBindingDimensions(outputIndex);

        const std::size_t inputElements =
            volume(inputDims);

        const std::size_t outputElements =
            volume(outputDims);

        // Our current engines expose FP32 I/O.
        if (
            engine->getBindingDataType(inputIndex)
                != nvinfer1::DataType::kFLOAT ||
            engine->getBindingDataType(outputIndex)
                != nvinfer1::DataType::kFLOAT
        ) {
            throw std::runtime_error(
                "This example expects FP32 engine I/O"
            );
        }

        // -------------------------
        // Load reference data
        // -------------------------

        std::vector<float> hostInput =
            loadFloatFile(inputPath);

        std::vector<float> reference =
            loadFloatFile(referencePath);

        if (hostInput.size() != inputElements) {
            throw std::runtime_error(
                "Input element count mismatch"
            );
        }

        if (reference.size() != outputElements) {
            throw std::runtime_error(
                "Reference element count mismatch"
            );
        }

        std::vector<float> hostOutput(
            outputElements
        );

        // -------------------------
        // GPU memory
        // -------------------------

        void* deviceInput = nullptr;
        void* deviceOutput = nullptr;

        const std::size_t inputBytes =
            inputElements * sizeof(float);

        const std::size_t outputBytes =
            outputElements * sizeof(float);

        checkCuda(
            cudaMalloc(
                &deviceInput,
                inputBytes
            ),
            "cudaMalloc input"
        );

        checkCuda(
            cudaMalloc(
                &deviceOutput,
                outputBytes
            ),
            "cudaMalloc output"
        );

        std::vector<void*> bindings(nbBindings);

        bindings[inputIndex] = deviceInput;
        bindings[outputIndex] = deviceOutput;

        cudaStream_t stream;

        checkCuda(
            cudaStreamCreate(&stream),
            "cudaStreamCreate"
        );

        // -------------------------
        // H2D
        // -------------------------

        checkCuda(
            cudaMemcpyAsync(
                deviceInput,
                hostInput.data(),
                inputBytes,
                cudaMemcpyHostToDevice,
                stream
            ),
            "H2D"
        );

        // -------------------------
        // TensorRT inference
        // -------------------------

        if (!context->enqueueV2(
                bindings.data(),
                stream,
                nullptr)) {

            throw std::runtime_error(
                "TensorRT enqueueV2 failed"
            );
        }

        // -------------------------
        // D2H
        // -------------------------

        checkCuda(
            cudaMemcpyAsync(
                hostOutput.data(),
                deviceOutput,
                outputBytes,
                cudaMemcpyDeviceToHost,
                stream
            ),
            "D2H"
        );

        checkCuda(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize"
        );

        // -------------------------
        // Accuracy comparison
        // -------------------------

        double sumAbs = 0.0;
        double sumSquared = 0.0;
        double maxAbs = 0.0;

        for (std::size_t i = 0;
             i < outputElements;
             ++i) {

            const double diff =
                static_cast<double>(hostOutput[i]) -
                static_cast<double>(reference[i]);

            const double absDiff =
                std::abs(diff);

            sumAbs += absDiff;
            sumSquared += diff * diff;

            maxAbs =
                std::max(maxAbs, absDiff);
        }

        const double meanAbs =
            sumAbs / outputElements;

        const double rmse =
            std::sqrt(
                sumSquared / outputElements
            );

        const auto trtTop1 =
            std::distance(
                hostOutput.begin(),
                std::max_element(
                    hostOutput.begin(),
                    hostOutput.end()
                )
            );

        const auto pytorchTop1 =
            std::distance(
                reference.begin(),
                std::max_element(
                    reference.begin(),
                    reference.end()
                )
            );

        std::cout << "\n=== Accuracy ===\n";

        std::cout
            << "Max absolute error : "
            << maxAbs << '\n';

        std::cout
            << "Mean absolute error: "
            << meanAbs << '\n';

        std::cout
            << "RMSE               : "
            << rmse << '\n';

        std::cout
            << "PyTorch Top-1      : "
            << pytorchTop1 << '\n';

        std::cout
            << "TensorRT Top-1     : "
            << trtTop1 << '\n';

        std::cout
            << "Top-1 match        : "
            << (pytorchTop1 == trtTop1
                    ? "YES"
                    : "NO")
            << '\n';

        // -------------------------
        // Cleanup
        // -------------------------

        cudaStreamDestroy(stream);

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

        context->destroy();
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