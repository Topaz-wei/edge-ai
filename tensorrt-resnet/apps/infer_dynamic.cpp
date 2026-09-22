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


std::vector<char> loadEngineFile(
    const std::string& path
) {
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


std::vector<float> loadFloatFile(
    const std::string& path
) {
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
            "Invalid float binary file"
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


std::size_t volume(
    const nvinfer1::Dims& dims
) {
    std::size_t result = 1;

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error(
                "Unresolved dynamic dimension"
            );
        }

        result *= dims.d[i];
    }

    return result;
}


void printDims(
    const nvinfer1::Dims& dims
) {
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
    if (argc != 5) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <engine.plan>"
            << " <input.bin>"
            << " <reference.bin>"
            << " <batch>\n";

        return 1;
    }

    try {
        const std::string enginePath = argv[1];
        const std::string inputPath = argv[2];
        const std::string referencePath = argv[3];

        const int batch = std::stoi(argv[4]);

        if (batch <= 0) {
            throw std::runtime_error(
                "Batch must be positive"
            );
        }

        // --------------------------------
        // Load Engine
        // --------------------------------

        auto engineData =
            loadEngineFile(enginePath);

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

        auto* context =
            engine->createExecutionContext();

        if (!context) {
            throw std::runtime_error(
                "Failed to create execution context"
            );
        }

        // --------------------------------
        // Find bindings
        // --------------------------------

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

        if (inputIndex < 0 ||
            outputIndex < 0) {

            throw std::runtime_error(
                "Input/output binding not found"
            );
        }

        std::cout
            << "Engine input shape : ";

        printDims(
            engine->getBindingDimensions(
                inputIndex
            )
        );

        std::cout << '\n';

        // --------------------------------
        // Print profile
        // --------------------------------

        const int profileIndex = 0;

        const auto minDims =
            engine->getProfileDimensions(
                inputIndex,
                profileIndex,
                nvinfer1::OptProfileSelector::kMIN
            );

        const auto optDims =
            engine->getProfileDimensions(
                inputIndex,
                profileIndex,
                nvinfer1::OptProfileSelector::kOPT
            );

        const auto maxDims =
            engine->getProfileDimensions(
                inputIndex,
                profileIndex,
                nvinfer1::OptProfileSelector::kMAX
            );

        std::cout << "Profile MIN        : ";
        printDims(minDims);
        std::cout << '\n';

        std::cout << "Profile OPT        : ";
        printDims(optDims);
        std::cout << '\n';

        std::cout << "Profile MAX        : ";
        printDims(maxDims);
        std::cout << '\n';

        if (
            batch < minDims.d[0] ||
            batch > maxDims.d[0]
        ) {
            throw std::runtime_error(
                "Requested batch is outside "
                "optimization profile range"
            );
        }

        // --------------------------------
        // CUDA stream
        // --------------------------------

        cudaStream_t stream;

        checkCuda(
            cudaStreamCreate(&stream),
            "cudaStreamCreate"
        );

        // --------------------------------
        // Select optimization profile
        // --------------------------------

        if (!context->setOptimizationProfileAsync(
                profileIndex,
                stream)) {

            throw std::runtime_error(
                "Failed to set optimization profile"
            );
        }

        // --------------------------------
        // Set actual runtime shape
        // --------------------------------

        nvinfer1::Dims4 actualInputDims{
            batch,
            3,
            224,
            224
        };

        if (!context->setBindingDimensions(
                inputIndex,
                actualInputDims)) {

            throw std::runtime_error(
                "setBindingDimensions failed"
            );
        }

        if (!context->allInputDimensionsSpecified()) {
            throw std::runtime_error(
                "Not all dynamic dimensions "
                "were specified"
            );
        }

        // IMPORTANT:
        // Runtime dimensions must come from
        // IExecutionContext, not ICudaEngine.
        const auto inputDims =
            context->getBindingDimensions(
                inputIndex
            );

        const auto outputDims =
            context->getBindingDimensions(
                outputIndex
            );

        std::cout
            << "\nRuntime input shape : ";

        printDims(inputDims);

        std::cout
            << "\nRuntime output shape: ";

        printDims(outputDims);

        std::cout << "\n";

        // --------------------------------
        // Validate datatype
        // --------------------------------

        if (
            engine->getBindingDataType(inputIndex)
                != nvinfer1::DataType::kFLOAT ||
            engine->getBindingDataType(outputIndex)
                != nvinfer1::DataType::kFLOAT
        ) {
            throw std::runtime_error(
                "This example expects FP32 I/O"
            );
        }

        // --------------------------------
        // Calculate buffer sizes
        // --------------------------------

        const std::size_t inputElements =
            volume(inputDims);

        const std::size_t outputElements =
            volume(outputDims);

        const std::size_t inputBytes =
            inputElements * sizeof(float);

        const std::size_t outputBytes =
            outputElements * sizeof(float);

        std::cout
            << "Input elements       : "
            << inputElements << '\n';

        std::cout
            << "Output elements      : "
            << outputElements << '\n';

        // --------------------------------
        // Load host data
        // --------------------------------

        auto hostInput =
            loadFloatFile(inputPath);

        auto reference =
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

        for (float x : reference) {
            if (!std::isfinite(x)) {
                throw std::runtime_error(
                    "Reference contains NaN/Inf"
                );
            }
        }

        std::vector<float> hostOutput(
            outputElements
        );

        // --------------------------------
        // Allocate GPU memory
        // --------------------------------

        void* deviceInput = nullptr;
        void* deviceOutput = nullptr;

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

        std::vector<void*> bindings(
            nbBindings,
            nullptr
        );

        bindings[inputIndex] =
            deviceInput;

        bindings[outputIndex] =
            deviceOutput;

        // --------------------------------
        // H2D
        // --------------------------------

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

        // --------------------------------
        // TensorRT inference
        // --------------------------------

        if (!context->enqueueV2(
                bindings.data(),
                stream,
                nullptr)) {

            throw std::runtime_error(
                "enqueueV2 failed"
            );
        }

        // --------------------------------
        // D2H
        // --------------------------------

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

        // --------------------------------
        // Numerical validation
        // --------------------------------

        double maxAbs = 0.0;
        double sumAbs = 0.0;
        double sumSquared = 0.0;

        for (std::size_t i = 0;
             i < outputElements;
             ++i) {

            if (!std::isfinite(hostOutput[i])) {
                throw std::runtime_error(
                    "TensorRT output contains NaN/Inf"
                );
            }

            const double diff =
                static_cast<double>(
                    hostOutput[i]
                ) -
                static_cast<double>(
                    reference[i]
                );

            const double absDiff =
                std::abs(diff);

            maxAbs =
                std::max(
                    maxAbs,
                    absDiff
                );

            sumAbs += absDiff;
            sumSquared += diff * diff;
        }

        const double meanAbs =
            sumAbs / outputElements;

        const double rmse =
            std::sqrt(
                sumSquared / outputElements
            );

        // --------------------------------
        // Top-1 validation per sample
        // --------------------------------

        if (outputDims.nbDims != 2) {
            throw std::runtime_error(
                "Expected output [batch, classes]"
            );
        }

        const int outputBatch =
            outputDims.d[0];

        const int classes =
            outputDims.d[1];

        int top1Matches = 0;

        for (int b = 0;
             b < outputBatch;
             ++b) {

            const auto trtBegin =
                hostOutput.begin() +
                b * classes;

            const auto refBegin =
                reference.begin() +
                b * classes;

            const int trtTop1 =
                std::distance(
                    trtBegin,
                    std::max_element(
                        trtBegin,
                        trtBegin + classes
                    )
                );

            const int refTop1 =
                std::distance(
                    refBegin,
                    std::max_element(
                        refBegin,
                        refBegin + classes
                    )
                );

            if (trtTop1 == refTop1) {
                ++top1Matches;
            }

            std::cout
                << "Sample "
                << b
                << ": PyTorch="
                << refTop1
                << ", TensorRT="
                << trtTop1
                << '\n';
        }

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
            << "Top-1 matches      : "
            << top1Matches
            << "/"
            << outputBatch
            << '\n';

        // --------------------------------
        // Cleanup
        // --------------------------------

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

        cudaStreamDestroy(stream);

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