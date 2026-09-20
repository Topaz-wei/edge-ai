#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <random>
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


std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open engine: " + path);
    }

    file.seekg(0, std::ios::end);
    const std::size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    return data;
}


std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;

    for (int i = 0; i < dims.nbDims; ++i) {
        result *= dims.d[i];
    }

    return result;
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr
            << "Usage: " << argv[0]
            << " <engine.plan>\n";
        return 1;
    }

    const std::string enginePath = argv[1];

    // 1. Load serialized TensorRT engine
    auto engineData = loadEngineFile(enginePath);

    // 2. Create TensorRT runtime
    nvinfer1::IRuntime* runtime =
        nvinfer1::createInferRuntime(gLogger);

    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime\n";
        return 1;
    }

    // 3. Deserialize engine
    nvinfer1::ICudaEngine* engine =
        runtime->deserializeCudaEngine(
            engineData.data(),
            engineData.size()
        );

    if (!engine) {
        std::cerr << "Failed to deserialize engine\n";
        runtime->destroy();
        return 1;
    }

    // 4. Create execution context
    nvinfer1::IExecutionContext* context =
        engine->createExecutionContext();

    if (!context) {
        std::cerr << "Failed to create execution context\n";
        engine->destroy();
        runtime->destroy();
        return 1;
    }

    const int nbBindings = engine->getNbBindings();

    std::cout << "Bindings: " << nbBindings << '\n';

    int inputIndex = -1;
    int outputIndex = -1;

    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine->getBindingName(i);
        const auto dims = engine->getBindingDimensions(i);

        std::cout
            << "Binding " << i
            << ": " << name
            << " [";

        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];

            if (j + 1 != dims.nbDims)
                std::cout << "x";
        }

        std::cout << "] ";

        if (engine->bindingIsInput(i)) {
            std::cout << "INPUT\n";
            inputIndex = i;
        } else {
            std::cout << "OUTPUT\n";
            outputIndex = i;
        }
    }

    if (inputIndex < 0 || outputIndex < 0) {
        std::cerr << "Could not find input/output binding\n";
        return 1;
    }

    const auto inputDims =
        engine->getBindingDimensions(inputIndex);

    const auto outputDims =
        engine->getBindingDimensions(outputIndex);

    const std::size_t inputElements = volume(inputDims);
    const std::size_t outputElements = volume(outputDims);

    const std::size_t inputBytes =
        inputElements * sizeof(float);

    const std::size_t outputBytes =
        outputElements * sizeof(float);

    std::cout
        << "Input elements : "
        << inputElements << '\n';

    std::cout
        << "Output elements: "
        << outputElements << '\n';

    // 5. Host buffers
    std::vector<float> hostInput(inputElements);
    std::vector<float> hostOutput(outputElements);

    std::mt19937 generator(0);
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (auto& x : hostInput) {
        x = distribution(generator);
    }

    // 6. Device buffers
    void* deviceInput = nullptr;
    void* deviceOutput = nullptr;

    cudaMalloc(&deviceInput, inputBytes);
    cudaMalloc(&deviceOutput, outputBytes);

    std::vector<void*> bindings(nbBindings);

    bindings[inputIndex] = deviceInput;
    bindings[outputIndex] = deviceOutput;

    // 7. CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 8. H2D
    cudaMemcpyAsync(
        deviceInput,
        hostInput.data(),
        inputBytes,
        cudaMemcpyHostToDevice,
        stream
    );

    // 9. TensorRT inference
    const bool success =
        context->enqueueV2(
            bindings.data(),
            stream,
            nullptr
        );

    if (!success) {
        std::cerr << "TensorRT inference failed\n";
        return 1;
    }

    // 10. D2H
    cudaMemcpyAsync(
        hostOutput.data(),
        deviceOutput,
        outputBytes,
        cudaMemcpyDeviceToHost,
        stream
    );

    cudaStreamSynchronize(stream);

    std::cout << "\nFirst 10 outputs:\n";

    for (int i = 0; i < 10; ++i) {
        std::cout
            << i << ": "
            << hostOutput[i]
            << '\n';
    }

    // Cleanup
    cudaStreamDestroy(stream);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}