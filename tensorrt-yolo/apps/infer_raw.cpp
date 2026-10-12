#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#include "include/infer_raw.hpp"


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


std::string dimsToString(const nvinfer1::Dims& dims)
{
    std::string s = "[";

    for (int i = 0; i < dims.nbDims; ++i) {
        s += std::to_string(dims.d[i]);

        if (i != dims.nbDims - 1) {
            s += "x";
        }
    }

    s += "]";
    return s;
}


std::size_t getNumElements(const nvinfer1::Dims& dims)
{
    std::size_t elements = 1;

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error(
                "Invalid tensor dimension"
            );
        }

        elements *= static_cast<std::size_t>(
            dims.d[i]
        );
    }

    return elements;
}


void saveFloatBinary(
    const std::string& path,
    const std::vector<float>& data)
{
    std::ofstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to open output file: " + path
        );
    }

    file.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(
            data.size() * sizeof(float)
        )
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to write output file: " + path
        );
    }
}


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <engine.plan>"
            << " <input.bin>"
            << " <output.bin>";

        return 1;
    }

    try {
        const std::string enginePath = argv[1];
        const std::string inputPath = argv[2];
        const std::string outputPath = argv[3];

        // -------------------------
        // Load TensorRT engine
        // -------------------------

        auto engineData =
            loadEngineFile(enginePath);

        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};

        if (!runtime) {
            throw std::runtime_error(
                "Failed to create TensorRT runtime"
            );
        }

        std::unique_ptr<nvinfer1::ICudaEngine> engine{
            runtime->deserializeCudaEngine(engineData.data(), engineData.size())
        };

        if (!engine) {
            throw std::runtime_error(
                "Failed to deserialize engine"
            );
        }

        std::unique_ptr<nvinfer1::IExecutionContext> context{
            engine->createExecutionContext()
        };

        if (!context) {
            throw std::runtime_error(
                "Failed to create execution context"
            );
        }

        // -------------------------
        // Find input/output
        // -------------------------

        const int nbIOTensors =
            engine->getNbIOTensors();

        const char* inputName = nullptr;
        const char* outputName = nullptr;

        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name =
                engine->getIOTensorName(i);

            const auto mode =
                engine->getTensorIOMode(name);

            if (mode ==
                nvinfer1::TensorIOMode::kINPUT) {
                inputName = name;
            }
            else if (
                mode ==
                nvinfer1::TensorIOMode::kOUTPUT) {
                outputName = name;
            }
        }

        if (!inputName || !outputName) {
            throw std::runtime_error(
                "Failed to find input/output tensors"
            );
        }

        auto inputDims =
            engine->getTensorShape(inputName);

        auto outputDims =
            engine->getTensorShape(outputName);

        const std::size_t inputElements =
            getNumElements(inputDims);

        const std::size_t outputElements =
            getNumElements(outputDims);

        std::cout
            << "Input tensor  : "
            << inputName
            << "\n";

        std::cout
            << "Input shape   : "
            << dimsToString(inputDims)
            << "\n";

        std::cout
            << "Input elems   : "
            << inputElements
            << "\n\n";

        std::cout
            << "Output tensor : "
            << outputName
            << "\n";

        std::cout
            << "Output shape  : "
            << dimsToString(outputDims)
            << "\n";

        std::cout
            << "Output elems  : "
            << outputElements
            << "\n";

        // Our current engines expose FP32 I/O.
        if (
            engine->getTensorDataType(inputName) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType(outputName) != nvinfer1::DataType::kFLOAT
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

        if (hostInput.size() != inputElements) {
            throw std::runtime_error(
                "Input element count mismatch"
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

        // 设置 tensor → address 映射
        if (!context->setTensorAddress(
                inputName,
                deviceInput)) {
            throw std::runtime_error(
                "setTensorAddress failed for input"
            );
        }

        if (!context->setTensorAddress(
                outputName,
                deviceOutput)) {
            throw std::runtime_error(
                "setTensorAddress failed for output"
            );
        }

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

        if (!context->enqueueV3(stream)) {
            throw std::runtime_error(
                "TensorRT enqueueV3 failed"
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

        saveFloatBinary(
            outputPath,
            hostOutput
        );

        std::cout
            << "Output saved to: "
            << outputPath
            << "\n";

        // -------------------------
        // Cleanup
        // -------------------------

        cudaStreamDestroy(stream);

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

    } catch (const std::exception& e) {
        std::cerr
            << "ERROR: "
            << e.what()
            << '\n';

        return 1;
    }

    return 0;
}