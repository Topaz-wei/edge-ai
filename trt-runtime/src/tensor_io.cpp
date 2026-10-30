// 本文件实现通用张量体积校验、CUDA 错误检查及二进制文件读写。
#include "edge/trt/tensor_io.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>
namespace edge::trt {
void checkCuda(
    cudaError_t error,
    const char* operation
) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

std::vector<char> loadEngineFile(
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error("Failed to open engine: " + path);
    }

    file.seekg(
        0,
        std::ios::end
    );

    const auto end = file.tellg();

    if (end <= 0)
        throw std::runtime_error("Invalid engine size: " + path);

    const std::size_t size = static_cast<std::size_t>(end);

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<char> data(size);

    file.read(
        data.data(),
        size
    );

    if (!file)
        throw std::runtime_error("Failed to read engine: " + path);

    return data;
}

std::vector<float> loadFloatFile(
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    file.seekg(
        0,
        std::ios::end
    );

    const auto end = file.tellg();

    if (end <= 0)
        throw std::runtime_error("Invalid float file size: " + path);

    const std::size_t bytes = static_cast<std::size_t>(end);

    if (bytes % sizeof(float) != 0) {
        throw std::runtime_error("Invalid float binary file: " + path);
    }

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<float> data(bytes / sizeof(float));

    file.read(
        reinterpret_cast<char*>(data.data()),
        bytes
    );

    if (!file)
        throw std::runtime_error("Failed to read float file: " + path);

    return data;
}

std::string dimsToString(
    const nvinfer1::Dims& dims
) {
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

std::size_t getNumElements(
    const nvinfer1::Dims& dims
) {
    if (dims.nbDims < 0 || dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
        throw std::runtime_error("Invalid tensor rank");
    }

    std::size_t elements = 1;

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Invalid tensor dimension");
        }

        if (elements > std::numeric_limits<std::size_t>::max() /
                           static_cast<std::size_t>(dims.d[i])) {
            throw std::runtime_error("Tensor element count overflow");
        }

        elements *= static_cast<std::size_t>(dims.d[i]);
    }

    return elements;
}

void saveFloatBinary(
    const std::string& path,
    const std::vector<float>& data
) {
    std::ofstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    file.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(float))
    );

    if (!file) {
        throw std::runtime_error("Failed to write output file: " + path);
    }
}


}
