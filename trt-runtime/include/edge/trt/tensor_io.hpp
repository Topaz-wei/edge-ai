#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
namespace edge::trt {
void checkCuda(cudaError_t error, const char* operation);
std::vector<char> loadEngineFile(const std::string& path);
std::vector<float> loadFloatFile(const std::string& path);
std::string dimsToString(const nvinfer1::Dims& dims);
std::size_t getNumElements(const nvinfer1::Dims& dims);
void saveFloatBinary(const std::string& path, const std::vector<float>& data);
}
