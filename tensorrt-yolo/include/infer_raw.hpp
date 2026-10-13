#ifndef INFER_RAW_HPP_
#define INFER_RAW_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

void checkCuda(cudaError_t error, const char* operation);
std::vector<char> loadEngineFile(const std::string& path);
std::vector<float> loadFloatFile(const std::string& path);
std::string dimsToString(const nvinfer1::Dims& dims);
std::size_t getNumElements(const nvinfer1::Dims& dims);
void saveFloatBinary(const std::string& path, const std::vector<float>& data);
// One-shot compatibility wrapper: constructs a runner per call.
// Use TrtRunner directly for repeated inference / benchmarking.
std::vector<float> inferRaw(const std::string& enginePath, const std::string& inputPath);

// Accept preprocessed input directly, without an intermediate binary file.
std::vector<float> inferRaw(const std::string& enginePath, const std::vector<float>& hostInput);

#endif
