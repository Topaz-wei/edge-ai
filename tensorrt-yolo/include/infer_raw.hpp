#ifndef INFER_RAW_HPP_
#define INFER_RAW_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

#include "edge/trt/tensor_io.hpp"
using edge::trt::checkCuda;
using edge::trt::loadEngineFile;
using edge::trt::loadFloatFile;
using edge::trt::dimsToString;
using edge::trt::getNumElements;
using edge::trt::saveFloatBinary;
// One-shot compatibility wrapper: constructs a runner per call.
// Use TrtRunner directly for repeated inference / benchmarking.
std::vector<float> inferRaw(const std::string& enginePath, const std::string& inputPath);

// Accept preprocessed input directly, without an intermediate binary file.
std::vector<float> inferRaw(const std::string& enginePath, const std::vector<float>& hostInput);

#endif
