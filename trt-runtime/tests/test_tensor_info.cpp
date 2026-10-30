// 本测试验证张量体积计算及非法维度、rank 的边界。
#include "edge/trt/runner.hpp"
#include "edge/trt/tensor_io.hpp"
#include <iostream>
#include <stdexcept>
int main() {
    using namespace edge::trt;
    nvinfer1::Dims shape{4, {1,3,640,640}};
    if (getNumElements(shape) != 1228800) return 1;
    for (int invalid : {0, -1}) {
        shape.d[2] = invalid;
        bool rejected = false;
        try { getNumElements(shape); } catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) return 2;
    }
    nvinfer1::Dims malformed{};
    malformed.nbDims = nvinfer1::Dims::MAX_DIMS + 1;
    bool rankRejected = false;
    try {
        getNumElements(malformed);
    } catch (const std::runtime_error& error) {
        rankRejected = std::string(error.what()) == "Invalid tensor rank";
    }
    if (!rankRejected) {
        std::cerr << "Rank must be rejected before dimension access\n";
        return 3;
    }
    std::cout << "Tensor volume and unsupported dimensions checked\n";
}
