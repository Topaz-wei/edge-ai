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
    std::cout << "Tensor volume and unsupported dimensions checked\n";
}
