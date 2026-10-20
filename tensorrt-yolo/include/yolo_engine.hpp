#pragma once
#include "include/trt_runner.hpp"
inline bool isFixedYolo(const TrtRunner& runner) {
    const auto& in = runner.inputInfo().shape;
    const auto& out = runner.outputInfo().shape;
    return in.nbDims == 4 && in.d[0] == 1 && in.d[1] == 3 && in.d[2] == 640 &&
           in.d[3] == 640 && out.nbDims == 3 && out.d[0] == 1 && out.d[1] == 84 && out.d[2] == 8400;
}
