#include "include/trt_runner.hpp"
#include "include/infer_raw.hpp"
#include "include/yolo_preprocess.hpp"
#include "include/postprocess_raw.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: test_trt_runner engine image baseline.bin");
        const auto input = preprocess(std::string(argv[2]));
        const auto baseline = loadFloatFile(argv[3]);
        TrtRunner runner(argv[1]);
        if (runner.inputElements() != input.size() || runner.outputElements() != baseline.size()) {
            throw std::runtime_error("Unexpected I/O element count");
        }
        bool rejected = false;
        try { runner.infer({}); } catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Invalid input accepted");
        for (int i = 0; i < 5; ++i) {
            TrtTimings timings;
            auto output = runner.infer(input, i % 2 ? &timings : nullptr);
            if (output.size() != baseline.size() ||
                std::memcmp(output.data(), baseline.data(), output.size() * sizeof(float)) != 0) {
                throw std::runtime_error("Output differs from pre-refactor baseline");
            }
            if (i % 2 && (!std::isfinite(timings.h2dMs) || timings.h2dMs < 0 ||
                          !std::isfinite(timings.inferenceMs) || timings.inferenceMs <= 0 ||
                          !std::isfinite(timings.d2hMs) || timings.d2hMs < 0)) {
                throw std::runtime_error("Invalid CUDA timings");
            }
            if (i == 4) printDetections(postprocess(output));
        }
        std::cout << "5 repeated timed/untimed calls bitwise equal to baseline; input rejection passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
