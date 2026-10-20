#include "edge/trt/runner.hpp"
#include "edge/trt/tensor_io.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <engine.plan>"
            << " <input.bin>"
            << " <reference.bin>\n";

        return 1;
    }

    try {
        const std::string enginePath = argv[1];
        const std::string inputPath = argv[2];
        const std::string referencePath = argv[3];

        edge::trt::Runner runner(enginePath);
        const auto hostInput = edge::trt::loadFloatFile(inputPath);
        const auto reference = edge::trt::loadFloatFile(referencePath);
        const auto outputElements = runner.outputElements();
        if (reference.size() != outputElements)
            throw std::runtime_error("Reference element count mismatch");
        const auto hostOutput = runner.infer(hostInput);

        // -------------------------
        // Accuracy comparison
        // -------------------------

        double sumAbs = 0.0;
        double sumSquared = 0.0;
        double maxAbs = 0.0;

        for (std::size_t i = 0;
             i < outputElements;
             ++i) {

            const double diff =
                static_cast<double>(hostOutput[i]) -
                static_cast<double>(reference[i]);

            const double absDiff =
                std::abs(diff);

            sumAbs += absDiff;
            sumSquared += diff * diff;

            maxAbs =
                std::max(maxAbs, absDiff);
        }

        const double meanAbs =
            sumAbs / outputElements;

        const double rmse =
            std::sqrt(
                sumSquared / outputElements
            );

        const auto trtTop1 =
            std::distance(
                hostOutput.begin(),
                std::max_element(
                    hostOutput.begin(),
                    hostOutput.end()
                )
            );

        const auto pytorchTop1 =
            std::distance(
                reference.begin(),
                std::max_element(
                    reference.begin(),
                    reference.end()
                )
            );

        std::cout << "\n=== Accuracy ===\n";

        std::cout
            << "Max absolute error : "
            << maxAbs << '\n';

        std::cout
            << "Mean absolute error: "
            << meanAbs << '\n';

        std::cout
            << "RMSE               : "
            << rmse << '\n';

        std::cout
            << "PyTorch Top-1      : "
            << pytorchTop1 << '\n';

        std::cout
            << "TensorRT Top-1     : "
            << trtTop1 << '\n';

        std::cout
            << "Top-1 match        : "
            << (pytorchTop1 == trtTop1
                    ? "YES"
                    : "NO")
            << '\n';

    } catch (const std::exception& e) {
        std::cerr
            << "ERROR: "
            << e.what()
            << '\n';

        return 1;
    }

    return 0;
}