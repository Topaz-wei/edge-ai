#include "include/infer_raw.hpp"
#include "include/trt_runner.hpp"

#include <iostream>
#include <stdexcept>

std::vector<float> inferRaw(const std::string& enginePath, const std::string& inputPath) {
    return inferRaw(enginePath, loadFloatFile(inputPath));
}

std::vector<float> inferRaw(const std::string& enginePath, const std::vector<float>& input) {
    TrtRunner runner(enginePath);
    return runner.infer(input);
}

#ifndef YOLO_NO_MAIN
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <engine.plan> <input.bin> <output.bin>\n";
        return 1;
    }

    try {
        const auto output = inferRaw(argv[1], argv[2]);
        saveFloatBinary(argv[3], output);
        std::cout << "Output saved to: " << argv[3] << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
#endif
