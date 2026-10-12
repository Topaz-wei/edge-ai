#include "include/infer_raw.hpp"
#include "include/postprocess_raw.hpp"

#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <engine.plan> <input.bin>\n";
        return 1;
    }

    try {
        const auto output = inferRaw(argv[1], argv[2]);
        const auto detections = postprocess(output);
        std::cout << std::fixed << std::setprecision(6);
        printDetections(detections);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
