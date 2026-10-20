#include "include/infer_raw.hpp"
#include "include/trt_runner.hpp"

#include <fstream>
#include <iostream>

int main(
    int argc,
    char** argv
) {
    if (argc != 3)
        return 2;

    try {
        // Keep the engine/context alive across images instead of deserializing
        // an engine for each call to the one-shot yolo_infer_raw executable.
        TrtRunner runner(argv[1]);
        std::ifstream manifest(argv[2]);
        if (!manifest)
            throw std::runtime_error("Cannot open manifest");

        // The manifest alternates input.bin and output.bin paths, one per line.
        std::string input, output;
        while (std::getline(manifest, input) && std::getline(manifest, output)) {
            auto prediction = runner.infer(loadFloatFile(input));
            saveFloatBinary(output, prediction);

            std::cout << input << " -> " << output << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
