#include "edge/trt/runner.hpp"
#include "edge/trt/tensor_io.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Expected engine input baseline");
        edge::trt::Runner runner(argv[1]);
        const auto& in = runner.inputInfo();
        if (in.name.empty() || in.shape.nbDims != 4 || in.shape.d[2] != 224 ||
            runner.outputElements() != 1000 || in.dtype != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error("Unexpected ResNet I/O");
        const auto input = edge::trt::loadFloatFile(argv[2]);
        const auto reference = edge::trt::loadFloatFile(argv[3]);
        for (int i=0;i<3;++i) {
            const auto output = runner.infer(input);
            if (output.size()!=reference.size() || std::memcmp(output.data(),reference.data(),output.size()*sizeof(float)))
                throw std::runtime_error("ResNet raw differs from original inference");
        }
        bool rejected=false;
        try { runner.infer({}); } catch(const std::runtime_error&) { rejected=true; }
        if(!rejected) throw std::runtime_error("Bad input accepted");
        std::cout << "ResNet metadata, repeated bitwise output and invalid input passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
