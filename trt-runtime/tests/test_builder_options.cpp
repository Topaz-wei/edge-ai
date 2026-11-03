// 本测试验证公共构建配置拒绝危险输入，尤其禁止无校准器的隐式 INT8 构建。
#include "edge/trt/builder.hpp"
#include <iostream>
#include <stdexcept>

using namespace edge::trt;

void rejects(const BuildOptions& options) {
    try {
        validateBuildOptions(options);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Invalid options were accepted");
}

int main() {
    try {
        BuildOptions options;
        options.onnxPath = "model.onnx";
        options.enginePath = "model.engine";
        options.inputs = {{"input", {-1, 3, 256, 256}, {1, 3, 256, 256},
                           {2, 3, 256, 256}, {4, 3, 256, 256}, {}}};
        options.outputs = {{"x", {-1, 21, 512}}, {"y", {-1, 21, 512}}};
        validateBuildOptions(options);
        auto invalid = options;
        invalid.precision = Precision::INT8;
        rejects(invalid);
        invalid = options;
        invalid.workspaceBytes = 0;
        rejects(invalid);
        invalid = options;
        invalid.inputs[0].optimum[0] = 5;
        rejects(invalid);
        invalid = options;
        invalid.inputs[0].maximum[2] = 512;
        rejects(invalid);
        invalid = options;
        invalid.inputs[0].minimum.pop_back();
        rejects(invalid);
        invalid = options;
        invalid.outputs[1].name = "x";
        rejects(invalid);
        invalid = options;
        invalid.inputs[0].shape[0] = 0;
        rejects(invalid);
        std::cout << "Builder options checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
