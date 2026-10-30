// 本测试覆盖 C++ 解码的平均分、并列峰值、无效点和 ROI 几何边界。
#include "rtmpose/reference.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
    const auto region = rtmpose::makeRegion({20, 70, 180, 230});
    if (region.center.x != 100 || region.center.y != 150 || region.scale != 200) {
        throw std::runtime_error("ROI geometry mismatch");
    }
    std::vector<float> x(21 * 512, 0);
    std::vector<float> y(21 * 512, 0);
    x[20] = 0.8F;
    y[40] = 0.2F;
    x[512 + 5] = x[512 + 9] = 0.6F;
    y[512 + 7] = y[512 + 10] = 0.4F;
    const auto points = rtmpose::decode(
        x,
        y,
        {region}
    );
    if (points[0] != 10 || points[1] != 20 || std::abs(points[4] - 0.5) > 1e-7 ||
        points[6] != 2.5 || points[7] != 3.5 || points[12] != -0.5 ||
        points[14] != -0.390625 || points[15] != 49.609375 || points[17] != 0) {
        throw std::runtime_error("SimCC decode contract mismatch");
    }
    bool rejected = false;
    try {
        rtmpose::makeRegion({10, 10, 5, 20});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("Invalid ROI accepted");
    }
    std::cout << "PASS: ROI and SimCC boundary cases\n";
}
