// 本文件声明 RTMPose 专用预处理与解码，遵循冻结 RGB/SimCC 契约。
#pragma once
#include <array>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace rtmpose {
struct Region {
    std::array<double, 4> bbox;
    cv::Point2d center;
    double scale;
};
Region makeRegion(const std::array<double, 4>& bbox);
std::vector<float> preprocess(
    const cv::Mat& bgr,
    const Region& region
);
// 每点六列：input x/y、source x/y、score、valid；source 坐标保留双精度。
std::vector<double> decode(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<Region>& regions
);
std::vector<float> loadNpy(const std::string& path);
} // namespace rtmpose
