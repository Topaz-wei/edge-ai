// 本文件实现冻结的 RGB 仿射预处理、SimCC 解码及单手 FP32 NPY 读取。
#include "rtmpose/reference.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace rtmpose {
Region makeRegion(const std::array<double, 4>& bbox) {
    for (double value : bbox) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Nonfinite bbox");
        }
    }
    if (bbox[2] <= bbox[0] || bbox[3] <= bbox[1]) {
        throw std::runtime_error("Nonpositive bbox size");
    }
    return {bbox, {(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2},
            1.25 * std::max(bbox[2] - bbox[0], bbox[3] - bbox[1])};
}

std::vector<float> preprocess(
    const cv::Mat& bgr,
    const Region& region
) {
    if (bgr.empty() || bgr.type() != CV_8UC3 || region.bbox[2] <= 0 ||
        region.bbox[3] <= 0 || region.bbox[0] >= bgr.cols || region.bbox[1] >= bgr.rows) {
        throw std::runtime_error("Invalid image or bbox outside image");
    }
    cv::Mat rgb;
    cv::cvtColor(
        bgr,
        rgb,
        cv::COLOR_BGR2RGB
    );
    // 与 rtmlib 一样先将前两个控制点量化为 float32，再构造第三点。
    cv::Point2f source[3];
    source[0] = cv::Point2f(region.center);
    source[1] = cv::Point2f(
        region.center.x,
        region.center.y - region.scale / 2
    );
    const auto direction = source[0] - source[1];
    source[2] = source[1] + cv::Point2f(-direction.y, direction.x);
    const cv::Point2f destination[3]{{128, 128}, {128, 0}, {0, 0}};
    const auto affine = cv::getAffineTransform(
        source,
        destination
    );
    cv::Mat crop;
    cv::warpAffine(
        rgb,
        crop,
        affine,
        cv::Size(256, 256),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );
    const double mean[3]{123.675, 116.28, 103.53};
    const double deviation[3]{58.395, 57.12, 57.375};
    std::vector<float> result(3 * 256 * 256);
    for (int row = 0; row < 256; ++row) {
        const auto* pixels = crop.ptr<cv::Vec3b>(row);
        for (int column = 0; column < 256; ++column) {
            for (int channel = 0; channel < 3; ++channel) {
                result[channel * 256 * 256 + row * 256 + column] = static_cast<float>(
                    (static_cast<double>(pixels[column][channel]) - mean[channel]) / deviation[channel]
                );
            }
        }
    }
    return result;
}

std::vector<double> decode(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<Region>& regions
) {
    if (x.size() != regions.size() * 21 * 512 || x.size() != y.size()) {
        throw std::runtime_error("SimCC shape mismatch");
    }
    for (std::size_t index = 0; index < x.size(); ++index) {
        if (!std::isfinite(x[index]) || !std::isfinite(y[index])) {
            throw std::runtime_error("Nonfinite SimCC");
        }
    }
    std::vector<double> decoded;
    decoded.reserve(regions.size() * 21 * 6);
    for (std::size_t hand = 0; hand < regions.size(); ++hand) {
        for (int point = 0; point < 21; ++point) {
            const auto offset = (hand * 21 + point) * 512;
            const auto firstX = x.begin() + offset;
            const auto firstY = y.begin() + offset;
            const auto peakX = std::max_element(
                firstX,
                firstX + 512
            );
            const auto peakY = std::max_element(
                firstY,
                firstY + 512
            );
            const float score = 0.5F * (*peakX + *peakY);
            const double u = score > 0 ? (peakX - firstX) / 2.0 : -0.5;
            const double v = score > 0 ? (peakY - firstY) / 2.0 : -0.5;
            const auto& region = regions[hand];
            decoded.insert(
                decoded.end(),
                {u, v, u / 256 * region.scale + region.center.x - region.scale / 2,
                 v / 256 * region.scale + region.center.y - region.scale / 2,
                 score, score > 0 ? 1.0 : 0.0}
            );
        }
    }
    return decoded;
}

std::vector<float> loadNpy(const std::string& path) {
    std::ifstream file(
        path,
        std::ios::binary
    );
    unsigned char prefix[8]{};
    file.read(
        reinterpret_cast<char*>(prefix),
        8
    );
    if (!file || std::string(reinterpret_cast<char*>(prefix), 6) != std::string("\x93NUMPY", 6) ||
        (prefix[6] != 1 && prefix[6] != 2) || prefix[7] != 0) {
        throw std::runtime_error("Unsupported NPY header: " + path);
    }
    const int lengthBytes = prefix[6] == 1 ? 2 : 4;
    std::uint32_t length = 0;
    for (int index = 0; index < lengthBytes; ++index) {
        unsigned char byte = 0;
        file.read(
            reinterpret_cast<char*>(&byte),
            1
        );
        length |= static_cast<std::uint32_t>(byte) << (8 * index);
    }
    if (!file || length > 65536) {
        throw std::runtime_error("Invalid NPY header length");
    }
    std::string header(length, '\0');
    file.read(
        header.data(),
        length
    );
    if (!std::regex_search(header, std::regex("'descr'\\s*:\\s*'<f4'")) ||
        !std::regex_search(header, std::regex("'fortran_order'\\s*:\\s*False")) ||
        !std::regex_search(header, std::regex("'shape'\\s*:\\s*\\(1,\\s*3,\\s*256,\\s*256,?\\s*\\)"))) {
        throw std::runtime_error("Expected little-endian contiguous FP32 [1,3,256,256] NPY");
    }
    std::vector<float> values(3 * 256 * 256);
    file.read(
        reinterpret_cast<char*>(values.data()),
        values.size() * sizeof(float)
    );
    if (!file || file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("NPY payload length mismatch");
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Nonfinite NPY input");
        }
    }
    return values;
}
} // namespace rtmpose
