#include "include/yolo_preprocess.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void meanRegionMask(cv::Mat& rgb, const std::vector<cv::Point>& polygon) {
    // Recompute the whole-image mean on each call. NumPy's astype(uint8)
    // truncates; passing a fractional Scalar directly to OpenCV would round.
    const cv::Scalar mean = cv::mean(rgb);
    const cv::Scalar fill(
        static_cast<unsigned char>(mean[0]),
        static_cast<unsigned char>(mean[1]),
        static_cast<unsigned char>(mean[2]));
    cv::Mat mask = cv::Mat::zeros(rgb.size(), CV_8UC1);
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(1));
    rgb.setTo(fill, mask);
}
} // namespace

std::vector<float> preprocess(const std::string& imagePath) {
    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (bgr.empty()) {
        throw std::runtime_error("Cannot decode image: " + imagePath);
    }
    return preprocess(bgr);
}

std::vector<float> preprocess(const cv::Mat& businessImage) {
    if (businessImage.rows != 640 || businessImage.cols != 360 || businessImage.type() != CV_8UC3) {
        throw std::runtime_error("Expected business BGR image: width=360, height=640");
    }

    cv::Mat rgb, bgr;
    cv::cvtColor(businessImage, rgb, cv::COLOR_BGR2RGB);
    meanRegionMask(rgb, {{240, 0}, {360, 0}, {360, 200}, {300, 100}});
    meanRegionMask(rgb, {{0, 150}, {65, 106}, {95, 237}, {6, 327}});
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    // Fixed business geometry: ratio=1, no resize, left/right padding=140.
    cv::Mat letterboxed;
    cv::copyMakeBorder(bgr, letterboxed, 0, 0, 140, 140,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    constexpr std::size_t planeSize = 640 * 640;
    std::vector<float> tensor(3 * planeSize);
    for (int y = 0; y < 640; ++y) {
        const auto* row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < 640; ++x) {
            for (int c = 0; c < 3; ++c) {
                tensor[c * planeSize + y * 640 + x] =
                    static_cast<float>(row[x][c]) / 255.0F;
            }
        }
    }
    return tensor;
}

#ifndef YOLO_NO_MAIN
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage:\n  " << argv[0] << " <business_image> <input.bin>\n";
        return 1;
    }
    try {
        const auto tensor = preprocess(argv[1]);
        std::ofstream file(argv[2], std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open output file: " + std::string(argv[2]));
        }
        file.write(reinterpret_cast<const char*>(tensor.data()),
                   static_cast<std::streamsize>(tensor.size() * sizeof(float)));
        file.close();
        if (!file) {
            throw std::runtime_error("Failed to write output file: " + std::string(argv[2]));
        }
        std::cout << "FP32 [1,3,640,640] saved to: " << argv[2] << '\n';
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
#endif
