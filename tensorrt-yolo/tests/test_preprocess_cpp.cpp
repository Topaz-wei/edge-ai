#include "include/yolo_preprocess.hpp"
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected business image path");
        auto image = cv::imread(argv[1], cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        auto original = image.clone();
        const auto fromPath = preprocess(std::string(argv[1]));
        if (preprocess(image) != fromPath) throw std::runtime_error("Mat/path mismatch");
        if (cv::norm(image, original, cv::NORM_INF) != 0) throw std::runtime_error("Input mutated");
        cv::Mat padded(640, 362, CV_8UC3);
        auto roi = padded(cv::Rect(1, 0, 360, 640));
        image.copyTo(roi);
        if (preprocess(roi) != fromPath) throw std::runtime_error("Strided ROI mismatch");
        for (auto bad : {cv::Mat(), cv::Mat(360, 640, CV_8UC3), cv::Mat(640, 360, CV_32FC3)}) {
            bool rejected = false;
            try { preprocess(bad); } catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Invalid image accepted");
        }
        std::cout << "Mat/path, non-mutation, strided ROI and invalid images passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
