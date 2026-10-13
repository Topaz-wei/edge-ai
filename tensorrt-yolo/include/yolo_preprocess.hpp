#ifndef YOLO_PREPROCESS_HPP_
#define YOLO_PREPROCESS_HPP_

#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

// Business BGR image (width=360, height=640) -> RGB FP32 NCHW [1,3,640,640].
// Matches prepare_yolo_reference.py with input_stage="business".
std::vector<float> preprocess(const std::string& imagePath);
// Does not modify the caller's image; also accepts strided cv::Mat ROIs.
std::vector<float> preprocess(const cv::Mat& businessImage);

#endif
