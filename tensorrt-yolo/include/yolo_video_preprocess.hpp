#pragma once
#include <opencv2/core.hpp>
#include <string>
namespace video {
// CPU-only geometry validation/preparation. Output is immutable while in use.
// Business mode shares the input storage; camera mode owns a new image.
void validateBusinessInput(const cv::Mat& input,const std::string& stage);
cv::Mat toBusinessFrame(const cv::Mat& input,const std::string& stage);
}
