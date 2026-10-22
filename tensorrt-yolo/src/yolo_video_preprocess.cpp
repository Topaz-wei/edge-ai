#include "include/yolo_video_preprocess.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>
namespace video {
void validateBusinessInput(const cv::Mat& input,const std::string& stage) {
    if(input.empty() || input.type()!=CV_8UC3)
        throw std::runtime_error("Expected a nonempty CV_8UC3 BGR frame");
    if(stage=="business") {
        if(input.cols!=360 || input.rows!=640)
            throw std::runtime_error("Business input must be width=360 height=640");
    } else if(stage=="camera") {
        if(static_cast<int>(input.rows*(640.0/input.cols))!=360)
            throw std::runtime_error("Camera geometry must produce width=360 height=640");
    } else throw std::runtime_error("Input stage must be business or camera");
}
cv::Mat toBusinessFrame(const cv::Mat& input,const std::string& stage) {
    validateBusinessInput(input,stage);
    if(stage=="business")return input;
    // Match the existing reference: transpose, not a 90-degree rotation.
    cv::Mat transposed,business;
    cv::transpose(input,transposed);
    cv::resize(transposed,business,cv::Size(360,640),0.0,0.0,cv::INTER_AREA);
    return business;
}
}
