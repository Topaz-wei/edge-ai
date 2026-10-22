#pragma once
#include "edge/video/model.hpp"
namespace edge::video {
// Optional compatibility presentation transform, supplied by a model application.
// New config-driven entry points leave this empty and display source coordinates.
// Called on the main thread for first-frame validation, then on the producer.
// Returned pixels must remain immutable/alive until that frame is consumed.
struct VideoHooks {
    std::function<cv::Mat(const cv::Mat&)> transform;
    bool legacyPresentation=false;
};
int runVideo(const VideoConfig& config,const ModelRegistry& registry,const VideoHooks& hooks={});
void renderResults(cv::Mat& canvas,const std::vector<InstanceResult>& results,bool legacy);
}
