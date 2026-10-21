#pragma once
#include "edge/video/config.hpp"
#include <opencv2/core.hpp>
#include <functional>
#include <map>
#include <memory>
#include <cstdint>
namespace edge::video {
// Input is borrowed and read-only by contract, including cv::Mat's underlying storage.
struct Frame { const cv::Mat& bgr; uint64_t id; double timestampMs; };
struct DetectionResult { cv::Rect2f box; int classId; float score; std::string label; };
struct ClassificationResult { int classId; float score; std::string label; };
struct ModelResult {
    std::vector<DetectionResult> detections;
    std::vector<ClassificationResult> classifications;
    std::map<std::string,double> timingsMs;
};
class IModel {
public:
    virtual ~IModel()=default;
    virtual void validateFrame(const cv::Mat&) const {}
    virtual ModelResult infer(const Frame& frame)=0;
};
using ModelFactory=std::function<std::unique_ptr<IModel>(const ModelConfig&)>;
using ModelRegistry=std::map<std::string,ModelFactory>;
struct InstanceResult { std::string id; ModelResult result; double wallMs; };
class ModelSet {
public:
    ModelSet(const std::vector<ModelConfig>& configs, const ModelRegistry& registry);
    void validateFrame(const cv::Mat& frame) const;
    std::vector<InstanceResult> infer(const Frame& frame);
private:
    std::vector<std::pair<std::string,std::unique_ptr<IModel>>> models_;
};
}
