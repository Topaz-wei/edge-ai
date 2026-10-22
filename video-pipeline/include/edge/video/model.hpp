#pragma once
#include "edge/video/config.hpp"
#include <opencv2/core.hpp>
#include <functional>
#include <map>
#include <memory>
#include <cstdint>
namespace edge::video {
// Owns a reference-counted image header. Pixels stay immutable until consumption finishes.
struct Frame { cv::Mat bgr; uint64_t id; double timestampMs; };
struct DetectionResult { cv::Rect2f box; int classId; float score; std::string label; };
struct ClassificationResult { int classId; float score; std::string label; };
struct ModelResult {
    std::vector<DetectionResult> detections;
    std::vector<ClassificationResult> classifications;
    std::map<std::string,double> timingsMs;
};
// Model-specific subclasses own CPU-prepared tensors/images and mapping metadata.
struct PreparedInput {
    explicit PreparedInput(const Frame& frame):source(frame){}
    virtual ~PreparedInput()=default;
    Frame source;
};
struct PreparedFrame {
    Frame source;
    std::vector<std::unique_ptr<PreparedInput>> inputs;
};
class IModel {
public:
    virtual ~IModel()=default;
    virtual void validateFrame(const cv::Mat&) const {}
    // CPU-only; may run concurrently with infer on this same instance. Never
    // mutate shared inference buffers or source pixels; return per-frame storage.
    virtual std::unique_ptr<PreparedInput> prepare(const Frame& frame) const {
        return std::make_unique<PreparedInput>(frame);
    }
    // Consumes prepared data; GPU submission/transfers and result packaging live here.
    // Calls to infer on a given instance remain serialized.
    virtual ModelResult infer(const PreparedInput& input)=0;
};
using ModelFactory=std::function<std::unique_ptr<IModel>(const ModelConfig&)>;
using ModelRegistry=std::map<std::string,ModelFactory>;
struct InstanceResult { std::string id; ModelResult result; double wallMs; };
class ModelSet {
public:
    ModelSet(const std::vector<ModelConfig>& configs, const ModelRegistry& registry);
    void validateFrame(const cv::Mat& frame) const;
    PreparedFrame prepare(const Frame& frame) const;
    std::vector<InstanceResult> infer(const PreparedFrame& frame);
private:
    std::vector<std::pair<std::string,std::unique_ptr<IModel>>> models_;
};
}
