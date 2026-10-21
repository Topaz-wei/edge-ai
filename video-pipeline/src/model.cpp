#include "edge/video/model.hpp"
#include <chrono>
#include <set>
#include <stdexcept>
namespace edge::video {
ModelSet::ModelSet(const std::vector<ModelConfig>& configs,const ModelRegistry& registry) {
    if(configs.empty())throw std::runtime_error("No enabled models");
    std::set<std::string> ids;
    // Validate every registration/path before allocating any GPU resources.
    for(const auto& c:configs) {
        if(c.id.empty() || !ids.insert(c.id).second)throw std::runtime_error("Empty or duplicate model id");
        if(!registry.count(c.type))throw std::runtime_error("Unknown model type: "+c.type);
        if(!std::filesystem::is_regular_file(c.engine))throw std::runtime_error("Missing engine: "+c.engine.string());
    }
    for(const auto& c:configs) {
        auto model=registry.at(c.type)(c);
        if(!model)throw std::runtime_error("Model factory returned null: "+c.id);
        models_.emplace_back(c.id,std::move(model));
    }
}
void ModelSet::validateFrame(const cv::Mat& frame) const {
    if (frame.empty() || frame.type() != CV_8UC3)
        throw std::runtime_error("Expected nonempty BGR uint8 frame");
    for (const auto& model : models_) model.second->validateFrame(frame);
}
std::vector<InstanceResult> ModelSet::infer(const Frame& frame) {
    if(frame.bgr.empty() || frame.bgr.type()!=CV_8UC3)throw std::runtime_error("Expected nonempty BGR uint8 frame");
    std::vector<InstanceResult> results;
    results.reserve(models_.size());
    for(auto& m:models_) {
        auto begin=std::chrono::steady_clock::now();
        auto result=m.second->infer(frame);
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        results.push_back({m.first,std::move(result),ms});
    }
    return results;
}
}
