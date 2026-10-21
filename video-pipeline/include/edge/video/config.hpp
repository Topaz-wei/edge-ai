#pragma once
#include "edge/video/capture_options.hpp"
#include <filesystem>
#include <vector>
namespace edge::video {
struct ModelConfig {
    std::string id, type;
    std::filesystem::path engine;
    std::string inputStage;
    bool graph=true;
};
struct VideoConfig : CaptureOptions {
    int schemaVersion=1;
    std::vector<ModelConfig> models;
};
VideoConfig loadConfig(const std::filesystem::path& path);
}
