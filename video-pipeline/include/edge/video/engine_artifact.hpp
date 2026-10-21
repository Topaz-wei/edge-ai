#pragma once
#include <filesystem>
#include <map>
#include <string>
namespace edge::video {
// Generic artifact audit. Runtime-specific values are supplied by the adapter.
// An optional ENGINE.meta.yaml must include engine_sha256; every present key
// must match a known current value. All metadata values are strings.
std::map<std::string,std::string> inspectEngineArtifact(
    const std::filesystem::path& engine,
    const std::map<std::string,std::string>& current);
}
