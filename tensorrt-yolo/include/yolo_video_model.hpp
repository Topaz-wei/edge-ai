#pragma once
#include "edge/video/model.hpp"
std::unique_ptr<edge::video::IModel> makeYoloModel(const edge::video::ModelConfig& config);
