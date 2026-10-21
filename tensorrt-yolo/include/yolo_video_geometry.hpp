#pragma once
#include "include/video_input.hpp"
namespace yolo {
inline cv::Rect2f toSourceBox(const cv::Rect2f& box,cv::Size source,const std::string& stage) {
    if(stage=="business") {
        if(source!=cv::Size(360,640))throw std::runtime_error("Invalid business geometry");
        return box;
    }
    if(stage!="camera" || source.width<=0 || source.height<=0 ||
       static_cast<int>(source.height*(640.0/source.width))!=360)
        throw std::runtime_error("Invalid camera geometry");
    const float sx=source.width/640.f, sy=source.height/360.f;
    return {box.y*sx,box.x*sy,box.height*sx,box.width*sy};
}
}
