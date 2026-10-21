#include "include/video_input.hpp"
#include "include/yolo_video_model.hpp"
#include "edge/video/pipeline.hpp"
#include <iostream>
int main(int argc,char** argv) {
    if(argc<4) {
        std::cerr<<"Usage: yolo_video_cuda ENGINE SOURCE business|camera [--output FILE] [--no-display] [--no-graph] [--max-frames N] [--timing-csv CSV] [--capture-width W --capture-height H] [--capture-fps FPS] [--capture-fourcc MJPG] [--capture-buffers N]\n";
        return 1;
    }
    try {
        const auto old=video::parse(std::vector<std::string>(argv+1,argv+argc));
        edge::video::VideoConfig config;
        static_cast<edge::video::CaptureOptions&>(config)=old;
        config.models.push_back({"person","yolo",old.engine,"business",old.graph});
        edge::video::VideoHooks hooks;
        hooks.legacyPresentation=true;
        hooks.transform=[stage=old.stage](const cv::Mat& frame){return video::toBusinessFrame(frame,stage);};
        return edge::video::runVideo(config,{{"yolo",makeYoloModel}},hooks);
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
