#include "edge/video/pipeline.hpp"
#include "include/yolo_video_model.hpp"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::runtime_error("Usage: edge_video CONFIG.yaml");
        return edge::video::runVideo(edge::video::loadConfig(argv[1]),{{"yolo",makeYoloModel}});
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
