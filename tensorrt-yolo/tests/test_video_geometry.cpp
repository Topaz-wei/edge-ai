#include "include/yolo_video_geometry.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main(){
    try {
        cv::Rect2f box(10,20,30,40);
        if(yolo::toSourceBox(box,cv::Size(360,640),"business")!=box)throw std::runtime_error("Business mapping");
        auto b=yolo::toSourceBox(box,cv::Size(1280,720),"camera");
        if(b!=cv::Rect2f(40,20,80,60))throw std::runtime_error("Transpose mapping");
        b=yolo::toSourceBox(box,cv::Size(1281,721),"camera");
        if(std::abs(b.x-20*1281.f/640)>1e-4 || std::abs(b.y-10*721.f/360)>1e-4)throw std::runtime_error("Rounded resize mapping");
        std::cout<<"Business and camera geometry mapping passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
