#include "edge/video/pipeline.hpp"
#include <opencv2/videoio.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace edge::video;
struct Observer: IModel {
    int& calls; int expected; std::vector<std::string>& order; std::string name;
    Observer(int& c,int e,std::vector<std::string>& o,std::string n):calls(c),expected(e),order(o),name(n){}
    ModelResult infer(const Frame& f) override {
        if(f.bgr.at<cv::Vec3b>(0,0)[0]!=expected) throw std::runtime_error("Input modified");
        ++calls; order.push_back(name);
        return {{{cv::Rect2f(1,1,5,5),0,.8f,"test"}}, {}, {}};
    }
};
struct RejectGeometry : IModel {
    void validateFrame(const cv::Mat&) const override { throw std::runtime_error("Unsupported geometry"); }
    ModelResult infer(const Frame& f) override { validateFrame(f.bgr); return {}; }
};
int main() {
    auto dir=std::filesystem::temp_directory_path()/("edge-video-"+std::to_string(getpid()));
    std::filesystem::create_directories(dir); std::ofstream(dir/"dummy.plan")<<"fixture";
    try {
        int calls=0; std::vector<std::string> order;
        ModelRegistry registry{{"fixture",[&](const ModelConfig& c){return std::make_unique<Observer>(calls,32,order,c.id);}}};
        std::vector<ModelConfig> configs{{"a","fixture",dir/"dummy.plan","",false},{"b","fixture",dir/"dummy.plan","",false}};
        ModelSet models(configs,registry); cv::Mat frame(64,64,CV_8UC3,cv::Scalar(32,32,32));
        auto result=models.infer({frame,7,0});
        if(calls!=2 || order!=std::vector<std::string>{"a","b"} || result[0].id!="a" || result[1].id!="b") throw std::runtime_error("Dispatch mismatch");
        bool rejected=false; auto invalid=configs; invalid[0].type="unknown";
        try{ModelSet bad(invalid,registry);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("Unknown model accepted");
        auto input=(dir/"input.avi").string(); cv::VideoWriter writer(input,cv::VideoWriter::fourcc('M','J','P','G'),30,frame.size());
        if(!writer.isOpened()) throw std::runtime_error("Fixture encoder unavailable");
        for(int i=0;i<4;++i)writer.write(frame); writer.release();
        VideoConfig c; c.source=input; c.display=false; c.maxFrames=3; c.models=configs; c.timingCsv=(dir/"times.csv").string();
        calls=0; order.clear(); runVideo(c,registry);
        if(calls!=6)throw std::runtime_error("Frame limit/multi-model dispatch failed");
        std::ifstream csv(c.timingCsv);std::string line;int lines=0;while(std::getline(csv,line))++lines;
        if(lines!=4)throw std::runtime_error("CSV row count");
        rejected=false;try{runVideo(c,registry);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("Existing output accepted");
        c.timingCsv.clear(); c.output=(dir/"must-not-exist.mp4").string();
        ModelRegistry invalidGeometry{{"fixture",[](const ModelConfig&){return std::make_unique<RejectGeometry>();}}};
        rejected=false;
        try{runVideo(c,invalidGeometry);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected || std::filesystem::exists(c.output))throw std::runtime_error("Invalid geometry created output");
        c.output.clear();
        c.timingCsv.clear();c.source=(dir/"missing.avi").string();
        rejected=false;try{runVideo(c,registry);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("Missing source accepted");
        std::filesystem::remove_all(dir);std::cout<<"Dispatch, input preservation, frame limits and output protection passed\n";
    } catch(const std::exception& e){std::filesystem::remove_all(dir);std::cerr<<e.what()<<'\n';return 1;}
}
