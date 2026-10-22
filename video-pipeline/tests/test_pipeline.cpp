#include "edge/video/pipeline.hpp"
#include <opencv2/videoio.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <sstream>
using namespace edge::video;
struct Observer: IModel {
    int& calls; int expected; std::vector<std::string>& order; std::string name;
    Observer(int& c,int e,std::vector<std::string>& o,std::string n):calls(c),expected(e),order(o),name(n){}
    ModelResult infer(const PreparedInput& input) override {
        const auto& f=input.source;
        if(f.bgr.at<cv::Vec3b>(0,0)[0]!=expected) throw std::runtime_error("Input modified");
        ++calls; order.push_back(name);
        return {{{cv::Rect2f(1,1,5,5),0,.8f,"test"}}, {}, {}};
    }
};
struct RejectGeometry : IModel {
    void validateFrame(const cv::Mat&) const override { throw std::runtime_error("Unsupported geometry"); }
    ModelResult infer(const PreparedInput& input) override {
        const auto& f=input.source; validateFrame(f.bgr); return {}; }
};
// Fails if preparation remains serialized with inference, or a live slot is reused.
struct OverlapState {
    std::mutex mutex;
    std::condition_variable changed;
    uint64_t prepared=0, consumed=0;
};
struct OverlapModel : IModel {
    OverlapState& state;
    explicit OverlapModel(OverlapState& s):state(s){}
    std::unique_ptr<PreparedInput> prepare(const Frame& frame) const override {
        auto input=std::make_unique<PreparedInput>(frame);
        std::lock_guard<std::mutex> lock(state.mutex);
        if(frame.id > state.consumed+2)throw std::runtime_error("More than two live frames");
        state.prepared=frame.id;
        state.changed.notify_all();
        return input;
    }
    ModelResult infer(const PreparedInput& input) override {
        const auto id=input.source.id;
        const auto pixels=input.source.bgr.clone();
        if(pixels.at<cv::Vec3b>(0,0)[0]!=id*32)throw std::runtime_error("Frame pixels/identity mismatch");
        std::unique_lock<std::mutex> lock(state.mutex);
        if(id==1 && !state.changed.wait_for(lock,std::chrono::seconds(2),[&]{return state.prepared>=2;}))
            throw std::runtime_error("Next-frame preparation did not overlap inference");
        if(id!=state.consumed+1)throw std::runtime_error("Frame order changed");
        if(cv::norm(pixels,input.source.bgr,cv::NORM_INF)!=0)throw std::runtime_error("Live frame overwritten");
        state.consumed=id;
        return {};
    }
};
struct FailingModel : IModel {
    bool failPrepare;
    int& calls;
    FailingModel(bool p,int& c):failPrepare(p),calls(c){}
    std::unique_ptr<PreparedInput> prepare(const Frame& frame) const override {
        if(failPrepare && frame.id==2)throw std::runtime_error("prepare failed");
        return std::make_unique<PreparedInput>(frame);
    }
    ModelResult infer(const PreparedInput& input) override {
        if(!failPrepare && input.source.id==2)throw std::runtime_error("infer failed");
        ++calls; return {};
    }
};
int main() {
    auto dir=std::filesystem::temp_directory_path()/("edge-video-"+std::to_string(getpid()));
    std::filesystem::create_directories(dir); std::ofstream(dir/"dummy.plan")<<"fixture";
    try {
        int calls=0; std::vector<std::string> order;
        ModelRegistry registry{{"fixture",[&](const ModelConfig& c){return std::make_unique<Observer>(calls,32,order,c.id);}}};
        std::vector<ModelConfig> configs{{"a","fixture",dir/"dummy.plan","",false},{"b","fixture",dir/"dummy.plan","",false}};
        ModelSet models(configs,registry); cv::Mat frame(64,64,CV_8UC3,cv::Scalar(32,32,32));
        auto prepared=models.prepare({frame,7,0});
        auto result=models.infer(prepared);
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
        // EOF and a one-frame limit must not require a second frame.
        c.models.resize(1);c.maxFrames=0;calls=0;
        runVideo(c,registry);
        if(calls!=4)throw std::runtime_error("EOF frame count");
        c.maxFrames=1;calls=0;runVideo(c,registry);
        if(calls!=1)throw std::runtime_error("Single frame limit");
        for(bool failPrepare:{true,false}) {
            calls=0;c.maxFrames=4;
            ModelRegistry failing{{"fixture",[&](const ModelConfig&){return std::make_unique<FailingModel>(failPrepare,calls);}}};
            rejected=false;
            try{runVideo(c,failing);}catch(const std::runtime_error& e){
                rejected=std::string(e.what())==(failPrepare?"prepare failed":"infer failed");
            }
            if(!rejected || calls!=1)throw std::runtime_error("Ordered error propagation failed");
        }
        auto varying=(dir/"varying.avi").string();
        writer.open(varying,cv::VideoWriter::fourcc('F','F','V','1'),30,frame.size());
        if(!writer.isOpened())throw std::runtime_error("Lossless fixture encoder unavailable");
        for(int i=1;i<=4;++i)writer.write(cv::Mat(64,64,CV_8UC3,cv::Scalar(i*32,i*32,i*32)));
        writer.release();c.source=varying;c.timingCsv=(dir/"overlap.csv").string();
        OverlapState state;
        ModelRegistry overlapping{{"fixture",[&](const ModelConfig&){return std::make_unique<OverlapModel>(state);}}};
        c.models.resize(1); c.maxFrames=3;
        runVideo(c,overlapping);
        if(state.prepared!=3 || state.consumed!=3)throw std::runtime_error("Speculative frames beyond limit");
        std::ifstream overlapCsv(c.timingCsv);
        std::getline(overlapCsv,line);
        if(line.find("prepare_ms,queue_wait_ms")==std::string::npos)throw std::runtime_error("Missing timing stages");
        std::vector<std::vector<double>> rows;
        while(std::getline(overlapCsv,line)) {
            std::istringstream row(line);std::string field;std::vector<double> values;
            while(std::getline(row,field,','))values.push_back(std::stod(field));
            if(values.size()!=16 || std::abs(values[9]-(values[2]-values[1]))>0.00001 || values[13]<0 || values[14]<0)
                throw std::runtime_error("Invalid latency/timing row");
            rows.push_back(std::move(values));
        }
        if(rows.size()!=3 || rows[1][1]>=rows[0][2])throw std::runtime_error("Timing does not show overlapping frames");
        c.timingCsv.clear();c.source=(dir/"missing.avi").string();
        rejected=false;try{runVideo(c,registry);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("Missing source accepted");
        std::filesystem::remove_all(dir);std::cout<<"Dispatch, input preservation, frame limits and output protection passed\n";
    } catch(const std::exception& e){std::filesystem::remove_all(dir);std::cerr<<e.what()<<'\n';return 1;}
}
