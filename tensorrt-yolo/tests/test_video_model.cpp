#include "include/yolo_video_model.hpp"
#include "include/cuda_yolo.hpp"
#include "include/yolo_video_geometry.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("Expected engine image");
        auto original=cv::imread(argv[2],cv::IMREAD_COLOR|cv::IMREAD_IGNORE_ORIENTATION);
        for(bool graph:{false,true}) {
            CudaYoloPipeline reference(argv[1],graph);
            auto expected=reference.infer(original);
            for(const std::string stage:{"business","camera"}) {
                cv::Mat frame;
                if(stage=="business")frame=original.clone();else cv::transpose(original,frame);
                auto before=frame.clone();
                auto model=makeYoloModel({"test","yolo",argv[1],stage,graph});
                auto result=model->infer({frame,1,0});
                if(cv::norm(before,frame,cv::NORM_INF)!=0 || result.detections.size()!=expected.size())
                    throw std::runtime_error("Input or detection count differs");
                for(std::size_t i=0;i<expected.size();++i) {
                    const auto& e=expected[i]; const auto& b=e.businessBox;
                    auto box=yolo::toSourceBox({b.x1,b.y1,b.x2-b.x1,b.y2-b.y1},frame.size(),stage);
                    const auto& actual=result.detections[i];
                    if(actual.box!=box || actual.score!=e.confidence || actual.classId!=e.classId)
                        throw std::runtime_error("Adapter changes detection");
                }
            }
        }
        std::cout<<"YOLO video adapter Graph on/off, camera/business and input preservation passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
