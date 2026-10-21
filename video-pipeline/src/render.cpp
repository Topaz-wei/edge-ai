#include "edge/video/pipeline.hpp"
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iomanip>
namespace edge::video {
void renderResults(cv::Mat& image,const std::vector<InstanceResult>& results,bool legacy) {
    int textY=70;
    for(const auto& instance:results) {
        for(const auto& d:instance.result.detections) {
            const auto& b=d.box;
            cv::rectangle(image,cv::Point(cvRound(b.x),cvRound(b.y)),
                cv::Point(cvRound(b.x+b.width),cvRound(b.y+b.height)),cv::Scalar(0,255,0),2);
            std::ostringstream label;
            if(!legacy)label<<instance.id<<": ";
            label<<d.label<<" "<<std::fixed<<std::setprecision(3)<<d.score;
            cv::putText(image,label.str(),cv::Point(cvRound(b.x),std::max(20,cvRound(b.y)-5)),
                cv::FONT_HERSHEY_SIMPLEX,.5,cv::Scalar(0,255,0),1,cv::LINE_AA);
        }
        for(const auto& c:instance.result.classifications) {
            cv::putText(image,instance.id+": "+c.label+" "+std::to_string(c.score),{8,textY},
                cv::FONT_HERSHEY_SIMPLEX,.5,cv::Scalar(0,255,0),1,cv::LINE_AA);textY+=20;
        }
    }
}
}
