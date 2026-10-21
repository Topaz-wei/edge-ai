#pragma once
#include <string>
#include <cstddef>
#include <algorithm>
#include <cctype>
#include <cmath>
namespace edge::video {
struct CaptureOptions {
    std::string source, output, timingCsv;
    bool camera=false;
    int cameraIndex=0, captureWidth=0, captureHeight=0, captureFps=0, captureBuffers=2;
    std::string captureFourcc;
    bool display=true;
    std::size_t maxFrames=0;
};
inline bool digits(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(),s.end(),[](unsigned char c){return std::isdigit(c);});
}
inline bool cameraSource(const std::string& s) {
    return digits(s) || s.rfind("/dev/video",0)==0 || s.rfind("/dev/v4l/",0)==0;
}
inline double outputFps(double value) { return std::isfinite(value)&&value>1 ? value:30; }
inline double fps(double ms) { return ms>0 ? 1000/ms:0; }
}
