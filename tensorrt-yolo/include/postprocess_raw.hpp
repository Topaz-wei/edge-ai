#ifndef POSTPROCESS_RAW_HPP_
#define POSTPROCESS_RAW_HPP_

#include <vector>

struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
};


struct Detection {
    int candidate;
    int classId;
    float confidence;

    Box modelBox;
    Box businessBox;
};


Box xywhToXyxy(float x, float y, float w, float h);
float boxArea(const Box& box);
float boxIoU(const Box& a, const Box& b);
Box restoreLetterbox(const Box& modelBox);
std::vector<Detection> postprocess(const std::vector<float>& output, bool verbose = true);
void printBox(const Box& box);
void printDetections(const std::vector<Detection>& detections);

#endif
