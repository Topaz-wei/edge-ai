#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/postprocess_raw.hpp"


constexpr int NUM_CLASSES = 80;
constexpr int NUM_CANDIDATES = 8400;
constexpr int OUTPUT_CHANNELS = 84;

constexpr int PERSON_CLASS_ID = 0;

constexpr float CONF_THRESHOLD = 0.7F;
constexpr float IOU_THRESHOLD = 0.7F;

// 固定 ONNX 输入尺寸
constexpr float MODEL_WIDTH = 640.0F;
constexpr float MODEL_HEIGHT = 640.0F;

// 当前业务图尺寸
constexpr float BUSINESS_WIDTH = 360.0F;
constexpr float BUSINESS_HEIGHT = 640.0F;

// 360x640 -> 640x640
// ratio = 1
// 左右 padding = 140
constexpr float LETTERBOX_RATIO = 1.0F;
constexpr float PAD_X = 140.0F;
constexpr float PAD_Y = 0.0F;


static std::vector<float> loadFloatFile(
    const std::string& path)
{
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to open file: " + path
        );
    }

    file.seekg(
        0,
        std::ios::end
    );

    const std::size_t bytes =
        static_cast<std::size_t>(
            file.tellg()
        );

    if (bytes % sizeof(float) != 0) {
        throw std::runtime_error(
            "Invalid float binary file"
        );
    }

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<float> data(
        bytes / sizeof(float)
    );

    file.read(
        reinterpret_cast<char*>(
            data.data()
        ),
        static_cast<std::streamsize>(
            bytes
        )
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to read file"
        );
    }

    return data;
}


Box xywhToXyxy(
    float x,
    float y,
    float w,
    float h)
{
    Box box;

    box.x1 = x - w / 2.0F;
    box.y1 = y - h / 2.0F;
    box.x2 = x + w / 2.0F;
    box.y2 = y + h / 2.0F;

    return box;
}


float boxArea(
    const Box& box)
{
    const float width =
        std::max(
            0.0F,
            box.x2 - box.x1
        );

    const float height =
        std::max(
            0.0F,
            box.y2 - box.y1
        );

    return width * height;
}


float boxIoU(
    const Box& a,
    const Box& b)
{
    const float x1 =
        std::max(
            a.x1,
            b.x1
        );

    const float y1 =
        std::max(
            a.y1,
            b.y1
        );

    const float x2 =
        std::min(
            a.x2,
            b.x2
        );

    const float y2 =
        std::min(
            a.y2,
            b.y2
        );

    const float interWidth =
        std::max(
            0.0F,
            x2 - x1
        );

    const float interHeight =
        std::max(
            0.0F,
            y2 - y1
        );

    const float intersection =
        interWidth * interHeight;

    const float unionArea =
        boxArea(a) +
        boxArea(b) -
        intersection;

    if (unionArea <= 0.0F) {
        return 0.0F;
    }

    return intersection / unionArea;
}


Box restoreLetterbox(
    const Box& modelBox)
{
    Box box;

    box.x1 =
        (modelBox.x1 - PAD_X) /
        LETTERBOX_RATIO;

    box.y1 =
        (modelBox.y1 - PAD_Y) /
        LETTERBOX_RATIO;

    box.x2 =
        (modelBox.x2 - PAD_X) /
        LETTERBOX_RATIO;

    box.y2 =
        (modelBox.y2 - PAD_Y) /
        LETTERBOX_RATIO;

    // clip 到业务图范围
    box.x1 = std::clamp(
        box.x1,
        0.0F,
        BUSINESS_WIDTH
    );

    box.x2 = std::clamp(
        box.x2,
        0.0F,
        BUSINESS_WIDTH
    );

    box.y1 = std::clamp(
        box.y1,
        0.0F,
        BUSINESS_HEIGHT
    );

    box.y2 = std::clamp(
        box.y2,
        0.0F,
        BUSINESS_HEIGHT
    );

    return box;
}


std::vector<Detection> postprocess(
    const std::vector<float>& output)
{
    const std::size_t expectedElements =
        static_cast<std::size_t>(
            OUTPUT_CHANNELS
        ) *
        NUM_CANDIDATES;

    if (output.size() != expectedElements) {
        throw std::runtime_error(
            "Unexpected output element count"
        );
    }

    std::vector<Detection> candidates;

    // --------------------------------------------------
    // Output layout:
    //
    // [84, 8400]
    //
    // channel 0 -> x
    // channel 1 -> y
    // channel 2 -> w
    // channel 3 -> h
    //
    // channel 4..83 -> 80 class scores
    //
    // flattened:
    // output[channel * 8400 + candidate]
    // --------------------------------------------------

    for (
        int candidate = 0;
        candidate < NUM_CANDIDATES;
        ++candidate)
    {
        int bestClass = -1;
        float bestScore = -1.0F;

        // ------------------------------------------
        // 找这个 candidate 的最高分类
        // ------------------------------------------

        for (
            int classId = 0;
            classId < NUM_CLASSES;
            ++classId)
        {
            const int channel =
                4 + classId;

            const float score =
                output[
                    channel *
                    NUM_CANDIDATES +
                    candidate
                ];

            if (score > bestScore) {
                bestScore = score;
                bestClass = classId;
            }
        }

        // ------------------------------------------
        // 历史业务逻辑：
        //
        // class == person
        // confidence >= 0.7
        // ------------------------------------------

        if (
            bestClass != PERSON_CLASS_ID ||
            bestScore < CONF_THRESHOLD)
        {
            continue;
        }

        // ------------------------------------------
        // bbox
        // ------------------------------------------

        const float x =
            output[
                0 * NUM_CANDIDATES +
                candidate
            ];

        const float y =
            output[
                1 * NUM_CANDIDATES +
                candidate
            ];

        const float w =
            output[
                2 * NUM_CANDIDATES +
                candidate
            ];

        const float h =
            output[
                3 * NUM_CANDIDATES +
                candidate
            ];

        Detection det;

        det.candidate =
            candidate;

        det.classId =
            bestClass;

        det.confidence =
            bestScore;

        det.modelBox =
            xywhToXyxy(
                x,
                y,
                w,
                h
            );

        candidates.push_back(det);
    }

    std::cout
        << "Candidates before NMS: "
        << candidates.size()
        << "\n";

    // --------------------------------------------------
    // 按 confidence 从高到低排序
    // --------------------------------------------------

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](
            const Detection& a,
            const Detection& b)
        {
            return (
                a.confidence >
                b.confidence
            );
        }
    );

    // --------------------------------------------------
    // NMS
    // --------------------------------------------------

    std::vector<Detection> kept;

    for (
        const auto& current :
        candidates)
    {
        bool suppressed = false;

        for (
            const auto& selected :
            kept)
        {
            const float iou =
                boxIoU(
                    current.modelBox,
                    selected.modelBox
                );

            if (
                iou >
                IOU_THRESHOLD)
            {
                suppressed = true;
                break;
            }
        }

        if (!suppressed) {
            Detection det =
                current;

            det.businessBox =
                restoreLetterbox(
                    det.modelBox
                );

            kept.push_back(det);
        }
    }

    return kept;
}


void printBox(
    const Box& box)
{
    std::cout
        << "["
        << box.x1
        << ", "
        << box.y1
        << ", "
        << box.x2
        << ", "
        << box.y2
        << "]";
}


void printDetections(
    const std::vector<Detection>& detections)
{
    std::cout
        << "\nDetections: "
        << detections.size()
        << "\n";

    for (
        std::size_t i = 0;
        i < detections.size();
        ++i)
    {
        const auto& det =
            detections[i];

        std::cout
            << "\nDetection "
            << i
            << "\n";

        std::cout
            << "candidate : "
            << det.candidate
            << "\n";

        std::cout
            << "class     : person\n";

        std::cout
            << "confidence: "
            << det.confidence
            << "\n";

        std::cout
            << "model box : ";

        printBox(
            det.modelBox
        );

        std::cout
            << "\n";

        std::cout
            << "business  : ";

        printBox(
            det.businessBox
        );

        std::cout
            << "\n";
    }
}


#ifndef YOLO_NO_MAIN
int main(
    int argc,
    char** argv)
{
    if (argc != 2) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <output.bin>\n";

        return 1;
    }

    try {
        const std::string outputPath =
            argv[1];

        std::vector<float> output =
            loadFloatFile(
                outputPath
            );

        std::cout
            << std::fixed
            << std::setprecision(6);

        std::cout
            << "Output elements: "
            << output.size()
            << "\n";

        const auto detections =
            postprocess(
                output
            );

        printDetections(
            detections
        );

    } catch (
        const std::exception& e)
    {
        std::cerr
            << "ERROR: "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}
#endif
