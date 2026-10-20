#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace video {

struct Options {
    std::string engine;
    std::string source;
    std::string stage;
    std::string output;
    bool camera = false;
    int cameraIndex = 0;
    int captureWidth = 0;
    int captureHeight = 0;
    int captureFps = 0;
    std::string captureFourcc;
    bool display = true;
    bool graph = true;
    std::size_t maxFrames = 0;  // Zero means no limit when the option is omitted.
};

inline bool digits(const std::string& value) {
    return !value.empty() && std::all_of(
        value.begin(),
        value.end(),
        [](unsigned char c) { return std::isdigit(c); }
    );
}

inline Options parse(const std::vector<std::string>& args) {
    if (args.size() < 3)
        throw std::runtime_error("Expected engine, source and input stage");

    Options options;
    options.engine = args[0];
    options.source = args[1];
    options.stage = args[2];
    if (options.stage != "business" && options.stage != "camera")
        throw std::runtime_error("Input stage must be business or camera");

    options.camera = digits(options.source) ||
        options.source.rfind("/dev/video", 0) == 0 ||
        options.source.rfind("/dev/v4l/", 0) == 0;
    if (digits(options.source))
        options.cameraIndex = std::stoi(options.source);

    std::set<std::string> seen;
    for (std::size_t i = 3; i < args.size(); ++i) {
        const auto& key = args[i];
        if (!seen.insert(key).second)
            throw std::runtime_error("Duplicate option: " + key);

        if (key == "--no-display") {
            options.display = false;
        } else if (key == "--no-graph") {
            options.graph = false;
        } else if (key.rfind("--capture-", 0) == 0) {
            if (!options.camera)
                throw std::runtime_error("Capture settings require a camera device");
            if (++i >= args.size())
                throw std::runtime_error("Missing value: " + key);
            if (key == "--capture-fourcc") {
                if (args[i].size() != 4)
                    throw std::runtime_error("Capture FOURCC must have four characters");
                options.captureFourcc = args[i];
            } else {
                if (!digits(args[i]))
                    throw std::runtime_error("Capture setting requires a positive integer");
                const int value = std::stoi(args[i]);
                if (value <= 0)
                    throw std::runtime_error("Capture setting must be positive");
                if (key == "--capture-width")
                    options.captureWidth = value;
                else if (key == "--capture-height")
                    options.captureHeight = value;
                else if (key == "--capture-fps")
                    options.captureFps = value;
                else
                    throw std::runtime_error("Unknown option: " + key);
            }
        } else if (key == "--output" || key == "--max-frames") {
            if (++i >= args.size() || args[i].empty() || args[i].rfind("--", 0) == 0)
                throw std::runtime_error("Missing value: " + key);

            if (key == "--output") {
                options.output = args[i];
            } else {
                if (!digits(args[i]))
                    throw std::runtime_error("--max-frames requires a positive integer");
                const auto count = std::stoull(args[i]);
                if (count == 0 || count > std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error("Invalid --max-frames value");
                options.maxFrames = static_cast<std::size_t>(count);
            }
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }

    if ((options.captureWidth == 0) != (options.captureHeight == 0))
        throw std::runtime_error("Capture width and height must be supplied together");

    return options;
}

inline cv::Mat toBusinessFrame(
    const cv::Mat& input,
    const std::string& stage
) {
    if (input.empty() || input.type() != CV_8UC3)
        throw std::runtime_error("Expected a nonempty CV_8UC3 BGR frame");

    if (stage == "business") {
        if (input.cols != 360 || input.rows != 640)
            throw std::runtime_error("Business input must be width=360 height=640");
        return input;
    }
    if (stage != "camera")
        throw std::runtime_error("Input stage must be business or camera");

    // Match prepare_yolo_reference.py: transpose (not rotation), then resize.
    cv::Mat transposed;
    cv::transpose(input, transposed);
    const double scale = 640.0 / transposed.rows;
    const int width = static_cast<int>(transposed.cols * scale);
    if (width != 360)
        throw std::runtime_error("Camera geometry must produce width=360 height=640");

    cv::Mat business;
    cv::resize(
        transposed,
        business,
        cv::Size(width, 640),
        0.0,
        0.0,
        cv::INTER_AREA
    );
    return business;
}

inline double outputFps(double sourceFps) {
    return std::isfinite(sourceFps) && sourceFps > 1.0 ? sourceFps : 30.0;
}

inline double fps(double milliseconds) {
    return milliseconds > 0.0 ? 1000.0 / milliseconds : 0.0;
}

}  // namespace video
