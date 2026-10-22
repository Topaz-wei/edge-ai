#pragma once
#include "edge/video/capture_options.hpp"

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

using edge::video::digits;
using edge::video::fps;
using edge::video::outputFps;
struct Options : edge::video::CaptureOptions {
    std::string engine, stage;
    bool graph=true;
};

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
                else if (key == "--capture-buffers")
                    options.captureBuffers = value;
                else if (key == "--capture-fps")
                    options.captureFps = value;
                else
                    throw std::runtime_error("Unknown option: " + key);
            }
        } else if (key == "--output" || key == "--max-frames" || key == "--timing-csv") {
            if (++i >= args.size() || args[i].empty() || args[i].rfind("--", 0) == 0)
                throw std::runtime_error("Missing value: " + key);

            if (key == "--timing-csv") {
                options.timingCsv = args[i];
            } else if (key == "--output") {
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

} // namespace video
