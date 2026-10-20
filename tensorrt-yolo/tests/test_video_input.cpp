#include "include/video_input.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdexcept>

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("video input assertion failed");
}

template <typename Function>
void rejects(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected);
}

int main() {
    cv::Mat business(640, 360, CV_8UC3);
    cv::randu(business, 0, 255);
    require(cv::norm(video::toBusinessFrame(business, "business"), business) == 0);

    cv::Mat raw;
    cv::transpose(business, raw);
    require(cv::norm(video::toBusinessFrame(raw, "camera"), business) == 0);

    // A wider 16:9 source must preserve the historical transpose + area resize.
    cv::Mat large(720, 1280, CV_8UC3);
    cv::randu(large, 0, 255);
    cv::Mat expected;
    cv::transpose(large, expected);
    cv::resize(expected, expected, cv::Size(360, 640), 0, 0, cv::INTER_AREA);
    require(cv::norm(video::toBusinessFrame(large, "camera"), expected) == 0);

    rejects([&] { video::toBusinessFrame(raw, "business"); });
    rejects([&] { video::toBusinessFrame(business, "unknown"); });
    rejects([] { video::toBusinessFrame(cv::Mat(), "business"); });
    rejects([] { video::toBusinessFrame(cv::Mat(640, 360, CV_8UC1), "business"); });
    rejects([] { video::toBusinessFrame(cv::Mat(480, 640, CV_8UC3), "camera"); });

    auto options = video::parse({"engine.plan", "0", "camera", "--no-display", "--max-frames", "3"});
    require(options.camera && options.cameraIndex == 0 && !options.display && options.maxFrames == 3);
    require(!video::parse({"e", "./0", "business"}).camera);
    rejects([] { video::parse({"e", "v", "wrong"}); });
    rejects([] { video::parse({"e", "v", "business", "--output"}); });
    rejects([] { video::parse({"e", "v", "business", "--max-frames", "0"}); });
    rejects([] { video::parse({"e", "v", "business", "--max-frames", "3x"}); });
    rejects([] { video::parse({"e", "99999999999999999999999", "camera"}); });
    rejects([] { video::parse({"e", "v", "business", "--no-graph", "--no-graph"}); });
    rejects([] { video::parse({"e", "v", "business", "--typo"}); });
    auto usb = video::parse({
        "e", "/dev/v4l/by-id/usb-test-video-index0", "camera",
        "--capture-width", "1280", "--capture-height", "720",
        "--capture-fps", "30", "--capture-fourcc", "MJPG"
    });
    require(usb.camera && usb.captureWidth == 1280 && usb.captureHeight == 720);
    require(usb.captureFps == 30 && usb.captureFourcc == "MJPG");
    rejects([] { video::parse({"e", "v.mp4", "camera", "--capture-fps", "30"}); });
    rejects([] { video::parse({"e", "8", "camera", "--capture-width", "1280"}); });
    rejects([] { video::parse({"e", "8", "camera", "--capture-fps", "0"}); });
    rejects([] { video::parse({"e", "8", "camera", "--capture-fourcc", "MJPEG"}); });
    require(video::outputFps(0) == 30 && video::outputFps(25) == 25);
    require(video::outputFps(std::numeric_limits<double>::infinity()) == 30);
    require(video::fps(0) == 0);
    std::cout << "video input tests passed\n";
}
