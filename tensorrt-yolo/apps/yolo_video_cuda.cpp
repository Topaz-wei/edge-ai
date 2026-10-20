#include "include/cuda_yolo.hpp"
#include "include/video_input.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(
    Clock::time_point begin,
    Clock::time_point end
) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void drawDetections(
    cv::Mat& image,
    const std::vector<Detection>& detections
) {
    for (const auto& detection : detections) {
        const auto& box = detection.businessBox;
        cv::rectangle(
            image,
            cv::Point(cvRound(box.x1), cvRound(box.y1)),
            cv::Point(cvRound(box.x2), cvRound(box.y2)),
            cv::Scalar(0, 255, 0),
            2
        );

        std::ostringstream label;
        label << "person " << std::fixed << std::setprecision(3)
              << detection.confidence;
        cv::putText(
            image,
            label.str(),
            cv::Point(cvRound(box.x1), std::max(20, cvRound(box.y1) - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            1,
            cv::LINE_AA
        );
    }
}

void drawTiming(
    cv::Mat& image,
    const std::string& name,
    double milliseconds,
    int y
) {
    std::ostringstream text;
    text << name << " " << std::fixed << std::setprecision(1)
         << milliseconds << " ms / " << video::fps(milliseconds) << " FPS";
    cv::putText(
        image,
        text.str(),
        cv::Point(8, y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.42,
        cv::Scalar(0, 255, 255),
        1,
        cv::LINE_AA
    );
}

int run(const video::Options& options) {
    // Reject existing output before opening any writer: source files and engines
    // must never be silently truncated, including through a symlink alias.
    if (!options.output.empty()) {
        if (std::filesystem::exists(options.output) ||
            std::filesystem::is_symlink(std::filesystem::symlink_status(options.output)))
            throw std::runtime_error("Output already exists: " + options.output);
    }

    cv::VideoCapture capture;
    if (options.camera) {
        if (video::digits(options.source))
            capture.open(options.cameraIndex, cv::CAP_V4L2);
        else
            capture.open(options.source, cv::CAP_V4L2);
    }
    else
        capture.open(options.source);
    if (!capture.isOpened())
        throw std::runtime_error("Cannot open video source: " + options.source);

    // Set compressed format before dimensions/FPS; YUYV has lower USB bandwidth limits.
    if (options.camera) {
        auto set = [&](int property, double value, const char* name) {
            if (!capture.set(property, value))
                throw std::runtime_error(std::string("Camera rejected setting: ") + name);
        };
        if (!options.captureFourcc.empty()) {
            const auto& code = options.captureFourcc;
            set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(code[0], code[1], code[2], code[3]), "FOURCC");
        }
        if (options.captureWidth) {
            set(cv::CAP_PROP_FRAME_WIDTH, options.captureWidth, "width");
            set(cv::CAP_PROP_FRAME_HEIGHT, options.captureHeight, "height");
        }
        if (options.captureFps)
            set(cv::CAP_PROP_FPS, options.captureFps, "FPS");
        if (!capture.set(cv::CAP_PROP_BUFFERSIZE, 1))
            std::cerr << "Warning: camera did not accept one capture buffer\n";
    }

    const double sourceFps = capture.get(cv::CAP_PROP_FPS);
    const double writeFps = video::outputFps(sourceFps);
    std::cout << "Source FPS: " << sourceFps << "; writer FPS: " << writeFps << '\n';

    // Validate the first decoded frame before allocating GPU resources.
    cv::Mat frame;
    if (!capture.read(frame) || frame.empty())
        throw std::runtime_error("Source has no decodable first frame");
    std::cout << "Actual capture: " << frame.cols << "x" << frame.rows
              << "; backend: " << capture.getBackendName() << std::endl;
    if (options.captureWidth &&
        (frame.cols != options.captureWidth || frame.rows != options.captureHeight))
        throw std::runtime_error("Actual camera dimensions differ from requested dimensions");
    video::toBusinessFrame(frame, options.stage);

    CudaYoloPipeline pipeline(options.engine, options.graph);
    cv::VideoWriter writer;
    if (!options.output.empty()) {
        const auto parent = std::filesystem::path(options.output).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        writer.open(
            options.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            writeFps,
            cv::Size(360, 640)
        );
        if (!writer.isOpened())
            throw std::runtime_error("Cannot open output video: " + options.output);
    }

    std::size_t count = 0;
    double pipelineEma = 0.0;
    double loopEma = 0.0;
    constexpr double alpha = 0.1;

    // First-frame setup is excluded from steady-loop timing. The first pipeline
    // call is still included; this demo is not a warmed-up benchmark.
    while (true) {
        const auto loopBegin = Clock::now();
        if (count > 0 && !capture.read(frame)) {
            if (options.camera)
                throw std::runtime_error("Camera frame read failed");
            std::cout << "Video read ended (EOF or decoder failure)\n";
            break;
        }

        cv::Mat business = video::toBusinessFrame(frame, options.stage);
        const auto inferBegin = Clock::now();
        const auto detections = pipeline.infer(business);
        const double pipelineMs = elapsedMs(inferBegin, Clock::now());
        pipelineEma = count == 0 ? pipelineMs : alpha * pipelineMs + (1.0 - alpha) * pipelineEma;

        bool quit = false;
        if (options.display || writer.isOpened()) {
            // Keep annotations out of subsequent preprocessing/inference.
            cv::Mat visualization = business.clone();
            drawDetections(visualization, detections);
            drawTiming(visualization, "Pipeline EMA", pipelineEma, 24);
            if (count > 0)
                drawTiming(visualization, "Loop EMA (prev)", loopEma, 44);

            if (writer.isOpened())
                writer.write(visualization);
            if (options.display) {
                cv::imshow("YOLO TensorRT", visualization);
                const int key = cv::waitKey(1);
                quit = key == 27 || key == 'q';
            }
        }

        const double loopMs = elapsedMs(loopBegin, Clock::now());
        loopEma = count == 0 ? loopMs : alpha * loopMs + (1.0 - alpha) * loopEma;
        ++count;  // Count the processed frame even when q/Esc was pressed.

        if (count == 1 || count % 30 == 0)
            std::cout << "Frame " << count << ": detections=" << detections.size()
                      << ", pipeline_ms=" << pipelineMs << std::endl;

        if (quit || (options.maxFrames && count >= options.maxFrames))
            break;
    }

    // Flush encoded output before reporting completion. Capture/writer also use RAII.
    writer.release();
    if (options.display)
        cv::destroyAllWindows();

    std::cout << "Frames: " << count << '\n'
              << std::fixed << std::setprecision(3)
              << "Pipeline EMA: " << pipelineEma << " ms, " << video::fps(pipelineEma) << " FPS\n"
              << "Full loop EMA: " << loopEma << " ms, " << video::fps(loopEma) << " FPS\n";
    return 0;
}

}  // namespace

int main(
    int argc,
    char** argv
) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine.plan> <video_path|camera_index> <business|camera>"
                  << " [--output output.mp4] [--no-display] [--no-graph]"
                  << " [--max-frames N]"
                  << " [--capture-width W --capture-height H]"
                  << " [--capture-fps FPS] [--capture-fourcc MJPG]\n";
        return 1;
    }

    try {
        return run(video::parse(std::vector<std::string>(argv + 1, argv + argc)));
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
