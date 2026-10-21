#include "edge/video/pipeline.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <array>
#include <time.h>
#include <iostream>
#include <sstream>

namespace edge::video {
namespace {

using Clock = std::chrono::steady_clock;

// V4L2 frame timestamps on this Linux device use CLOCK_MONOTONIC. Keep the
// native timestamp separate from the relative wall-clock timings in the CSV.
double monotonicMs() {
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        throw std::runtime_error("Cannot read CLOCK_MONOTONIC");
    return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}

double elapsedMs(
    Clock::time_point begin,
    Clock::time_point end
) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void drawTiming(
    cv::Mat& image,
    const std::string& name,
    double milliseconds,
    int y
) {
    std::ostringstream text;
    text << name << " " << std::fixed << std::setprecision(1)
         << milliseconds << " ms / " << fps(milliseconds) << " FPS";
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

} // namespace

int runVideo(const VideoConfig& options, const ModelRegistry& registry, const VideoHooks& hooks) {
    // Reject existing output before opening any writer: source files and engines
    // must never be silently truncated, including through a symlink alias.
    if (!options.output.empty()) {
        if (std::filesystem::exists(options.output) ||
            std::filesystem::is_symlink(std::filesystem::symlink_status(options.output)))
            throw std::runtime_error("Output already exists: " + options.output);
    }

    if (!options.timingCsv.empty()) {
        if (std::filesystem::exists(options.timingCsv) ||
            std::filesystem::is_symlink(std::filesystem::symlink_status(options.timingCsv)))
            throw std::runtime_error("Timing CSV already exists: " + options.timingCsv);
        if (!options.output.empty() &&
            std::filesystem::weakly_canonical(options.output) ==
                std::filesystem::weakly_canonical(options.timingCsv))
            throw std::runtime_error("Timing CSV and video output must differ");
    }

    ModelSet models(options.models, registry);
    cv::VideoCapture capture;
    if (options.camera) {
        if (digits(options.source))
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
        set(cv::CAP_PROP_BUFFERSIZE, options.captureBuffers, "buffers");
        const int actualBuffers = cvRound(capture.get(cv::CAP_PROP_BUFFERSIZE));
        std::cout << "Capture buffers: requested=" << options.captureBuffers
                  << ", actual=" << actualBuffers << std::endl;
        if (actualBuffers != options.captureBuffers)
            throw std::runtime_error("Actual capture buffer count differs from request");
    }

    const double sourceFps = capture.get(cv::CAP_PROP_FPS);
    const double writeFps = outputFps(sourceFps);
    std::cout << "Source FPS: " << sourceFps << "; writer FPS: " << writeFps << '\n';

    // Validate the first decoded frame for every model before creating outputs.
    cv::Mat frame;
    if (!capture.read(frame) || frame.empty())
        throw std::runtime_error("Source has no decodable first frame");
    std::cout << "Actual capture: " << frame.cols << "x" << frame.rows
              << "; backend: " << capture.getBackendName() << std::endl;
    if (options.captureWidth &&
        (frame.cols != options.captureWidth || frame.rows != options.captureHeight))
        throw std::runtime_error("Actual camera dimensions differ from requested dimensions");
    const auto firstView = hooks.transform ? hooks.transform(frame) : frame;
    models.validateFrame(firstView);
    const auto viewSize = firstView.size();
    cv::VideoWriter writer;
    if (!options.output.empty()) {
        const auto parent = std::filesystem::path(options.output).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        writer.open(
            options.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            writeFps,
            viewSize
        );
        if (!writer.isOpened())
            throw std::runtime_error("Cannot open output video: " + options.output);
    }

    std::size_t count = 0;
    double pipelineEma = 0.0;
    double loopEma = 0.0;
    constexpr double alpha = 0.1;
    // Retain measurements in memory; write CSV after capture to avoid per-frame disk I/O.
    // start,end,read,geometry,pipeline,draw,encode,display,total,
    // capture timestamp, age at read return, age at completion (all milliseconds).
    // Ages are diagnostic, not exposure-to-screen latency; -1 means unavailable.
    std::vector<std::array<double, 12>> timings;
    std::vector<std::vector<double>> modelTimings;
    const auto runBegin = Clock::now();

    // First-frame setup is excluded from steady-loop timing. The first pipeline
    // call is still included; this demo is not a warmed-up benchmark.
    while (true) {
        const auto loopBegin = Clock::now();
        std::array<double, 12> timing{};
        timing[9] = timing[10] = timing[11] = -1.0;
        timing[0] = elapsedMs(runBegin, loopBegin);
        const auto readBegin = Clock::now();
        if (count > 0 && !capture.read(frame)) {
            if (options.camera)
                throw std::runtime_error("Camera frame read failed");
            std::cout << "Video read ended (EOF or decoder failure)\n";
            break;
        }

        const auto readEnd = Clock::now();
        timing[2] = count == 0 ? 0.0 : elapsedMs(readBegin, readEnd);
        if (options.camera && !options.timingCsv.empty()) {
            const double timestamp = capture.get(cv::CAP_PROP_POS_MSEC);
            const double age = monotonicMs() - timestamp;
            // Reject unsupported/foreign-clock values instead of reporting fake latency.
            if (std::isfinite(timestamp) && timestamp > 0.0 && age >= 0.0 && age < 10000.0) {
                timing[9] = timestamp;
                timing[10] = age;
            }
        }
        const auto geometryBegin = Clock::now();
        cv::Mat business = hooks.transform ? hooks.transform(frame) : frame;
        if (business.size() != viewSize) throw std::runtime_error("Video dimensions changed");
        timing[3] = elapsedMs(geometryBegin, Clock::now());
        const auto inferBegin = Clock::now();
        const auto results = models.infer({business, count + 1, timing[9]});
        std::size_t detectionCount = 0;
        for (const auto& r : results) detectionCount += r.result.detections.size();
        const double pipelineMs = elapsedMs(inferBegin, Clock::now());
        timing[4] = pipelineMs;
        pipelineEma = count == 0 ? pipelineMs : alpha * pipelineMs + (1.0 - alpha) * pipelineEma;

        bool quit = false;
        if (options.display || writer.isOpened()) {
            // Keep annotations out of subsequent preprocessing/inference.
            const auto drawBegin = Clock::now();
            cv::Mat visualization = business.clone();
            renderResults(visualization, results, hooks.legacyPresentation);
            drawTiming(visualization, "Pipeline EMA", pipelineEma, 24);
            if (count > 0)
                drawTiming(visualization, "Loop EMA (prev)", loopEma, 44);

            timing[5] = elapsedMs(drawBegin, Clock::now());
            if (writer.isOpened()) {
                const auto encodeBegin = Clock::now();
                writer.write(visualization);
                timing[6] = elapsedMs(encodeBegin, Clock::now());
            }
            if (options.display) {
                const auto displayBegin = Clock::now();
                cv::imshow(hooks.legacyPresentation ? "YOLO TensorRT" : "edge-ai", visualization);
                const int key = cv::waitKey(1);
                quit = key == 27 || key == 'q';
                timing[7] = elapsedMs(displayBegin, Clock::now());
            }
        }

        const auto loopEnd = Clock::now();
        const double loopMs = elapsedMs(loopBegin, loopEnd);
        timing[1] = elapsedMs(runBegin, loopEnd);
        timing[8] = loopMs;
        if (timing[9] >= 0.0)
            timing[11] = monotonicMs() - timing[9];
        if (!options.timingCsv.empty()) {
            timings.push_back(timing);
            std::vector<double> modelRow;
            for (const auto& r : results) modelRow.push_back(r.wallMs);
            modelTimings.push_back(std::move(modelRow));
        }
        loopEma = count == 0 ? loopMs : alpha * loopMs + (1.0 - alpha) * loopEma;
        ++count;  // Count the processed frame even when q/Esc was pressed.

        if (count == 1 || count % 30 == 0)
            std::cout << "Frame " << count << ": detections=" << detectionCount
                      << ", pipeline_ms=" << pipelineMs << std::endl;

        if (quit || (options.maxFrames && count >= options.maxFrames))
            break;
    }

    const double runMs = elapsedMs(runBegin, Clock::now());
    const auto flushBegin = Clock::now();

    // Flush encoded output before reporting completion. Capture/writer also use RAII.
    writer.release();
    const double flushMs = elapsedMs(flushBegin, Clock::now());
    if (options.display)
        cv::destroyAllWindows();

    if (!options.timingCsv.empty()) {
        const auto parent = std::filesystem::path(options.timingCsv).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        std::ofstream csv(options.timingCsv);
        csv << "frame,start_ms,end_ms,read_ms,geometry_ms,pipeline_ms,draw_ms,encode_ms,display_ms,total_ms,capture_timestamp_ms,read_age_ms,completion_age_ms";
        if (!hooks.legacyPresentation)
            for (const auto& m : options.models) csv << ',' << m.id << "_wall_ms";
        csv << '\n';
        csv << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < timings.size(); ++i) {
            csv << i + 1;
            for (double value : timings[i])
                csv << ',' << value;
            if (!hooks.legacyPresentation)
                for (double value : modelTimings[i]) csv << ',' << value;
            csv << '\n';
        }
        csv.close();
        if (!csv)
            throw std::runtime_error("Cannot save timing CSV");
    }

    std::cout << "Measured loop wall: " << runMs << " ms; actual processing FPS/QPS: "
              << (runMs > 0 ? count * 1000.0 / runMs : 0.0) << '\n'
              << "Writer finalization: " << flushMs << " ms\n";
    std::cout << "Frames: " << count << '\n'
              << std::fixed << std::setprecision(3)
              << "Pipeline EMA: " << pipelineEma << " ms, " << fps(pipelineEma) << " FPS\n"
              << "Full loop EMA: " << loopEma << " ms, " << fps(loopEma) << " FPS\n";
    return 0;
}

}  // namespace
