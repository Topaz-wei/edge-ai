#include "include/cuda_yolo.hpp"
#include "include/infer_raw.hpp"
#include "include/yolo_preprocess.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

void compare(
    const std::vector<Detection>& a,
    const std::vector<Detection>& b
) {
    if (a.size() != b.size())
        throw std::runtime_error("Detection count mismatch");

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].candidate != b[i].candidate || a[i].classId != b[i].classId ||
            a[i].confidence != b[i].confidence)
            throw std::runtime_error("Detection order/class/score mismatch");

        const auto boxesA = {
            a[i].modelBox,
            a[i].businessBox
        };
        const auto boxesB = {
            b[i].modelBox,
            b[i].businessBox
        };
        auto q = boxesB.begin();

        for (auto x : boxesA) {
            auto y = *q++;

            if (x.x1 != y.x1 || x.y1 != y.y1 || x.x2 != y.x2 || x.y2 != y.y2)
                throw std::runtime_error("Detection box mismatch");
        }
    }
}

int main(
    int argc,
    char** argv
) {
    try {
        if (argc != 3)
            throw std::runtime_error("Usage: test_cuda_yolo engine image");

        TrtRunner runner(argv[1]);

        CudaYoloOps ops(runner.stream());

        auto image = cv::imread(
            argv[2],
            cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION
        );

        if (image.empty())
            throw std::runtime_error("Image missing");

        cv::Mat random(
            640,
            360,
            CV_8UC3
        );

        cv::RNG rng(42);

        rng.fill(
            random,
            cv::RNG::UNIFORM,
            0,
            256
        );

        cv::Mat padded(
            640,
            362,
            CV_8UC3
        );

        auto roi = padded(
            cv::Rect(
                1,
                0,
                360,
                640
            )
        );

        random.copyTo(roi);

        std::vector<cv::Mat> images{
            image,
            random,
            roi,
            cv::Mat(
                640,
                360,
                CV_8UC3,
                cv::Scalar(
                    10,
                    123,
                    255
                )
            ),
            cv::Mat::zeros(
                640,
                360,
                CV_8UC3
            )
        };

        for (auto frame : images) {
            auto original = frame.clone();

            auto expected = preprocess(frame);

            ops.upload(frame);

            ops.preprocess(runner.deviceInput());

            std::vector<float> actual(expected.size());

            checkCuda(
                cudaMemcpyAsync(
                    actual.data(),
                    runner.deviceInput(),
                    actual.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    runner.stream()
                ),
                "test tensor copy"
            );

            checkCuda(
                cudaStreamSynchronize(runner.stream()),
                "test tensor sync"
            );

            if (std::memcmp(
                    actual.data(),
                    expected.data(),
                    actual.size() * sizeof(float)
                ))
                throw std::runtime_error("CUDA preprocess not bitwise equal");

            if (cv::norm(
                    frame,
                    original,
                    cv::NORM_INF
                ) != 0)
                throw std::runtime_error("Input mutated");
        }

        bool rejected = false;

        try {
            ops.upload(cv::Mat());
        } catch (const std::runtime_error&) {
            rejected = true;
        }

        if (!rejected)
            throw std::runtime_error("Invalid image accepted");

        std::cout
            << "Preprocess: real/random/ROI/constants bitwise equal; invalid input rejected\n";

        std::vector<float> raw(
            84 * 8400,
            0.f
        );

        auto put = [&](int i,
                       float x,
                       float y,
                       float w,
                       float h,
                       float score) {
            raw[i] = x;
            raw[8400 + i] = y;
            raw[2 * 8400 + i] = w;
            raw[3 * 8400 + i] = h;
            raw[4 * 8400 + i] = score;
        };
        auto testPost = [&]() {
            const auto expected = postprocess(
                raw,
                false
            );

            checkCuda(
                cudaMemcpyAsync(
                    const_cast<float*>(runner.deviceOutput()),
                    raw.data(),
                    raw.size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    runner.stream()
                ),
                "test raw upload"
            );

            ops.postprocess(runner.deviceOutput());

            compare(
                expected,
                ops.download()
            );
        };
        testPost(); // No survivors.
        put(0,
            150,
            200,
            80,
            80,
            .9f);

        put(1,
            150,
            200,
            80,
            80,
            .9f); // Tie -> candidate 0.
        put(2,
            350,
            500,
            10,
            10,
            .7f);

        put(3,
            400,
            500,
            10,
            10,
            .69999f);

        put(4,
            450,
            500,
            10,
            10,
            .8f);

        raw[5 * 8400 + 4] = .9f; // Other class wins.
        put(5,
            8.5f,
            30,
            17,
            10,
            .8f);

        put(6,
            11.5f,
            30,
            17,
            10,
            .8f); // IoU exactly .7 -> keep both.
        put(7,
            10,
            600,
            40,
            100,
            .85f); // Clipping at left/bottom edges.
        testPost();

        std::fill(
            raw.begin(),
            raw.end(),
            0.f
        );

        for (int i = 0; i < 512; ++i)
            put(i,
                float(i % 32) * 18,
                float(i / 32) * 18,
                4,
                4,
                .8f);

        testPost(); // No silent 300-detection cap; deterministic ties.
        std::cout
            << "Postprocess: empty/ties/class argmax/confidence/IoU boundary/clipping/512 survivors exact\n";

        runner.prepareGraph();

        for (int iteration = 0; iteration < 4; ++iteration) {
            const auto& frame = images[iteration % 2];
            auto expected = runner.infer(preprocess(frame));

            ops.upload(frame);

            ops.preprocess(runner.deviceInput());

            runner.enqueueDevice(iteration % 2 == 0);

            std::vector<float> actual(expected.size());

            checkCuda(
                cudaMemcpyAsync(
                    actual.data(),
                    runner.deviceOutput(),
                    actual.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    runner.stream()
                ),
                "test output copy"
            );

            checkCuda(
                cudaStreamSynchronize(runner.stream()),
                "test output sync"
            );

            if (std::memcmp(
                    actual.data(),
                    expected.data(),
                    actual.size() * sizeof(float)
                ))
                throw std::runtime_error("CUDA/Graph raw output differs");

            ops.postprocess(runner.deviceOutput());

            compare(
                postprocess(
                    expected,
                    false
                ),
                ops.download()
            );
        }

        for (bool graph :
            {false,
                true}) {
            CudaYoloPipeline pipeline(
                argv[1],
                graph
            );

            for (int i = 0; i < 3; ++i) {
                const auto& frame = images[i % 2];
                auto expected = postprocess(
                    runner.infer(preprocess(frame)),
                    false
                );

                CudaPipelineTimings timing;
                compare(
                    expected,
                    pipeline.infer(
                        frame,
                        i % 2 ? &timing : nullptr
                    )
                );

                if (i % 2 &&
                    (!std::isfinite(timing.inferenceMs) || timing.inferenceMs <= 0))
                    throw std::runtime_error("Invalid GPU timing");
            }
        }

        std::cout
            << "Changing frames: CUDA/Graph raw bitwise equal; full pipeline graph on/off exact\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';

        return 1;
    }
}
