#include "include/trt_runner.hpp"
#include "include/yolo_preprocess.hpp"
#include "include/postprocess_raw.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>


void drawDetections(
    cv::Mat& image,
    const std::vector<Detection>& detections)
{
    for (const auto& det : detections) {

        const auto& box = det.businessBox;

        const cv::Point p1(
            static_cast<int>(
                std::lround(box.x1)
            ),
            static_cast<int>(
                std::lround(box.y1)
            )
        );

        const cv::Point p2(
            static_cast<int>(
                std::lround(box.x2)
            ),
            static_cast<int>(
                std::lround(box.y2)
            )
        );

        // -------------------------
        // bbox
        // -------------------------
        cv::rectangle(
            image,
            p1,
            p2,
            cv::Scalar(0, 255, 0),
            2
        );

        // -------------------------
        // label
        // -------------------------
        std::ostringstream oss;

        oss
            << "person "
            << std::fixed
            << std::setprecision(3)
            << det.confidence;

        const std::string label =
            oss.str();

        int baseline = 0;

        const cv::Size textSize =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                1,
                &baseline
            );

        int labelX = p1.x;

        int labelY =
            std::max(
                p1.y - 5,
                textSize.height + 5
            );

        // label 背景
        cv::rectangle(
            image,
            cv::Point(
                labelX,
                labelY - textSize.height - 5
            ),
            cv::Point(
                labelX + textSize.width + 5,
                labelY + baseline
            ),
            cv::Scalar(0, 255, 0),
            cv::FILLED
        );

        // label 文字
        cv::putText(
            image,
            label,
            cv::Point(
                labelX + 2,
                labelY - 2
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 0, 0),
            1,
            cv::LINE_AA
        );
    }
}


int main(
    int argc,
    char** argv)
{
    if (argc != 4) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <engine.plan>"
            << " <business_image>"
            << " <output_image>\n";

        return 1;
    }

    try {
        const std::string enginePath =
            argv[1];

        const std::string imagePath =
            argv[2];

        const std::string outputPath =
            argv[3];

        TrtRunner runner(enginePath);

        // --------------------------------
        // 1. Preprocess
        // --------------------------------
        const auto input =
            preprocess(
                imagePath
            );

        // --------------------------------
        // 2. TensorRT inference
        // --------------------------------
        const auto output =
            runner.infer(input);

        // --------------------------------
        // 3. YOLO postprocess
        // --------------------------------
        const auto detections =
            postprocess(
                output,
                true
            );

        std::cout
            << std::fixed
            << std::setprecision(6);

        printDetections(
            detections
        );

        // --------------------------------
        // 4. Load original business image
        // --------------------------------
        cv::Mat image =
            cv::imread(
                imagePath,
                cv::IMREAD_COLOR |
                cv::IMREAD_IGNORE_ORIENTATION
            );

        if (image.empty()) {
            throw std::runtime_error(
                "Failed to load image for visualization: "
                + imagePath
            );
        }

        // --------------------------------
        // 5. Draw detections
        // --------------------------------
        drawDetections(
            image,
            detections
        );

        // --------------------------------
        // 6. Save result
        // --------------------------------
        if (!cv::imwrite(
                outputPath,
                image))
        {
            throw std::runtime_error(
                "Failed to save visualization: "
                + outputPath
            );
        }

        std::cout
            << "\nVisualization saved to: "
            << outputPath
            << "\n";

    } catch (
        const std::exception& e)
    {
        std::cerr
            << "ERROR: "
            << e.what()
            << '\n';

        return 1;
    }

    return 0;
}