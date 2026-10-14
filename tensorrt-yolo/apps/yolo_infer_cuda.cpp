#include "include/cuda_yolo.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(
    int argc,
    char** argv
) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine.plan> <business_image> <output_image> [--no-graph]\n";

        return 1;
    }

    try {
        if (argc == 5 && std::string(argv[4]) != "--no-graph")
            throw std::runtime_error("Unknown option");

        auto image = cv::imread(
            argv[2],
            cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION
        );

        if (image.empty())
            throw std::runtime_error("Cannot decode image");

        CudaYoloPipeline pipeline(
            argv[1],
            argc == 4
        );

        const auto detections = pipeline.infer(image);

        std::cout << std::fixed << std::setprecision(6);

        printDetections(detections);

        for (const auto& d : detections) {
            const auto& b = d.businessBox;
            cv::rectangle(
                image,
                cv::Point(
                    cvRound(b.x1),
                    cvRound(b.y1)
                ),
                cv::Point(
                    cvRound(b.x2),
                    cvRound(b.y2)
                ),
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                2
            );

            cv::putText(
                image,
                "person " + std::to_string(d.confidence),
                cv::Point(
                    cvRound(b.x1),
                    std::max(
                        16,
                        cvRound(b.y1) - 5
                    )
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                .5,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                1,
                cv::LINE_AA
            );
        }

        if (!cv::imwrite(
                argv[3],
                image
            ))
            throw std::runtime_error("Cannot save output image");

        std::cout << "Visualization saved to: " << argv[3] << '\n';
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';

        return 1;
    }
}
