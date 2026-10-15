#include "include/calibrator.hpp"
#include "include/yolo_preprocess.hpp"
#include "include/infer_raw.hpp"

#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

int main(
    int argc,
    char** argv
) {
    try {
        if (argc != 3)
            throw std::runtime_error("Usage: test_calibrator temp_dir [cpu|gpu]");
        const std::string mode = argv[2];
        if (mode != "cpu" && mode != "gpu")
            throw std::runtime_error("Mode must be cpu or gpu");

        const fs::path root = argv[1];
        fs::create_directories(root / "images/sub");
        const auto imagePath = root / "images/sub/a.png";
        cv::Mat image(
            640,
            360,
            CV_8UC3
        );
        cv::RNG rng(42);
        rng.fill(
            image,
            cv::RNG::UNIFORM,
            0,
            256
        );
        if (!cv::imwrite(
                imagePath.string(),
                image
            ))
            throw std::runtime_error("Cannot write test image");
        const auto model = root / "model.onnx";
        std::ofstream(model) << "model fingerprint fixture";
        const auto cache = root / "calibration.cache";
        auto make = [&]() {
            return std::make_unique<YOLOCalibrator>(
                (root / "images").string(),
                model.string(),
                "images",
                cache.string()
            );
        };
        auto calibrator = make();
        if (calibrator->imageCount() != 1 || calibrator->getBatchSize() != 1)
            throw std::runtime_error("Recursive enumeration/batch size failed");
        std::size_t length = 123;
        if (calibrator->readCalibrationCache(length) || length != 0)
            throw std::runtime_error("Unexpected cache hit");
        const char bytes[] = "calibration cache test payload";
        calibrator->writeCalibrationCache(
            bytes,
            sizeof(bytes)
        );
        calibrator->saveCache();
        auto cached = make();
        if (!cached->readCalibrationCache(length) || length != sizeof(bytes))
            throw std::runtime_error("Cache reuse failed");
        cached->verifyComplete();

        std::ofstream(cache, std::ios::app) << "corrupted";
        auto corrupted = make();
        if (corrupted->readCalibrationCache(length) || length != 0)
            throw std::runtime_error("Corrupt cache accepted");
        calibrator->saveCache();

        std::ofstream(
            model,
            std::ios::app
        ) << "changed";
        auto changed = make();
        if (changed->readCalibrationCache(length) || length != 0)
            throw std::runtime_error("Model change failed to invalidate cache");
        image.at<cv::Vec3b>(
            0,
            0
        )[0] ^= 1;
        cv::imwrite(
            imagePath.string(),
            image
        );
        changed->writeCalibrationCache(
            bytes,
            sizeof(bytes)
        );
        changed->saveCache(); // Fingerprint was computed before image change.
        auto changedImage = make();
        if (changedImage->readCalibrationCache(length))
            throw std::runtime_error("Image change failed to invalidate cache");
        if (std::string(argv[2]) == "gpu") {
            void* binding = nullptr;
            const char* name = "images";
            if (!changedImage->getBatch(
                    &binding,
                    &name,
                    1
                )) {
                changedImage->throwIfFailed();
                throw std::runtime_error("Calibration batch missing");
            }
            auto expected = preprocess(image);
            std::vector<float> actual(expected.size());
            checkCuda(
                cudaMemcpy(
                    actual.data(),
                    binding,
                    actual.size() * sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "test calibration copy"
            );
            if (actual != expected)
                throw std::runtime_error(
                    "Calibration tensor differs from CPU reference"
                );
            if (changedImage->getBatch(
                    &binding,
                    &name,
                    1
                ))
                throw std::runtime_error("Unexpected extra batch");
            changedImage->throwIfFailed();
            auto wrongName = make();
            const char* bad = "wrong";
            if (wrongName->getBatch(
                    &binding,
                    &bad,
                    1
                ))
                throw std::runtime_error("Wrong binding accepted");
            bool failed = false;
            try {
                wrongName->throwIfFailed();
            } catch (const std::exception&) {
                failed = true;
            }
            if (!failed)
                throw std::runtime_error("Callback error was silently swallowed");
        }
        cv::imwrite(
            imagePath.string(),
            cv::Mat(
                32,
                32,
                CV_8UC3
            )
        );
        bool rejected = false;
        try {
            make();
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Wrong image geometry accepted");
        std::cout << "Calibrator tests passed (" << argv[2] << ")\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
