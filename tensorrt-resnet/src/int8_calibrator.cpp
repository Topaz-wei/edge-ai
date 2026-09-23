#include "int8_calibrator.h"

#include <fstream>
#include <iostream>
#include <stdexcept>


static std::vector<float> loadFloatFile(
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to open calibration data: "
            + path
        );
    }

    file.seekg(
        0,
        std::ios::end
    );

    const std::size_t bytes =
        file.tellg();

    if (bytes % sizeof(float) != 0) {
        throw std::runtime_error(
            "Invalid calibration file"
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
        bytes
    );

    return data;
}


Int8EntropyCalibrator::Int8EntropyCalibrator(
    const std::string& dataPath,
    const std::string& cachePath,
    std::size_t elementsPerSample
)
    : data_(loadFloatFile(dataPath)),
      cachePath_(cachePath),
      elementsPerSample_(elementsPerSample) {

    if (
        elementsPerSample_ == 0 ||
        data_.size() % elementsPerSample_ != 0
    ) {
        throw std::runtime_error(
            "Calibration data size mismatch"
        );
    }

    numSamples_ =
        data_.size() / elementsPerSample_;

    const std::size_t inputBytes =
        elementsPerSample_
        * sizeof(float);

    const auto error =
        cudaMalloc(
            &deviceInput_,
            inputBytes
        );

    if (error != cudaSuccess) {
        throw std::runtime_error(
            "cudaMalloc failed for calibrator"
        );
    }

    std::cout
        << "Calibration samples: "
        << numSamples_
        << '\n';
}


Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    if (deviceInput_) {
        cudaFree(deviceInput_);
    }
}


int Int8EntropyCalibrator::getBatchSize()
    const noexcept {

    // Static ONNX input:
    // [1, 3, 224, 224]

    return 1;
}


bool Int8EntropyCalibrator::getBatch(
    void* bindings[],
    const char* names[],
    int nbBindings
) noexcept {

    if (currentSample_ >= numSamples_) {
        return false;
    }

    if (nbBindings != 1) {
        std::cerr
            << "Expected one calibration input\n";

        return false;
    }

    const float* src =
        data_.data()
        + currentSample_
          * elementsPerSample_;

    const std::size_t bytes =
        elementsPerSample_
        * sizeof(float);

    const auto error =
        cudaMemcpy(
            deviceInput_,
            src,
            bytes,
            cudaMemcpyHostToDevice
        );

    if (error != cudaSuccess) {
        std::cerr
            << "Calibration cudaMemcpy failed\n";

        return false;
    }

    bindings[0] = deviceInput_;

    ++currentSample_;

    if (
        currentSample_ == 1 ||
        currentSample_ % 10 == 0
    ) {
        std::cout
            << "Calibration: "
            << currentSample_
            << "/"
            << numSamples_
            << '\n';
    }

    return true;
}


const void*
Int8EntropyCalibrator::readCalibrationCache(
    std::size_t& length
) noexcept {

    std::ifstream file(
        cachePath_,
        std::ios::binary
    );

    if (!file) {
        length = 0;

        std::cout
            << "Calibration cache not found. "
            << "Running calibration.\n";

        return nullptr;
    }

    file.seekg(
        0,
        std::ios::end
    );

    const std::size_t size =
        file.tellg();

    file.seekg(
        0,
        std::ios::beg
    );

    cache_.resize(size);

    file.read(
        cache_.data(),
        size
    );

    length = cache_.size();

    std::cout
        << "Loaded calibration cache: "
        << cachePath_
        << '\n';

    return cache_.data();
}


void
Int8EntropyCalibrator::writeCalibrationCache(
    const void* cache,
    std::size_t length
) noexcept {

    std::ofstream file(
        cachePath_,
        std::ios::binary
    );

    if (!file) {
        std::cerr
            << "Failed to write calibration cache\n";

        return;
    }

    file.write(
        static_cast<const char*>(cache),
        length
    );

    std::cout
        << "Calibration cache saved: "
        << cachePath_
        << '\n';
}