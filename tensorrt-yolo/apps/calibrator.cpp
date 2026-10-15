#include "include/calibrator.hpp"
#include "include/cuda_yolo.hpp"
#include "include/infer_raw.hpp"

#include <NvInferVersion.h>
#include <opencv2/imgcodecs.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
constexpr std::size_t inputBytes = 3 * 640 * 640 * sizeof(float);

std::string hex(
    const unsigned char* bytes,
    unsigned int size
) {
    std::ostringstream result;
    for (unsigned int i = 0; i < size; ++i) {
        result << std::hex << std::setfill('0') << std::setw(2) << unsigned(bytes[i]);
    }
    return result.str();
}

std::string sha256File(
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );
    if (!file)
        throw std::runtime_error("Cannot read for SHA256: " + path);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(
        EVP_MD_CTX_new(),
        EVP_MD_CTX_free
    );
    if (!ctx || EVP_DigestInit_ex(
                    ctx.get(),
                    EVP_sha256(),
                    nullptr
                ) != 1)
        throw std::runtime_error("Cannot initialize SHA256");
    std::array<char, 65536> buffer{};
    while (file.read(
               buffer.data(),
               buffer.size()
           ) ||
           file.gcount()) {
        if (EVP_DigestUpdate(
                ctx.get(),
                buffer.data(),
                file.gcount()
            ) != 1)
            throw std::runtime_error("SHA256 update failed");
    }
    if (!file.eof())
        throw std::runtime_error("SHA256 read failed: " + path);
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(
            ctx.get(),
            hash,
            &length
        ) != 1)
        throw std::runtime_error("SHA256 final failed");
    return hex(
        hash,
        length
    );
}

std::string sha256Bytes(
    const void* data,
    std::size_t size
) {
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (EVP_Digest(
            data,
            size,
            hash,
            &length,
            EVP_sha256(),
            nullptr
        ) != 1)
        throw std::runtime_error("SHA256 failed");
    return hex(
        hash,
        length
    );
}

std::vector<char> readBytes(
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );
    if (!file)
        throw std::runtime_error("Cannot read: " + path);
    std::vector<char> result((std::istreambuf_iterator<char>(file)), {});
    if (file.bad())
        throw std::runtime_error("Read failed: " + path);
    return result;
}

void writeAtomic(
    const std::string& path,
    const void* bytes,
    std::size_t size
) {
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty())
        fs::create_directories(parent);
    const std::string temporary = path + ".tmp";
    std::ofstream file(
        temporary,
        std::ios::binary | std::ios::trunc
    );
    if (!file)
        throw std::runtime_error("Cannot open: " + temporary);
    file.write(
        static_cast<const char*>(bytes),
        size
    );
    file.close();
    if (!file)
        throw std::runtime_error("Cannot write: " + temporary);
    fs::rename(
        temporary,
        path
    );
}
}

YOLOCalibrator::YOLOCalibrator(
    const std::string& imageDirectory,
    const std::string& onnxPath,
    const std::string& inputName,
    const std::string& cachePath
)
    : inputName_(inputName),
      cachePath_(cachePath) {
    if (inputName.empty() || cachePath.empty())
        throw std::runtime_error("Empty input/cache name");
    if (!fs::is_directory(imageDirectory))
        throw std::runtime_error("Calibration directory missing: " + imageDirectory);
    for (const auto& entry : fs::recursive_directory_iterator(imageDirectory)) {
        if (!entry.is_regular_file())
            continue;
        auto extension = entry.path().extension().string();
        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(), [](unsigned char c) {
                return std::tolower(c);
            }
        );
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
            extension == ".bmp" || extension == ".tif" || extension == ".tiff" ||
            extension == ".webp") {
            images_.push_back(entry.path().string());
        }
    }
    std::sort(
        images_.begin(),
        images_.end()
    );
    if (images_.empty())
        throw std::runtime_error("Calibration directory contains no images");
    std::ostringstream signature;
    signature << "YOLO-entropy2-business-fixed640-v1\n"
              << "TensorRT=" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.'
              << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD << '\n'
              << "ONNX=" << sha256File(onnxPath) << '\n'
              << "CUDA-preprocess=" << YOLO_PREPROCESS_SHA256 << '\n'
              << "OpenCV=" << CV_VERSION << '\n'
              << "Input=" << inputName << ":1,3,640,640:FP32\n"
              << "Flags=INT8,FP16-fallback,TF32-off\n";
    for (const auto& path : images_) {
        const auto image = cv::imread(
            path,
            cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION
        );
        if (image.empty() || image.rows != 640 || image.cols != 360 ||
            image.type() != CV_8UC3)
            throw std::runtime_error(
                "Expected 360x640 business calibration image: " + path
            );
        imageHashes_.push_back(sha256File(path));
        signature << fs::relative(
                         path,
                         imageDirectory
                     )
                         .generic_string()
                  << ' ' << imageHashes_.back() << '\n';
    }
    fingerprint_ = signature.str();
    std::cout << "Calibration images: " << images_.size()
              << " (batch=1, CUDA preprocessing)\n";
}

YOLOCalibrator::~YOLOCalibrator() {
    if (stream_)
        cudaStreamSynchronize(stream_);
    preprocessing_.reset();
    if (deviceInput_)
        cudaFree(deviceInput_);
    if (stream_)
        cudaStreamDestroy(stream_);
}

int YOLOCalibrator::getBatchSize() const noexcept {
    return 1;
}

std::size_t YOLOCalibrator::imageCount() const {
    return images_.size();
}

void YOLOCalibrator::throwIfFailed() const {
    if (failure_)
        std::rethrow_exception(failure_);
}

bool YOLOCalibrator::getBatch(
    void* bindings[],
    const char* names[],
    int nbBindings
) noexcept {
    if (failure_)
        return false;
    try {
        if (nbBindings != 1 || !bindings || !names || !names[0] ||
            inputName_ != names[0])
            throw std::runtime_error("Calibration input binding name/count mismatch");
        if (nextImage_ == images_.size())
            return false;
        if (!stream_)
            checkCuda(
                cudaStreamCreate(&stream_),
                "calibration stream"
            );
        if (!deviceInput_)
            checkCuda(
                cudaMalloc(
                    &deviceInput_,
                    inputBytes
                ),
                "calibration input"
            );
        if (!preprocessing_)
            preprocessing_ = std::make_unique<CudaYoloOps>(stream_);
        const auto& path = images_[nextImage_];
        if (sha256File(path) != imageHashes_[nextImage_])
            throw std::runtime_error("Calibration image changed during build: " + path);
        const auto image = cv::imread(
            path,
            cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION
        );
        preprocessing_->upload(image);
        preprocessing_->preprocess(static_cast<float*>(deviceInput_));
        // Builder consumes this binding on its own stream; finish preparation first.
        checkCuda(
            cudaStreamSynchronize(stream_),
            "calibration preprocess sync"
        );
        bindings[0] = deviceInput_;
        ++nextImage_;
        if (nextImage_ == 1 || nextImage_ % 25 == 0 || nextImage_ == images_.size())
            std::cout << "Calibration batch " << nextImage_ << '/' << images_.size()
                      << std::endl;
        return true;
    } catch (...) {
        failure_ = std::current_exception();
        return false;
    }
}

const void* YOLOCalibrator::readCalibrationCache(
    std::size_t& length
) noexcept {
    length = 0;
    if (failure_)
        return nullptr;
    try {
        if (!fs::exists(cachePath_) || !fs::exists(cachePath_ + ".meta"))
            return nullptr;
        const auto metadata = readBytes(cachePath_ + ".meta");
        auto bytes = readBytes(cachePath_);
        const auto expected = fingerprint_ + "CacheSHA256=" +
                              sha256Bytes(
                                  bytes.data(),
                                  bytes.size()
                              ) +
                              '\n';
        if (bytes.empty() || std::string(
                                 metadata.begin(),
                                 metadata.end()
                             ) != expected) {
            std::cout << "Calibration cache mismatch; recalibrating\n";
            return nullptr;
        }
        cache_ = std::move(bytes);
        cacheUsed_ = true;
        length = cache_.size();
        std::cout << "Using validated calibration cache: " << cachePath_ << '\n';
        return cache_.data();
    } catch (...) {
        failure_ = std::current_exception();
        return nullptr;
    }
}

void YOLOCalibrator::writeCalibrationCache(
    const void* data,
    std::size_t length
) noexcept {
    if (failure_)
        return;
    try {
        if (!data || !length)
            throw std::runtime_error("TensorRT returned empty calibration cache");
        const auto* begin = static_cast<const char*>(data);
        cache_.assign(
            begin,
            begin + length
        );
    } catch (...) {
        failure_ = std::current_exception();
    }
}

void YOLOCalibrator::verifyComplete() const {
    throwIfFailed();
    if (!cacheUsed_ && nextImage_ != images_.size())
        throw std::runtime_error("TensorRT did not consume all calibration images");
    if (cache_.empty())
        throw std::runtime_error("TensorRT produced no calibration cache");
}

void YOLOCalibrator::saveCache() {
    throwIfFailed();
    if (cache_.empty())
        throw std::runtime_error("No calibration cache to save");
    const auto metadata = fingerprint_ + "CacheSHA256=" +
                          sha256Bytes(
                              cache_.data(),
                              cache_.size()
                          ) +
                          '\n';
    writeAtomic(
        cachePath_,
        cache_.data(),
        cache_.size()
    );
    writeAtomic(
        cachePath_ + ".meta",
        metadata.data(),
        metadata.size()
    );
    std::cout << "Calibration cache saved: " << cachePath_ << '\n';
}
