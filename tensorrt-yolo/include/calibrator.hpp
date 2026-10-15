#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <memory>
#include <string>
#include <vector>

class CudaYoloOps;

// TensorRT 8.x implicit INT8 PTQ, fixed640 business images, batch size 1.
class YOLOCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    YOLOCalibrator(
        const std::string& imageDirectory,
        const std::string& onnxPath,
        const std::string& inputName,
        const std::string& cachePath
    );

    ~YOLOCalibrator() override;

    YOLOCalibrator(const YOLOCalibrator&) = delete;
    YOLOCalibrator& operator=(const YOLOCalibrator&) = delete;

    int getBatchSize() const noexcept override;

    bool getBatch(
        void* bindings[],
        const char* names[],
        int nbBindings
    ) noexcept override;

    const void* readCalibrationCache(std::size_t& length) noexcept override;

    void writeCalibrationCache(
        const void* data,
        std::size_t length
    ) noexcept override;

    void throwIfFailed() const;
    void verifyComplete() const;
    void saveCache();
    std::size_t imageCount() const;

private:
    std::vector<std::string> images_;
    std::vector<std::string> imageHashes_;
    std::string inputName_;
    std::string cachePath_;
    std::string fingerprint_;
    std::vector<char> cache_;
    std::exception_ptr failure_;
    std::size_t nextImage_{0};
    bool cacheUsed_{false};
    cudaStream_t stream_{nullptr};
    void* deviceInput_{nullptr};
    std::unique_ptr<CudaYoloOps> preprocessing_;
};
