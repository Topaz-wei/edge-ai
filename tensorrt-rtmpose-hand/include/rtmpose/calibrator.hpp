// 本文件定义 RTMPose 固定张量校准数据与 EntropyCalibrator2；不重新预处理输入。
#pragma once
#include <NvInfer.h>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace rtmpose {
struct CalibrationTensor {
    std::filesystem::path path;
    std::string sha256;
};
struct CalibrationData {
    std::vector<CalibrationTensor> tensors;
    std::string provenanceJson;
};
std::string calibrationFileHash(const std::filesystem::path& path);
CalibrationData auditCalibration(const std::filesystem::path& module);

class HandCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    HandCalibrator(
        CalibrationData data,
        std::filesystem::path cache
    );
    ~HandCalibrator() override;
    HandCalibrator(const HandCalibrator&) = delete;
    HandCalibrator& operator=(const HandCalibrator&) = delete;
    int getBatchSize() const noexcept override;
    bool getBatch(
        void* bindings[],
        const char* names[],
        int count
    ) noexcept override;
    const void* readCalibrationCache(std::size_t& length) noexcept override;
    void writeCalibrationCache(
        const void* data,
        std::size_t length
    ) noexcept override;
    void throwIfFailed() const;
    void verifyComplete() const;
    void saveCache();
    std::size_t consumed() const { return next_; }
private:
    CalibrationData data_;
    std::filesystem::path cachePath_;
    std::vector<char> cache_;
    std::exception_ptr failure_;
    std::size_t next_ = 0;
    bool cacheUsed_ = false;
    void* device_ = nullptr;
};
}
