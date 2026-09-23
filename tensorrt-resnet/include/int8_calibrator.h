#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>


class Int8EntropyCalibrator
    : public nvinfer1::IInt8EntropyCalibrator2 {

public:
    Int8EntropyCalibrator(
        const std::string& dataPath,
        const std::string& cachePath,
        std::size_t elementsPerSample
    );

    ~Int8EntropyCalibrator() override;

    int getBatchSize() const noexcept override;

    bool getBatch(
        void* bindings[],
        const char* names[],
        int nbBindings
    ) noexcept override;

    const void* readCalibrationCache(
        std::size_t& length
    ) noexcept override;

    void writeCalibrationCache(
        const void* cache,
        std::size_t length
    ) noexcept override;

private:
    std::vector<float> data_;
    std::vector<char> cache_;

    std::string cachePath_;

    std::size_t elementsPerSample_{0};
    std::size_t numSamples_{0};
    std::size_t currentSample_{0};

    void* deviceInput_{nullptr};
};