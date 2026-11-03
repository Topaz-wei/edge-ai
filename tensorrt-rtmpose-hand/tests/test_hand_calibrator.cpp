// CPU 校准器契约测试：错误绑定名被捕获重抛，未完成消费不能保存 cache。
#include "rtmpose/calibrator.hpp"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <unistd.h>

int main() {
    try {
        rtmpose::CalibrationData data;
        data.tensors.push_back({"missing.npy", "unused"});
        data.provenanceJson = "{}";
        rtmpose::HandCalibrator calibrator(
            data,
            "/tmp/unused-rtmpose-test-cache"
        );
        if (calibrator.getBatchSize() != 1) {
            throw std::runtime_error("Expected calibration batch one");
        }
        void* bindings[]{nullptr};
        const char* names[]{"wrong"};
        if (calibrator.getBatch(
            bindings,
            names,
            1
        )) {
            throw std::runtime_error("Wrong binding accepted");
        }
        bool rejected = false;
        try {
            calibrator.throwIfFailed();
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Callback failure was swallowed");
        }
        rtmpose::HandCalibrator incomplete(
            data,
            "/tmp/unused-rtmpose-test-cache"
        );
        rejected = false;
        try {
            incomplete.verifyComplete();
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Incomplete calibration accepted");
        }
        // 独立临时fixture验证cache身份匹配和篡改拒绝，不接触正式校准cache。
        const auto folder = std::filesystem::temp_directory_path() /
            ("hand-cache-test-" + std::to_string(getpid()));
        std::filesystem::create_directory(folder);
        const auto cachePath = folder / "fixture.cache";
        {
            std::ofstream cache(cachePath);
            cache << "synthetic-unit-test-payload";
        }
        const auto metadata = "{\n\"cache_sha256\":\"" + rtmpose::calibrationFileHash(cachePath) +
            "\",\n\"identity\":{}}\n";
        {
            std::ofstream meta(cachePath.string() + ".meta.json");
            meta << metadata;
        }
        rtmpose::HandCalibrator validCache(
            data,
            cachePath
        );
        std::size_t length = 0;
        if (!validCache.readCalibrationCache(length) || length == 0) {
            throw std::runtime_error("Matching cache rejected");
        }
        validCache.verifyComplete();
        {
            std::ofstream cache(cachePath);
            cache << "tampered";
        }
        rtmpose::HandCalibrator badCache(
            data,
            cachePath
        );
        length = 999;
        if (badCache.readCalibrationCache(length) || length != 0) {
            throw std::runtime_error("Tampered cache accepted");
        }
        rejected = false;
        try {
            badCache.throwIfFailed();
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Cache failure swallowed");
        }
        std::filesystem::remove_all(folder);
        std::cout << "Calibrator contract checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
