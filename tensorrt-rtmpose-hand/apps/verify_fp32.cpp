// 本程序通过公共 Runner 验证 RTMPose FP32；支持 NPY 与原图预处理两条独立路径。
#include "rtmpose/reference.hpp"
#include "edge/trt/runner.hpp"
#include "edge/trt/tensor_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>

// 将实际运行的模型、样本清单及输出绑定到完成记录，避免历史文件冒充新结果。
std::string fileSha256(const std::filesystem::path& path) {
    std::ifstream file(
        path,
        std::ios::binary
    );
    if (!file) {
        throw std::runtime_error("Cannot hash file: " + path.string());
    }
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
        EVP_MD_CTX_new(),
        EVP_MD_CTX_free
    );
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("Cannot initialize SHA256");
    }
    std::array<char, 65536> bytes{};
    while (file) {
        file.read(
            bytes.data(),
            bytes.size()
        );
        if (EVP_DigestUpdate(context.get(), bytes.data(), file.gcount()) != 1) {
            throw std::runtime_error("Cannot update SHA256");
        }
    }
    if (!file.eof()) {
        throw std::runtime_error("Cannot read hash input");
    }
    unsigned char hash[EVP_MAX_MD_SIZE]{};
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(context.get(), hash, &length) != 1) {
        throw std::runtime_error("Cannot finalize SHA256");
    }
    std::ostringstream hex;
    for (unsigned int index = 0; index < length; ++index) {
        hex << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[index]);
    }
    return hex.str();
}

struct Sample {
    std::string id;
    std::string tensor;
    std::string image;
    rtmpose::Region region;
};

int main(
    int argc,
    char** argv
) {
    try {
        if (argc != 5) {
            throw std::runtime_error("Usage: rtmpose_fp32 engine|prepare cases.json output_dir npy|image");
        }
        const std::string mode = argv[4];
        if (mode != "npy" && mode != "image") {
            throw std::runtime_error("Unknown input mode");
        }
        const std::filesystem::path output(argv[3]);
        std::filesystem::create_directories(output);
        std::filesystem::remove(output / "complete.json");
        const bool prepareOnly = std::string(argv[1]) == "prepare";
        const auto engineHash = prepareOnly ? std::string{} : fileSha256(argv[1]);
        const auto casesHash = fileSha256(argv[2]);
        cv::FileStorage manifest(
            argv[2],
            cv::FileStorage::READ
        );
        if (!manifest.isOpened()) {
            throw std::runtime_error("Cannot open case manifest");
        }
        std::vector<Sample> samples;
        for (const auto& node : manifest["samples"]) {
            std::array<double, 4> bbox{};
            int axis = 0;
            for (const auto& value : node["bbox"]) {
                if (axis >= 4) {
                    throw std::runtime_error("Invalid bbox length");
                }
                bbox[axis++] = static_cast<double>(value);
            }
            if (axis != 4) {
                throw std::runtime_error("Invalid bbox length");
            }
            samples.push_back({static_cast<std::string>(node["id"]),
                               static_cast<std::string>(node["tensor"]),
                               static_cast<std::string>(node["image"]), rtmpose::makeRegion(bbox)});
        }
        if (samples.empty()) {
            // 零手图不调用 rtmlib 的整图兜底，也不启动 TensorRT。
            return 0;
        }
        std::vector<std::vector<float>> tensors;
        for (const auto& sample : samples) {
            auto tensor = mode == "npy" ? rtmpose::loadNpy(sample.tensor) : rtmpose::preprocess(
                cv::imread(sample.image),
                sample.region
            );
            edge::trt::saveFloatBinary(
                (output / (sample.id + "_input.bin")).string(),
                tensor
            );
            tensors.push_back(std::move(tensor));
        }
        if (std::string(argv[1]) == "prepare") {
            std::cout << "Prepared " << samples.size() << " inputs with OpenCV " << CV_VERSION << '\n';
            return 0;
        }
        edge::trt::Runner runner(argv[1]);
        if (runner.inputInfo().name != "input" || runner.tensors().size() != 3) {
            throw std::runtime_error("RTMPose I/O names/count mismatch");
        }
        cv::FileStorage metadata(
            (output / "runtime.json").string(),
            cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON
        );
        metadata << "opencv" << CV_VERSION << "profile" << runner.profileIndex();
        metadata << "tensors" << "[";
        for (const auto& info : runner.tensors()) {
            metadata << "{" << "name" << info.name;
            metadata.write(
                "engine_shape",
                edge::trt::dimsToString(info.engineShape)
            );
            metadata << "capacity_bytes" << static_cast<double>(info.capacityBytes) << "}";
        }
        metadata << "]";
        const auto* inputAddress = runner.deviceInput();
        // 6 表示调用方请求六只手，按 MAX=4 分块，保持原顺序，不丢弃尾批。
        for (int requestedBatch : {1, 2, 3, 4, 6, 1}) {
            const int chunk = std::min(requestedBatch, 4);
            for (std::size_t start = 0; start < samples.size();) {
                const int requestRemaining = requestedBatch - static_cast<int>(start % requestedBatch);
                const int count = std::min<std::size_t>(
                    std::min(chunk, requestRemaining),
                    samples.size() - start
                );
                runner.setInputShape(nvinfer1::Dims4{count, 3, 256, 256});
                if (runner.deviceInput() != inputAddress) {
                    throw std::runtime_error("Input address changed with shape");
                }
                std::vector<float> packed;
                std::vector<rtmpose::Region> regions;
                for (int offset = 0; offset < count; ++offset) {
                    packed.insert(
                        packed.end(),
                        tensors[start + offset].begin(),
                        tensors[start + offset].end()
                    );
                    regions.push_back(samples[start + offset].region);
                }
                for (const std::string name : {"simcc_x", "simcc_y"}) {
                    const auto& info = runner.tensorInfo(name);
                    if (info.dtype != nvinfer1::DataType::kFLOAT || info.shape.nbDims != 3 ||
                        info.shape.d[0] != count || info.shape.d[1] != 21 || info.shape.d[2] != 512 ||
                        info.activeBytes != static_cast<std::size_t>(count * 21 * 512 * sizeof(float))) {
                        throw std::runtime_error("RTMPose output contract mismatch");
                    }
                }
                const auto raw = runner.inferNamed(packed);
                const auto prefix = "b" + std::to_string(requestedBatch) + "_" + std::to_string(start);
                for (const auto& entry : raw) {
                    edge::trt::saveFloatBinary(
                        (output / (prefix + "_" + entry.first + ".bin")).string(),
                        entry.second
                    );
                }
                const auto decoded = rtmpose::decode(
                    raw.at("simcc_x"),
                    raw.at("simcc_y"),
                    regions
                );
                std::ofstream file(
                    output / (prefix + "_decoded.bin"),
                    std::ios::binary
                );
                file.write(
                    reinterpret_cast<const char*>(decoded.data()),
                    decoded.size() * sizeof(double)
                );
                if (!file) {
                    throw std::runtime_error("Cannot write decoded output");
                }
                start += count;
            }
        }
        metadata.release();
        if (fileSha256(argv[1]) != engineHash || fileSha256(argv[2]) != casesHash) {
            throw std::runtime_error("Engine or manifest changed during inference");
        }
        cv::FileStorage completion(
            (output / "complete.tmp.json").string(),
            cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON
        );
        if (!completion.isOpened()) {
            throw std::runtime_error("Cannot write completion record");
        }
        completion << "engine_sha256" << engineHash << "cases_sha256" << casesHash
                   << "mode" << mode << "samples" << static_cast<int>(samples.size()) << "files" << "[";
        for (const auto& artifact : std::filesystem::directory_iterator(output)) {
            if (artifact.path().extension() == ".bin" || artifact.path().filename() == "runtime.json") {
                completion << "{" << "path" << artifact.path().filename().string()
                           << "sha256" << fileSha256(artifact.path()) << "}";
            }
        }
        completion << "]";
        completion.release();
        std::filesystem::rename(
            output / "complete.tmp.json",
            output / "complete.json"
        );
        std::cout << "PASS: " << samples.size() << " samples, batches 1/2/3/4, >MAX chunking, mode=" << mode << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
