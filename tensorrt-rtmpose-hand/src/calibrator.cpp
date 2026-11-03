// 本文件按冻结清单核验 NPY 并提供单手 GPU 输入；校准 cache 与数据和实现身份绑定。
#include "rtmpose/calibrator.hpp"
#include "rtmpose/reference.hpp"
#include <cuda_runtime_api.h>
#include <openssl/evp.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace rtmpose {
namespace fs = std::filesystem;
namespace {
std::vector<char> bytes(const fs::path& path) {
    std::ifstream stream(
        path,
        std::ios::binary
    );
    if (!stream) {
        throw std::runtime_error("Cannot read: " + path.string());
    }
    std::vector<char> result{std::istreambuf_iterator<char>(stream), {}};
    if (stream.bad()) {
        throw std::runtime_error("Read error: " + path.string());
    }
    return result;
}

std::string hashBytes(const std::vector<char>& values) {
    unsigned char hash[EVP_MAX_MD_SIZE]{};
    unsigned int length = 0;
    if (EVP_Digest(
        values.data(),
        values.size(),
        hash,
        &length,
        EVP_sha256(),
        nullptr
    ) != 1) {
        throw std::runtime_error("Cannot compute SHA256");
    }
    std::ostringstream text;
    for (unsigned int index = 0; index < length; ++index) {
        text << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[index]);
    }
    return text.str();
}

std::vector<std::string> order(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Cannot read tensor order");
    }
    std::vector<std::string> result;
    std::set<std::string> unique;
    std::string line;
    while (std::getline(
        stream,
        line
    )) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || !unique.insert(line).second) {
            throw std::runtime_error("Empty or duplicate tensor order entry");
        }
        result.push_back(line);
    }
    if (!stream.eof()) {
        throw std::runtime_error("Cannot finish reading tensor order");
    }
    return result;
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

std::string cacheMetadata(
    const std::vector<char>& cache,
    const std::string& provenance
) {
    return "{\n\"cache_sha256\":\"" + hashBytes(cache) + "\",\n\"identity\":" + provenance + "}\n";
}

void writeBytes(
    const fs::path& path,
    const char* data,
    std::size_t size
) {
    std::ofstream file(
        path,
        std::ios::binary
    );
    file.write(
        data,
        size
    );
    file.close();
    if (!file) {
        throw std::runtime_error("Cannot write: " + path.string());
    }
}
}

std::string calibrationFileHash(const fs::path& path) {
    return hashBytes(bytes(path));
}

CalibrationData auditCalibration(const fs::path& module) {
    const auto root = module / "calibration";
    const auto calibration = order(root / "calibration_tensor_order.txt");
    const auto development = order(root / "development_tensor_order.txt");
    if (calibration.size() != 352 || development.size() != 55) {
        throw std::runtime_error("Expected 352 calibration and 55 development entries");
    }
    const std::set<std::string> dev(development.begin(), development.end());
    // 锚定已提交且经审核的 coverage 版本，避免同时改清单和manifest绕过冻结划分。
    if (calibrationFileHash(root / "coverage.json") != "c61eb53810afe3294998ff6d76afae6102f9bbc05e23fa8596d265a6683a4304") {
        throw std::runtime_error("Approved coverage snapshot changed");
    }
    // 本机OpenCV JSON不支持coverage中的null；这里直接冻结已审核版本的四项摘要。
    const std::map<std::string, std::string> frozenFiles{
        {"hand_boxes.jsonl", "c7156b34c77b7f4a65cdc3c39febe70aee6dfc362212088cfeaf7a8e68e3d9ed"},
        {"split.json", "9a550d03d31aff3d62e0f421eef752d47db3bc890e34e68cffe81d63e641b193"},
        {"calibration_tensor_order.txt", "1b3e332a5da5c7bd5fa2773c63db57d0355cd8888ef2d3301ff486c1d2ff742f"},
        {"development_tensor_order.txt", "bfa5c9cbe26f315ad1c55acc35feb04101c06caa5ead69a3af26ad53a64abcdd"},
    };
    for (const auto& entry : frozenFiles) {
        if (calibrationFileHash(root / entry.first) != entry.second) {
            throw std::runtime_error("Frozen coverage identity mismatch: " + entry.first);
        }
    }
    const auto modelHash = calibrationFileHash(module / "models/rtmpose_hand.onnx");
    const auto referenceHash = calibrationFileHash(module / "configs/rtmpose_hand_reference.json");
    if (modelHash != "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29" ||
        referenceHash != "3036459e1f6bc230c9059f34ea260070928669f54f9b1bfef3e031279aef0aec") {
        throw std::runtime_error("Frozen model/reference identity changed");
    }
    std::map<std::string, std::pair<std::string, std::string>> manifest;
    std::ifstream file(root / "hand_boxes.jsonl");
    std::string line;
    while (std::getline(
        file,
        line
    )) {
        cv::FileStorage row(
            line,
            cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON
        );
        const std::string name = row["tensor_path"];
        const std::string split = row["split"];
        if (static_cast<std::string>(row["review_status"]) != "approved" ||
            static_cast<std::string>(row["decision"]) != "keep" ||
            !manifest.emplace(name, std::make_pair(split, static_cast<std::string>(row["tensor_sha256"]))).second) {
            throw std::runtime_error("Unapproved or duplicate hand manifest entry");
        }
    }
    if (!file.eof() || manifest.size() != 407) {
        throw std::runtime_error("Invalid hand manifest");
    }
    for (const auto& name : development) {
        if (manifest.at(name).first != "development") {
            throw std::runtime_error("Development order split mismatch");
        }
    }
    CalibrationData result;
    cv::FileStorage identity(
        "identity.json",
        cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON
    );
    identity << "schema_version" << 1 << "model_sha256" << modelHash << "reference_sha256" << referenceHash;
    identity << "hand_boxes_sha256" << calibrationFileHash(root / "hand_boxes.jsonl");
    identity << "tensor_order_sha256" << calibrationFileHash(root / "calibration_tensor_order.txt");
    identity << "development_order_sha256" << calibrationFileHash(root / "development_tensor_order.txt");
    identity << "split_sha256" << calibrationFileHash(root / "split.json");
    identity << "cpp_preprocess_sha256" << calibrationFileHash(module / "src/reference.cpp");
    identity << "python_preprocess_sha256" << calibrationFileHash(module / "scripts/verify_reference_config.py");
    identity << "tensor_generation_sha256" << calibrationFileHash(module / "scripts/rebuild_reference_tensors.py");
    identity << "tensorrt_version_integer" << getInferLibVersion() << "cuda_headers_version" << CUDART_VERSION;
    identity << "calibrator_source_sha256" << calibrationFileHash(module / "src/calibrator.cpp");
    identity << "calibrator_header_sha256" << calibrationFileHash(module / "include/rtmpose/calibrator.hpp");
    identity << "int8_entry_sha256" << calibrationFileHash(module / "apps/build_int8_engine.cpp");
    identity << "common_builder_sha256" << calibrationFileHash(module.parent_path() / "trt-runtime/src/builder.cpp");
    identity << "common_builder_header_sha256" << calibrationFileHash(module.parent_path() / "trt-runtime/include/edge/trt/builder.hpp");
    identity << "executable_sha256" << calibrationFileHash("/proc/self/exe");
    identity << "coverage_sha256" << calibrationFileHash(root / "coverage.json");
    identity << "int8_config_sha256" << calibrationFileHash(module / "configs/rtmpose_hand_builder_int8.json");
    identity << "algorithm" << "IInt8EntropyCalibrator2" << "calibration_batch" << 1;
    identity << "calibration_shape" << "[1,3,256,256]" << "inference_profile" << "1/2/4";
    identity << "workspace_mib" << 2048 << "fp16" << 0 << "tf32" << 0;
    identity << "input_name" << "input" << "input_dtype" << "FP32";
    identity << "ordered_tensors" << "[";
    std::set<fs::path> canonical;
    for (const auto& name : calibration) {
        const fs::path relative(name);
        if (relative.parent_path() != fs::path("calibration_set/tensors") || relative.extension() != ".npy" || dev.count(name)) {
            throw std::runtime_error("Calibration path not isolated from development");
        }
        const auto path = fs::canonical(root / relative);
        if (path.parent_path() != fs::canonical(root / "calibration_set/tensors") || !canonical.insert(path).second) {
            throw std::runtime_error("Duplicate or escaping calibration tensor path");
        }
        const auto& entry = manifest.at(name);
        if (entry.first != "calibration" || calibrationFileHash(path) != entry.second) {
            throw std::runtime_error("Calibration split/hash mismatch: " + name);
        }
        // 复用冻结 NPY 解析：严格检查 little-endian FP32、NCHW、payload长度及有限值。
        loadNpy(path.string());
        result.tensors.push_back({path, entry.second});
        identity << "{" << "path" << name << "sha256" << entry.second << "}";
    }
    identity << "]";
    result.provenanceJson = identity.releaseAndGetString();
    return result;
}

HandCalibrator::HandCalibrator(
    CalibrationData data,
    fs::path cache
) : data_(std::move(data)), cachePath_(std::move(cache)) {
    if (data_.tensors.empty() || data_.provenanceJson.empty()) {
        throw std::runtime_error("Empty calibration dataset");
    }
}

HandCalibrator::~HandCalibrator() {
    if (device_) {
        cudaFree(device_);
    }
}

int HandCalibrator::getBatchSize() const noexcept { return 1; }

bool HandCalibrator::getBatch(
    void* bindings[],
    const char* names[],
    int count
) noexcept {
    if (failure_) {
        return false;
    }
    try {
        if (count != 1 || !bindings || !names || !names[0] || std::string(names[0]) != "input") {
            throw std::runtime_error("Expected single calibration binding named input");
        }
        if (cacheUsed_ || next_ == data_.tensors.size()) {
            return false;
        }
        const auto& item = data_.tensors[next_];
        if (calibrationFileHash(item.path) != item.sha256) {
            throw std::runtime_error("Tensor changed after audit");
        }
        const auto tensor = loadNpy(item.path.string());
        const auto size = tensor.size() * sizeof(float);
        if (!device_) {
            checkCuda(cudaMalloc(
                &device_,
                size
            ));
        }
        // 同步 H2D 保证返回时数据可供 TensorRT 使用，不做 /255、通道转换或再次裁剪。
        checkCuda(cudaMemcpy(
            device_,
            tensor.data(),
            size,
            cudaMemcpyHostToDevice
        ));
        bindings[0] = device_;
        ++next_;
        return true;
    } catch (...) {
        failure_ = std::current_exception();
        return false;
    }
}

const void* HandCalibrator::readCalibrationCache(std::size_t& length) noexcept {
    length = 0;
    if (failure_) {
        return nullptr;
    }
    try {
        const auto metadataPath = cachePath_.string() + ".meta.json";
        if (!fs::exists(cachePath_) && !fs::exists(metadataPath)) {
            return nullptr;
        }
        auto cache = bytes(cachePath_);
        const auto meta = bytes(metadataPath);
        if (cache.empty() || std::string(meta.begin(), meta.end()) != cacheMetadata(cache, data_.provenanceJson)) {
            throw std::runtime_error("Cache identity mismatch; use a new cache path");
        }
        cache_ = std::move(cache);
        cacheUsed_ = true;
        length = cache_.size();
        return cache_.data();
    } catch (...) {
        failure_ = std::current_exception();
        return nullptr;
    }
}

void HandCalibrator::writeCalibrationCache(
    const void* data,
    std::size_t length
) noexcept {
    if (failure_) {
        return;
    }
    try {
        if (!data || length == 0) {
            throw std::runtime_error("Empty TensorRT calibration cache");
        }
        const auto* first = static_cast<const char*>(data);
        cache_.assign(
            first,
            first + length
        );
    } catch (...) {
        failure_ = std::current_exception();
    }
}

void HandCalibrator::throwIfFailed() const {
    if (failure_) {
        std::rethrow_exception(failure_);
    }
}

void HandCalibrator::verifyComplete() const {
    throwIfFailed();
    if ((!cacheUsed_ && next_ != data_.tensors.size()) || cache_.empty()) {
        throw std::runtime_error("Calibration incomplete or cache absent");
    }
}

void HandCalibrator::saveCache() {
    verifyComplete();
    if (cacheUsed_) {
        return;
    }
    const auto metadataPath = cachePath_.string() + ".meta.json";
    if (fs::exists(cachePath_) || fs::exists(metadataPath)) {
        throw std::runtime_error("Refusing to overwrite calibration cache");
    }
    const auto metadata = cacheMetadata(
        cache_,
        data_.provenanceJson
    );
    writeBytes(
        cachePath_,
        cache_.data(),
        cache_.size()
    );
    // 若两文件写入被中断，下次读取会拒绝缺失/不匹配的meta，不会复用半成品。
    writeBytes(
        metadataPath,
        metadata.data(),
        metadata.size()
    );
}
}
