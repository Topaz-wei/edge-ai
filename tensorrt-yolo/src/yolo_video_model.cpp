#include "include/yolo_video_model.hpp"
#include "include/yolo_video_geometry.hpp"
#include "include/cuda_yolo.hpp"
#include "edge/trt/tensor_io.hpp"
#include "edge/video/engine_artifact.hpp"
#include <iostream>
namespace {
class YoloModel final : public edge::video::IModel {
    std::string stage_;
    CudaYoloPipeline pipeline_;
public:
    explicit YoloModel(const edge::video::ModelConfig& c):stage_(c.inputStage),pipeline_(c.engine.string(),c.graph) {
        int device=0, cudaVersion=0;
        cudaDeviceProp props{};
        edge::trt::checkCuda(cudaGetDevice(&device), "get device");
        edge::trt::checkCuda(cudaGetDeviceProperties(&props, device), "device properties");
        edge::trt::checkCuda(cudaRuntimeGetVersion(&cudaVersion), "CUDA version");
        const auto info = edge::video::inspectEngineArtifact(c.engine, {
            {"tensorrt_version", std::to_string(getInferLibVersion())},
            {"cuda_runtime_version", std::to_string(cudaVersion)},
            {"gpu_name", props.name},
            {"gpu_compute_capability", std::to_string(props.major)+"."+std::to_string(props.minor)},
            {"input_name", pipeline_.inputInfo().name},
            {"input_shape", edge::trt::dimsToString(pipeline_.inputInfo().shape)},
            {"output_name", pipeline_.outputInfo().name},
            {"output_shape", edge::trt::dimsToString(pipeline_.outputInfo().shape)},
            {"io_dtype", "float32"},
            {"preprocess_version", "yolo-fixed640-business-v1"},
            {"preprocess_sha256", YOLO_PREPROCESS_SHA256}
        });
        std::cout << "Model " << c.id << " engine " << c.engine << '\n';
        for (const auto& field : info) std::cout << "  " << field.first << ": " << field.second << '\n';
    }
    void validateFrame(const cv::Mat& frame) const override {
        video::toBusinessFrame(frame, stage_);
    }
    edge::video::ModelResult infer(const edge::video::Frame& frame) override {
        const auto business=video::toBusinessFrame(frame.bgr,stage_);
        // Preserve original video inference sync behavior: timing is measured by host scheduler.
        auto detections=pipeline_.infer(business);
        edge::video::ModelResult result;
        for(const auto& d:detections) {
            const auto& b=d.businessBox;
            result.detections.push_back({yolo::toSourceBox({b.x1,b.y1,b.x2-b.x1,b.y2-b.y1},frame.bgr.size(),stage_),
                                        d.classId,d.confidence,"person"});
        }
        return result;
    }
};
}
std::unique_ptr<edge::video::IModel> makeYoloModel(const edge::video::ModelConfig& config) {
    if(config.inputStage!="camera" && config.inputStage!="business")
        throw std::runtime_error("YOLO requires input_stage camera or business");
    return std::make_unique<YoloModel>(config);
}
