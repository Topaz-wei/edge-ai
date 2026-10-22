# 公共视频流水线

C++17 / OpenCV；核心库不依赖 TensorRT、CUDA 或 YOLO。
`edge_video` 应用在组合层注册 YOLO factory，实例由配置选择。

## 配置

采用 OpenCV FileStorage YAML（保留 `%YAML:1.0`）。布尔值严格使用 `0/1`，
source/路径使用字符串。未知字段、重复 id、空 models、非法类型和数值会报错。

```yaml
%YAML:1.0
---
schema_version: 1
source: "../datasets/example.avi"
display: 0
max_frames: 100
output: "../outputs/example.mp4"
timing_csv: "../outputs/example.csv"
models:
  - id: detector_fp16
    type: yolo
    engine: "../tensorrt-yolo/engines/yolov8m_fp16.plan"
    input_stage: business
    graph: 1
  - id: detector_int8
    type: yolo
    engine: "../tensorrt-yolo/debug/trial007_sub_model22/yolov8m_int8.plan"
    input_stage: business
    graph: 0
```

路径相对于配置文件目录，与启动 cwd 无关。数字字符串（如 `"0"`）、`/dev/video*`、
`/dev/v4l/*` 作为摄像头源，URL 保留原样。数字文件名写成 `"./0"`。
`max_frames` 省略表示不限帧数，显式设置必须为正整数。
摄像头专属选项：`capture_width/capture_height` 成对，`capture_fps`、
`capture_buffers`（默认 2）为正整数，`capture_fourcc` 为四字符字符串。
实际缓冲数和请求不一致时报错，不自动改变功耗模式或频率。

所有模型初始化完成后才创建输出；已有输出和符号链接拒绝覆盖，CSV 与视频不能同路径。
新入口在 source 帧绘制结果。YOLO camera 模式在 CPU 准备阶段转换为业务图后推理，
再按实际缩放比例逆转置检测框；business 模式要求 360×640 BGR 帧。
旧 YOLO 命令由适配层提供 business 画布变换，共享视频核心不硬编码业务几何。

## 接入接口

- `IModel::prepare(const Frame&) const -> unique_ptr<PreparedInput>`：只做 CPU 准备，返回逐帧独立的数据；可与同一实例的 `infer()` 并发，不能修改共享推理缓冲或输入像素。
- `IModel::infer(const PreparedInput&) -> ModelResult`：消费准备结果，负责 GPU 提交、传输、同步和结果封装；不再执行 CPU 图像变换。同一模型的 infer 仍然串行调用。
- `Frame` 持有 `cv::Mat` 引用计数和帧号/时间戳；`PreparedInput` 保存源帧，模型可派生自己的输入类型持有图像或张量。复制 Mat 头不复制像素，调用方必须保持底层像素只读直至消费结束。
- `ModelResult` 分别存储 detection/classification，不强制所有任务返回检测框。检测框使用源帧坐标。
- 每个配置 id 创建独立模型对象；同帧各模型按配置顺序 prepare、按配置顺序 infer。
- 固定两个槽位：准备线程读帧/解码、执行 `VideoHooks::transform` 和所有模型的 prepare；主线程 infer、绘制、编码和显示。第 N+1 帧准备与第 N 帧消费交叠，槽位消费结束前不复用，不主动丢帧。
- EOF 和准备异常按帧顺序交付；退出时通知并 join 准备线程。OpenCV 的阻塞 read 无法通用地中断，退出可能需要等待正在进行的读取完成。`max_frames` 不会额外读取或准备超过上限的帧。
- `VideoHooks` 仅用于已有应用的展示变换，不是 YAML 隐含全局预处理；第一帧验证时在主线程执行，之后在准备线程执行，返回的像素必须在消费完成前保持有效、不可覆盖。
- YOLO 的 CPU 几何处理位于 `tensorrt-yolo/include/yolo_video_preprocess.hpp` 和 `src/yolo_video_preprocess.cpp`，保留转置及 INTER_AREA 缩放语义。

CSV 新增 `prepare_ms`（全部模型 CPU prepare）和 `queue_wait_ms`（准备好到主线程取走）。
`geometry_ms` 为兼容入口的画布变换；新入口的 YOLO 几何处理计入 `prepare_ms`。
`pipeline_ms` 和 `<id>_wall_ms` 为消费端 infer 的 host wall time，包含传输/等待/结果封装，不是纯 GPU kernel 时间。
`start_ms` 是准备线程开始处理该帧的时刻，`end_ms` 是消费完成时刻，`total_ms=end_ms-start_ms` 包括排队等待。
相邻帧时间区间可以重叠，不能用 `1000/mean(total_ms)` 代表吞吐；吞吐看 actual processing FPS/QPS，Full loop EMA 是消费循环间隔。
第一帧读取、画布变换及验证仍是初始化，排除在稳态计时外；第一次模型 prepare/infer 包含在计时内，未做 warmup。
frame age 仅在相机时间戳通过校验时有效，否则 -1；不是曝光到显示延迟。
计时数据在内存中累计并在结束时保存，长时间不限帧运行时请留意 CSV 内存使用。

独立核心测试：`cmake -S video-pipeline -B video-pipeline/build && cmake --build video-pipeline/build && ctest --test-dir video-pipeline/build --output-on-failure`。
`test_video_model` 需要 GPU 和已存在的 YOLO engine，故不自动放入 CPU CTest。

## Engine 审计与可选元数据

YOLO 适配器启动时记录 engine SHA-256、实际输入输出名称/形状、FP32 I/O、
TensorRT/CUDA 运行时版本、GPU 名称/计算能力以及预处理版本/源码哈希。
这只在初始化时执行，不进入逐帧路径。

若 engine 为 `model.plan`，可放置 `model.plan.meta.yaml`：

```yaml
%YAML:1.0
---
engine_sha256: "替换为构建产物的真实 SHA-256"
tensorrt_version: "8502"
cuda_runtime_version: "11040"
input_shape: "[1x3x640x640]"
output_shape: "[1x84x8400]"
io_dtype: "float32"
```

sidecar 存在时必须包含 engine_sha256；其余字段按需声明，所有值必须为字符串，
每个声明必须与当前实测值严格一致。完整字段名见启动日志；未知或重复字段也会报错。
版本数值仅为本机示例，不能替代实际构建环境。没有 sidecar 的旧 engine 不伪造历史元数据，
仍通过实际反序列化、I/O 检查并记录当前信息。此处采用保守的精确匹配策略，
不会自动推断跨 TensorRT 版本或 GPU 的兼容模式。
审计 helper 使用 OpenSSL Crypto，公共 TensorRT runtime 仍仅依赖 TensorRT/CUDA。

配置整数字面量限定为有符号 32 位范围；在交给 FileStorage 前检查，防止解析时溢出。
模型可实现 `validateFrame`，第一帧会在创建输出前经过所有模型验证，避免错误几何留下空视频。
跨行 quoted scalar 不属于本配置格式；路径等字符串请写在单行上。普通未引用字符串中
的 apostrophe 不会影响后续数值检查。
