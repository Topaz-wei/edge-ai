# Shared Inference and Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在 dev 分支提取公共 TensorRT 与视频模块，用原有 engine 验证 YOLO 数值不变。

**Architecture:** 公共固定形状 runtime、模型适配器、配置驱动视频入口分层。
先实现并验证 runtime，再以其稳定接口实现视频模块；保留旧命令行兼容。

**Tech Stack:** C++17、CUDA、TensorRT（本机已有版本）、CMake、OpenCV、Python unittest。

**Spec:** [书面设计](../specs/2026-10-20-shared-inference-design.md)

## Global Constraints

- 分支 dev，不重建或移动 engine，不覆盖既有基线。
- 固定形状、单输入单输出、FP32 linear device I/O；不增加动态 shape 支持。
- 保持 YOLO 算法、阈值、stream/sync 语义和 `--fmad=false`。
- 公共 runtime 不依赖 OpenCV；视频核心不依赖具体模型。
- 相对配置路径以配置目录为基准；无效配置必须报错。
- 多模型顺序处理同帧，不引入级联、并发、隐式跳帧或动态共享库加载。
- 每步验证后提交；GPU 或数据不可用必须报告，不以替代测试声称数值回归通过。

## Review Focus

- 从不同工作目录加载同一配置，应解析到同一 engine/source/output（任务 4）。
- 多实例共享输入时，一个适配器不得污染后续模型输入（任务 5）。
- camera→business 后的检测框显示在原帧上应正确映射（任务 5）。
- 不支持的 engine 必须在推理前明确报错，不能按 float 错读（任务 2）。
- 新旧构建入口同时使用时不能重复定义 target，禁用视频不应引入 GUI 依赖（任务 6）。

## Task 1: 冻结修改前基线

**Files:** 新建 `scripts/regression_shared_runtime.py`、
`tensorrt-yolo/benchmarks/shared-runtime/README.md`；
本地产物使用 `tensorrt-yolo/cache/shared-runtime-baseline/`。

**Interfaces:** Python CLI `capture --output DIR` / `compare --baseline DIR --output DIR`，
manifest 记录文件哈希、命令、返回码、环境和阶段结果；已有目录拒绝覆盖。

- [x] 检查 CUDA、TensorRT、OpenCV、功耗模式及验证集清单；保存工具版本。
- [x] 在独立 `build-baseline` 目录编译当前代码，记录旧可执行程序位置。
  命令：`cmake -S tensorrt-yolo -B tensorrt-yolo/build-baseline -DCMAKE_BUILD_TYPE=Release`
  和 `cmake --build tensorrt-yolo/build-baseline -j4`；预期成功，不执行 yolo_build。
- [x] 捕获 `engines/yolov8m_fp16.plan` 与
  `debug/trial007_sub_model22/yolov8m_int8.plan` 的哈希、固定参考 input.bin、
  test.jpg 及验证集输入哈希；通过旧 `yolo_infer_raw ENGINE INPUT OUTPUT` 保存 raw。
- [x] 保存旧 `yolo_infer_cuda` 的检测结果，覆盖 Graph 开/关；
  运行旧 benchmark 记录 warmup 后耗时及波动。先读取各 CLI 用法再固定完整命令到 manifest。
- [x] 对可用验证集生成旧 batch 输出；使用现有验证工具记录原口径指标和标签覆盖。
- [x] 为 ResNet 旧固定推理增加独立基线捕获副本（放 cache，不改产品文件），
  在已有 hostOutput 处保存二进制，使用现有固定 FP16 engine 和 data/input.bin。
- [x] 验证 capture 拒绝覆盖、compare 对改变一个字节的复制产物返回失败；
  不变产物返回成功。记录 capture 命令及所有不具备的测试条件，提交脚本与说明。

## Task 2: 提取公共 runtime 并完成 YOLO 回归

**Files:** 新建 `trt-runtime/{CMakeLists.txt,README.md}`、
`trt-runtime/include/edge/trt/{runner.hpp,tensor_io.hpp}`、
`trt-runtime/src/{runner.cpp,tensor_io.cpp}`、
`cmake/FindTensorRT.cmake`；修改 YOLO CMake、runner 头/源及引用处、
`src/cuda_pipeline.cpp`，新增 `tensorrt-yolo/include/yolo_engine.hpp`。

**Interfaces:** `edge::trt::Runner` 保留原 TrtRunner 构造、infer、stream、
deviceInput/deviceOutput、prepareGraph/enqueueDevice 接口；
增加 `TensorInfo {std::string name; nvinfer1::Dims shape; nvinfer1::DataType dtype;}`
及 `const TensorInfo& inputInfo() const` / `outputInfo() const`。
`bool isFixedYolo(const edge::trt::Runner&)` 属于 YOLO；
旧头通过 using alias 保持兼容。二进制工具归 edge::trt，旧声明兼容转发。

- [x] 增加 metadata/YOLO shape 边界测试，验证非 YOLO 形状不会被误接受；
  新公共头不存在时先确认编译失败。
- [x] 迁移 runner 与 I/O 工具，不改内存分配、执行、Graph 捕获和同步顺序。
  保留 dtype/format/数量/形状校验，使用 TensorInfo 实现模型层校验。
- [x] 创建 `edge_trt_runtime` target，原 trt_runner 保持兼容 target；
  公共库只链接 TensorRT/CUDA，不链接 ONNX parser/OpenCV。
- [x] 编译新旧入口；运行 `ctest --test-dir tensorrt-yolo/build --output-on-failure`。
- [x] 对两种 engine 运行 `test_trt_runner ENGINE IMAGE BASELINE` 和
  `test_cuda_yolo ENGINE IMAGE`；预期逐字节与旧 raw 一致及 CUDA 一致性通过。
  对支持边界使用现有不匹配 engine 或元信息 fixture，确认拒绝而非错误读取。
- [x] 用任务 1 compare 完成固定图及可用验证集对比；检查输入、raw、检测结果、
  校准源码指纹逻辑与 engine 哈希，提交公共层和迁移。

## Task 3: 用 ResNet 验证复用

**Files:** 修改 `tensorrt-resnet/apps/infer.cpp`、`CMakeLists.txt`、`README.md`；
新增 `tensorrt-resnet/tests/test_shared_runner.cpp`。

**Interfaces:** 原 trt_infer CLI 和指标输出保留，内部改用 edge::trt::Runner。
动态入口与构建器保持原实现。测试 CLI `test_shared_runner ENGINE INPUT BASELINE`。

- [x] 新建测试，断言固定 ResNet engine 元信息、完整 raw 与任务 1 基线逐字节一致，
  错误输入长度抛异常；迁移前先验证公共 target 尚未接入时失败。
- [x] 替换固定推理中的重复执行管理，保留参考输出比较和 top1 计算。
- [x] `cmake -S tensorrt-resnet -B tensorrt-resnet/build-shared` 后构建；
  运行新增测试与旧 CLI，预期数值一致；构建动态目标验证未破坏现有程序。
- [x] 检查 export/reference 脚本是否提供真实图像预处理与类别映射：
  若只是随机 Tensor，本次明确标记 ResNet 仅完成 runtime 复用，不伪造视频分类语义。
- [x] 提交 ResNet 复用及其验证记录。

## Task 4: 模型接口与严格配置

**Files:** 新建 `video-pipeline/include/edge/video/{model.hpp,config.hpp}`、
`video-pipeline/src/config.cpp`、`video-pipeline/tests/test_config.cpp`、
`video-pipeline/CMakeLists.txt`、`configs/yolo-video.yaml`。

**Interfaces:**
- `Frame {const cv::Mat& bgr; uint64_t id; double timestampMs;}`。
- `DetectionResult {cv::Rect2f box; int classId; float score; std::string label;}`，
  box 必须是展示输入帧坐标。
- `ClassificationResult {int classId; float score; std::string label;}`。
- `ModelResult {std::vector<DetectionResult> detections;
  std::vector<ClassificationResult> classifications;
  std::map<std::string,double> timingsMs;}`。
- `IModel::infer(const Frame&) -> ModelResult`，虚析构；
  每个模型对象持有其配置 id，由调度器附加到输出。
- `ModelConfig {std::string id,type; std::filesystem::path engine;
  std::string inputStage; bool graph;}`。
- `VideoConfig` 保存 schemaVersion、source、采集参数、display、
  output、timingCsv、maxFrames 和 `std::vector<ModelConfig> models`。
- `VideoConfig loadConfig(const std::filesystem::path&)`；
  `ModelFactory` 为 `std::function<std::unique_ptr<IModel>(const ModelConfig&)>`，
  registry 为 type 到 factory 的 map。

- [x] 写 config 测试：schema=1、重复 id、未知字段/类型由对应层拒绝、
  空 models、非正 max_frames、非法 camera 设置、错误 YAML 类型均有明确结果。
  改变 cwd 后相对路径解析不变；纯数字 source 保留摄像头编号。
- [x] 运行新增测试确认未实现失败；用 FileStorage 实现严格字段集合与类型校验，
  把相对 engine、文件 source、输出路径转成配置目录下的路径。
- [x] 配置示例使用相对于 configs 的现有混合 INT8 engine，
  显式 camera/business 和 graph。提供无显示有限帧示例。
- [x] 编译并运行配置测试，通过后提交接口、解析器和示例。

## Task 5: 视频核心、YOLO 适配与兼容入口

**Files:** 新建 `video-pipeline/include/edge/video/pipeline.hpp`、
`video-pipeline/src/{pipeline.cpp,render.cpp}`、
`video-pipeline/apps/edge_video.cpp`、
`video-pipeline/tests/test_pipeline.cpp`；
新增 `tensorrt-yolo/include/yolo_video_model.hpp`、
`tensorrt-yolo/src/yolo_video_model.cpp`；
拆分原 video_input.hpp 并精简 apps/yolo_video_cuda.cpp。

**Interfaces:** `int runVideo(const VideoConfig&, const ModelRegistry&)`；
YOLO 提供 `std::unique_ptr<IModel> makeYoloModel(const ModelConfig&)`。
edge_video 负责组装 registry，核心库不 include YOLO。
旧 YOLO CLI 使用公共采集/输出能力的兼容模式，保留 business 画布及原参数含义。

- [x] 使用两个假模型写流水线测试：均观察到原始输入、调用顺序固定、
  frame id 一致、模型实例独立、结果带正确 id、空源/未知类型初始化失败。
- [x] 增加确定性坐标测试：business 输入恒等映射；
  camera 输入业务框逆实际 x/y resize 比例并转置，覆盖非整数缩放取整。
  验证旧入口仍输出 360×640 business 画布及原检测结果。
- [x] 将采集、buffer 校验、无窗口、输出拒绝覆盖、计时、有限帧停止迁入核心；
  展示与录像仅在需要时复制输入，模型不得原地修改输入。
- [x] YOLO 适配器持有 CudaYoloPipeline，复用原 geometry/预处理；
  将结果映射到新入口展示坐标，保持 raw/检测算法和默认阈值。
- [x] 测试失败先确认，再实现对应代码；执行 CPU 配置/流水线测试。
- [x] 用固定离线视频对旧入口、兼容入口和新配置入口分别运行 Graph 开/关；
  比较逐帧推理结果，测试无显示/录像/CSV/重复模型实例/输出路径冲突。
  真实摄像头可用时再验证采集参数和帧年龄；否则明确标记未测试。
- [x] 重跑 YOLO 数值回归和耗时比较，提交公共视频与适配层。

## Task 6: 集成构建、目录约定与最终验收

**Files:** 新建根 `CMakeLists.txt`、`video-pipeline/README.md`；
更新根 README、YOLO README/VIDEO_PIPELINE.md、.gitignore、
回归报告及各模块说明。

**Interfaces:** 根选项 `EDGE_BUILD_YOLO`、`EDGE_BUILD_RESNET`、
`EDGE_BUILD_VIDEO`；视频开启需要对应已注册模型目标。
默认提供 YOLO，ResNet 视频注册仅在其真实预处理具备时加入。

- [x] 根构建与两个子项目独立构建各验证一次，不能重复定义公共 target。
  构建仅 runtime、关闭视频、完整 YOLO+ResNet 三种配置，检查依赖隔离。
- [x] 运行 ctest、已有 Python unittest 和新增测试；
  Python 环境按仓库实际依赖选择，完整命令写入报告。
- [x] 运行任务 1 compare；全部 engine 哈希不变，逐字节差异为零，
  检测结果一致；报告验证集标签范围和性能统计。
- [x] 文档明确源码/产物边界、配置路径规则、接入新模型步骤、
  runtime 限制、旧入口兼容性和未完成的硬件验项。
- [x] 检查 git diff --check、意外生成物及 .gitignore；
  只提交源码、配置、测试、文档，不提交 engine/大数据。
- [x] 完成最终代码审查并修复发现的问题；提交最终报告，汇报实际完成与限制。


## 执行结果（2026-10-21）

任务 1–6 已实施；实际证据和限定范围见
[回归报告](../../../tensorrt-yolo/benchmarks/shared-runtime/README.md)。
9 项根 CTest、22 项既有 Python 测试和 2 项基线工具测试通过；GPU 数值及视频路径已验证。
241 个产物逐字节一致，115 图评测指标不变，engine 哈希不变。

实施裁定：用户指定在现有 dev 工作区执行；配置/视频任务按共享接口组合提交。
ResNet 只验证固定 Tensor 路径，因为源项目未定义真实图像分类预处理。
元数据审计 helper 复用 OpenCV/OpenSSL，由模型适配器提供 TRT/CUDA 实测信息，
保持公共 runtime 不依赖 OpenCV、公共视频核心不依赖 CUDA。
独立审查发现的整数溢出、首帧错误留下空视频、启动审计缺失均已补测试修复；
整数检查同时覆盖未引用字符串中的 apostrophe，不接受跨行 quoted scalar。
物理摄像头/桌面显示未复测。性能实测有波动且新版略慢，未声称性能完全不变；
具体数据保留在报告，未借此次提取更改算法或 engine。
