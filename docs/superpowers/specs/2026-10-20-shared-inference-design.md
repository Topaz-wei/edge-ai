# edge-ai 公共推理与视频入口设计

状态：用户已确认并完成实现（2026-10-21）；验证范围见 benchmarks/shared-runtime 报告。工作分支：`dev`；起点：`282645d`。

## 目标与约束

将 YOLO 中与模型无关的 TensorRT 执行能力提取到根目录公共库，
将视频采集、调度、输出提取到根目录视频模块。
模型独立保留量化、预处理、后处理和验证能力，使用配置选择已构建 engine。
保持已有 YOLO 的数值行为及命令行入口。

本次不重新构建 engine，不修改网络、量化策略、阈值、NMS、CUDA 算法
或 `--fmad=false` 等数值相关编译选项。构建指编译程序，不运行 engine 构建器。
不移动现有 engine、数据集或历史报告，以免破坏已有脚本路径。

## 方案选择

采用公共静态库、编译期模型注册和运行时配置选择。
仅移动文件不足以消除 YOLO 耦合；动态共享库插件则引入当前不必要的 ABI、
版本管理和生命周期成本，因此这两种方式不作为第一版方案。

## 目录与依赖

- `trt-runtime/`：TensorRT engine 加载、Tensor 元信息、buffer、stream、
  enqueue、CUDA Graph、计时；不得依赖 OpenCV 或模型目录。
- `video-pipeline/`：视频采集、顺序调度、显示、录像和时间统计；
  定义模型适配接口，不直接依赖 YOLO 实现。
- `configs/`：运行组合示例，采用 OpenCV FileStorage 支持的 YAML。
- `cmake/`：公共依赖发现。
- 根 `CMakeLists.txt`：按选项组合公共库、模型和视频可执行程序。
- `tensorrt-yolo/`、`tensorrt-resnet/`：模型自身代码与产物归属。

各模块拥有 `include/src/tests/CMakeLists.txt/README.md`，应用入口保留 `apps/`。
模型 `configs/` 保存模型配置，`scripts/` 保存准备和评测工具。
`models/engines/calibration/data` 分别表示源模型、部署引擎、校准配置与清单、
样例或数据引用；`benchmarks/docs` 保存报告与说明。
`build/cache/logs/debug/profiles` 按需生成，不为目录统一制造空文件。
profiles 表示分析产物，TensorRT shape profile 配置另放 configs。
必须检查 Git 忽略规则，使应追踪的配置和校准说明不会被误忽略。

现有模型目录可独立配置构建；通过 target 存在性检查避免重复引入公共库。
视频依赖保持可选，关闭视频时不要求 videoio/highgui。

## 公共 TensorRT 库边界

第一版沿用当前 TrtRunner 的固定形状、单输入单输出、
FP32 linear device I/O 限制，明确拒绝不支持的 engine。
INT8/FP16 内部计算与 FP32 I/O 不冲突。
暴露输入输出名称、形状、类型供适配器验证。
移除公共类中的 `isFixedYolo()`，将 fixed640 输出规格检查移至 YOLO。
保留 stream、buffer 和同步语义，不引入额外 GPU/CPU 拷贝。
每个实例独占 context/stream，不声称线程安全。
raw 二进制读写与维度工具按职责拆分，避免公共库依赖 YOLO 的 infer_raw 头文件。

YOLO CUDA 预处理、遮罩、检测解码、NMS、校准预处理全部留在 YOLO。
构建器和校准器第一版不强行泛化。
NVTX、CUDA 错误处理等只提取确实可独立复用的部分。

## 模型接口与视频执行

模型工厂按类型注册；每个配置实例创建自己的模型对象。
接口接收只读 BGR 帧和帧标识/时间，返回带模型实例标识的结构化结果及计时。
结果类型明确区分检测和分类，不将所有结果硬转成 Detection。
模型负责其预处理及输出语义，渲染通过适配器或对应结果渲染器完成。

YOLO 的 camera→business 转置/resize 和 360×640 校验留在适配层，
并明确其检测框处于 business 坐标系。
统一展示采用原始输入帧，适配器提供到展示帧的映射；
历史 YOLO 命令保留原 business 画布及文字语义，不改变其输出约定。
无法确定坐标变换时明确报错，不猜测方向。

初版每帧依次运行全部启用模型，不引入后台无界队列、并发或隐式跳帧。
共享帧不得被模型原地修改。只有可视化需要时才复制图像。
保留摄像头缓冲设置及实际值校验、无窗口模式、有限帧运行、
输出拒绝覆盖和帧年龄统计；帧年龄不标称为曝光到显示延迟。
CSV 区分各模型耗时和整帧耗时。
输出打开前校验配置和所有模型，避免运行一半才发现未知模型或缺失 engine。

## 配置约定

配置包含 schema 版本、source、采集设置、display/output/max_frames，
以及有唯一 id、type、engine、模型选项的 models 列表。
相对路径统一相对于配置所在目录解析。
未知字段、未知类型、重复 id、空模型列表、非法数值必须报错。
YOLO input_stage 明确为 camera 或 business，graph 选项保持可配置。
预处理版本和 Tensor 规格由适配器约定，错误 engine 在初始化时拒绝。
engine 环境元数据存在时校验；缺少历史元数据时不伪造，记录其哈希、
实际 I/O 与当前运行环境，由实际反序列化和规格检查确定能否运行。
同帧多模型与检测后裁剪的级联是不同语义，本版只实现前者。

## ResNet 验证范围

使用现有固定形状 engine 和已有输入 Tensor，接入公共 runner，
与修改前的 ResNet 原始输出比较，证明公共层没有依赖 YOLO shape。
ResNet 动态输入程序保持原实现，本版不扩展公共层的动态 shape。
视频分类适配必须复用仓库可核实的预处理和类别定义；
若资料不足，明确报告并保留 Tensor 级验证，不臆造预处理或类别标签。

## 验收与回归

所有基线在修改产品代码前生成，记录提交、engine/input 哈希、工具链与环境。
至少覆盖现有 FP16 和视频文档使用的混合 INT8 engine
`debug/trial007_sub_model22/yolov8m_int8.plan`。

1. 保存固定图及可用验证集的输入 Tensor、原始输出和检测结果；
   不覆盖历史基线，使用独立回归目录。
2. 复用 test_trt_runner 对同一 engine 的新旧输出做逐字节比较；
   保留 timed/untimed 和重复调用检查。
3. 运行预处理、后处理、CUDA Graph 开/关和现有 CUDA 一致性测试；
   发现不一致先定位原因，不自动放宽阈值。
4. 可用完整验证集按原口径跑新旧结果并比较指标；
   缺少标签时仅报告输出一致性，不声称验证了 mAP。
5. ResNet 固定 engine 新旧输出比较。
6. 配置合法/非法路径、多实例、视频无窗口有限帧、旧命令兼容、
   camera/business 几何及输出保护测试。
7. 同一功耗模式和输入、充分 warmup，比较模型耗时和整帧延迟；
   记录统计波动，不以单次 FPS 判断性能退化。
8. 无可用摄像头或桌面时明确记录未验项，用固定离线视频验证流水线。

不修改既有 engine，结束时再次核对哈希。
GPU 不可访问时记录阻塞原因，不用 CPU 冒充 GPU 回归。

## 实施顺序

1. 冻结基线和环境。
2. 提取公共 runtime，保持 YOLO 原命令并完成数值回归。
3. 接入 ResNet 固定推理，验证公共边界。
4. 提取视频能力和模型接口，增加配置入口与测试。
5. 更新目录职责、构建说明和回归报告。

每一步独立验证；不将算法优化、engine 重建和结构迁移混在一起。
书面设计审阅后再编写具体实现计划。
