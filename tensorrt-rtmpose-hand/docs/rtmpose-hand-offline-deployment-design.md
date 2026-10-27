# RTMPose-Hand 离线量化部署方案 v1

日期：2026-10-27。状态：设计提案，已创建目标模块的 docs 目录用于保存本文；尚未实现部署代码、复制模型/数据或构建 engine。

## 1. 目标与边界

在 `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand` 新增独立模块，完成 RTMPose-Hand 的 FP32 参考、TensorRT FP32/FP16/INT8 engine、离线手部关键点推理、数值一致性与性能验证。

第一阶段固定 batch=1，每次推理一只手，支持一张原图中的多只手顺序推理。Python 用于资产准备、参考输出和评测，C++ 用于 TensorRT 构建及离线部署。暂不修改视频流水线，不量化 RTMDet，不接摄像头，不训练模型，不支持动态 batch 或 DLA。

RTMDet-Hand 仅作为离线手框生成辅助；现有 YOLO 人体检测器不参与本阶段。人体框不能替代手框。RTMPose-Hand 不是全身姿态模型。

## 2. 已核查事实与方案选择

- 本机 `/usr/include/aarch64-linux-gnu/NvInferVersion.h` 声明 TensorRT 8.5.2.2；实施前还需记录实际链接库、GPU、驱动、CUDA 和 Python 环境，确认一致。
- RTMPose ONNX：opset 11，FP32 输入 `input: [batch,3,256,256]`，两个 FP32 输出 `simcc_x`、`simcc_y`，最后一维均为 512。关键点维在原图元数据中是符号维，必须由 ONNX Runtime 实跑确认是 21，不能仅靠文件名假定。
- RTMDet ONNX 输入为 `[1,3,320,320]`，用于获得手框。
- 公共 runtime 当前只接受固定形状、单输入、单输出、FP32 线性设备 I/O，因此必须扩展多输出支持。
- 源模型和 edge-ai 归档模型已经逐文件核对 SHA-256，一致。

已选择“先离线量化部署，后续再接视频”的路线。相较直接接实时检测级联，该路线将手框固定，能够单独定位预处理、关键点解码与量化误差。实现上建议保留 ONNX 双输出并扩展公共 runtime；不为了迁就单输出接口拼接输出，也不在模型目录复制一套通用 runtime。

## 3. 模型资产及绝对路径

### 3.1 必需：RTMPose-Hand

优先复制来源：

`/home/ssd/projects/edge-ai/tensorrt-yolo/models/comfort_sensing_air_conditioner/data/weights/rtmpose_hand.onnx`

等价原始来源：

`/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data/weights/rtmpose_hand.onnx`

计划目标：

`/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/models/rtmpose_hand.onnx`

SHA-256：`39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29`

### 3.2 必需辅助：RTMDet-Hand

优先复制来源：

`/home/ssd/projects/edge-ai/tensorrt-yolo/models/comfort_sensing_air_conditioner/data/weights/rtmdet_hand.onnx`

等价原始来源：

`/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data/weights/rtmdet_hand.onnx`

计划目标：

`/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/models/rtmdet_hand.onnx`

SHA-256：`568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2`

通过 ONNX Runtime FP32 离线检测，结果审核后冻结为 bbox manifest。无需先构建其 engine。

### 3.3 可选归档：历史 RTMPose FP16 engine

来源：

`/home/ssd/projects/edge-ai/tensorrt-yolo/models/comfort_sensing_air_conditioner/data/weights/engines/rtmpose_hand_fp16.engine`

等价原始来源：

`/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data/weights/engines/rtmpose_hand_fp16.engine`

可选目标：

`/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/engines/archive/rtmpose_hand_fp16.engine`

SHA-256：`2294ac2f7f353a484e465517c0bd3fc2c4b2b955a320ba0dd55726fda7d1e626`

此文件只有历史精度命名证据，不作为本次验收基线。新 engine 在目标设备重新构建；旧 engine 可选检查兼容性，加载失败不阻塞任务。

## 4. calibration 和 validation 的原图划分

原则：先按采集目录和近似重复组划分，再生成手框和裁剪；不可对全部手部 crop 随机拆分。以下是初始分组提案，不声称目录名等于独立人物或独立采集场次。实施时核对内容和采集元信息，跨组近似重复图必须整组归入同一集合或剔除，并重新记录数量。

### 4.1 calibration：331 张候选原图

| 来源绝对目录 | 数量 | 用途 |
| --- | ---: | --- |
| `/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data_collect/czq/images` | 203 | 车内实采 |
| `/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data_collect/10/images` | 66 | 车内实采 |
| `/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/camera/calibration/rgb` | 46 | RGB 原图，抽样为车内手掌 |
| `/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/camera/calibration/v2/rgb` | 16 | RGB 原图，抽样为车内双手 |

前两组在 edge-ai 中已有候选副本：

- `/home/ssd/projects/edge-ai/tensorrt-yolo/calibration/czq/images`
- `/home/ssd/projects/edge-ai/tensorrt-yolo/calibration/10/images`

其数量已核对，内容需在导入时重新与 vh3 逐文件校验，不能只凭目录同名复用。

### 4.2 validation：258 张候选原图，单独报告两类场景

主验证集（车内）：

| 来源绝对目录 | 数量 |
| --- | ---: |
| `/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data_collect/zbx/images` | 203 |

等价候选副本目录为 `/home/ssd/projects/edge-ai/tensorrt-yolo/calibration/zbx/images`。虽然父目录叫 calibration，本方案把这组数据用于 RTMPose validation；旧 YOLO 校准缓存及用途不沿用。导入时同样核验副本内容。

补充验证集（办公室/跨场景）：

| 来源绝对目录或匹配模式 | 数量 |
| --- | ---: |
| `/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/camera/calibration/v1/rgb` | 25 |
| `/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/rgb` | 10 |
| `/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/v1_H/rgb` | 4 |
| `/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/v2_H/rgb` | 7 |
| `/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/v3_H/rgb` | 7 |
| `/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/v1_H/t/rgb_*.jpg` | 2 |

补充组共 55 张；场景标签需逐张确认，不能仅由目录推定。现有 589 张候选文件之间 SHA-256 不重复；仍可能存在连续帧、同人物或近似重复。

### 4.3 划分后的使用约束

- 校准数据只用于激活分布估计；校准前可以剔除误检、空裁剪和损坏图片，保留合理的遮挡、不同尺度和低光照手部。
- 校准原图组中预留少量独立近似重复组作为开发检查集，比例建议约 10%，具体数量取决于分组；这些开发样本不进入校准缓存。331 是原图候选上限，不是最终校准 batch 数。
- 混合精度层选择、阈值与解码调试只使用开发集。冻结方案后运行 validation；若依据 validation 调整策略，报告必须说明该集已被用于调优，不能再声称是独立最终评测。
- validation 不进入 INT8 校准，不因 INT8 表现差而删图。检测失败、无手图和无效手框必须记录原因与覆盖率。
- 同一个人出现在不同目录并不自动算泄漏，但需要在报告中说明人物重叠及有限的泛化范围。
- 第一版不混入红外图、棋盘格、人脸照片及带绘图覆盖的视频帧。

## 5. 导入、存储与追溯

为了形成可独立搬运的模块，允许将所选原图和两个小型 ONNX 各复制一份到新模块；不再复制整套 YOLO 数据和模型目录。未获用户要求时不提交这些二进制资产到 Git。

原图目标目录：

- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/data/raw/comfort_sensing_air_conditioner/`：保留上文该项目下的相对路径。
- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/data/raw/Thermal_Comfort_Algorithm/`：保留上文该项目下的相对路径。

例如，czq 的目标目录为：

`/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/data/raw/comfort_sensing_air_conditioner/data_collect/czq/images`

计划生成：

- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/models/manifest.json`：模型源路径、目标相对路径、大小、SHA-256。
- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/data/manifest.jsonl`：原图源绝对路径、目标相对路径、尺寸、哈希、采集组、场景与拆分归属。
- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/calibration/split.json`：固定校准、开发、车内验证、补充验证分组。
- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/data/annotations/hand_boxes.jsonl`：原图 ID、手框、检测分数、检测模型哈希、人工修正和拒绝原因。
- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/data/prepared/`：每只手的 FP32 NCHW 输入张量与几何变换元信息。
- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/validation/reference/`：ONNX Runtime 原始输出、解码坐标及可视化。

跨机器运行使用目标相对路径定位资产，源绝对路径只用于追溯。重复导入按哈希验证后跳过；同名不同内容报错，禁止静默覆盖。

## 6. 预处理与输出基准

1. 锁定并记录 rtmlib 版本/源码 commit，显式传入本地两个 ONNX，禁止隐式下载其他权重。
2. 确认该版本与本地手部模型的配置匹配，保存 mean/std、BGR/RGB、框扩展系数、插值、padding、仿射规则、SimCC split ratio 和分数计算规则。不要只从通用默认值猜测。
3. 通过 RTMDet 生成候选框，抽样及异常项人工检查，验证集漏检手可人工补框并记录来源。所有精度版本使用同一份冻结手框。
4. 使用原始手框计算中心与尺度，做与参考一致的 256×256 仿射变换；归一化为 FP32 NCHW，保留逆变换信息。不是将整张图直接缩小到 256×256。
5. Python ONNX Runtime FP32 作为现有 ONNX 的参考实现；实跑确认输入/输出名、形状、关键点数及有限数值。保存两个 SimCC 原始输出与最终坐标。
6. C++ 预处理和解码分别与 Python 对照。先让不同推理后端使用完全相同的输入张量，再测试各自预处理路径，避免把几何变换误差误判为量化损失。
7. SimCC 解码、置信度及无效点处理沿用锁定参考，坐标通过对应逆变换还原到原图。不擅自新增 softmax、改变分数阈值或左右手翻转规则。

上游参考：https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/pose_estimation/rtmpose.py 。上游 main 仅供查阅，实施必须固定版本；当前环境尚未确认与 vh3 历史运行一致的 rtmlib 安装。

## 7. engine 构建与量化策略

### 7.1 固定形状和基线

保留原始 ONNX 不变，生成 batch=1 的部署派生模型：

`/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/models/rtmpose_hand_b1.onnx`

设置静态输入并做形状推断/检查，跑参考样本确认派生模型与源模型在 batch=1 时一致。若输出仍含符号维，以实测和构建器形状推导校验，不手工猜形状。

构建三个独立 engine，保留双输出且 I/O 为 FP32：

- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/engines/rtmpose_hand_b1_fp32.engine`
- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/engines/rtmpose_hand_b1_fp16.engine`
- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/engines/rtmpose_hand_b1_int8.engine`

FP32 对照明确禁用 TF32；FP16 允许必要的 FP32 fallback。所有构建在目标 Jetson 上完成，记录 builder 配置、workspace、版本和模型哈希。

### 7.2 INT8 PTQ

本机采用 TensorRT 8.5 的 `IInt8EntropyCalibrator2`，对不含 Q/DQ 的浮点 ONNX 做隐式 PTQ。开启 INT8 与 FP16 fallback；不把“启用了 INT8”写成“所有层均为 INT8”。该方案针对 8.5 环境，不承诺直接适用于未来 TensorRT 主版本。

每批提供一只手的 `[1,3,256,256]` FP32 输入，使用同一预处理实现，严格固定样本顺序。校准回调错误、损坏样本或未完整遍历应让构建失败；不使用随机数据校准，不使用 YOLO 的校准缓存。

校准产物：

- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/calibration/rtmpose_hand_b1_int8.cache`
- `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/calibration/rtmpose_hand_b1_int8.cache.meta.json`

meta 绑定 ONNX 哈希、样本和 bbox 哈希、预处理代码/参数、样本顺序、校准算法、batch、TensorRT/CUDA 与设备信息；任一关键字段改变即失效。

导出 engine layer inspector 信息，确认 INT8 层实际存在并记录混合精度。若开发集误差过大，再在开发集尝试对 SimCC 输出头或其他实测敏感层保留 FP16/FP32；不能预先认定某层一定敏感。选定配置后冻结再运行验证。

可参考现有代码（只借鉴错误处理、缓存追溯与构建框架）：

- `/home/ssd/projects/edge-ai/tensorrt-yolo/include/calibrator.hpp`
- `/home/ssd/projects/edge-ai/tensorrt-yolo/apps/calibrator.cpp`
- `/home/ssd/projects/edge-ai/tensorrt-yolo/apps/build_engine.cpp`
- `/home/ssd/projects/edge-ai/tensorrt-yolo/benchmarks/INT8_PTQ.md`

YOLO 业务遮罩、640×640 输入、letterbox、/255 处理及检测后处理不能复制到 RTMPose 校准路径。

## 8. C++ 部署与仓库改动范围

新增模型目录按需放置 `apps/`、`include/`、`src/`、`scripts/`、`tests/`、`configs/`、`benchmarks/`，不创建无实际用途的空目录。

公共 runtime 修改范围为 `/home/ssd/projects/edge-ai/trt-runtime`：

- 支持单输入、多个具名输出，按名字管理形状、缓冲和数据拷贝。
- 保持固定形状和 FP32 线性 I/O 限制，第一版不扩展动态形状。
- 保持现有单输出调用兼容；对多输出 engine 调用旧单输出接口时明确报错，不能静默取第一个输出。
- 双输出 CPU 拷贝和 device API 均有明确所有权；同步和 CUDA Graph 行为回归验证。

新增根 CMake 选项 `EDGE_BUILD_RTMPOSE_HAND`，默认关闭；离线构建可以关闭视频和 YOLO，只启用公共 runtime 和新模块。视频关键点结果及级联接口留给后续阶段。

离线入口至少支持：单张原图加 bbox、多张原图加 bbox manifest、已准备张量 benchmark。输出包括原图坐标下的关键点、分数、手框 ID、engine 信息与计时；可选写可视化图片。手框由离线 Python 工具准备，无需 C++ 先部署 RTMDet。

## 9. validation 与验收

### 9.1 数值一致性验证

目前没有确认存在可直接使用的标准 21 点手部真值。已有相机标定 JSON 是跨相机对应点，不是该任务的完整真值。

对固定手框和同一输入，比较 ORT FP32、TRT FP32、TRT FP16、TRT INT8：

- SimCC X/Y 的 MAE、最大绝对误差及非有限值计数。
- 关键点在 256×256 输入空间及原图空间的欧氏偏移，报告均值、中位数、P95、最大值；另报按手框尺度归一化偏移。
- 分数变化、有效关键点数量变化和阈值翻转率。不能只统计 INT8 高置信度点，否则会隐藏退化。
- 车内主验证与办公室补充验证分别报告，同时展示小手、遮挡和低光照样本。
- 无手、漏检、错误框、有效手框及有效关键点的总数全部保留在覆盖率报告中。

这部分叫“一致性验证”，不是 mAP/PCK 的真实精度。若需要真实精度，在冻结的验证子集中人工标注手框、21 点与可见性，明确不可见点规则和尺度归一化，再计算 PCK/NME。真值标注是可选增强，不能自动把 FP32 输出当人工真值。

建议在开发集阶段先锁定验收阈值，再看最终验证结果。可讨论的初始目标：INT8 对 FP16 的输入空间关键点偏移 P95 ≤ 2 px、有效点比例下降 ≤ 1 个百分点；这些只是工程提案，不是模型保证，需结合业务与可见性口径确认。不得根据最终验证表现反向放宽门槛。

### 9.2 性能验证

同设备、同功耗模式、同输入序列、同 batch，预热至少 100 次、测量至少 1000 次；记录温度、频率状态及原始数据。测试期间避免其他重负载，发生降频则记录并解释。

分别报告 GPU engine 时间（CUDA event）、H2D/D2H、CPU 预处理/解码、每只手端到端 wall time；图片解码/I/O 是否计入要明确。报告 median/P95、吞吐、engine 大小和 buffer/显存口径。

INT8 对 FP16 是否有收益以实测为准；若无显著收益或误差超阈值，保留 FP16 为推荐部署版本。离线 RTMDet 生成手框耗时单列，不混入 RTMPose engine 延迟。多手原图耗时按实际手数统计。

### 9.3 必要测试

- Python/C++ 预处理与坐标逆变换对照，包括越界框、边缘手、非正方形框和多手图。
- 双 SimCC 输出的读取、解码、无效值处理；零手图应正常返回空结果。
- 固定 batch 派生 ONNX 等价性；ORT 与 TRT FP32 基线对齐。
- 多输出 runtime 的 shape/dtype 错误、资源生命周期、旧单输出兼容和已有 YOLO/ResNet 相关回归。
- 缓存失效、校准失败传播、样本读取完整性与集合交叉检查。

## 10. 分阶段交付

1. 环境与资产核验：确定版本、复制两个 ONNX 和所选原图、生成 manifest；不覆盖来源。
2. 数据冻结：近似重复分组，划分 calibration/development/validation，检测并审核手框，输出实际可用手数。
3. 参考冻结：确认 21 点配置，生成 FP32 输入、SimCC 输出、坐标与可视化。
4. runtime 和离线 FP32/FP16：支持双输出、构建静态模型与基线 engine，通过预后处理与数值测试。
5. INT8：真实数据校准、缓存追溯、开发集敏感层分析，冻结最终配置。
6. 独立验证和性能对比：输出两类场景报告及 FP16/INT8 选择结论。
7. 文档交付：新模块 README、复现命令、来源与拆分 manifest、构建配置、engine 哈希及 benchmark 原始记录。最终报告位于 `/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/benchmarks/offline-quantization/README.md`。

实施前须落实的技术核查：本地 rtmlib/模型预处理配置、实跑输出关键点数、数据组的近似重复和人物重叠。它们属于第一阶段工作，不要求用户先手动整理。


## 11. 方案概要（补充自上一轮答复）

本节汇总上一轮答复的模型选择、数据划分、整体流程和验收结论；详细实施约束与完整资产路径见前文。

### 11.1 模型选择

目标目录：`/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand`。

| 模型 | 本阶段用途 | 是否量化 |
| --- | --- | --- |
| RTMPose-Hand | 估计手部关键点 | 构建 FP32、FP16、INT8 engine |
| RTMDet-Hand | 离线生成手框，准备手部输入 | 暂不量化，使用 ONNX Runtime FP32 |

优先复制以下两个模型，已确认与 vh3 原模型 SHA-256 一致：

```text
/home/ssd/projects/edge-ai/tensorrt-yolo/models/comfort_sensing_air_conditioner/data/weights/rtmpose_hand.onnx
/home/ssd/projects/edge-ai/tensorrt-yolo/models/comfort_sensing_air_conditioner/data/weights/rtmdet_hand.onnx
```

目标分别为：

```text
/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/models/rtmpose_hand.onnx
/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand/models/rtmdet_hand.onnx
```

历史 `rtmpose_hand_fp16.engine` 仅作可选归档，本次基线 engine 在目标设备重新构建。现有 YOLO 人体检测不参与这条离线流程。

### 11.2 calibration 与 validation 划分

按采集目录分组，避免将相邻图片生成的手部裁剪随机分到两边。

| 集合 | 来源分组 | 候选原图数 |
| --- | --- | ---: |
| 校准准备 | data_collect/czq、data_collect/10、comfort 的 calibration/rgb 和 calibration/v2/rgb | 331 |
| 车内验证 | data_collect/zbx | 203 |
| 补充验证 | comfort 的 calibration/v1/rgb，以及 Thermal 的 30 张 RGB 图片 | 55 |

331 张是校准候选原图数，不是最终校准样本数。先检查近似重复，再从其中留出少量开发检查组，其余用于生成校准手部输入。验证集保持独立。

校准来源绝对路径：

```text
/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data_collect/czq/images
/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data_collect/10/images
/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/camera/calibration/rgb
/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/camera/calibration/v2/rgb
```

车内主验证来源：

```text
/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data_collect/zbx/images
```

补充验证来源：

```text
/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/camera/calibration/v1/rgb
/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/rgb
/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/v1_H/rgb
/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/v2_H/rgb
/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/v3_H/rgb
/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm/camera/calibration/v1_H/t/rgb_*.jpg
```

589 张候选文件没有完全相同的哈希重复，但仍要检查连续帧和相似场景。目录划分只是初始方案，不能直接等同于人物、场次完全独立。复制时保留来源项目及相对目录结构，同时生成清单记录源路径、哈希、采集组和集合归属。

### 11.3 整体流程

```text
原图导入、去重与分组
        ↓
RTMDet-Hand 离线生成手框
        ↓
审核手框，冻结 bbox 清单
        ↓
按参考实现做手部仿射变换、归一化
        ↓
生成 1×3×256×256 FP32 输入
        ↓
ONNX Runtime FP32 参考输出
        ↓
TensorRT FP32 / FP16 基线
        ↓
使用 calibration 手部输入构建 INT8
        ↓
开发集检查与必要的混合精度调整
        ↓
冻结配置，在独立 validation 上比较
```

各精度版本使用同一份手框和同一份输入张量，先隔离量化误差，再验证完整预处理、解码和原图坐标还原。

### 11.4 量化部署方法

本机 TensorRT 头文件版本为 8.5.2.2，采用现有项目适用的方案：

- 使用 `IInt8EntropyCalibrator2` 做 INT8 PTQ。
- 固定 batch=1，一次处理一只手。
- INT8 允许 FP16 fallback，输入输出保留 FP32。
- 分别生成 FP32、FP16、INT8 engine。
- 校准缓存绑定模型、手框、样本、预处理和环境信息。
- 检查 engine 实际层精度，不能仅凭构建时开启 INT8 就认定量化成功。

YOLO 的校准程序只能参考框架，不能沿用其业务遮罩、640×640 预处理或校准缓存。

### 11.5 公共能力改造

`/home/ssd/projects/edge-ai/trt-runtime` 当前只支持单输出，而 RTMPose 有 `simcc_x` 和 `simcc_y` 两个输出。因此：

- 扩展公共 runtime，支持多个具名输出。
- 保留现有 YOLO 单输出调用兼容性。
- 将 RTMPose 输入固定为 batch=1，暂不扩展动态形状。
- 预处理、SimCC 解码、坐标还原放在新模型目录。
- 第一阶段不修改 `video-pipeline`。

### 11.6 部署验收与最终交付

验证分为两部分：

- 一致性：原始 SimCC 输出差异、关键点偏移、分数变化、有效点数量变化，分别报告车内与补充场景。
- 性能：GPU 推理时间、数据传输、预处理与解码、每只手端到端耗时，以及 FP16/INT8 对比。

现有数据没有确认可直接使用的标准 21 点真值，所以第一版只报告相对 FP32 的一致性。需要真实关键点精度时，再对独立验证子集人工标注。

最终交付包含可复现命令、资产清单、固定数据划分、三个 engine、离线推理程序和对比报告；是否推荐 INT8 部署，以精度与速度实测结果决定。
