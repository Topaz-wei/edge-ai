# TensorRT 模型部署研究

完整 CV 部署项目。已归档 vh3 历史模型，并建立 YOLOv8m fixed640 预处理与 PyTorch FP32 Golden；尚未实现 C++ 推理或 INT8 量化。下方模型盘点保留当时的核查边界，最新进展见「YOLO fixed640 Reference」。

## 来源与复制结果

盘点日期：2026-10-09。源目录：

- `/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner`
- `/home/ssd/code/vh3/src/py_algorithm/Thermal_Comfort_Algorithm`

模型按原项目相对路径完整保留，避免同名覆盖：

```text
models/
├── comfort_sensing_air_conditioner/
│   ├── data/weights/             # PTH/PT、ONNX、RTMO ZIP 与 engines/
│   └── others/LYY/model_and_scaler.pkl
├── Thermal_Comfort_Algorithm/models/
└── manifest.json                # 原路径、目标路径、字节数、SHA-256
```

共复制 **31 个文件，1,632,857,233 字节（约 1.52 GiB）**。每个文件均与源文件进行 SHA-256 比对。两项目中的 YuNet、原始 SFace、MiVOLO 分别内容相同，但保留各自来源副本。ZIP 不解包落盘，历史 engine 原样归档。相机标定 `camera_calibration.pkl` 不是学习模型，未复制；人脸特征数据库、样本、业务代码不在本次复制范围。

详细文件清单见 [models/manifest.json](models/manifest.json)。下文的“使用”指历史源码中的加载或实验用途，不代表 vh3 当前在线启用。

## 模型介绍与实际可做的量化推理

以下 C 表示 `models/comfort_sensing_air_conditioner/data/weights/`，T 表示 `models/Thermal_Comfort_Algorithm/models/`。所有 INT8 判断都是后续工程可行性评估，**不是已运行成功的结论**。非 YOLO 模型需要各自的预处理、输出解码与评测，不能直接套用 YOLO 后处理。

| 模型 / 文件 | 用途及本地证据 | 后续可以做什么 |
| --- | --- | --- |
| YOLOv8m：C/`yolov8m.pt`、`yolov8m.onnx` | 历史代码用于人员检测；ONNX 输入 `1×3×640×640`，输出 `1×84×8400`，为框与 80 类分数，业务可筛选 person。 | **首选实验**：已有 ONNX，可先做 FP32/FP16 基线，再做 INT8 PTQ；补齐缩放、类别过滤与 NMS，评估检测精度及端到端延迟。 |
| YuNet：C、T/`face_detection_yunet_2023mar.onnx` | 人脸框与 5 点关键点；图中输入固定 `1×3×640×640`，12 个输出覆盖三个尺度的类别、目标分数、框和关键点。历史 OpenCV 调用却设 `(320,320)`。 | **可做 INT8 候选**：先以真实 ONNX 形状建立基线，不能照搬 OpenCV 尺寸；复现解码/NMS。模型很小，收益需测量。 |
| SFace：C、T/`face_recognition_sface_2021dec.onnx`；C/`sface_cleaned.onnx` | 对齐的 `112×112` 人脸输出 128 维特征，再由相似度匹配身份；不是直接输出姓名的分类器。 | **可做，精度敏感**：优先检查 cleaned 版；其图输入仍列有两个 scalar，但它们是 initializer，实际非 initializer 输入仅 `data`。先验证原版与 cleaned 等价，再评估 INT8 的特征余弦相似度、误识/拒识率及阈值。 |
| MiVOLO：C、T/`model_imdb_age_gender_4.22.pth.tar` | 年龄回归与性别分类。权重元数据确认 `with_persons_model=False`、`no_gender=False`；结合源码及 `pos_embed=[1,14,14,384]`，此权重为 224 输入的人脸分支，不是双输入人体+人脸版本。年龄元数据为 min=1、max=95、avg=48。 | **可做，但先导出**：恢复本地 MiVOLO/timm 结构及年龄反归一化，导出 ONNX 后先验证 FP16；INT8 需检查注意力等算子和年龄 MAE、性别分类指标，可保留敏感层浮点。 |
| 6DRepNet：C/`6DRepNet_300W_LP_AFLW2000.pth`、`sixdrepnet.onnx` | RepVGG-B1g2 骨干的头部姿态估计；输入人脸 `B×3×224×224`。源码生成旋转矩阵再计算欧拉角，ONNX 输出名虽为 `ortho6d`，声明却是三维动态形状，不能按名称假定是 `B×6`。 | **已有 ONNX，可做 INT8 评估**：先确认输出含义及形状，保留必要的归一化、旋转运算为浮点；用角度误差评测，优先建立 FP16 基线。 |
| EfficientFaceCNN：C/`model_fold_1.pth` ～ `model_fold_5.pth` | 从 `64×64` 人脸做胖瘦/体型三分类；源码加载同结构五折模型，属于五份权重，不是五种网络，也不是连续 BMI 数值回归。 | **适合补充量化实验**：先导出一个折，验证分类与速度后再处理五折集成；保留原集成规则，对比单模型及集成的精度/耗时。 |
| LightweightCNN：C/`bmi_classifier.pth` | 早期 `others/air_conditioner_v1.py` 定义并加载，输入 `128×128` 人脸，三分类；与上面的 EfficientFaceCNN 结构不同。 | **可做，但先导出**：使用对应早期结构，不能用五折网络加载；适合作为小 CNN 的 INT8 对照，速度收益未必大。 |
| RepViT 衣物分类：C/`two_branch_repvit15_finetuning_b128_e30.pth`、`upperbody_repvit.onnx` | RepViT-M1.5 上装分类，224 输入。`orin_utils_v21.py` 的 forward 只返回类别；ONNX 实测输出 `category: B×6`。名称中的 two_branch 不意味着当前有属性输出。 | **优先候选**：已有 ONNX，做 6 类衣物分类 FP16/INT8；保持 SmartResize padding 和 ImageNet 归一化，评估 macro-F1。PTH 与 ONNX 数值等价性尚未测试。 |
| RepViT 类别+属性旧版：C/`best_upperbody_model_repvit15_6+35_cos_L_all.pth` | `others/air_conditioner_v10.py` 保留类别和属性分支，6 类、35 属性，属性按类别建模；权重序列化键含 `attribute_shared` / `attribute_heads`。 | **可做，但需单独恢复双输出结构并导出**；不能用目前只输出 6 类的 ONNX 替代，也不能直接用 v21 单输出结构加载。分别评估类别及多标签属性 F1。 |
| RTMDet-Hand：C/`rtmdet_hand.onnx` | 手部检测，输入 `1×3×320×320`，输出 `dets`（框+分数）和 `labels`，检测数动态。 | **可做 INT8 候选**：检查图内 NMS/动态输出的 TensorRT 支持，复现框到手部裁剪的处理。 |
| RTMPose-Hand：C/`rtmpose_hand.onnx` | 手部关键点定位，输入 `B×3×256×256`，输出 `simcc_x`、`simcc_y`，每轴长度 512；须解码 SimCC 并映射回原图。历史代码用 `rtmlib.Hand(mode='lightweight')`。 | **可做 INT8 候选**：与手检测组成两阶段链路，评估关键点误差；关键点分数/解码可保持浮点。历史 Hand 自动选择的模型与这些本地文件是否完全对应仍需验证。 |
| MediaPipe Palm：T/`palm_detection_mediapipe_2023feb.onnx` | 旧版手掌检测；NHWC 输入 `1×192×192×3`，2016 个候选，输出框与 7 点手掌关键点、分数；不是完整 21 点手姿模型。 | **可做 INT8 候选**：复现 `utils/mp_palmdet.py` 的 RGB、除 255、anchors 和 NMS；作为 RTMDet-Hand 的替代检测方案单独比较。 |
| RTMO-s / RTMO-m：C/`rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip`、`rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip` | 人体姿态估计实验包；每包实际含 `end2end.onnx`、`deploy.json`、`detail.json`、`pipeline.json`。ONNX 输入 `B×3×640×640`，输出 `dets` 和 `keypoints`；旧版源码有相关注释调用。 | **可做，第二阶段**：先解包并按部署配置验证推理，再评估 INT8；需检查动态图、NMS 与关键点解码。body7 是包名的一部分，不能据此断言只输出 7 点。 |
| 舒适温度回归：`models/comfort_sensing_air_conditioner/others/LYY/model_and_scaler.pkl` | 序列化内容识别到 sklearn `GradientBoostingRegressor`；测试代码加载 model/scaler，输入用户 ID、心率、年龄、性别编码、体型编码和衣物，预测舒适温度。 | **不列入常规 TensorRT INT8 路线**：属于树回归模型；优先保留 sklearn CPU 推理，恢复训练时编码与 scaler。转成 ONNX 也不等于 TensorRT 支持树算子。 |

历史 FP16 引擎共五个：`engines/yolov8m_fp16.engine`、`repvit_fp16.engine`、`sface_fp16.engine`、`sixdrepnet_fp16.engine`、`rtmpose_hand_fp16.engine`。它们用于归档参考，不代表在本机可直接反序列化运行；后续应依据目标 GPU、TensorRT/CUDA 版本验证兼容性，必要时从 ONNX 重建。FP16 是半精度浮点推理，不能当作已完成 INT8 量化。引擎兼容性依据 [NVIDIA TensorRT capabilities](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capabilities.html)。

## 推荐实施顺序与验收条件

1. **第一批：YOLOv8m、6 类 RepViT。** 已有 ONNX，检测与分类各形成一条完整的 FP32 → FP16 → INT8 实验链路。
2. **第二批：YuNet、SFace、6DRepNet、RTMDet-Hand + RTMPose-Hand、MediaPipe Palm、RTMO。** 按业务价值逐项推进，优先解决输出解码与精度评估。
3. **第三批：EfficientFaceCNN 五折、LightweightCNN、MiVOLO、旧版衣物属性分支。** 先恢复结构、导出 ONNX、核对原模型输出，再量化。MiVOLO 与双输出衣物网络准备成本更高。
4. **舒适温度树模型保留 CPU 路线。** 不把相机标定、温度读取、几何映射、PMV/规则计算及控制逻辑当作神经网络量化任务。

真正开展 INT8 前还需要：

- 确认目标设备、GPU 架构及 TensorRT/JetPack/CUDA 版本，按相应版本选择量化方式和支持精度，不能只因目录名含 orin 就假定部署环境。参见 [NVIDIA 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)。
- 准备真实车内场景校准数据，覆盖光照、遮挡、姿态及衣物；各模型使用其实际输入裁剪与归一化，不能所有模型共用同一份原图张量。另留独立带标注验证集，不用随机输入替代校准/精度验证。
- 优先评估显式 Q/DQ 的 INT8 PTQ；量化比例与敏感层保留浮点由实测决定。旧版 TensorRT 的 calibrator 路线与新版 API 有差异，不能无条件复制旧命令。参见 [TensorRT 10.x 量化说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html) 及 [11.x 迁移说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-c-api-patterns.html)。
- 先验证原框架/ONNX/TensorRT 的 FP32 一致性，再比较 FP16 与 INT8；同时记录任务精度、预处理+推理+后处理总延迟、吞吐、显存和 engine 大小。小模型及五折串行链路不保证 INT8 更快。

## 本次核查边界与源码索引

已读取历史调用、网络定义和 ONNX 图声明，检查 RTMO ZIP 内容；确认所读 ONNX 图未含 QuantizeLinear/DequantizeLinear 节点。MiVOLO 元数据采用 `weights_only=True` 读取；两个衣物 checkpoint 的该加载方式失败，因此只静态检查序列化键并结合源码说明，未宣称成功恢复模型。未导出、修改 ONNX、构建/运行 TensorRT engine，也未做校准、精度或速度测试。

主要证据路径（相对于上述源项目目录）：

- comfort：`analysis_ir_temperature.py`、`air_conditioner_v21.py`：模型组合、预处理及权重加载。
- comfort：`orin_utils_v21.py`：EfficientFaceCNN、单输出 RepViT；`others/air_conditioner_v1.py`：LightweightCNN；`others/air_conditioner_v10.py`：双输出 RepViT。
- comfort：`mivolo/model/mi_volo.py`、`sixdrepnet/model.py`：年龄性别及头姿模型结构/输出处理。
- comfort：`others/LYY/test_model.py`：舒适温度回归特征和 scaler 使用。
- Thermal：`det_all_one_person_v1.py`、`det_all_one_person_v2.py`、`utils/mp_palmdet.py`：人脸、年龄性别与手掌检测链路。

历史代码还引用 `yolov8x.pt`、`checkpoint-377.pth.tar` 等示例路径，但本次两个目录中没有找到这些权重；未下载或把它们列为已经具备的模型。

## YOLO fixed640 Reference（2026-10-09）

采用“双参考、单部署路径”：历史 `.pt` 的 `rect=True / auto=True` 是业务行为参考；部署与逐元素比较统一使用 `auto=False` 的 640×640 张量。典型竖向业务图高×宽为 640×360，历史 `.pt` 左右各补 12 像素，张量为 `[1,3,640,384]`；固定 ONNX 左右各补 140 像素，张量为 `[1,3,640,640]`。不能要求两种形状的 raw output 逐元素一致。

### 真实车内图片及输入阶段

vh3 内已有 `comfort_sensing_air_conditioner/data_collect/` 采集图片：

| 子目录 | JPG 数量 | 解码后 HWC |
| --- | ---: | --- |
| `czq/images/` | 203 | `[640,360,3]` |
| `zbx/images/` | 203 | `[640,360,3]` |
| `10/images/` | 66 | `[640,360,3]` |

共 472 张是文件计数，未做去重或逐张场景审查。已视觉确认 `czq/images/6.jpg` 为真实车内画面，原样复制到 [data/images/test.jpg](data/images/test.jpg)，来源和 SHA-256 记录在 [source.json](data/images/source.json)。

关键证据：`others/save_ac_data.py:173` 先调用 `process_rgb_and_infrared`，`:438` 保存处理后的 `frame`，而不是 `cap.read()` 的未处理帧。因此当前样本属于 **business 输入（未加 v21 两块遮罩）**，不能再转置一次。JPEG 已有压缩损失，这个基准固定的是 JPEG 解码后的像素，不声称恢复拍摄时的无损相机帧。相机原图入口使用合成数据与历史处理核对，尚未以确认未处理过的真实相机文件建立第二份端到端基准。

脚本强制显式 `--input-stage camera|business`：

- `camera`：BGR uint8 HWC → 宽高转置（等价于 `np.rot90(k=1)[::-1]`）→ `imutils.resize(height=640)` 等价实现，INTER_AREA。
- `business`：跳过以上操作，不猜测方向。两种入口都要求最终业务图 `[640,360,3]`，以保证历史固定像素遮罩适用；其他几何尺寸明确报错。
- 接着 BGR→RGB，依次用**当前整图**的每通道均值填充头枕和车窗多边形，写回 uint8 时截断小数；第二块重新计算均值。再转回 BGR。头枕坐标 `(240,0),(360,0),(360,200),(300,100)`；车窗 `(0,150),(65,106),(95,237),(6,327)`。
- LetterBox 保持比例、允许放大、INTER_LINEAR、居中、`auto=False`、`scale_fill=False`，边界采用 `round(d/2 ± 0.1)`，填充 114。
- BGR→RGB，HWC→NCHW，连续内存，float32 除以 255；无 ImageNet mean/std。文件解码不自动应用 EXIF 方向。

### 运行与产物

本机可用环境为 `/home/ssd/anaconda3/envs/py38/bin/python`；默认 `python` 不保证指向该环境。两个命令默认使用仓库根目录下的数据路径，不依赖当前工作目录；示例从仓库根目录执行。

```bash
/home/ssd/anaconda3/envs/py38/bin/python scripts/prepare_yolo_reference.py --input-stage business
/home/ssd/anaconda3/envs/py38/bin/python scripts/run_pytorch_reference.py --device cuda
```

已经生成下列文件，**脚本拒绝覆盖已有 Reference**。复跑请使用新目录，例如预处理加 `--output-dir data/reference_run2`，推理加 `--reference-dir data/reference_run2`。新相机原图使用 `--input path/to/camera.png --input-stage camera`。CPU 基准必须显式 `--device cpu`，不会在 CUDA 失败时静默切换。

```text
scripts/
├── prepare_yolo_reference.py
└── run_pytorch_reference.py
data/reference/
├── business.png                  # 无损中间图；避免再次 JPEG 压缩
├── masked.png
├── letterbox.png
├── input.npy                     # FP32 [1,3,640,640]
├── input.bin                     # little-endian float32, C order，无文件头
├── metadata.json                 # 输入阶段/来源哈希/几何/遮罩/版本/文件哈希
├── preprocess_verification.json  # 对原始历史类与 LetterBox 的独立验证记录
├── pytorch_output.npy            # FP32 [1,84,8400]
├── pytorch_output.bin            # little-endian float32, C order
├── pytorch_metadata.json         # 权重、输入、脚本、运行环境及模型状态
├── onnx_output.npy               # 本次额外 ORT 诊断输出
└── onnx_comparison.json          # 本次额外 ORT 对照记录
```

`input.bin` 为 4,915,200 字节，`pytorch_output.bin` 为 2,822,400 字节。后续 C++/trtexec 应按模型 binding 名称、形状和 FP32 类型读取这些二进制，不能把有文件头的 `.npy` 当 raw bin。现有根目录 `models/yolov8m.pt`、`models/yolov8m.onnx` 用于运行；其哈希记入各报告，原始归档仍保留来源目录。

PyTorch 脚本直接调用 `YOLO(...).model`，不经过 Predictor，不做第二次预处理；要求 Ultralytics 8.3.173，`eval()`、FP32、无 AMP，关闭 CUDA matmul/cuDNN TF32，关闭 cuDNN benchmark 并启用 deterministic。返回 tuple 时取第一个张量并检查形状、有限值；同一进程重复推理检查逐元素一致。默认模型实际融合状态也记入 metadata，不凭文件名推断。

这里的 raw 指 **pre-NMS、已解码**输出，不是原始卷积 logits：前 4 通道为模型输入坐标系中的中心 `xywh`，后 80 通道为类别概率，没有独立 objectness；不要重复 sigmoid 或再次做 DFL 解码。输出形状正确只是接口验证，不能代替模型数值对照。

### 本次实测与验证边界

- 输入：FP32 `[1,3,640,640]`，`r=1`，LTRB padding=`[140,0,140,0]`。
- 将原始 `MeanRegionMask` 与 Ultralytics 8.3.173 `LetterBox` 类从源码独立提取执行，遮罩图与输入 tensor 均逐元素一致，最大绝对差 **0**。另验证 3 种相机尺寸和 3 种 LetterBox 尺寸，包含缩放和奇数补边；没有导入带相机/控制副作用的完整历史业务模块。
- PyTorch：Orin CUDA，torch `2.1.0a0+41361538.nv23.06`，FP32 `[1,84,8400]`，同进程两次结果 bitwise 相同。该结果不保证跨设备/跨版本 bitwise 相同。
- 使用 ONNX Runtime `1.19.2` CPUExecutionProvider，对现有 ONNX 输入**同一份 input.npy** 做了额外诊断对照（这不是 TensorRT 对照）：

| 与 PyTorch Golden 比较 | 最大绝对差 | 平均绝对差 |
| --- | ---: | ---: |
| 全部元素 | 0.00140381 | 1.15514e-6 |
| 前 4 通道框坐标（像素） | 0.00140381 | 2.35606e-5 |
| 后 80 通道类别概率 | 1.64658e-6 | 3.48672e-8 |

上述为单图诊断值，不是预先确定的验收阈值，也不证明完整数据集的检测准确率。后续应分别报告框/概率误差，使用绝对+相对容差，避免用近零概率的相对误差或被大量低概率值稀释的全局均值作唯一判据。FP16/INT8 还需检测结果及真实验证集指标。

回归测试：`python3 -m unittest discover -s tests -v`，覆盖方向、逐次均值/截断、历史真实多边形顺序、固定/奇数补边、RGB/NCHW、business 入口、npy/bin 一致性及错误几何/覆盖拒绝。

### 坐标与后续路线修正

模型中心框 `xywh` 可按 `x_b=(x_m-left)/r`、`y_b=(y_m-top)/r`、`w_b=w_m/r`、`h_b=h_m/r` 还原。推荐转为 `xyxy` 后裁剪边界：连续框坐标使用 `[0,W]` 和 `[0,H]`；只有实际像素索引才使用 `[0,W-1]` 和 `[0,H-1]`。典型业务图只需 x 减 140。需要原始 camera 坐标时，再逆业务缩放并交换 x/y；非单位缩放须考虑宽度 `int()` 舍入造成的实际 x/y 比例差异。当前 business 样本没有原始相机帧尺寸证据，不能伪造该映射来源。

后处理要复现历史 `classes=[0], conf=0.7` 时，默认 `multi_label=False` 的 Ultralytics 是先选 80 类中的最高分/类别，再保留 person，并非只判断 person 通道 >0.7。默认 NMS IoU=0.7、max_det=300；这些参数与坐标还原应在后处理阶段单独验证，当前未实现 NMS。

路线：固定预处理与 PyTorch Golden（已生成）→ ONNX Runtime 诊断（已对照）→ C++ TensorRT FP32 raw 对照 → FP16 → 后处理与端到端 → 独立校准/验证数据上的 INT8。严格 FP32 对照时 TensorRT 也应关闭 TF32；引擎融合造成的小数值差异仍需容差判定。历史形状分支只作业务参考。

仓库已有 `logs/yolov8m_build_test.log` 记录 TensorRT 8.5.2 `--buildOnly` 成功、跳过推理；`yolov8m_parse.log` 中 `--skipInference` 命令失败。前者支持“曾构建成功”，不支持“推理数值已经验证”。本次没有新增 TensorRT 引擎、C++、FP16 或 INT8 测试。

当前 Git 规则忽略 `data/` 和 `models/`，数值样本已在本地落盘，未自动纳入版本控制；可追踪脚本和本节文档。真正长期共享 Golden 时需同时归档带哈希的样本、权重、环境与输出，单独提交脚本不足以固定数值基准。

## 单图 INT8 精度搜索

构建器支持 timing cache 和精确 FP16 层名单；Python checker、C++ 构建驱动的精筛及私有 Polygraphy 粗定位入口见 [PRECISION_SEARCH.md](docs/PRECISION_SEARCH.md)。默认只使用固定 input.bin 和冻结FP16输出，不读取 validation。
