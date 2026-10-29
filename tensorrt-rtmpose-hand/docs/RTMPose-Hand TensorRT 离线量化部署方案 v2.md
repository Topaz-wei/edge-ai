# RTMPose-Hand TensorRT 离线量化部署方案 v2

## 0. 当前实施状态与已有产物

2026-10-29 对齐：本文件作为后续实施基线。**模型导入、校准/开发集手框审核、裁剪和单手输入张量准备已完成；TensorRT INT8 校准尚未执行。** 下文的工程目标和最终验收项不代表已经实现。

| 阶段 | 当前状态 | 已有证据与剩余工作 |
| --- | --- | --- |
| Stage 1 环境与模型核验 | 部分完成，模型导入完成 | 两个原始 ONNX 已导入，SHA-256 与来源一致；已有 ONNX 检查和 ORT batch=1 零输入实跑记录。Python/ORT/rtmlib 版本及源码哈希已记录；头文件中可找到所需具名 API。ORT 动态 batch 1/2/3/4 与 reference 契约已核验；实际 TensorRT 链接库及 TRT 动态 batch 实跑仍待验证 |
| Stage 2 数据冻结 | 校准/开发数据划分、手框及裁剪已认可，覆盖度报告已生成 | 352 个校准张量、55 个开发张量已分目录保存；全部已检出手框保留，零检测不补框。`calibration/coverage.json` 已记录几何和亮度代理指标，未标注语义保持未知；validation 手框未准备，人物/采集场次独立性不作额外保证 |
| Stage 3 Reference Freeze | 技术契约已冻结，人工关键点评审通过 | RGB 契约已核验；407 个张量已修正，55 个开发样本已保存 ORT FP32 输出、解码与原图坐标，10 个代表样本已生成可视化；详见 [核验记录](reference-freeze.md) |
| Stage 4 Runtime vNext | 尚未扩展 | 现有 runtime 已使用具名 Tensor API 和 `enqueueV3()`，但仍限制固定 shape、单输入/单输出和 FP32 线性设备 I/O；动态 shape、多个输入/输出、profile 与容量管理待实现 |
| Stage 5 TRT FP32 / FP16 | 未开始 | 未生成本模块 engine，也未做动态 batch 和后端一致性验证 |
| Stage 6 INT8 PTQ | 未开始 | 无本模块 INT8 cache、engine 或 inspector 结果 |
| Stage 7 最终验证 | 未开始 | validation 的 258 张原图已导入并隔离，尚未生成手框及参考输出；GT 增强项尚未实施 |
| Stage 8 Benchmark | 未开始 | 尚无本模块精度/批次性能报告 |

当前资产入口：

- [模型清单](../models/manifest.json)、[环境与模型检查记录](../models/environment_check.json)。
- [数据集使用说明](../calibration/README.md)、[已认可划分](../calibration/split.json)、[用户认可记录](../calibration/approval.json)。
- [手框与张量清单](../calibration/hand_boxes.jsonl)、[校准顺序表](../calibration/calibration_tensor_order.txt)、[开发集顺序表](../calibration/development_tensor_order.txt)。
- [逐帧报告](../calibration/index.html)、[总览报告](../calibration/overview.html)、[最近一次完整性检查](../calibration/verification.json)。

已认可的数据划分、手框和裁剪不需要重新审核。Reference 核验确认模型要求 RGB，已保留划分与裁剪并重建全部 407 个张量及相关哈希、覆盖度报告。55 个开发样本的参考链路已通过自动检查，关键点叠图已通过用户评审；此结果不代表量化或 GT 精度验收。

v2 相比 v1 的技术范围变化仍按本文件执行：部署采用原始动态 batch ONNX、初始 profile 1/2/4、公共 runtime 动态及多 Tensor 扩展，不再以固定 batch=1 派生模型为正式路线。现有 `[1,3,256,256]` 单手张量可保留，后续按 batch 维组装动态输入，不需要仅因开放动态 batch 重做手部裁剪。

## 1. 项目目标

在 Jetson AGX Orin 上建立独立工程：

```text
/home/ssd/projects/edge-ai/tensorrt-rtmpose-hand
```

完成 RTMPose-Hand 的：

```text
ONNX Runtime FP32 reference
        ↓
TensorRT FP32
        ↓
TensorRT FP16
        ↓
TensorRT INT8 PTQ
        ↓
必要时 Mixed Precision
        ↓
离线关键点验证
        ↓
性能 Benchmark
```

同时借此升级公共：

```text
/home/ssd/projects/edge-ai/trt-runtime
```

使其从目前面向固定 shape、单输出模型的 runtime，升级为：

```text
name-based Tensor API
+
static / dynamic shape
+
single / multiple inputs
+
single / multiple outputs
+
Optimization Profile
+
enqueueV3
```

本阶段重点不是实时视频，而是：

> 正确理解并实现 Top-down Hand Pose 的 ROI、Affine、SimCC、多输出、Dynamic Batch 和 INT8 部署。

原始方案已经明确 RTMPose-Hand 是本阶段的核心模型，RTMDet-Hand 只用于离线产生手框；不量化 RTMDet、不修改现有视频流水线。

---

# 2. 第一阶段项目边界

本阶段包含：

| 内容 | 是否做 |
|---|---|
| RTMPose-Hand ORT FP32 reference | 是 |
| TensorRT FP32 | 是 |
| TensorRT FP16 | 是 |
| TensorRT INT8 PTQ | 是 |
| 必要的 mixed precision | 可选 |
| Dynamic Batch | **是** |
| Dynamic H/W | **否** |
| 多输出 Runtime | **是** |
| `enqueueV3()` | **是，作为主执行接口** |
| RTMDet TensorRT 部署 | 否 |
| 摄像头实时级联 | 暂不做 |
| CUDA Graph for RTMPose Dynamic Batch | 第一版不做 |
| DLA | 不做 |
| 模型训练/QAT | 不做 |

RTMPose-Hand 的部署形式固定为：

```text
input:
[N, 3, 256, 256]

N:
dynamic

C/H/W:
static
```

即：

> 只开放 Batch 动态，不开放输入分辨率动态。

---

# 3. 整体架构

完整离线流程：

```text
原始 RGB/BGR 图片
        │
        ▼
RTMDet-Hand
ONNX Runtime FP32
        │
        ▼
hand bbox
        │
   人工审核/修正
        │
        ▼
冻结 bbox manifest
        │
        ├──────────────┐
        ▼              ▼
hand ROI 0          hand ROI 1 ...
        │              │
        └──────┬───────┘
               ▼
     center / scale
               ↓
       affine transform
               ↓
       normalization
               ↓
    [N,3,256,256] FP32
               ↓
      RTMPose-Hand
      TensorRT Engine
               ↓
       ┌───────┴───────┐
       ▼               ▼
   simcc_x           simcc_y
 [N,K,512]         [N,K,512]
       └───────┬───────┘
               ↓
          SimCC Decode
               ↓
      256×256 keypoints
               ↓
        inverse affine
               ↓
     原图关键点坐标
```

`K` 已通过原始 ONNX 的 ORT batch=1 实跑确认为 21：输入 `input` 为 FP32 `[1,3,256,256]`，输出 `simcc_x`、`simcc_y` 均为 FP32 `[1,21,512]`，输出有限。原始输入 batch 维为符号维。该记录是零输入形状冒烟检查，不代表真实样本精度、batch=2/3/4 或 TensorRT 动态 batch 已验证；后续仍按实际 runtime shape 核验，不能仅根据文件名猜测。

---

# 4. 模型资产

## 4.1 RTMPose-Hand

状态：已导入，原始 ONNX 保持不变；已重新核对目标文件 SHA-256 与下列记录一致。

优先来源：

```text
/home/ssd/projects/edge-ai/tensorrt-yolo/models/
comfort_sensing_air_conditioner/data/weights/
rtmpose_hand.onnx
```

目标：

```text
tensorrt-rtmpose-hand/models/rtmpose_hand.onnx
```

当前已记录 SHA-256：

```text
39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29
```

## 4.2 RTMDet-Hand

状态：已导入，哈希核对通过；已通过 ORT FP32 为校准/开发来源的 331 张原图生成候选框，完成用户审核。

仅作为手框生成器：

```text
models/rtmdet_hand.onnx
```

当前已记录 SHA-256：

```text
568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2
```

使用：

```text
RTMDet-Hand ONNX
        ↓
ONNX Runtime FP32
        ↓
hand bbox
```

不构建 TensorRT engine，不进入本阶段 INT8 分析。

这样能够把：

```text
detector error
```

和：

```text
pose quantization error
```

隔离开。

这也是原方案选择冻结统一手框的主要目的。

---

# 5. 数据集设计

继续使用当前按**原始采集组**而不是 crop 随机拆分的方案。

初始导入候选（历史数量）：

| Split | 原图数量 |
|---|---:|
| Calibration preparation | 331 |
| In-car validation | 203 |
| Cross-scene validation | 55 |

上述 331 张已经进一步划分，不再全部属于 calibration。当前用户认可的数据如下：

| Split | 来源原图数 | 有已保留手框的原图数 | 零检测原图数 | 手部裁剪/张量数 |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 285 | 242 | 43 | 352 |
| Development | 46 | 45 | 1 | 55 |
| In-car validation | 203 | 尚未检测 | 尚未检测 | 尚未生成 |
| Cross-scene validation | 55 | 尚未检测 | 尚未检测 | 尚未生成 |

原始图片已全部导入 `data/raw/`。当前划分以 `calibration/split.json` 为准；`data/manifest.jsonl` 及 v1 中的候选数量保留导入历史，不代替当前分组。

核心原则：

```text
先按 raw image / capture group 划分
              ↓
再生成 hand bbox
              ↓
再生成 hand crop
```

禁止：

```text
所有 hand crop
      ↓
random shuffle
      ↓
calibration / validation
```

否则连续帧中极其相似的手可能泄漏到两个集合。

---

# 6. Calibration Dataset 进一步冻结

当前状态：校准集的 352 个已认可手框及对应单手张量已就绪，位于 `calibration/calibration_set/`。固定读取顺序见 `calibration/calibration_tensor_order.txt`；手框、变换矩阵、来源和张量哈希见 `calibration/hand_boxes.jsonl`。开发集的 55 个张量位于 `calibration/development_set/`，使用独立顺序表。

用户已明确认可全部检测到的手部裁剪，保留 5 对重叠框；44 张零检测原图（校准 43、开发 1）不补框，记录为 `skip`。零检测不等于已确认无手，也不能直接统计为已证实的漏检。这里冻结的是原图归属与已审核手框；张量预处理已通过 Stage 3 的 RGB 参考契约核验。

331 张只是：

```text
calibration candidate raw images
```

真正的 INT8 calibration dataset 应定义为：

```text
冻结 bbox 后生成的
RTMPose input tensors
```

即：

```text
[1,3,256,256] FP32 hand tensors
```

已生成（以现有认可数据为输入）：

```text
calibration/coverage.json
```

报告分别统计 calibration/development，记录输入清单和脚本哈希，保留逐手测量值。校准集 285 张来源原图中有 242 张贡献 352 个张量；每图保留框数 0/1/2+ 为 43/134/108。开发集 46 张来源原图贡献 55 个张量，0/1/2+ 为 1/35/10。几何边缘按框与四周 5% 边缘带相交定义，低亮度按排除仿射填充后的裁剪平均亮度小于 60 定义；两者均是代理指标，不取代人工语义标注。详见 [覆盖度报告](../calibration/coverage.json) 和 [统计口径](../calibration/README.md)。

至少统计：

```text
raw image 数
有效 hand bbox 数
最终 calibration tensor 数
0 / 1 / 2+ hand image 数
bbox size 分布
边缘手数量
遮挡样本数量
低光照数量
reject 数量及原因
```

报告以当前 352 个校准 crop/tensor 为主体，同时说明其 285 张来源原图中实际贡献手框的 242 张及跳过的 43 张。55 个开发张量应单列，不能合并计入校准覆盖度。遮挡、左右手、真实漏检等未经确认的语义属性标为 unknown；尺寸、亮度或边缘位置等自动统计必须记录定义，不能将代理指标冒充人工标注。

校准前允许删除：

```text
RTMDet 明显误检
损坏图片
非法 bbox
空 crop
```

但不能因为：

```text
姿态困难
遮挡严重
INT8 表现不好
```

而删除正常业务样本。

Validation 数据不得进入 calibration。

---

# 7. Development / Validation 隔离

已从 calibration candidate group 中预留整个 `comfort_sensing_air_conditioner/camera/calibration/rgb` 来源目录，共 46 张原图、55 个裁剪，占原始 331 张候选的约 13.9%。用户已认可此划分，不再为凑齐 10% 重新随机拆分。这部分作为：

```text
development set
```

用途：

```text
INT8 error analysis
threshold tuning
mixed precision layer selection
decode/debug
```

最终：

```text
validation
```

只能在配置冻结之后运行。

如果根据 validation 结果再次修改：

```text
precision
threshold
layer fallback
decode rule
```

则必须明确：

> 该 validation 已经成为 development data，不再属于独立最终评测。

这一点继续沿用 v1 的设计。

---

# 8. RTMPose Reference Contract

这是本项目最重要的基线之一。

当前状态：`configs/rtmpose_hand_reference.json` 已冻结；官方同哈希模型包确认 RGB，几何与解码锁定本地 rtmlib 0.0.15。核验报告见 `configs/rtmpose_hand_reference.verification.json`，复现方式见 [Reference Freeze](reference-freeze.md)。

新增：

```text
configs/rtmpose_hand_reference.json
```

冻结以下信息：

```text
ONNX SHA256

input:
    name
    dtype
    shape

outputs:
    simcc_x name
    simcc_y name
    dtype
    shapes

input_size:
    256x256

color:
    BGR/RGB

normalization:
    mean
    std

bbox processing:
    center
    scale
    scale factor

affine:
    interpolation
    padding
    border value

SimCC:
    split_ratio
    argmax rule
    keypoint score rule
    invalid point rule

coordinate transform:
    model → original image
```

这些参数必须从：

```text
本地实际历史实现
+
锁定版本 rtmlib
+
实际 ONNX
```

中确认。

不得根据通用 RTMPose 示例猜测。

特别是：

```text
bbox expansion
affine
mean/std
BGR/RGB
SimCC split_ratio
score calculation
```

必须全部冻结。原方案对此已经明确要求。

---

# 9. Python FP32 Reference

Python 第一阶段负责建立 Ground Reference：

```text
raw image
 +
frozen bbox
        ↓
reference preprocess
        ↓
input tensor
        ↓
ORT FP32
        ↓
simcc_x
simcc_y
        ↓
reference decode
        ↓
original image keypoints
```

保存：

```text
validation/reference/

input/*.bin
raw/simcc_x/*.bin
raw/simcc_y/*.bin
decoded/*.json
visualization/*.jpg
```

Reference 必须记录：

```text
ONNX hash
bbox manifest hash
reference config hash
script hash
ORT version
NumPy/OpenCV version
```

---

# 10. TensorRT Runtime vNext

## 10.1 总体目标

当前代码基线：`trt-runtime/src/runner.cpp` 已使用 `getNbIOTensors()`、具名 Tensor 查询、地址绑定及 `enqueueV3()`。`trt-runtime/include/edge/trt/runner.hpp` 仍声明固定 shape、单输入/单输出、FP32 线性设备 I/O 限制。因此本阶段是在已有具名执行路径上扩展能力，而不是重新做一次 V2 到 V3 的接口迁移。

公共：

```text
trt-runtime
```

升级为：

```text
TensorRT name-based generic runtime
```

不再以：

```text
binding[0]
binding[1]
```

作为主要 abstraction。

统一采用：

```text
tensor name
tensor mode
dtype
shape
device address
capacity
```

---

# 11. Runtime Tensor Abstraction

建议至少定义：

```cpp
struct TensorMeta {
    std::string name;
    nvinfer1::TensorIOMode mode;
    nvinfer1::DataType dtype;

    nvinfer1::Dims engineShape;
    nvinfer1::Dims runtimeShape;

    bool dynamic;
};

struct TensorBuffer {
    TensorMeta meta;

    void* device{nullptr};

    size_t capacityBytes{0};
    size_t activeBytes{0};
};
```

不要让：

```text
RTMPose
YOLO
ResNet
```

进入公共 runtime 的代码。

公共 runtime 只认识：

```text
Tensor
```

---

# 12. Static + Dynamic Shape 统一处理

加载 engine 后：

```text
getNbIOTensors()
      ↓
getIOTensorName()
      ↓
getTensorIOMode()
      ↓
getTensorDataType()
      ↓
getTensorShape()
```

检查 engine shape 中是否存在：

```text
-1
```

然后：

```text
              Engine
                 │
       ┌─────────┴─────────┐
       │                   │
     Static              Dynamic
       │                   │
engine shape          runtime shape
       │                   │
       │              setInputShape()
       │                   │
       └─────────┬─────────┘
                 ▼
          resolve outputs
                 ↓
        setTensorAddress()
                 ↓
           enqueueV3()
```

Static model 是 dynamic-capable runtime 的特殊情况。

因此：

```text
YOLO
ResNet
RTMPose
```

全部使用同一个 runner。

---

# 13. enqueueV3 作为统一执行接口

新 runtime 不再新增基于：

```text
enqueueV2()
```

的功能。

统一：

```cpp
context->setTensorAddress(
    inputName,
    inputBuffer
);

context->setTensorAddress(
    outputName,
    outputBuffer
);

context->enqueueV3(stream);
```

对于动态模型，在此之前增加：

```cpp
context->setInputShape(...);
```

必要时：

```text
inferShapes()
```

检查所有 shape 已经完全解析。

旧代码如果仍然需要 `enqueueV2()`，只作为兼容历史实现保留，不再作为 runtime vNext 的设计基础。

---

# 14. RTMPose Dynamic Batch

正式部署继续使用原始 ONNX：

```text
rtmpose_hand.onnx
```

不再要求：

```text
rtmpose_hand_b1.onnx
```

作为正式部署模型。

预计输入：

```text
[-1,3,256,256]
```

初版 Optimization Profile 建议：

```text
MIN = [1,3,256,256]
OPT = [2,3,256,256]
MAX = [4,3,256,256]
```

实际 profile 在 RTMDet 完成数据统计后最终冻结。

如果业务数据显示：

```text
99.9% image <= 2 hands
```

也可以考虑：

```text
1 / 2 / 2
```

当前 `1 / 2 / 4` 是工程初始配置，不作为永久结论。

---

# 15. Dynamic Batch Runtime 行为

例如一张图产生：

```text
2 hands
```

则：

```text
ROI 0
ROI 1
  │
  └── stack
       ↓
[2,3,256,256]
       ↓
setInputShape()
       ↓
RTMPose
       ↓
simcc_x [2,K,512]
simcc_y [2,K,512]
```

如果：

```text
0 hands
```

则：

```text
skip RTMPose inference
```

如果：

```text
1 hand
```

则：

```text
batch = 1
```

如果 hand 数超过 profile MAX：

第一版建议：

```text
按 MAX batch 分块
```

例如：

```text
6 hands

batch 4
+
batch 2
```

禁止静默丢弃多余 hand。

---

# 16. Dynamic Buffer 管理

第一版不采用：

```text
shape变化
→ cudaFree
→ cudaMalloc
```

而采用：

> 按 optimization profile MAX shape 一次分配。

例如 MAX batch=4：

```text
input capacity:
4×3×256×256

simcc_x:
4×K×512

simcc_y:
4×K×512
```

运行：

```text
batch1 → 使用部分容量
batch2 → 使用部分容量
batch4 → 使用全部容量
```

因此：

```text
dynamic shape
≠
dynamic cuda allocation
```

runtime 保存：

```text
capacityBytes
activeBytes
```

每次 inference 只根据实际 runtime shape 更新：

```text
activeBytes
```

---

# 17. Dynamic Output Shape

不能在 RTMPose 业务代码里假设：

```cpp
N * 21 * 512
```

公共 runtime 应：

```text
setInputShape()
        ↓
context->getTensorShape(output)
        ↓
resolve actual output shape
        ↓
calculate active bytes
```

RTMPose 层再验证：

```text
K == expected keypoint count
last_dim == expected SimCC dimension
```

即：

```text
generic runtime
负责 shape

RTMPose adapter
负责语义
```

---

# 18. Optimization Profile 管理

第一版 RTMPose：

```text
Profile 0
MIN=1
OPT=2
MAX=4
```

runtime 仍显式管理 profile：

```cpp
setOptimizationProfileAsync(
    profileIndex,
    stream
);
```

第一版：

```text
profileIndex = 0
```

但 API 不应硬编码“永远只有 profile0”。

这样未来：

```text
profile0 → small batch
profile1 → large batch
```

无需重构 runtime。

---

# 19. CUDA Graph 策略

现有 YOLO static pipeline 可以继续使用 CUDA Graph。

RTMPose Dynamic Batch 第一版：

```text
enqueueV3
+
NO CUDA Graph
```

原因不是 `enqueueV3()` 不支持 Graph，而是避免同时引入：

```text
dynamic shape
+
multi-output
+
profile
+
Graph cache
```

增加调试复杂度。

后续若需要，可实现：

```text
GraphCache
├── batch1
├── batch2
├── batch3
└── batch4
```

每个 shape 单独 capture。

---

# 20. TensorRT Engine

正式构建：

```text
engines/
├── rtmpose_hand_fp32.engine
├── rtmpose_hand_fp16.engine
└── rtmpose_hand_int8.engine
```

三者使用相同：

```text
ONNX
optimization profile
input/output contract
```

FP32：

```text
TF32 OFF
```

FP16：

```text
FP16 enabled
允许必要 FP32 fallback
```

INT8：

```text
INT8 enabled
FP16 fallback enabled
```

不能把：

```text
builder 开启 INT8
```

等同于：

```text
整个 network 全 INT8
```

必须保存 TensorRT layer inspector 信息验证实际 precision。

原方案对三个独立 engine、FP32 禁用 TF32 和 FP16 fallback 已经做了明确要求。

---

# 21. INT8 PTQ

继续使用当前 Jetson TensorRT 8.5 环境适用的：

```text
IInt8EntropyCalibrator2
```

校准数据：

```text
frozen hand tensors
[1,3,256,256]
```

每批一只手。

建立独立 calibration profile：

```text
MIN = [1,3,256,256]
OPT = [1,3,256,256]
MAX = [1,3,256,256]
```

而 inference optimization profile：

```text
1 / 2 / 4
```

二者职责分离。

INT8 cache：

```text
calibration/
├── rtmpose_hand_int8.cache
└── rtmpose_hand_int8.cache.meta.json
```

meta 至少绑定：

```text
ONNX SHA256
reference config SHA256
bbox manifest SHA256
calibration tensor list + SHA256
preprocess implementation SHA256
sample order
TRT version
CUDA version
device
calibration algorithm
```

关键条件变化：

```text
cache INVALID
```

禁止复用 YOLO calibration cache 或 YOLO preprocessing。原设计已经明确二者必须完全隔离。

---

# 22. Mixed Precision 策略

不预先认定：

```text
SimCC head 一定敏感
```

第一步直接：

```text
FP16
vs
INT8
```

在 development set 比较。

只有 INT8 明显超出预先定义的误差 gate 时：

```text
layer inspector
+
precision sensitivity
```

再进行：

```text
FP16 fallback
```

最终：

```text
INT8
或
Mixed INT8/FP16
```

由数据决定。

不再复制 YOLO 项目那种超长 precision search，除非 RTMPose 确实出现明显 INT8 degradation。

---

# 23. C++ RTMPose Adapter

公共 runtime 之外，新工程负责 RTMPose 语义：

```text
bbox
center/scale
affine
normalization
batch packing
SimCC decode
inverse affine
keypoint score
```

建议模块：

```text
include/
    rtmpose_preprocess.hpp
    rtmpose_decode.hpp
    rtmpose_types.hpp

src/
    rtmpose_preprocess.cpp
    rtmpose_decode.cpp

apps/
    rtmpose_infer.cpp
    rtmpose_benchmark.cpp
```

公共 `trt-runtime` 不知道什么是：

```text
bbox
hand
SimCC
keypoint
```

---

# 24. 离线程序

至少支持三种方式：

```text
单图 + 手工 bbox

多图 + bbox manifest

prepared tensor benchmark
```

例如：

```text
rtmpose_infer
    engine
    image
    bbox-manifest
```

输出：

```text
image_id
hand_id
bbox
keypoint coordinates
keypoint scores
engine
batch
timings
```

可选：

```text
draw keypoints
draw skeleton
save visualization
```

---

# 25. 数值一致性验证

固定：

```text
same bbox
same input tensor
```

比较：

```text
ORT FP32
TRT FP32
TRT FP16
TRT INT8
```

第一层比较：

```text
simcc_x
simcc_y

MAE
RMSE
max absolute error
P95 error
non-finite count
```

第二层比较：

```text
decoded keypoint
```

报告：

```text
mean displacement
median displacement
P95 displacement
max displacement
normalized displacement
score difference
valid point count change
threshold flip rate
```

并分别报告：

```text
in-car validation
cross-scene validation
```

不能只对 INT8 仍为高置信度的点计算误差，否则会隐藏严重 confidence degradation。

这部分依然定义为：

> Deployment Consistency Validation

而不是：

> Pose Accuracy Evaluation。

原方案对此边界定义是正确的。

---

# 26. 小规模真实 GT——推荐增强项

建议从 validation 中选：

```text
约 50~100 hand instances
```

人工标注：

```text
hand bbox
K keypoints
visibility
```

冻结：

```text
annotation protocol
invisible-point rule
normalization rule
```

然后报告：

```text
PCK
NME
```

这样最终有两层结论：

```text
A. 部署一致性

ORT FP32
vs
TRT FP32/FP16/INT8


B. 模型真实精度

prediction
vs
human GT
```

如果时间有限，这部分可以作为增强项，不阻塞第一阶段工程完成。

---

# 27. Dynamic Batch 验证

除了 precision，还必须验证 batch。

相同 N 个 hand tensor：

```text
逐个 batch=1 推理
```

与：

```text
一次 batch=N 推理
```

比较：

```text
simcc_x
simcc_y
decoded keypoints
```

理想情况下：

```text
batch1 sequential
≈
dynamic batched
```

至少测试：

```text
N=1
N=2
N=3
N=4
```

并测试：

```text
N > MAX
```

确认 chunking 正确。

---

# 28. Runtime 回归测试

升级公共 runtime 后必须验证：

```text
原 YOLO static single-output engine
原 ResNet static engine
RTMPose dynamic multi-output engine
```

至少测试：

```text
static shape
dynamic shape
wrong shape
out-of-profile shape
wrong dtype
missing tensor
multi-output
resource lifecycle
repeated inference
profile switching
```

多输出 engine 如果调用旧 single-output API：

```text
必须明确报错
```

禁止：

```text
静默取第一个 output
```

这继续沿用原方案中的兼容原则。

---

# 29. 性能 Benchmark

设备统一：

```text
Jetson AGX Orin
MAXN
jetson_clocks
```

每个 precision：

```text
warmup >= 100
measure >= 1000
```

报告：

```text
H2D
TensorRT inference
D2H
CPU preprocessing
SimCC decode
per-hand E2E
per-image E2E
P50
P95
throughput
engine size
```

分别 benchmark：

```text
batch=1
batch=2
batch=4
```

同时计算：

```text
latency / batch

hands per second

latency per hand
```

因为 dynamic batching 的价值不能只看：

```text
一次 inference latency
```

还要看：

```text
per-hand throughput
```

---

# 30. 业务级性能指标

最终至少给：

```text
1 hand / image
2 hands / image
```

两种典型场景。

例如：

```text
RTMDet 时间
+
ROI prepare
+
RTMPose batch1/batch2
+
decode
```

不过 RTMDet ONNX Runtime 时间必须单列，不混入 RTMPose engine benchmark。

原方案已经规定手数相关延迟需要单独解释。

---

# 31. 工程目录建议

下列目录结合现有资产布局与后续目标；除第 0 节确认的产物外，engine、reference config、C++ 程序、测试和 benchmark 等均为待实现内容，不因列入此树而视为已存在。

```text
tensorrt-rtmpose-hand/
├── CMakeLists.txt
│
├── models/
│   ├── rtmpose_hand.onnx
│   ├── rtmdet_hand.onnx
│   └── manifest.json
│
├── engines/
│   ├── rtmpose_hand_fp32.engine
│   ├── rtmpose_hand_fp16.engine
│   └── rtmpose_hand_int8.engine
│
├── configs/
│   └── rtmpose_hand_reference.json
│
├── data/
│   ├── raw/
│   └── manifest.jsonl
│
├── calibration/
│   ├── calibration_set/
│   │   ├── crops/                   # 352 张校准裁剪，已有
│   │   └── tensors/                 # 352 个单手张量，已有
│   ├── development_set/
│   │   ├── crops/                   # 55 张开发裁剪，已有
│   │   └── tensors/                 # 55 个单手张量，已有
│   ├── frames/
│   ├── detections/
│   ├── contact_sheets/
│   ├── provenance/
│   ├── hand_boxes.jsonl
│   ├── frames.jsonl
│   ├── split.json
│   ├── approval.json
│   ├── summary.json
│   ├── verification.json
│   ├── calibration_tensor_order.txt
│   ├── development_tensor_order.txt
│   ├── index.html
│   ├── overview.html
│   ├── README.md
│   ├── coverage.json                # 已生成，校准/开发分别统计
│   ├── rtmpose_hand_int8.cache       # 待执行 INT8 校准后生成
│   └── rtmpose_hand_int8.cache.meta.json
│
├── validation/
│   ├── reference/
│   ├── development/
│   ├── in_car/
│   ├── cross_scene/
│   └── gt/
│
├── include/
│   ├── rtmpose_types.hpp
│   ├── rtmpose_preprocess.hpp
│   └── rtmpose_decode.hpp
│
├── src/
│   ├── rtmpose_preprocess.cpp
│   └── rtmpose_decode.cpp
│
├── apps/
│   ├── rtmpose_infer.cpp
│   └── rtmpose_benchmark.cpp
│
├── scripts/
│   ├── prepare_assets.py
│   ├── generate_hand_boxes.py
│   ├── prepare_reference.py
│   └── validate_engines.py
│
├── tests/
│
├── benchmarks/
│   └── offline-quantization/
│       └── README.md
│
└── README.md
```

公共：

```text
edge-ai/trt-runtime/
```

独立维护，不复制进该项目。

当前校准/开发手框统一使用 `calibration/hand_boxes.jsonl`，不再为套用早期目录提案复制到 `data/annotations/`；已准备的输入保留在两个集合的 `tensors/` 中，不重复存入 `data/prepared/`。两份顺序表中的路径相对 `calibration/`。将来 `validation/development/` 保存开发集的推理结果和比较报告，不再存放一套开发输入副本。表中的 Python 脚本为后续可复现工具目标；现有 `provenance/` 脚本是历史一次性脚本，不代表这些工具已经实现。

---

# 32. 实施阶段

## Stage 1 — 环境与模型核验

当前：模型导入、哈希和 batch=1 ONNX/ORT 冒烟检查已完成；本机头文件已确认包含下列四个 API。尚须完成实际链接库/运行验证、动态 batch 检查，以及本地 rtmlib 与历史配置的核实，不重复导入模型。

确认：

```text
TensorRT
CUDA
GPU
driver
Python
ONNX
ORT
OpenCV
rtmlib commit
```

检查：

```text
enqueueV3
setTensorAddress
setInputShape
getNbIOTensors
```

在本机 TensorRT 头文件中均存在。

实跑 RTMPose ONNX，确认：

```text
input name
input shape
output names
output shapes
K
dtype
finite values
```

---

## Stage 2 — 数据冻结

当前：校准/开发原图已检测，划分、手框和裁剪已经用户认可；352/55 个张量已隔离保存。近似重复候选有记录，按用户意见保留相似裁剪和重叠框。覆盖度报告及未知属性说明已生成，不能把零检测图片直接标成确定漏检，也不重新要求用户审核已认可的裁剪。validation 目前仅冻结原图归属，其手框与输入仍待准备。

```text
raw images
→ near-duplicate grouping
→ calibration/dev/validation split
→ RTMDet
→ bbox review
→ freeze hand_boxes.jsonl
→ calibration coverage
```

---

## Stage 3 — Reference Freeze

当前：技术契约已冻结，55 个 development 样本已完成真实 ORT 输出、解码及坐标还原；10 个代表样本已有叠图，已通过用户人工评审。最终 validation 的 258 张原图未参与本次模型推理，输入/手框准备与最终评测仍待实施。

完成：

```text
Python preprocess
ORT FP32
SimCC decode
inverse affine
visualization
```

冻结：

```text
rtmpose_hand_reference.json
```

---

## Stage 4 — Runtime vNext

完成：

```text
name-based tensor
multi-output
static/dynamic
optimization profile
persistent MAX buffers
enqueueV3
```

回归：

```text
YOLO
ResNet
```

---

## Stage 5 — TRT FP32 / FP16

建立：

```text
ORT FP32
        ↓
TRT FP32
        ↓
TRT FP16
```

先完成：

```text
raw-output consistency
decoded keypoint consistency
dynamic batch consistency
```

此阶段通过后再进入 INT8。

---

## Stage 6 — INT8 PTQ

```text
real hand calibration tensors
        ↓
EntropyCalibrator2
        ↓
cache
        ↓
INT8 engine
        ↓
layer inspector
        ↓
development validation
```

如果误差不可接受：

```text
precision sensitivity
→ selective FP16 fallback
```

否则直接保留 INT8。

---

## Stage 7 — Final Validation

配置冻结后运行：

```text
in-car validation
cross-scene validation
```

输出：

```text
FP32
FP16
INT8/Mixed
```

完整一致性报告。

如完成 GT 子集，再补：

```text
PCK
NME
```

---

## Stage 8 — Benchmark

在：

```text
MAXN + jetson_clocks
```

下比较：

```text
FP32
FP16
INT8/Mixed
```

以及：

```text
batch1
batch2
batch4
```

最后决定推荐 engine。

---

# 33. 最终验收标准

以下均为项目最终验收目标，勾号表示要求清单，不表示当前已通过；当前完成状态以第 0 节为准。

项目完成至少满足：

```text
✓ 原始 RTMPose-Hand ONNX contract 已冻结

✓ Python reference 可复现

✓ C++ preprocess 与 Python 对齐

✓ SimCC decode 与 Python 对齐

✓ 公共 TRT runtime 支持 static + dynamic

✓ 公共 runtime 支持 named multiple outputs

✓ 统一使用 enqueueV3

✓ 原有 YOLO/ResNet runtime 回归通过

✓ dynamic batch 1~MAX 正常运行

✓ batch=N 与 sequential batch1 数值一致

✓ TRT FP32 与 ORT FP32 对齐

✓ FP16 精度通过

✓ INT8 PTQ 使用真实 hand 数据完成 calibration

✓ calibration cache 可追溯

✓ engine layer precision 有 inspector 证据

✓ validation 与 calibration 严格隔离

✓ FP16/INT8 精度及性能均有最终报告

✓ 推荐 engine 由实测决定，而不是预先指定 INT8
```

---

# 34. 项目的核心学习成果

完成之后，这个项目不应该被描述成：

> “把 RTMPose 转成了 TensorRT。”

真正的技术路线应当是：

```text
Top-down Hand Pose
        ↓
ROI / Affine Geometry
        ↓
SimCC Representation
        ↓
Multi-output TensorRT
        ↓
Dynamic Batch
        ↓
Optimization Profile
        ↓
Name-based Tensor API
        ↓
enqueueV3
        ↓
Persistent Dynamic Buffers
        ↓
FP32 / FP16 / INT8
        ↓
Deployment Consistency Validation
```

它和之前 YOLOv8m 项目形成非常好的技能递进：

```text
YOLOv8m
static shape
single output
detection
CUDA pipeline
INT8 mixed precision

        ↓

RTMPose-Hand
dynamic batch
multi-output
ROI/Affine
SimCC
keypoint regression
generic TensorRT runtime
enqueueV3

        ↓

LLM
dynamic sequence
multiple tensors
KV cache
更复杂 runtime
```

---

# 35. 当前最终技术决策

本项目正式采用：

```text
原始 RTMPose-Hand dynamic-batch ONNX

TensorRT:
    dynamic explicit batch
    H/W fixed

Optimization Profile:
    初始 1 / 2 / 4
    根据实际 hand-count 数据最终冻结

Runtime:
    static + dynamic
    named input/output
    multiple outputs
    persistent MAX-size buffers

Execution:
    enqueueV3

CUDA Graph:
    RTMPose 第一版关闭

INT8:
    EntropyCalibrator2
    calibration batch=1
    独立 fixed calibration profile

Validation:
    deployment consistency 为必做
    small GT PCK/NME 为推荐增强

RTMDet:
    ORT FP32 only
    offline bbox generation

Video:
    本阶段不做
```

这版作为 **RTMPose-Hand v2 实施基线**，后续除非第一阶段环境/模型核验发现实际 ONNX contract 与当前假设不一致，否则不再改变整体技术路线。
