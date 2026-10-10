# TensorRT ResNet18 Deployment on NVIDIA Jetson AGX Orin

本项目用于系统学习和实践 NVIDIA Jetson 平台上的深度学习模型部署流程，以 **ResNet18** 为实验模型，完整覆盖：

```text
PyTorch
    ↓
ONNX
    ↓
TensorRT Engine Build
    ↓
FP32 / FP16
    ↓
Dynamic Shape
    ↓
INT8 PTQ
    ↓
C++ Runtime
    ↓
Accuracy Validation
    ↓
Benchmark
    ↓
TensorRT Engine Inspector
    ↓
trtexec Layer Profiling
    ↓
Nsight Systems
    ↓
Nsight Compute
    ↓
SM / Warp / Occupancy / Waves / Batch Scaling Analysis
```

该项目的目标不只是“让模型运行起来”，而是理解：

- PyTorch 模型如何转换为 TensorRT Engine；
- TensorRT Builder 和 Runtime 各自负责什么；
- Dynamic Shape 与 Optimization Profile 如何工作；
- FP32、FP16、INT8 的性能差异从哪里产生；
- TensorRT 内部 Layer、Tactic、CUDA Kernel 之间是什么关系；
- GPU 中 Grid、Block、Warp、SM、Occupancy、Wave 等概念如何对应实际推理性能；
- 如何使用 Nsight Systems / Nsight Compute 定位性能瓶颈。

---

# 1. Environment

实验平台：

```text
Device        : NVIDIA Jetson AGX Orin
GPU SMs       : 14
Compute Cap.  : 8.7
Ubuntu        : 20.04
JetPack       : 5.1.2
L4T           : 35.4.1

CUDA          : 11.4
cuDNN         : 8.6
TensorRT      : 8.5.2
GCC/G++       : 9.4

PyTorch       : 2.1.0a0+41361538.nv23.06
```

Python 环境使用 `uv` 管理。

项目路径：

```text
/home/ssd/projects/edge-ai/tensorrt-resnet/
```

性能测试时建议固定 Jetson 功耗模式和 GPU/CPU/EMC 时钟：

```bash
sudo nvpmodel -q
sudo jetson_clocks
```

否则不同时间进行的 benchmark 绝对 latency 可能存在一定差异。

编译方法:

```text
rm -rf build

cmake -S . -B build

cmake --build build -j$(nproc)
```

---

# 2. Project Structure

项目主要结构：

```text
tensorrt-resnet/
├── apps/
│   ├── build_engine.cpp
│   ├── build_int8.cpp
│   ├── infer.cpp
│   ├── infer_dynamic.cpp
│   ├── benchmark_dynamic.cpp
│   └── inspect_engine.cpp
│
├── include/
├── scripts/
├── models/
├── engines/
├── calibration/
├── data/
├── benchmarks/
├── logs/
├── CMakeLists.txt
└── README.md
```

主要程序职责：

| 程序 | 作用 |
|---|---|
| `build_engine.cpp` | ONNX → FP32/FP16 TensorRT Engine |
| `build_int8.cpp` | INT8 PTQ Calibration + INT8 Engine Build |
| `infer.cpp` | 静态 Shape C++ TensorRT Runtime |
| `infer_dynamic.cpp` | Dynamic Shape Runtime + Accuracy Validation |
| `benchmark_dynamic.cpp` | Dynamic Batch Benchmark |
| `inspect_engine.cpp` | TensorRT Engine Inspector |

---

# 3. End-to-End Implementation Flow

整个工程可以分成两个阶段：

```text
Build Phase
ONNX → TensorRT Engine

Runtime Phase
TensorRT Engine → GPU Inference
```

这是理解 TensorRT 最重要的整体结构。

---

# 4. PyTorch → ONNX

首先使用 PyTorch 创建 ResNet18，并导出 ONNX。

逻辑流程：

```text
PyTorch Model
    ↓
torch.onnx.export()
    ↓
ONNX Model
```

ONNX 是一种中间表示（Intermediate Representation，IR）。

它主要保存：

```text
Graph
Operators
Weights
Tensor Shape
Tensor Type
Attributes
```

例如 ResNet 中：

```text
Conv
 ↓
BatchNorm
 ↓
ReLU
 ↓
MaxPool
 ↓
Residual Block
 ↓
GEMM
```

都会被转换为 ONNX Graph 中的算子节点。

使用 Netron 可以查看：

```text
Tensor Shape
Operator
Weights
Bias
Attributes
Graph Connection
```

例如：

```text
Conv
W<64×3×7×7>
B<64>
```

表示该卷积：

```text
Output Channels = 64
Input Channels  = 3
Kernel          = 7×7

Weights:
64 × 3 × 7 × 7

Bias:
64
```

---

# 5. PyTorch Golden Reference

为了验证 TensorRT 推理正确性，需要保存 PyTorch 输出作为 Golden Reference。

本实验中曾发现：

```text
PyTorch CPU Conv2d
→ conv1 开始产生 NaN
```

而：

```text
PyTorch CUDA
→ 输出正常
```

因此最终使用：

```text
PyTorch CUDA FP32
```

作为 Golden Reference。

验证内容包括：

```text
Max Absolute Error
Mean Absolute Error
RMSE
Top-1 Match
```

---

# 6. ONNX → TensorRT Engine

## 6.1 逻辑数据流

从模型转换角度理解：

```text
ONNX File
    ↓
NvOnnxParser
    ↓
INetworkDefinition
    ↓
TensorRT Builder
    +
IBuilderConfig
    ↓
buildSerializedNetwork()
    ↓
IHostMemory
    ↓
Serialized TensorRT Engine
    ↓
.engine / .plan
```

---

## 6.2 实际 C++ 对象创建顺序

代码中实际上通常先创建 Builder：

```text
IBuilder
    ↓
INetworkDefinition
    ↓
IBuilderConfig
    ↓
NvOnnxParser::IParser
    ↓
parser->parse()
    ↓
buildSerializedNetwork()
```

典型代码结构：

```cpp
std::unique_ptr<nvinfer1::IBuilder> builder{
    nvinfer1::createInferBuilder(logger)
};

std::unique_ptr<nvinfer1::INetworkDefinition> network{
    builder->createNetworkV2(flags)
};

std::unique_ptr<nvinfer1::IBuilderConfig> config{
    builder->createBuilderConfig()
};

std::unique_ptr<nvonnxparser::IParser> parser{
    nvonnxparser::createParser(
        *network,
        logger
    )
};
```

解析：

```cpp
parser->parse(...);
```

构建：

```cpp
std::unique_ptr<nvinfer1::IHostMemory> serializedEngine{
    builder->buildSerializedNetwork(
        *network,
        *config
    )
};
```

最后将 `serializedEngine` 写入磁盘：

```text
resnet18_cpp_fp32.plan
resnet18_cpp_fp16.plan
resnet18_int8.plan
```

---

# 7. TensorRT Builder 中几个核心对象

## IBuilder

负责：

```text
选择 Tactic
搜索 Kernel Implementation
构建 TensorRT Engine
```

---

## INetworkDefinition

保存 TensorRT 对模型计算图的内部表示。

它可以理解成：

```text
TensorRT Network IR
```

ONNX Parser 的工作就是：

```text
ONNX Graph
    ↓
INetworkDefinition
```

---

## IBuilderConfig

保存 Builder 配置，例如：

```text
FP16
INT8
Workspace
Optimization Profile
Tactic Sources
```

例如：

```cpp
config->setFlag(
    nvinfer1::BuilderFlag::kFP16
);
```

或者：

```cpp
config->setFlag(
    nvinfer1::BuilderFlag::kINT8
);
```

---

## NvOnnxParser

负责：

```text
ONNX
 ↓
TensorRT Network
```

Parser 解析出的权重在 Builder 完成 Engine 构建之前必须保持有效。

---

# 8. Serialized Engine

`buildSerializedNetwork()` 返回：

```cpp
nvinfer1::IHostMemory
```

里面存放序列化后的 TensorRT Engine。

最终写为：

```text
*.engine
```

或者：

```text
*.plan
```

这两个扩展名在本质上没有区别。

都表示：

> Serialized TensorRT Engine

`.plan` / `.engine` 更多是命名习惯。

---

# 9. TensorRT Runtime Flow

模型构建完成之后进入 Runtime。

完整 Runtime 流程：

```text
TensorRT .plan
    ↓
读取 Engine Binary
    ↓
createInferRuntime()
    ↓
IRuntime
    ↓
deserializeCudaEngine()
    ↓
ICudaEngine
    ↓
createExecutionContext()
    ↓
IExecutionContext
    ↓
Query Tensor Name / Shape
    ↓
cudaMalloc
    ↓
cudaStreamCreate
    ↓
cudaMemcpyAsync H2D
    ↓
setTensorAddress()
    ↓
enqueueV3()
    ↓
cudaMemcpyAsync D2H
    ↓
cudaStreamSynchronize()
    ↓
Output
```

简化为：

```text
Disk
 ↓
Serialized Engine

CPU
 ↓
IRuntime
 ↓
ICudaEngine
 ↓
IExecutionContext

GPU
 ↓
H2D
 ↓
TensorRT Inference
 ↓
D2H
```

---

# 10. IRuntime / ICudaEngine / IExecutionContext

这三个对象非常容易混淆。

## IRuntime

主要负责：

```text
Serialized Engine
        ↓
Deserialize
        ↓
ICudaEngine
```

---

## ICudaEngine

代表：

> 已经完成 TensorRT 优化和 Tactic 选择的模型。

它包含：

```text
Network
Tensor Information
Optimization Profiles
TensorRT Layers
Selected Tactics
Kernel Configuration
```

---

## IExecutionContext

代表：

> 某一次实际运行 Engine 的执行上下文。

负责：

```text
Runtime Shape
Optimization Profile
Tensor Address
Inference Execution
```

因此关系可以理解为：

```text
IRuntime
   ↓
ICudaEngine
   ↓
IExecutionContext
   ↓
Inference
```

---

# 11. TensorRT Object Lifetime

TensorRT 对象已经从手动：

```cpp
context->destroy();
engine->destroy();
runtime->destroy();
```

迁移为：

```cpp
std::unique_ptr<>
```

例如：

```cpp
std::unique_ptr<nvinfer1::IRuntime> runtime{
    nvinfer1::createInferRuntime(logger)
};

std::unique_ptr<nvinfer1::ICudaEngine> engine{
    runtime->deserializeCudaEngine(...)
};

std::unique_ptr<nvinfer1::IExecutionContext> context{
    engine->createExecutionContext()
};
```

因为 C++ 局部变量按声明顺序的反方向析构：

```text
创建：

runtime
 ↓
engine
 ↓
context


销毁：

context
 ↓
engine
 ↓
runtime
```

正好满足 TensorRT 生命周期依赖关系。

---

# 12. Binding API → Tensor Name API

最开始 Runtime 使用 TensorRT 旧 Binding API：

```text
getNbBindings()
getBindingDimensions()
setBindingDimensions()
bindings[index]
enqueueV2()
```

后来升级为 TensorRT Tensor Name API：

```text
getNbIOTensors()
getIOTensorName()
getTensorIOMode()
getTensorShape()
setInputShape()
setTensorAddress()
enqueueV3()
```

旧模型：

```text
Tensor
 ↓
Binding Index
 ↓
bindings[index]
 ↓
Device Pointer
 ↓
enqueueV2()
```

新模型：

```text
Tensor Name
 ↓
Device Pointer
 ↓
setTensorAddress()
 ↓
enqueueV3()
```

例如：

```cpp
context->setTensorAddress(
    inputName,
    deviceInput
);

context->setTensorAddress(
    outputName,
    deviceOutput
);

context->enqueueV3(stream);
```

这种方式不再需要：

```cpp
std::vector<void*> bindings;
```

---

# 13. CUDA Runtime Flow

实际 GPU 推理核心：

```cpp
cudaMemcpyAsync(
    deviceInput,
    hostInput,
    inputBytes,
    cudaMemcpyHostToDevice,
    stream
);

context->enqueueV3(stream);

cudaMemcpyAsync(
    hostOutput,
    deviceOutput,
    outputBytes,
    cudaMemcpyDeviceToHost,
    stream
);

cudaStreamSynchronize(stream);
```

对应：

```text
CPU Memory
     │
     │ H2D
     ▼
GPU Memory
     │
     │ TensorRT Inference
     ▼
GPU Output
     │
     │ D2H
     ▼
CPU Memory
```

---

# 14. CUDA Stream

CUDA 默认是异步执行模型。

调用：

```cpp
cudaMemcpyAsync()
```

或者：

```cpp
enqueueV3()
```

并不意味着 GPU 已经执行完成。

CPU 只是把任务提交到 CUDA Stream：

```text
CUDA Stream

H2D
 ↓
Kernel 1
 ↓
Kernel 2
 ↓
...
 ↓
D2H
```

因此最后需要：

```cpp
cudaStreamSynchronize(stream);
```

确保：

```text
GPU work completed
```

之后 CPU 才能安全读取输出结果。

---

# 15. Dynamic Shape

静态 Engine：

```text
Input:
[1,3,224,224]
```

Dynamic Engine：

```text
Input:
[-1,3,224,224]
```

其中：

```text
-1
```

表示动态维。

本项目 Optimization Profile：

```text
MIN = [1,3,224,224]
OPT = [4,3,224,224]
MAX = [8,3,224,224]
```

含义：

```text
MIN
允许的最小 Shape

MAX
允许的最大 Shape

OPT
TensorRT Builder 重点进行 Tactic 优化的 Shape
```

`OPT` 是：

```text
Optimal / Optimization Shape
```

不是 `Option`。

---

# 16. Dynamic Runtime Flow

对于 Dynamic Engine：

```text
Engine Shape
[-1,3,224,224]
```

Runtime 首先：

```cpp
context->setInputShape(
    inputName,
    actualDims
);
```

例如 Batch=4：

```text
[4,3,224,224]
```

然后：

```cpp
context->getTensorShape(
    outputName
);
```

得到：

```text
[4,1000]
```

因此需要区分：

```text
engine->getTensorShape()
```

和：

```text
context->getTensorShape()
```

前者返回模型定义：

```text
[-1,3,224,224]
```

后者返回当前 Runtime 实际 Shape：

```text
[4,3,224,224]
```

---

# 17. Dynamic Shape Validation

实际验证：

```text
Batch 1
Input  : [1,3,224,224]
Output : [1,1000]
Top-1  : 1/1


Batch 4
Input  : [4,3,224,224]
Output : [4,1000]
Top-1  : 4/4


Batch 8
Input  : [8,3,224,224]
Output : [8,1000]
Top-1  : 8/8
```

FP16 最大误差约：

```text
B1:
5.62e-4

B4:
4.43e-4

B8:
7.40e-4
```

---

# 18. FP32 / FP16

FP32 Engine：

```text
FP32 weights
FP32 tensors
```

但 TensorRT 默认允许 TF32，因此 Ampere / Orin 上的 FP32 Engine 实际可以选择：

```text
TF32 Tensor Core
```

Nsight Systems 中实际观察到：

```text
f32f32_tf32f32
```

因此更准确地称为：

```text
FP32 Engine
(TF32 allowed)
```

FP16 Engine 中观察到：

```text
f16f16
h16816
```

说明实际使用了 FP16 Tensor Core Kernel。

---

# 19. INT8 PTQ

INT8 使用 PTQ：

```text
Post Training Quantization
```

基本流程：

```text
FP32 ONNX
    ↓
Calibration Dataset
    ↓
IInt8EntropyCalibrator2
    ↓
Activation Distribution
    ↓
Quantization Scale
    ↓
Calibration Cache
    ↓
INT8 TensorRT Engine
```

Calibrator 主要实现：

```text
getBatchSize()
getBatch()
readCalibrationCache()
writeCalibrationCache()
```

这些函数不是程序主动循环调用，而是：

```text
TensorRT Builder
```

在构建 INT8 Engine 时自动回调。

第二次构建可以直接读取：

```text
Calibration Cache
```

避免重新 Calibration。

---

# 20. INT8 Calibration Dataset

当前项目为了学习 Calibration 机制，使用：

```text
100 × [3,224,224]
```

随机输入作为 Calibration Dataset。

这只能用于：

> 学习 INT8 PTQ 流程和量化机制。

真实项目必须使用：

```text
具有代表性的真实数据集
```

否则量化 Scale 无法代表真实输入分布。

---

# 21. INT8 Accuracy

当前 INT8 Engine 验证结果：

```text
Max Absolute Error : 0.001718
MAE                : 0.000419
RMSE               : 0.000524
Top-1              : Match
```

INT8 相比 FP16 数值误差增大属于正常现象。

---

# 22. CHW32 INT8

TensorRT Engine Inspector 中观察到：

```text
Thirty-two wide channel vectorized row major Int8 format
```

以及 Nsight Kernel：

```text
nchw_vect_c_32
```

说明 TensorRT 内部使用：

```text
CHW32
```

格式。

逻辑 NCHW：

```text
[N,C,H,W]
```

CHW32 可以近似理解为：

```text
[N,C/32,H,W,32]
```

也就是每 32 个 Channel 进行一组向量化。

例如：

```text
C = 64

Group 0:
C0 ~ C31

Group 1:
C32 ~ C63
```

这样更适合 INT8 Tensor Core 的向量化访存和计算。

如果 Channel 不是 32 的倍数，会发生内部 Padding。

---

# 23. Reformat

由于外部输入仍是：

```text
FP32 CHW
```

而 TensorRT 内部 INT8 Kernel 可能需要：

```text
INT8 CHW32
```

因此需要：

```text
FP32 NCHW
    ↓
Quantize
    ↓
Layout Conversion
    ↓
INT8 CHW32
```

Nsight Systems 中可以看到：

```text
genericReformat
```

以及：

```text
float → signed char
```

相关 Kernel。

因此：

> INT8 计算虽然更快，但格式转换和量化也会产生额外开销。

---

# 24. Engine Inspector

TensorRT Engine Inspector 用于查看优化后的 TensorRT Engine。

需要注意：

```text
ONNX Node
≠
TensorRT Engine Layer
≠
CUDA Kernel
```

关系：

```text
PyTorch Module
    ↓
ONNX Node
    ↓
TensorRT Optimized Layer
    ↓
Tactic
    ↓
CUDA Kernel
```

例如 ONNX：

```text
Conv
 ↓
Add
 ↓
ReLU
```

TensorRT 可能 Fuse 为：

```text
Conv + Add + ReLU
```

一个 TensorRT Layer。

一个 TensorRT Layer 又可能产生：

```text
1 个或多个 CUDA Kernel
```

---

# 25. TensorRT Layer Fusion

例如 Residual Block：

```text
Conv
 ↓
BatchNorm
 ↓
Add
 ↓
ReLU
```

TensorRT Engine Inspector 中可能看到：

```text
Conv + Add + ReLU
```

并显示：

```text
LayerType: CaskConvolution
HasResidual: 1
Activation: RELU
```

这说明 TensorRT 已经进行 Layer Fusion。

Fusion 可以减少：

```text
Kernel Launch
Intermediate Memory
Memory Traffic
```

---

# 26. Tactic

一个 TensorRT Layer 并不只有一种实现方式。

例如 Conv 可以选择：

```text
Implicit GEMM
Winograd
cuDNN
Tensor Core Kernel
Custom TRT Kernel
```

TensorRT Builder 会 benchmark 多个候选：

```text
Tactic A
Tactic B
Tactic C
...
```

最后选择最快的 Tactic。

因此：

```text
ONNX Operator
```

描述的是：

> 算什么。

而：

```text
TensorRT Tactic
```

描述的是：

> 怎么算。

---

# 27. trtexec Layer Profiling

`trtexec --dumpProfile` 显示的是：

> TensorRT Engine Layer

不是 ONNX Node。

典型：

```bash
trtexec \
    --loadEngine=xxx.plan \
    --warmUp=1000 \
    --duration=10 \
    --noDataTransfers \
    --useSpinWait \
    --dumpProfile \
    --separateProfileRun
```

主要用于：

```text
Layer latency distribution
```

不要直接把 Layer Profile 总时间和正常 benchmark latency 做严格比较，因为：

```text
Layer Profiler
```

本身会引入额外 profiling overhead。

---

# 28. Benchmark vs Profiler

本项目中区分：

```text
C++ Benchmark
→ 测真实 latency / throughput

trtexec --dumpProfile
→ 看 TensorRT Layer 时间分布

Nsight Systems
→ 看 CUDA Kernel / Timeline

Nsight Compute
→ 看单个 Kernel 微架构指标
```

因此：

> 最终性能数字以 Benchmark 为准，而不是 Profiler 下的 latency。

---

# 29. Nsight Systems

Nsight Systems 用于分析：

```text
CPU Timeline
CUDA API
Kernel Launch
Memcpy
Synchronization
GPU Kernel
```

主要回答：

> 时间花在哪？

例如 FP32 Kernel 中观察到：

```text
f32f32_tf32f32
```

FP16：

```text
f16f16
```

INT8：

```text
i8i8_i8i32
```

因此确认：

```text
FP32 Engine → TF32 Tensor Core
FP16 Engine → FP16 Tensor Core
INT8 Engine → INT8 Tensor Core
```

---

# 30. INT8 Extra Kernel Overhead

Nsight Systems 中还发现 INT8 有大量：

```text
padBiasWeights
genericReformat
float → INT8
```

并且 INT8 Kernel Launch 数比 FP16 更多。

因此：

```text
INT8 math faster
```

并不等于：

```text
End-to-End 一定同比例变快
```

额外：

```text
Quantization
Reformat
Layout Conversion
Helper Kernel
Kernel Launch
```

都会消耗时间。

---

# 31. Nsight Compute

Nsight Compute 用于分析：

> 单个 CUDA Kernel 为什么快或慢。

关注：

```text
Compute Throughput
Memory Throughput
Occupancy
Active Warps
Waves Per SM
Scheduler Statistics
Register Usage
Shared Memory
```

---

# 32. CUDA Execution Hierarchy

CUDA Kernel 的结构：

```text
Kernel
  ↓
Grid
  ↓
Block
  ↓
Thread
```

GPU 实际执行：

```text
Thread
 ↓
32 Threads
 ↓
Warp
 ↓
SM
```

因此：

```text
Grid
= Kernel 中所有 Blocks

Block
= 一组 Threads

Warp
= 32 Threads

SM
= 实际执行 Warp 的 GPU 硬件单元
```

---

# 33. Jetson AGX Orin SM

当前设备 Nsight 报告：

```text
SMs = 14
Compute Capability = 8.7
```

Compute Capability 8.7：

```text
Warp Size = 32 Threads

Maximum Resident Warps / SM
= 48
```

因此：

```text
48 × 32
= 1536 resident threads / SM
```

---

# 34. Occupancy

Occupancy：

```text
Active Warps
---------------------
Maximum Warps per SM
```

例如某个 FP16 Kernel：

```text
Theoretical Active Warps = 8
Maximum Warps             = 48
```

所以：

```text
8 / 48
= 16.67%
```

即：

```text
Theoretical Occupancy
= 16.67%
```

如果实际只有：

```text
4 Active Warps
```

那么：

```text
4 / 48
= 8.33%
```

即：

```text
Achieved Occupancy
≈ 8.33%
```

---

# 35. 为什么每个 SM 最多驻留 2 个 Block？

FP16 Kernel：

```text
Block Limit SM         = 16
Block Limit Registers  = 2
Block Limit Shared Mem = 2
Block Limit Warps      = 12
```

最终上限：

```text
min(
    16,
    2,
    2,
    12
)
=
2 blocks / SM
```

主要受到：

```text
Registers
+
Shared Memory
```

限制。

FP16：

```text
Registers / Thread
= 186

Shared Memory / Block
≈ 62.46 KB
```

因此每个 SM 无法同时驻留大量 Blocks。

---

# 36. Grid / Block / Warp / SM / Wave

FP16 Batch=1：

```text
Grid Size
= 13 Blocks

Block Size
= 128 Threads
```

每个 Block：

```text
128 / 32
= 4 Warps
```

一个 SM 最多：

```text
2 Blocks
```

所以：

```text
2 × 4
= 8 Warps / SM
```

整个 GPU：

```text
14 SM
×
2 Blocks / SM
=
28 Blocks
```

可以把这 28 Blocks 看作一个完整 Wave。

实际 Grid：

```text
13 Blocks
```

所以：

```text
Waves Per SM
≈
13 / 28
=
0.46
```

Nsight Compute 实际：

```text
Waves Per SM = 0.46
```

---

# 37. Batch Scaling

同一个 FP16 Kernel：

| Metric | B1 | B4 | B8 |
|---|---:|---:|---:|
| Grid Size | 13 | 49 | 98 |
| Waves/SM | 0.46 | 1.75 | 3.50 |
| Achieved Occupancy | 8.32% | 15.02% | 15.53% |
| Active Warps/SM | 4.00 | 7.21 | 7.46 |
| Compute | 38.76% | 52.22% | 61.09% |
| Memory | 46.43% | 61.60% | 71.50% |
| SM Busy | 47.55% | 60.65% | 63.71% |
| Kernel Duration | 34.11 us | 84.67 us | 138.69 us |

说明：

```text
Batch ↑
   ↓
Grid Size ↑
   ↓
Blocks ↑
   ↓
Waves ↑
   ↓
Active Warps ↑
   ↓
GPU Utilization ↑
```

---

# 38. 为什么 Batch 大了 Latency 反而更高？

因为一个 Kernel 一次要处理更多 Samples：

```text
B1
34.11 us

B4
84.67 us

B8
138.69 us
```

总 latency 增大属于正常现象。

但平均每 sample：

```text
B1:
34.11 / 1
= 34.11 us/sample

B4:
84.67 / 4
≈ 21.17 us/sample

B8:
138.69 / 8
≈ 17.34 us/sample
```

所以：

```text
Latency ↑
Throughput ↑
```

这就是部署中的：

> Latency / Throughput Trade-off

实时业务往往更关心：

```text
Batch=1 latency
```

离线批处理则更关心：

```text
samples/s
```

---

# 39. Compute-bound

如果：

```text
Compute ≈ 90%
Memory  ≈ 30%
```

说明：

```text
计算单元已经接近满载
```

这时继续提高 Memory Bandwidth 帮助有限。

因为：

```text
数据供应已经足够快
```

真正限制性能的是：

```text
CUDA Core / Tensor Core
```

的计算能力。

---

# 40. Memory-bound

如果：

```text
Compute ≈ 30%
Memory  ≈ 90%
```

说明：

```text
Memory Subsystem
```

已经接近满载。

计算单元大量时间在：

```text
等待数据
```

因此 GPU 算力没有被完全利用。

基本过程：

```text
Load Data
 ↓
Cache Miss
 ↓
DRAM
 ↓
Warp Stall
 ↓
No Eligible Warp
 ↓
SM Idle
```

---

# 41. Parallelism / Latency Bound

当前 Batch=1 INT8 Representative Kernel：

```text
Compute ≈ 27%
Memory  ≈ 30%
Waves/SM = 0.46
Occupancy ≈ 8.3%
```

Compute 和 Memory 都不高。

因此不是典型：

```text
Compute-bound
```

也不是：

```text
Memory-bound
```

而主要属于：

```text
Insufficient Parallelism
/
Latency Bound
```

即：

> GPU 很强，但当前 workload 没有足够多的 Blocks / Warps 去持续填满 GPU。

---

# 42. Why FP16 → INT8 Speedup Is Limited

实际整体趋势：

```text
FP32/TF32
   ↓
FP16
明显提升

FP16
   ↓
INT8
继续提升，但幅度明显减小
```

原因包括：

```text
Batch=1
+
Small Grid
+
Low Waves/SM
+
Low Occupancy
+
Reformat
+
Quantization
+
CHW32 Layout Conversion
+
padBiasWeights
+
More Kernel Launches
```

因此 INT8 的理论算力提升不能全部转换为 End-to-End Latency 提升。

---

# 43. Latest Regression Benchmark

一次 `enqueueV3 + tensor-name API` 改造后的回归测试：

| Precision | Compute Latency | Samples/s |
|---|---:|---:|
| FP32 / TF32 | 1.413 ms | 707.6 |
| FP16 | 0.820 ms | 1219.0 |
| INT8 | 0.703 ms | 1422.9 |

性能趋势：

```text
FP32 → FP16
Latency ≈ -42%

FP16 → INT8
Latency ≈ -14%

FP32 → INT8
Latency ≈ -50%
```

注意：

> GPU Frequency、Power Mode、Temperature 等都会影响绝对 latency。

因此正式性能报告应该在统一：

```text
nvpmodel
jetson_clocks
```

条件下重新测试。

---

# 44. Accuracy Results

静态 FP32：

```text
Max Abs Error ≈ 4.43e-5
Top-1 Match   = YES
```

FP16：

```text
Max Abs Error ≈ 1.5e-4
Top-1 Match   = YES
```

INT8：

```text
Max Abs Error ≈ 1.72e-3
Top-1 Match   = YES
```

数值误差趋势符合：

```text
FP32
 <
FP16
 <
INT8
```

---

# 45. Important Limitation

当前 ResNet18 使用：

```text
Random Weights
+
Random Inputs
```

因此：

```text
Top-1 = 601
```

并没有真实 ImageNet 分类语义。

这里 Top-1 的意义只是：

> PyTorch 和 TensorRT 是否选择了相同的最大输出位置。

因此本项目重点是：

```text
Deployment Pipeline
Numerical Consistency
Performance Analysis
```

而不是模型真实 Accuracy。

后续 YOLO 项目将使用：

```text
真实模型
真实图片
真实后处理
真实检测结果
```

进行部署。

---

# 46. Frequently Asked Questions / Learning Notes

以下记录项目学习过程中遇到的核心问题。

---

## Q1. TorchScript 和 ONNX 有什么区别？

TorchScript：

```text
PyTorch Ecosystem IR
```

更贴近 PyTorch Runtime。

ONNX：

```text
Framework-independent IR
```

更适合：

```text
PyTorch
 ↓
ONNX
 ↓
TensorRT / OpenVINO / Other Runtime
```

本项目使用 ONNX 作为 TensorRT 输入。

---

## Q2. 什么是 IR？

IR：

```text
Intermediate Representation
```

即：

> 中间表示。

用于连接：

```text
Frontend Framework
```

和：

```text
Backend Compiler / Runtime
```

例如：

```text
PyTorch
 ↓
ONNX IR
 ↓
TensorRT
```

---

## Q3. `do_constant_folding` 是什么？

Constant Folding：

> 在模型转换阶段提前计算不会变化的常量表达式。

例如：

```text
Constant A
+
Constant B
```

可以在 Export 阶段提前计算，减少 Runtime 工作。

---

## Q4. 什么叫模型静态化？

本质是尽可能把：

```text
Runtime Dynamic Behavior
```

提前确定下来。

例如：

```text
Fixed Operator
Fixed Graph
Fixed Tensor Type
Known Shape
```

这样编译器可以做更多优化。

---

## Q5. NCHW / NHWC 怎么理解？

NCHW：

```text
N = Batch
C = Channel
H = Height
W = Width
```

例如：

```text
[1,3,224,224]
```

NHWC：

```text
[1,224,224,3]
```

逻辑 Shape 和物理 Tensor Layout 不一定完全等价。

TensorRT 内部可能自动转换为：

```text
NHWC
CHW32
Vectorized Format
```

以适配 Tensor Core。

---

## Q6. `.engine` 和 `.plan` 有区别吗？

本质没有。

都是：

```text
Serialized TensorRT Engine
```

只是文件扩展名命名习惯不同。

---

## Q7. MIN / OPT / MAX 为什么需要 OPT？

MIN / MAX：

```text
定义允许的 Shape 范围
```

OPT：

```text
Builder 重点进行 Tactic 优化的 Shape
```

例如：

```text
MIN = B1
OPT = B4
MAX = B8
```

TensorRT 会更加关注 B4 的性能。

---

## Q8. Profile Index 为什么是 0？

因为当前 Engine 只有：

```text
1 Optimization Profile
```

数组 index 从 0 开始：

```text
Profile 0
```

如果以后建立多个 Profile：

```text
Profile 0
Profile 1
Profile 2
```

才需要切换其他 profileIndex。

---

## Q9. `trtexec --warmUp=1000` 是什么？

表示正式 Benchmark 前：

```text
Warm Up 1000 ms
```

让：

```text
GPU
CUDA
Cache
TensorRT
Clock
```

进入相对稳定状态。

---

## Q10. `--duration=10` 是什么？

表示正式 Benchmark：

```text
持续约 10 秒
```

而不是执行 10 次。

---

## Q11. `2>&1 | tee` 是什么？

```bash
command 2>&1 | tee log.txt
```

表示：

```text
stderr
  ↓
stdout
  ↓
tee
```

最终：

```text
终端显示
+
写入日志
```

---

## Q12. `jetson_clocks` 是什么？

用于尽可能固定：

```text
CPU
GPU
EMC
```

频率，以减少 DVFS 对 Benchmark 的影响。

主要用于：

```text
Stable Performance Measurement
```

---

## Q13. TensorRT Layer 和 ONNX Node 是一回事吗？

不是。

```text
ONNX Node
    ↓
TensorRT Optimization
    ↓
Fusion / Reformat / Tactic
    ↓
TensorRT Layer
```

一个 TensorRT Layer 可能包含多个 ONNX Node。

---

## Q14. TensorRT Layer 和 CUDA Kernel 是一回事吗？

也不是。

关系：

```text
TensorRT Layer
    ↓
Tactic
    ↓
CUDA Kernel(s)
```

一个 TensorRT Layer 可以产生一个或多个 CUDA Kernels。

---

## Q15. 为什么 INT8 输入输出还是 FP32？

外部 I/O：

```text
FP32
```

并不意味着 Engine 内部也是 FP32。

内部可能：

```text
FP32 Input
 ↓
Quantize
 ↓
INT8 CHW32
 ↓
INT8 Tensor Core
 ↓
Dequantize
 ↓
FP32 Output
```

---

## Q16. 为什么 INT8 不一定比 FP16 快两倍？

因为实际时间不仅包含：

```text
Matrix / Conv Arithmetic
```

还包含：

```text
Kernel Launch
Reformat
Quantization
Memory
Synchronization
Helper Kernel
```

低精度只加速其中部分工作。

---

## Q17. 为什么低精度不一定带来更高 Occupancy？

Occupancy 取决于：

```text
Registers
Shared Memory
Block Size
Warp Limit
```

INT8 Kernel 可能为了更高单 Block 效率使用更多：

```text
Registers
Shared Memory
```

因此：

```text
INT8
```

不代表：

```text
Occupancy 必然更高
```

---

## Q18. 为什么 GPU 利用率高不代表一定更快？

例如：

```text
TF32
Compute = 43%

INT8
Compute = 27%
```

但 INT8 仍然更快。

因为利用率描述的是：

> 当前硬件资源忙碌程度。

并不直接代表：

> 完成同样工作所需时间。

INT8 单次工作更轻，因此即使 GPU 没完全利用，也可能更快完成任务。

---

## Q19. `Elapsed Cycles` 是什么？

例如：

```text
Elapsed Cycles = 15810
SM Frequency   = 603.87 cycle/us
```

则：

```text
Duration
≈
15810 / 603.87
≈
26.18 us
```

因此：

```text
Elapsed Cycles
```

表示 Kernel 大约经历了多少个 GPU Clock Cycles。

---

## Q20. QPS 是什么？

QPS：

```text
Queries Per Second
```

Batch=1：

```text
1 Query
≈
1 Sample
```

Batch=8：

```text
1 Query
=
8 Samples
```

因此比较不同 Batch 时：

```text
Samples/s
```

往往比 QPS 更有意义。

---

## Q21. Nsight Systems 和 Nsight Compute 有什么区别？

Nsight Systems：

```text
System / Timeline Level
```

回答：

> 时间花在哪里？

看：

```text
CUDA API
Kernel Launch
Memcpy
Synchronization
Timeline
```

Nsight Compute：

```text
Single Kernel Microarchitecture Level
```

回答：

> 这个 Kernel 为什么这么快或这么慢？

看：

```text
SM
Memory
Occupancy
Warp
Scheduler
Tensor Core
Registers
Shared Memory
```

---

## Q22. 为什么 NCU 下的模型 latency 不能当 Benchmark？

因为 Nsight Compute 为采集 Hardware Counters 会：

```text
Replay Kernel
```

例如：

```text
11 passes
```

同一个 Kernel 可能重复执行很多次。

因此：

```text
NCU Runtime Latency
```

不能代表真实推理性能。

真实 latency 应使用：

```text
C++ Benchmark
```

或正常：

```text
trtexec
```

测试。

---

# 47. Main Conclusions

通过本项目得到的核心认识：

```text
1. 模型部署不是“ONNX 转 Engine”这么简单。

2. TensorRT 性能来自：
   Graph Optimization
   Layer Fusion
   Tactic Selection
   Tensor Layout
   Tensor Core
   Precision

3. FP32 → FP16 通常具有明显收益。

4. FP16 → INT8 是否继续大幅加速，
   取决于模型规模、Batch、Layout、
   Kernel Launch、Reformat 和硬件利用率。

5. Batch=1 的 ResNet18 在 AGX Orin 上
   很容易出现 Small Grid。

6. Small Grid 会导致：
   Waves/SM 低
   Occupancy 低
   GPU Utilization 低

7. Batch 增大后：
   Grid Size ↑
   Waves ↑
   Occupancy ↑
   Compute/Memory Utilization ↑
   Throughput ↑

8. GPU Utilization 本身不是最终目标。

9. 实际部署最终需要在：
   Latency
   Throughput
   Power
   Accuracy
   Memory
   之间做权衡。

10. Benchmark、Layer Profiler、
    Nsight Systems 和 Nsight Compute
    分别解决不同层次的问题。
```

---

# 48. Learning Path Completed

当前 ResNet18 TensorRT 学习链：

```text
PyTorch                    ✅

PyTorch → ONNX             ✅

CUDA Golden Reference      ✅

TensorRT FP32              ✅

TensorRT FP16              ✅

C++ Engine Builder         ✅

C++ Runtime                ✅

Dynamic Shape              ✅

Optimization Profile       ✅

Dynamic Batch B1/B4/B8     ✅

INT8 PTQ                   ✅

Calibration Cache          ✅

Engine Inspector           ✅

Layer Profiling            ✅

Nsight Systems             ✅

Nsight Compute             ✅

SM / Warp / Block / Grid   ✅

Occupancy / Waves          ✅

Batch Scaling Analysis     ✅

enqueueV3                  ✅

Tensor Name API            ✅

TensorRT Object RAII       ✅
```

---

# 49. Next Step

下一阶段：

```text
tensorrt-yolo/
```

目标是把本项目掌握的 TensorRT 部署能力迁移到真实视觉模型：

```text
YOLO
 ↓
PyTorch
 ↓
ONNX
 ↓
TensorRT FP16 / INT8
 ↓
C++ Runtime
 ↓
Image Preprocess
 ↓
Inference
 ↓
Decode
 ↓
NMS
 ↓
Bounding Boxes
 ↓
Benchmark
 ↓
Profiling
```

相比 ResNet18，YOLO 将进一步引入：

```text
真实图片输入
图像预处理
多输出 Tensor
Detection Decode
NMS
Bounding Box Visualization
真实 Accuracy / Performance Trade-off
```

这将作为下一阶段端侧视觉模型部署实践。