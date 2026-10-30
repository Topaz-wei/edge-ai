# TensorRT runtime

`edge_trt_runtime` 是 C++17 公共库，仅依赖 TensorRT runtime 和 CUDA runtime。
`edge::trt::Runner` 拥有一个 engine/context/stream，不可并发调用；模型预后处理和 builder/parser 不属于本库。

支持显式 batch、一个输入、一个或多个具名输出，以及 FP32 linear device I/O。
内部 FP16/INT8 计算仍可使用，但非 FP32 I/O、host I/O、shape tensor、隐式 batch、多输入及数据依赖输出明确拒绝。

## 动态形状与 profile

构造函数 `Runner(enginePath, profileIndex=0)` 枚举 I/O，读取 engine shape，并在所有 profile 的 MAX 输入下解析输出容量；设备缓冲按这些容量的最大值一次分配。随后选中指定 profile，默认实际形状为该 profile 的 OPT。

调用 `setInputShape(dims)` 后，从 context 重新解析各输出的实际 shape 与 `activeBytes`。只有有效数据参与传输，`capacityBytes` 和设备指针不随 batch/profile 切换而改变。`selectProfile(index)` 同步本 stream 后切换，并恢复该 profile 的 OPT 输入。第一版适用于输出体积由 profile MAX 输入界定的网络；若某次实际输出超过容量则报错，不扩容或越界访问。

`tensorInfo(name)` 返回名字、mode、dtype、engineShape、当前 shape、activeBytes、capacityBytes；`deviceTensor(name)` 按名字取得设备地址。未知名称、非法 rank、非正维度、固定维度不符及 profile 越界会报错。context 设置失败后需再次设置有效形状，禁止继续使用过期状态。

`inferNamed(input, timings)` 返回 `map<string, vector<float>>`，每个输出按名称访问。函数同步执行；异常时等待已提交的传输完成后才销毁主机缓冲。实例中所有设备提交与 profile 设置使用同一个 stream。

## 旧接口与 CUDA Graph

原 `infer()`、`inputInfo()`、`outputInfo()`、`inputElements()`、`outputElements()`、`deviceInput()`、`deviceOutput()` 保留。所有单输出相关 API 在多输出 engine 上明确报错，不取“第一个输出”。

`enqueueDevice()` 无主机复制或同步，调用方负责在本 stream 准备输入和消费输出。
`prepareGraph()` 继续支持原静态 engine；动态 engine 第一版明确拒绝 Graph。形状或 profile 设置会销毁已捕获 Graph，调用方需要重新准备。

## 构建与验证

独立构建：

```bash
cmake -S trt-runtime -B trt-runtime/build
cmake --build trt-runtime/build -j4
ctest --test-dir trt-runtime/build --output-on-failure
```

真实动态多输出 GPU 测试（需要 Jetson GPU 权限）：

```bash
build-shared/trt-runtime/test_dynamic_runner /tmp/edge-dynamic-fixture.plan
```

覆盖双输出名称和值、两个 profile、batch 切换、容量/地址稳定、错误输入、非法 rank、旧 API 误用、动态 Graph 拒绝、非 FP32 I/O 拒绝及实例重复创建/销毁。TensorRT 8.5 的 `getProfileShape(name, 1, ...)` 会输出内部 binding 索引修正警告；本库未用 binding 顺序推断输出语义。

已有模型回归通过 `bash scripts/test_runtime_legacy_gpu.sh` 执行，使用历史固定 engine、输入与输出基线，验证 ResNet/YOLO raw 逐字节一致及 YOLO 静态 Graph 开关一致性。RTMPose 实测见 [FP32 阶段报告](../tensorrt-rtmpose-hand/docs/trt-fp32-baseline.md)。
