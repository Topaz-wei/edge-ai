<!-- 本文记录公共 runtime 动态多输出改造与 RTMPose-Hand 首次 TensorRT FP32 实测。 -->

# Runtime vNext 与 RTMPose-Hand FP32 基线

本阶段已完成单输入、具名多输出、动态 shape、profile 管理、持久 MAX 缓冲，以及 RTMPose 的 TensorRT FP32 基线验证。C++ 预处理和解码均已与冻结参考对照；原 YOLO/ResNet 通过公共 Runner 完成兼容性回归。

本报告对应实现提交 `e7ee5b5`；[结果快照](../benchmarks/fp32-baseline.json) 保存具体源码和可执行文件哈希。原始输出位于本地 `validation/trt-fp32/`，不提交 engine、样本和原始二进制。当前结果不是性能 benchmark，也不是人工 GT 精度报告。FP16、INT8 和最终 validation 尚未开始。

## 1. 固定边界与构建配置

Reference SHA-256 始终为：

```text
3036459e1f6bc230c9059f34ea260070928669f54f9b1bfef3e031279aef0aec
```

[原 reference 配置](../configs/rtmpose_hand_reference.json) 未修改。新增的 [FP32 构建配置](../configs/rtmpose_hand_trt_fp32.json) 单独记录部署选择：

| 项目 | 实际设置 |
| --- | --- |
| 模型 | 原始 `models/rtmpose_hand.onnx`，未改图、未生成 b1 派生模型 |
| Engine | `engines/rtmpose_hand_fp32.engine` |
| 输入 | FP32 `input [N,3,256,256]` |
| 输出 | FP32 `simcc_x`、`simcc_y`，各 `[N,21,512]` |
| Profile 0 | MIN=1、OPT=2、MAX=4，只有 batch 动态 |
| 精度 | FP32，TF32 关闭，未启用 FP16/INT8 |
| Workspace | 1024 MiB |
| 动态 CUDA Graph | 不启用 |
| 超过 MAX | 保持顺序分块，例如 6 只手执行 4+2 |

环境为 Jetson AGX Orin、TensorRT 8.5.2、CUDA runtime 11.4。C++ 使用系统 OpenCV 4.5.4，Python reference 使用 OpenCV 4.13.0；两者在本次 55 个开发样本上的预处理结果逐元素一致。

构建使用本机 TensorRT 8.5 的 `trtexec --buildOnly --noTF32`，完整命令见 [build_fp32_engine.sh](../scripts/build_fp32_engine.sh)。构建器和 parser 留在模型工具侧，公共 runtime 不增加 OpenCV、ONNX parser 或模型依赖。

## 2. Runtime 实际改动

主要接口在 [runner.hpp](../../trt-runtime/include/edge/trt/runner.hpp)，实现位于 [runner.cpp](../../trt-runtime/src/runner.cpp)。

1. 枚举 `getNbIOTensors()`，建立名字、mode、dtype、engineShape、当前 shape、activeBytes、capacityBytes；不根据 binding 顺序猜测 X/Y。
2. 构造时遍历各 profile 的 MAX 输入，用 context 解析输出形状并取最大容量，一次分配全部设备缓冲。
3. 显式 `selectProfile(index)`，在执行 stream 上调用 `setOptimizationProfileAsync()`，将实际输入恢复至该 profile 的 OPT。
4. `setInputShape()` 先验证 rank、正维度、固定维度和 profile 范围，再解析实际输出 shape、检查 activeBytes 不超过 capacityBytes。
5. 按名称绑定设备地址，调用 `enqueueV3()`；同步 host API `inferNamed()` 返回按名称索引的输出。
6. 保留旧单输出 API；多输出 engine 调用 `infer()`、`outputInfo()`、`outputElements()` 或 `deviceOutput()` 明确报错。

每个 Runner 实例持有一个 context/stream，不支持并发调用。shape/profile 变化前同步本 stream；异常路径先同步已提交的复制再释放主机缓冲。设备指针在 batch/profile 切换期间保持不变。

本版仅支持一个输入、一个或多个输出，I/O 必须是 FP32 linear device tensor。多输入、shape tensor、host I/O、隐式 batch、非 FP32 I/O 和数据依赖输出不在本次支持范围。容量策略要求各 profile MAX 输入能够界定输出体积；若实际输出超过预分配容量会报错，不静默写越界。

动态 engine 禁止 `prepareGraph()`；旧静态 engine 继续支持 Graph。shape/profile 设置会失效已有 Graph，需要重新准备。

RTMPose profile MAX 对应容量：

| 张量 | capacityBytes |
| --- | ---: |
| input | 3,145,728 |
| simcc_x | 172,032 |
| simcc_y | 172,032 |

## 3. 两条验证路径

```text
路径 A：冻结 NPY → C++ NPY 读取 → 公共 Runner → TRT FP32 → 原始 SimCC
路径 B：原图+bbox → C++ RGB/affine/normalize → 公共 Runner → TRT FP32 → C++ decode
```

[reference.cpp](../src/reference.cpp) 实现模型专属预处理、NPY 读取和解码。其 RGB、1.25 倍扩展、正方形、三点仿射、双线性插值、零填充、双精度归一化后转 FP32、SimCC 平均分及无效点规则均保持冻结契约。

[verify_fp32.cpp](../apps/verify_fp32.cpp) 使用同一个 Runner 反复切换 batch；两种输入路径均处理全部 55 个 development 样本，依次覆盖 1、2、3、4、请求 6 后分块，以及返回 batch=1。尾批不丢弃。这里只检查路径正确性，不把执行时间作为正式性能结果。

[verify_trt_fp32.py](../scripts/verify_trt_fp32.py) 按固定开发顺序生成清单，并执行以下比较：

- C++ 生成输入与现用 RGB NPY 逐元素比较。
- TRT 两个原始 SimCC 输出与已保存的 ORT FP32 单样本参考输出比较。
- TRT 各 batch 与 TRT 单样本输出比较。
- 两个后端的输入空间坐标、原图坐标、分数、有效掩码比较，不过滤低分点。
- 对同一份 TRT SimCC，用 C++ 和 Python 分别解码，单独隔离解码实现误差。

所有测试只使用 development，calibration 未参与 pose 输出调参，最终 validation 的 258 张原图未参与本次运行。

## 4. 结果与验收口径

预先设置的开发阶段门槛见本地 `acceptance.json` 及结果快照。它们是工程一致性门槛，不代表模型真实精度标准；此次未为通过结果而修改 reference 或放宽门槛。

| 指标 | 最差实测值（两种路径、全部 batch） | 门槛 |
| --- | ---: | ---: |
| C++ 输入张量最大绝对差 | 0 | 0 |
| TRT vs ORT SimCC 最大绝对差 | 4.601478576660156e-5 | 1e-4 |
| 输入坐标平均 L2 偏移 | 0.00043290044413879514 px | 0.1 px |
| 输入坐标 P95 L2 偏移 | 0 px | 0.5 px |
| 输入坐标最大 L2 偏移 | 0.5 px | 1.0 px |
| 原图坐标最大 L2 偏移 | 0.28410179540514946 px | 记录项 |
| 分数最大绝对差 | 1.7821788787841797e-5 | 1e-4 |
| 有效掩码翻转数 | 0 | 0 |
| TRT batch vs 单样本 SimCC 最大绝对差 | 1.6689300537109375e-6 | 1e-4 |
| C++ vs Python 对同一 SimCC 解码的最大绝对差 | 0 | 1e-9 |

55×21 个关键点中仅一个点相对 ORT 偏移了输入空间的 0.5 像素，其余坐标一致。TensorRT FP32 与 ORT 不保证逐位相同；这里通过的是事先定义的容差检查，而不是声称两个后端输出完全相等。

## 5. 回归、安全边界与独立审查

验证结果：

- 全部项目构建成功，CTest 11/11 通过。
- `scripts/test_camera_analysis.py` 的 4 项离线分析测试通过，未运行摄像头实测。
- [test_dynamic_runner.cpp](../../trt-runtime/tests/test_dynamic_runner.cpp) 构建真实小型 GPU 网络，验证双输出名称和值、两个 profile、容量/地址稳定、非法 shape/rank、越界 batch、错误输入长度、未知名字、旧 API 误用、非 FP32 I/O 拒绝、动态 Graph 拒绝及重复创建/销毁。
- ResNet：`resnet18_cpp_fp16.plan`、既有 `data/input.bin` 与历史 raw 基线逐字节一致，重复调用及错误长度检查通过。
- YOLO：`yolov8m_fp16.plan` 固定图的 5 次 timed/untimed 推理均与历史 raw 逐字节一致。
- YOLO 静态 CUDA Graph：真实图、随机图、ROI、常量图预处理及逐帧 Graph 开关输出检查通过。

旧模型回归入口为 [test_runtime_legacy_gpu.sh](../../scripts/test_runtime_legacy_gpu.sh)，日志在仓库根目录 `logs/runtime-vnext/`。没有为本次回归重新构建或量化 ResNet/YOLO。

独立代码审查指出并已修复：

1. 错误 rank 必须在体积计算前拒绝，`getNumElements()` 也增加 `Dims::MAX_DIMS` 上界检查；新增失败用例后修复并通过。
2. 固定目录可能遗留旧输出。运行开始即删除旧 `complete.json`，记录实际 engine/cases 哈希；仅成功结束且身份未变后，才对原始输出记录 SHA-256 并原子发布完成记录。比较脚本核验全部哈希，比较前先将旧报告置为 `incomplete`。错误 engine 身份和篡改输出的负例均被拒绝。

独立复核确认上述修复解决原问题。运行身份记录保存在每条路径的 `complete.json`，不得用缺少完成记录的历史输出生成成功报告。

## 6. 已遇到的问题与实际限制

- 沙箱无法访问 Jetson GPU，设备测试在批准后于沙箱外执行；CPU 构建和测试仍在工作区内进行。
- 本机 TensorRT 8.5 不支持 `--skipInference`，构建脚本已改为本机支持的 `--buildOnly`。
- OpenCV FileStorage 会把以 `[` 开头的形状字符串误当结构标记，已改用字符串 `write()` 接口保存形状。
- 首次 ResNet 回归误用了 `resnet18_fp16.plan`，与历史基线并非同一 engine，导致逐字节比较失败；按历史报告改回 `resnet18_cpp_fp16.plan` 后通过，未修改基线或比较标准。
- TensorRT 对原始 ONNX 的 INT64 常量会提示转 INT32；构建正常完成。1 GiB workspace 限制会排除部分 tactic，本次不追求性能最优。
- TensorRT 8.5 在具名查询第二个 profile 时打印内部 binding 索引修正警告；测试仍通过，代码未靠 binding 顺序猜测输出语义。
- 旧模型 engine 会打印跨设备构建兼容性提示；实际反序列化、raw 比较和静态 Graph 测试通过。没有据此宣称跨设备通用或性能等价。

## 7. 复现命令

在仓库根目录执行。现有构建目录保留原 YOLO/ResNet/video 选项，只启用新增模块：

```bash
cmake -S . -B build-shared -DEDGE_BUILD_RTMPOSE=ON
cmake --build build-shared -j4
ctest --test-dir build-shared --output-on-failure

bash tensorrt-rtmpose-hand/scripts/build_fp32_engine.sh
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/scripts/verify_trt_fp32.py prepare

build-shared/tensorrt-rtmpose-hand/rtmpose_fp32 \
  tensorrt-rtmpose-hand/engines/rtmpose_hand_fp32.engine \
  tensorrt-rtmpose-hand/validation/trt-fp32/cases.json \
  tensorrt-rtmpose-hand/validation/trt-fp32/npy npy

build-shared/tensorrt-rtmpose-hand/rtmpose_fp32 \
  tensorrt-rtmpose-hand/engines/rtmpose_hand_fp32.engine \
  tensorrt-rtmpose-hand/validation/trt-fp32/cases.json \
  tensorrt-rtmpose-hand/validation/trt-fp32/image image

/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/scripts/verify_trt_fp32.py compare

build-shared/trt-runtime/test_dynamic_runner /tmp/edge-dynamic-fixture.plan
bash scripts/test_runtime_legacy_gpu.sh
python3 scripts/test_camera_analysis.py
```

GPU 命令需要设备访问权限。Python 核验依赖断言，不要使用 `python -O`。重建 engine 后需重新运行两条推理路径，不能沿用旧完成记录。运行记录和 raw 文件是本地实验产物，版本控制中的结果快照保存本轮实测摘要。

后续顺序是 TRT FP16 一致性，然后再进入 INT8。当前不改变 reference，不运行最终 validation，不将本次正确性测试当作正式 benchmark。
