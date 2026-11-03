<!-- 本文记录公共 C++ builder 与 2 GiB FP16 新实验；历史 1 GiB 结果保持独立。 -->
# 公共 C++ builder 与 2 GiB FP16 实验

本轮范围是统一 engine 构建代码，以及新 workspace 下的构建与开发集验证。
不调整精度门槛，不做局部 FP32 保留，不使用最终 validation 或 calibration 数据。

## 实现边界

- `trt-runtime/include/edge/trt/builder.hpp`：`BuildOptions`、FP32/FP16/INT8 选项和外部校准器接口。
- `trt-runtime/src/builder.cpp`：解析 ONNX、具名 I/O 契约检查、profile、workspace、精度设置及 engine 序列化。
- `tensorrt-rtmpose-hand/apps/build_engine.cpp`：读取模型配置、核验 ONNX/reference SHA，调用公共 API。
- `configs/rtmpose_hand_builder.json`：实际构建参数，2048 MiB、FP16、TF32 off、profile 1/2/4。
- `configs/rtmpose_hand_trt_fp16_cpp_2g.json`：新实验的验收参数；acceptance 与首次实验一致。

构建与推理使用两个独立目标：`edge_trt_builder` 和 `edge_trt_runtime`。
公共 builder 不依赖 OpenCV，也不负责读取图片或手框；模型入口用 OpenCV 读取 JSON。
原 Runner、reference 及 FP32 推理程序不改动，旧 YOLO/ResNet 构建入口暂未迁移。

INT8 是公共 API 的可选项：调用方必须提供校准器及 calibration shape，校准器生命周期覆盖构建。
RTMPose 模型校准器本轮尚未实现，所以选择 INT8 会明确报错，不会静默变成 FP16。
本轮没有真实 INT8 构建验证，也不宣称 INT8 已完成。

## 运行与追溯

```bash
cmake -S . -B build-shared
cmake --build build-shared --target rtmpose_build -j4
bash tensorrt-rtmpose-hand/scripts/build_fp16_engine.sh
bash tensorrt-rtmpose-hand/scripts/run_fp16_validation.sh
```

新的 engine 为 `engines/rtmpose_hand_fp16_cpp_2g.engine`，原始结果目录为
`validation/trt-fp16-cpp-2g/`。这些本地产物按 `.gitignore` 管理。
首次实验的 `rtmpose_hand_fp16.engine`、`validation/trt-fp16/` 以及旧 FP32 产物保持独立。

构建记录包含 builder 源文件、模型入口、配置、模型/reference、builder 可执行文件 SHA。
`build-complete.json` 将这些记录与 engine 和日志 SHA 绑定。比较前再次核验身份。
构建脚本拒绝复用已有实验目录，不能直接重复执行覆盖结果；新实验应使用独立路径。

验证覆盖 55 个 development 样本的 NPY 和原图+bbox 两条路径，batch 1/2/3/4/6（4+2）。
比较 ORT FP32、历史 TRT FP32 同 batch，以及 FP16 batched-vs-single，全部关键点参与统计。
历史 TRT FP32 使用 1 GiB，因此该对照不是仅改变精度的严格单变量实验；
也不能将新旧 FP16 的任何差异全部归因于 workspace，builder 与 tactic 选择也可能不同。

## 验证状态

公共 builder 已编译，CPU 参数检查及 12 项 CTest 通过；FP16 比较器 7 项测试通过。
真实 GPU 构建成功，55 个样本两条路径、全部 batch 均执行完成，10 组比较报告完整保存。
构建与实验任务完成，但 FP16 精度验收仍为 **failed**；验证脚本因此返回 1，这是门槛拒绝而非推理执行崩溃。

结果快照：[fp16-cpp-2g.json](../benchmarks/fp16-cpp-2g.json)。
Engine SHA-256：`18a95f59220a896918a085119f2c3916ac69ab609d2cf0ff9491fdf0adc3f57a`。

下表为两条路径、五种 batch 请求中各指标最大值；坐标单位为模型输入像素。

| 指标 | 相对 ORT FP32 | 相对历史 TRT FP32 | 门槛 |
| --- | ---: | ---: | ---: |
| SimCC X MAE | 0.00160416 | 0.00160422 | 0.001 |
| SimCC Y MAE | 0.00171522 | 0.00171523 | 0.001 |
| SimCC X max abs | 0.03595591 | 0.03595066 | 0.05 |
| SimCC Y max abs | 0.02933919 | 0.02937043 | 0.05 |
| 关键点平均偏移 | 0.34366682 | 0.34384611 | 0.25 |
| 关键点 P95 偏移 | 0.79497474 | 0.79497474 | 0.5 |
| 关键点最大偏移 | 23.58495331 | 23.58495331 | 2 |
| score max abs | 0.01319075 | 0.01318264 | 0.02 |
| valid-mask 翻转 | 0 | 0 | 0 |

两条路径输入 tensor 最大差为 0；同一 SimCC 的 C++/Python 解码差为 0。
FP16 batched-vs-single 的 SimCC、坐标和 score 差均为 0。
因此本轮已验证构建、输入与动态 batch 执行链，但没有解决低精度数值偏差。

Inspector 共 193 层：按可见 I/O tensor 格式分类，183 层 FP16、5 层 FP32、5 层混合。
这不是融合算子内部累加精度统计，也不等于纯 FP16 网络。
构建日志保留 workspace 排除部分 tactic 的提示，以及 54 weights 的 FP16 subnormal 提示和
7 weights 低于最小正 subnormal 的转换提示；尚未验证这些提示与关键点偏差之间的因果关系。

最终 validation 使用 0 张，calibration 使用 0 个 tensor；没有开始 INT8 或性能 benchmark。
本轮按要求止于构建与验证，不自动增加 workspace，也不修改局部精度或验收门槛。
