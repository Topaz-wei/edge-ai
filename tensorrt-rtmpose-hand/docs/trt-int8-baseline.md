<!-- 本文记录首轮 INT8 校准、构建和开发集一致性实测；精度验收失败。 -->
# RTMPose-Hand INT8 首轮基线

2026-11-05 完成剩余验证。**校准、engine 构建和开发集测试已完成，精度验收未通过。**
这是一条独立于 FP16 的实验分支，不作为最终部署推荐。

## 构建与数据

使用公共 C++ Builder，经 `rtmpose_int8` 入口调用；TensorRT 8.5.2，Workspace 2 GiB，
INT8 ON、FP16 OFF、TF32 OFF。推理 MIN/OPT/MAX 为 1/2/4，C/H/W 固定为 3/256/256；
独立校准 profile 的 MIN/OPT/MAX 均为 batch 1。输入输出仍为 FP32。

固定顺序的 352 个 RGB/affine/normalization 后 FP32 NPY 全部参与 EntropyCalibrator2，
构建日志记录 352 次校准 batch，校准阶段耗时约 68.72 秒（不是完整构建耗时）。
55 个 development 样本用于验证；258 张最终 validation 原图未使用。

本地产物：

- `engines/rtmpose_hand_int8.engine`
- `calibration/rtmpose_hand_int8.cache` 与 `.cache.meta.json`
- `validation/trt-int8/build.log`、`build-inputs.json`、`build-complete.json`
- `validation/trt-int8/layers-context.json`、`layer-summary.json`
- `validation/trt-int8/npy/`、`image/`：两条路径原始输出和完成清单。
- `validation/trt-int8/report.json`、`argmax-shifts.csv`：完整指标和所有点的 X/Y argmax 变化。

engine SHA-256：`ae9f9d058ba1d59d4c4e022b9cd0b86afe9b976b1a770e035769dbdb9e4b8c2a`。
配置、cache、日志、输入清单、比较器和输出文件摘要见[结果快照](../benchmarks/int8-baseline.json)。
原始二进制和日志按 `.gitignore` 留在本机，快照及本文纳入版本控制。

## Inspector 与构建提示

按层输入输出 tensor 格式分类，共 136 层：94 层 INT8、34 层 FP32、8 层 INT8/FP32 混合，
0 层包含 FP16 格式。此统计不表示融合算子的累加精度。
其中 Convolution 类层：61 层 INT8、4 层混合、1 层 FP32。
X/Y 输出分支共享的 `MatMul_251 || MatMul_252` 为混合格式，随后 reformat/reshape 为 FP32。

日志有 INT64 权重转 INT32 提示，以及 6 个 Constant tensor 缺失 scale/zero-point 的 fallback 提示
（214、217、218、233、236、295）。这些提示已完整保留；目前不能将它们认定为精度退化根因。

## 开发集结果

复用冻结 C++ 验证器，运行 NPY 和原图+bbox 两条路径，每条覆盖 batch 1/2/3/4 和 6→4+2。
与 ORT FP32、同公共 C++ Builder/2 GiB 的 TRT FP32，以及 INT8 逐样本结果分别比较。
保持 FP16 阶段已有门槛，未根据 INT8 结果放宽。RMSE 和 argmax shift 作为补充统计。

以下为 NPY batch 1；其余路径/batch 的完整结果见快照：

| 指标 | INT8 vs ORT FP32 | INT8 vs TRT FP32 2 GiB | 门槛 |
| --- | ---: | ---: | ---: |
| SimCC X MAE | 0.0626882 | 0.0626872 | ≤0.001 |
| SimCC Y MAE | 0.0611645 | 0.0611634 | ≤0.001 |
| SimCC X RMSE | 0.0960808 | 0.0960789 | 仅记录 |
| SimCC Y RMSE | 0.0931740 | 0.0931722 | 仅记录 |
| SimCC X Max Abs | 1.370091 | 1.370086 | ≤0.05 |
| SimCC Y Max Abs | 1.183375 | 1.183350 | ≤0.05 |
| 输入坐标平均 L2 / px | 2.929895 | 2.930328 | ≤0.25 |
| 输入坐标 P95 L2 / px | 7.826238 | 7.826238 | ≤0.5 |
| 输入坐标最大 L2 / px | 104.0 | 104.0 | ≤2 |
| Score Max Abs | 0.607160 | 0.607157 | ≤0.02 |
| Valid-mask 翻转 | 0 | 0 | 0 |

所有 1155 个关键点参与统计。相对 ORT，X argmax 改变 936 点，Y 改变 1000 点；
X/Y 最大偏移分别为 208/55 bins（每 bin 对应 0.5 输入像素）。
全部检查共产生 160 项数值门槛失败记录，这是跨路径、batch、两个 FP32 参考的重复检查项数量，不是失败样本数。

工程一致性检查通过：两条路径输入相对冻结 NPY 的最大误差 0；C++/Python decode 最大误差 0；
INT8 batched/single SimCC 最大误差 0。说明这次观察到的退化不能用输入准备或动态 batch 不一致解释，
但尚未定位到具体敏感层。

## 元数据格式限制

本轮发现旧校准器使用 OpenCV FileStorage 写字符串 `[1,3,256,256]` 时，被解释为序列开始。
因此 `inference_profile` 等字段与 `ordered_tensors` 被嵌套在 `identity.calibration_shape` 数组中。
数据和 352 个有序 tensor SHA 未丢失，准备快照与 cache meta 完全一致；比较器严格核验旧数组布局、
profile/精度设置及每个 tensor SHA 后读取，不改写原始 meta 或历史准备快照。

此问题影响元数据结构，不改变 C++ Builder 实际 shape/profile 参数或送入校准器的张量。
后续升级元数据生产格式时应使用真正的数值数组，并生成新版身份记录；旧 cache 不能冒充新版产物。
本轮保留原校准器源码、可执行文件和 cache 身份，未将元数据格式修复混入已完成实验。

## 复核命令与实现

首次构建实际使用（现有 engine/cache 不应覆盖）：

```bash
build-shared/tensorrt-rtmpose-hand/rtmpose_int8 build \
  tensorrt-rtmpose-hand \
  tensorrt-rtmpose-hand/engines/rtmpose_hand_int8.engine
```

已有原始输出可以直接复核：

```bash
/home/ssd/anaconda3/envs/py38/bin/python \
  tensorrt-rtmpose-hand/scripts/verify_trt_int8.py build-completion
/home/ssd/anaconda3/envs/py38/bin/python \
  tensorrt-rtmpose-hand/scripts/verify_trt_int8.py compare
```

第二个命令在当前结果下返回 1，表示精度门槛未通过，完整失败报告仍会写出。
Inspector 复用 `scripts/inspect_fp16_engine.py --engine <INT8 engine> --output <JSON>`；
真实推理复用 `rtmpose_fp32 <INT8 engine> <cases.json> <输出目录> npy|image`。
两者名称保留历史命名，实际通过输入 engine 路径选择精度。

主要代码：[C++ 入口](../apps/build_int8_engine.cpp)、[校准器](../src/calibrator.cpp)、
[公共 Builder](../../trt-runtime/src/builder.cpp)、[比较器](../scripts/verify_trt_int8.py)、
[统计测试](../tests/test_int8_validation.py)。

本轮 CTest 13/13、Python 20/20 通过，不能据此替代模型精度验收。
下一步适合按实际输出层和误差进行局部 fallback 实验；尚未执行 mixed precision、INT8 性能测试或最终 validation。
