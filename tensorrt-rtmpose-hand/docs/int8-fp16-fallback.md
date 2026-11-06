<!-- 说明 INT8 FP16 fallback 的配置入口和实验边界。 -->
# INT8 启用 FP16 fallback

2026-11-06：当前 `configs/rtmpose_hand_builder_int8.json` 已设置 `fp16=true`。
`apps/build_int8_engine.cpp` 将该值传入公共 Builder 的 `int8Fp16Fallback`，不再硬编码关闭。
公共 Builder 同时启用 INT8 和 FP16；这允许 TensorRT 选择 FP16 tactic，不强制任何指定层使用 FP16。
实际精度分布必须等新 engine 构建后通过 Inspector 确认。

Workspace 仍为 2 GiB，TF32 OFF，推理 profile 1/2/4，校准 batch 1，352 个冻结张量不变。
新 cache 路径为 `calibration/rtmpose_hand_int8_fp16.cache`，meta 为其路径加 `.meta.json`。
旧 INT8/FP32 fallback 的 engine、cache 和报告保留，不覆盖。

本次同时将校准身份 schema 升至 2，shape 写为真正的数值数组，FP16 字段来自当前配置，
修复旧版 FileStorage 嵌套问题。源码、可执行文件和配置身份均变化，因此不会复用旧 cache。
旧可执行文件另存于本地 `validation/trt-int8/rtmpose_int8.original`。
旧报告为历史快照；其比较器锁定的是旧源码和配置，不能直接用于本轮实验。

构建新实验的命令（在仓库根目录）：

```bash
build-shared/tensorrt-rtmpose-hand/rtmpose_int8 build \
  tensorrt-rtmpose-hand \
  tensorrt-rtmpose-hand/engines/rtmpose_hand_int8_fp16.engine
```

已启动本轮 GPU 构建。日志位于 `validation/trt-int8-fp16/build.log`，构建身份记录位于同目录 `build-inputs.json`。构建已完成，新 engine/cache 已发布；55 个开发样本验证完成，精度未通过。详见下文。


## 开发集精度验证结果

使用冻结的 55 个开发样本，NPY 和原图+bbox 两条路径，各覆盖 batch 1/2/3/4 和 6→4+2。
参考为 ORT FP32、同公共 C++ Builder/2 GiB 的 TRT FP32，并检查 batched 与 single 一致性。
全部 1155 个关键点参与统计，沿用已有低精度门槛。258 张最终 validation 未使用。

**结论：仅开启 FP16 fallback 后，精度仍未通过，主要误差没有消除。**
以下为 NPY batch 1 相对 ORT FP32 的结果；全部路径和 batch 见[结果快照](../benchmarks/int8-fp16-fallback.json)。

| 指标 | 原 INT8 + FP32 fallback | INT8 + FP16 fallback | 门槛 |
| --- | ---: | ---: | ---: |
| SimCC X MAE | 0.062688 | 0.061793 | ≤0.001 |
| SimCC Y MAE | 0.061164 | 0.060484 | ≤0.001 |
| SimCC X Max Abs | 1.370091 | 1.382840 | ≤0.05 |
| SimCC Y Max Abs | 1.183375 | 1.179487 | ≤0.05 |
| 关键点平均 L2 / 输入 px | 2.929895 | 2.907422 | ≤0.25 |
| 关键点 P95 L2 / 输入 px | 7.826238 | 7.850075 | ≤0.5 |
| 关键点最大 L2 / 输入 px | 104.0 | 102.5 | ≤2 |
| Score Max Abs | 0.607160 | 0.625318 | ≤0.02 |
| Valid-mask 翻转 | 0 | 0 | 0 |

相对同条件 TRT FP32：平均 2.907855 px、P95 7.850075 px、最大 102.5 px。
两条路径的输入最大误差均为 0，C++/Python decode 最大误差为 0，所有 batch 相对 INT8 single 的 SimCC 最大误差为 0。
共 160 项门槛失败，是不同路径/batch/参考的重复检查项，不是失败样本数。

Inspector 共 134 层，按输入输出 tensor 格式统计：

| 格式 | 层数 |
| --- | ---: |
| INT8 | 84 |
| FP16 | 40 |
| FP32 | 2 |
| INT8/FP32 混合 | 2 |
| INT8/FP16 混合 | 3 |
| FP16/FP32 混合 | 3 |

这些是 tensor 格式分类，不能等同于每个融合算子的累加精度。
FP16 已实际使用，但这是自动 tactic 选择，并未强制指定敏感层回退。
日志报告 58 项 FP16 subnormal 权重提示，以及 7 项低于最小 FP16 subnormal 的转换提示；尚未确定它们与误差的因果关系。

新旧 calibration cache 的 payload SHA-256 完全一致：
`eebe4f11433790161edf23fe708200b18c97cce7f853f5d9a900c70fd0699410`。
本轮独立重新校准生成，并非借用旧 meta；meta 因配置、源码和 schema 变化而不同。
因此之前的 [cache 范围分布图](../validation/calibration-ranges-int8/index.html) 数值也适用于新 cache。

Engine SHA-256：`7a5537db2dd23f0f3bb4bc4dcc087c5888e486ec36b46ca2dd06867d489759dc`。
原始 inspector、双路径输出、完成清单、完整 report.json 和逐点 argmax-shifts.csv 保存在 `validation/trt-int8-fp16/`。
验证工具为 [verify_trt_int8_fp16.py](../scripts/verify_trt_int8_fp16.py)，独立于旧基线入口，复用已有统计与解码函数。
复核命令：

```bash
/home/ssd/anaconda3/envs/py38/bin/python \
  tensorrt-rtmpose-hand/scripts/verify_trt_int8_fp16.py compare
```

当前返回码 1 表示数值验收失败，完整报告仍写出。Python 23 项测试通过；本轮未修改或重编译 runtime。
下一步应根据误差和层信息选择局部 precision constraint 实验；本轮未继续调层、运行最终 validation 或性能测试。
