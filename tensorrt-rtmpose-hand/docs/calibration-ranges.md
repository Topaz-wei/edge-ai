<!-- 本文区分 calibration cache 的范围分布与逐层激活频数直方图。 -->
# Calibration 范围可视化

TensorRT EntropyCalibrator2 内部会收集激活直方图并据此选择 scale。
当前保存的 calibration cache 是 tensor 名称和最终 scale 表，不包含直方图 bins/counts。
因此无法从该 cache 还原每层激活值的频数分布，也不能推算 clipping 比例。
参考 [NVIDIA TensorRT 8.5.3 Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/pdf/TensorRT-Developer-Guide.pdf)。

本轮提供的是**跨 tensor 的量化范围分布图**：每个有效 cache 条目贡献一个 scale，纵轴为 tensor 数量。
cache 的 8 位 hex 按 IEEE754 float32 解码；图中另以 `127 × scale` 展示换算后的正向对称阈值。
INT8 网格为 `[-128 × scale, 127 × scale]`。这些值不能称为采样激活的实测 min/max。

旧 INT8 baseline cache 有 276 个条目，其中 274 个 scale 为有限正数：

| 统计 | 127 × scale |
| --- | ---: |
| 最小 | 0.0000100024 |
| P10 | 0.836986 |
| P50 | 3.118859 |
| P90 | 13.195855 |
| 最大 | 3636.449789 |

两个条目的原始 hex 是 `7f800000`，解码为正无穷：

- `(Unnamed Layer* 220) [Shuffle]_output`
- `(Unnamed Layer* 238) [Shuffle]_output`

这两个条目保留在 CSV/HTML/summary 中，但不进入对数直方图。此记录不能单独证明它们是精度退化根因，
也不能推断它们对应实际 INT8 执行层；需结合 engine Inspector 与模型图分析。

本地查看入口：

- 旧 baseline：`validation/calibration-ranges-int8/index.html`
- 每个报告目录包含 PNG、SVG、完整 CSV 和带 cache SHA 的 summary.json。
- 浏览器直接打开 HTML，无需服务器或联网；表格支持按 tensor 名称筛选。

复现示例（输出目录必须不存在，防止覆盖已有结果）：

```bash
MPLCONFIGDIR=/tmp/rtmpose-mpl XDG_CACHE_HOME=/tmp/rtmpose-cache \
/home/ssd/anaconda3/envs/py38/bin/python \
  tensorrt-rtmpose-hand/scripts/plot_calibration_ranges.py \
  tensorrt-rtmpose-hand/calibration/rtmpose_hand_int8.cache \
  tensorrt-rtmpose-hand/validation/calibration-ranges-int8-new
```

若要查看逐层激活值直方图，需要另行采集 FP32 中间 tensor 输出，统计 bins/counts；
这样的图应标为独立 FP32 采样结果，而非 TensorRT EntropyCalibrator2 的内部原始直方图。
本轮尚未执行该采集。
