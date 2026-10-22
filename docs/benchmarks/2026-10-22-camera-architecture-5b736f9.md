# 公共模块提取的真实摄像头性能回归（5b736f9）

> 适用范围说明：本文主测试为 **1920×1080**，不能作为用户实际 **1280×720** 的性能结论。用户确认输入尺寸后，已完成[720p 串行/双缓冲正式复测](2026-10-22-camera-720p-double-buffer.md)；当前 720p 场景请以该报告为准。历史原始数据保留。

测试日期：2026-10-22。正式测试于北京时间 14:00:44–14:18:17 完成，27/27 次通过，engine 测试前后哈希一致。

**结论：在本机、单模型和本次采集条件下，未观察到公共模块提取导致平均吞吐下降。配置入口开启原始 1080p 画布显示时，比兼容入口平均慢 7.14%；这包含显示行为变化，不能归为 TensorRT 抽象层开销。未锁频且存在长尾，不能宣称零开销或已排除小幅性能回归。**

## 补充：main 的 30 FPS 与本次 1080p 结果为何不同

用户指出 main 开启摄像头显示能够达到 30 FPS 后，重新核对发现：本地 main 正是本报告基线 `282645d`；main 的 [CAMERA_BUFFER_RESULTS.md](../../tensorrt-yolo/docs/CAMERA_BUFFER_RESULTS.md) 中 30.030 FPS 的显示测试使用 **1280×720 输入**，本报告前述 27 次测试使用 **1920×1080 输入**。此前未将这一历史输入条件差异突出说明，不能把历史 30 FPS 与本报告 1080p 结果直接作为重构前后退化比例。用户此次实际启动命令尚未提供，不能认定其当前运行一定使用 720p。

为验证分辨率这一解释，使用相同已编译的 A/B/C 程序、相同混合 INT8 engine、Graph、MJPG、30 FPS、两个缓冲和开启显示，补测 1280×720。顺序 B→A→C，各 900 帧，排除前 150 帧，每个入口仅 1 轮；所有运行及 engine 哈希校验通过。

| 720p 输入、开启显示 | 实际稳态 FPS | geometry + pipeline 均值 ms | 完成帧龄均值 ms |
|---|---:|---:|---:|
| A：main / 提取前 | 29.928 | 13.964 | 57.834 |
| B：提取后兼容入口 | 29.967 | 14.757 | 58.423 |
| C：提取后配置入口 | 29.965 | 14.612 | 59.784 |

**同为 720p 时，三个入口均复现约 30 FPS，没有复现从 30 FPS 降至约 24–26 FPS 的现象。** 摄像头输入从 720p 增至 1080p，像素数增加到 2.25 倍，即使 YOLO 的最终输入尺寸不变，MJPG 解码、CPU 图像变换等工作量也会改变；配置入口还显示原始尺寸画布。这些是端到端工作量变化，不能全部归于 TensorRT 公共模块。

这组补测只有一轮且与 1080p 测试不在同一交叉调度内，能够确认三个入口在 720p 下都可达到源帧率上限，不能精确分离分辨率与 DVFS 等因素的贡献，也不能证明计算开销完全一致。若用户的原命令实际同为 1080p，仍须进一步对齐其命令、engine、二进制及功耗状态；目前不能解释这种情况下的 30 FPS 差异。

补测证据：[metadata.json](../../outputs/camera-architecture-720p-check-20261022/metadata.json)、[runs.json](../../outputs/camera-architecture-720p-check-20261022/runs.json)、[summary.json](../../outputs/camera-architecture-720p-check-20261022/summary.json)。同目录保存全部逐帧数据和遥测。复现时在本文测试命令上添加 `--suite display --width 1280 --height 720 --rounds 1`，并使用新输出目录。

## 对比对象

| 标记 | 源码提交 | 可执行入口 | 用途 |
|---|---|---|---|
| A | `282645d1bd684df4130409f20ce377a21b4266b0` | `yolo_video_cuda` | 公共模块提取前基线 |
| B | `5b736f9f308fbb115f73c234b642aba4cafdcf85` | `yolo_video_cuda` | 提取后兼容入口，与 A 保持相同显示方式 |
| C | `5b736f9f308fbb115f73c234b642aba4cafdcf85` | `edge_video` + YAML | 提取后配置驱动入口 |

两份源码使用 `git archive` 独立导出并重新编译可执行程序；**没有重建 engine**。当前工作区 HEAD 是 `15711c2`，其双缓冲优化不参与本次测试。本文也不把后续双缓冲测试作为架构提取的性能结果。

A→B 是最接近同等功能条件的公共模块提取对比；B→C 还包含配置入口和坐标映射等行为差别。所有入口只加载一个 YOLOv8m 模型。本文的 FP16 指 **YOLOv8m FP16，非 ResNet**。

## 环境与方法

- Jetson AGX Orin；Linux `5.10.120-tegra`；CUDA 11.4.315、OpenCV 4.5.4、G++ 9.4.0、CMake 4.3.2。
- LRCP F1080P USB 真实摄像头，uvcvideo，稳定设备路径 `/dev/v4l/by-id/usb-LRCP_F1080P_LRCP_F1080P_000000000001-video-index0`，本机对应 `/dev/video8`。
- 1920×1080、MJPG、30 FPS、V4L2 两个采集缓冲；每次运行核验实际分辨率、源 FPS、缓冲数和 MJPG 格式。
- `input_stage: camera`，CUDA Graph 开启。无录像、无图像文件保存；显示组使用本机 `DISPLAY=:1` 和真实桌面窗口。
- MAXN，保持原有 DVFS；未修改时钟、功耗或摄像头曝光。`jetson_clocks --show` 需要 root，因此无法确认锁频状态；不能将 MAXN 当成固定 GPU 频率。tegrastats 每秒采样，保留 CPU 频率、利用率和温度；本机输出没有 GPU MHz。
- 每场景每入口 3 轮，每次 900 帧，剔除最前 150 帧，保留后 750 帧及其全部异常值。无显示测试包含 FP16 和混合 INT8；显示测试包含混合 INT8。共 27 次、24300 帧，其中 20250 帧纳入稳态统计。
- 顺序按轮次轮换，使每个入口在每个场景中各占一次第一、第二、第三位置。运行全部串行，避免同时争用 GPU 或摄像头。详见原始 `schedule.json`。
- 吞吐按每轮 `(750−1) × 1000 / (最后 end_ms−第一保留帧 end_ms)` 计算，再取三轮算术平均。阶段均值/P95合并该场景三轮的保留帧，百分位使用线性插值。帧之间并非独立重复样本，不据此宣称统计显著性。
- 可比处理耗时是 **`geometry_ms + pipeline_ms`**，包含 CPU 图像变换、GPU 推理/后处理及主机等待，不是纯 TensorRT kernel 耗时。C 将图像变换放在模型内；不能单独对比三者的 `pipeline_ms`。
- 完成帧龄为主机单调时钟减 V4L2 capture timestamp，排除负值哨兵；它不是曝光到屏幕的实测延迟。
- A/B 显示处理后的 360×640 画布；C 在原始 1920×1080 图像上画框，像素数为前者的 9 倍。C 的显示比较是两种实际入口行为的比较，不是同画布条件下的抽象层成本测量。

## Engine 与代码一致性

| Engine | 路径 | SHA-256 |
|---|---|---|
| YOLOv8m FP16 | `tensorrt-yolo/engines/yolov8m_fp16.plan` | `5426f5c14084d42f5092e4079069a7d64d640de5a075bb18dd1452c3f7461b7a` |
| YOLOv8m 混合 INT8 | `tensorrt-yolo/debug/trial007_sub_model22/yolov8m_int8.plan` | `8353b32b5c9c03654418caab042758ea78045b505a32326c84987ab19c87b203` |

`cuda_yolo.cu`、`yolo_preprocess.cpp`、`postprocess_raw.cpp` 在 A/B/C 对应源码中哈希相同。构建均为 Release，C++/CUDA 使用 `-O3 -DNDEBUG`。TensorRT 在加载已有 engine 时打印设备兼容性警告，该日志保留；三个入口使用同一设备和同一文件。

实时场景会变化，各轮并不接收同一组像素。本测试验证性能和实际运行稳定性，不能据检测数相近证明 mAP 不变；精度一致性应使用固定样本回归，已有结果见 [shared-runtime 回归记录](../../tensorrt-yolo/benchmarks/shared-runtime/README.md)。

## 测试结果

### 平均吞吐

| 场景 | A 基线 FPS | B 兼容 FPS | C 配置 FPS | B/A | C/A | C/B |
|---|---:|---:|---:|---:|---:|---:|
| 无显示 / FP16 | 26.578 | 26.851 | 26.589 | +1.03% | +0.04% | -0.98% |
| 无显示 / 混合 INT8 | 26.935 | 27.782 | 28.084 | +3.15% | +4.27% | +1.08% |
| 显示 / 混合 INT8 | 25.400 | 25.983 | 24.129 | +2.30% | -5.00% | -7.14% |

这些百分比是本次观察值，不是已证明的架构加速比例。尤其 A 的 INT8 无显示第二轮存在 1.34 秒停顿，显示第三轮整体较慢；DVFS、桌面后台负载及实时画面差异均可能影响轮次。没有删除这些数据，也没有用最快轮次替代均值。

### 逐轮吞吐

| 场景 | 轮次 | A FPS | B FPS | C FPS |
|---|---:|---:|---:|---:|
| 无显示 / FP16 | 1 | 26.713 | 26.642 | 26.614 |
| 无显示 / FP16 | 2 | 26.552 | 27.305 | 27.020 |
| 无显示 / FP16 | 3 | 26.470 | 26.605 | 26.132 |
| 无显示 / 混合 INT8 | 1 | 27.361 | 28.187 | 28.241 |
| 无显示 / 混合 INT8 | 2 | 26.831 | 26.932 | 28.329 |
| 无显示 / 混合 INT8 | 3 | 26.613 | 28.228 | 27.681 |
| 显示 / 混合 INT8 | 1 | 26.330 | 25.519 | 23.947 |
| 显示 / 混合 INT8 | 2 | 25.903 | 26.361 | 24.232 |
| 显示 / 混合 INT8 | 3 | 23.968 | 26.070 | 24.209 |

### 分项耗时与完成帧龄

单位均为 ms，均值/P95 合并每组 2250 帧。读取含取帧等待与 MJPG 解码；处理是 geometry + pipeline；绘制含画布复制。

| 场景 | 入口 | 读取均值 | 处理均值 / P95 | 绘制均值 | 显示均值 | 完成帧龄均值 / P95 |
|---|---|---:|---:|---:|---:|---:|
| 无显示 / FP16 | A | 16.809 | 20.797 / 25.463 | 0.000 | 0.000 | 81.639 / 97.324 |
| 无显示 / FP16 | B | 16.752 | 20.479 / 25.057 | 0.000 | 0.000 | 81.151 / 97.048 |
| 无显示 / FP16 | C | 16.844 | 20.760 / 25.537 | 0.000 | 0.000 | 81.780 / 96.711 |
| 无显示 / 混合 INT8 | A | 16.783 | 19.750 / 24.555 | 0.000 | 0.000 | 80.645 / 94.922 |
| 无显示 / 混合 INT8 | B | 16.685 | 19.311 / 24.071 | 0.000 | 0.000 | 78.569 / 93.476 |
| 无显示 / 混合 INT8 | C | 16.734 | 18.871 / 23.200 | 0.000 | 0.000 | 77.496 / 92.839 |
| 显示 / 混合 INT8 | A | 16.615 | 20.583 / 26.890 | 0.284 | 1.931 | 86.731 / 107.532 |
| 显示 / 混合 INT8 | B | 16.851 | 19.165 / 23.316 | 0.282 | 1.895 | 83.985 / 100.118 |
| 显示 / 混合 INT8 | C | 16.294 | 18.429 / 22.399 | 1.191 | 5.272 | 91.869 / 109.605 |

显示组 B 的绘制+显示均值为 2.177 ms，C 为 6.463 ms，增加约 **4.286 ms/帧**。C 的可比处理均值没有同步增加。因此，配置入口的显示路径成本是有直接分项证据的主要差别；但没有同画布对照或系统追踪，不能精确分离画布大小、GUI 后端、调度与其他因素的因果贡献。

A/B 同画布显示的绘制+显示均值接近（2.215 vs 2.177 ms）。公共接口分发、结果对象分配及坐标映射仍有理论开销，当前端到端测量无法将其单独量化；不能根据这些均值断言抽象层完全免费。

### 长尾与环境波动

全部 27 次稳态窗口共出现以下 **3 次 >100 ms** 的相邻完成间隔；它们也都 >200 ms。其余窗口无 >100 ms 完成间隔。

| 运行 | 帧号 | 完成间隔 ms | 当前帧 total_ms | 前帧结束至当前帧开始的间隙 ms |
|---|---:|---:|---:|---:|
| display-mixed_int8-B-r1 | 271 | 671.411 | 37.741 | 633.670 |
| display-mixed_int8-C-r1 | 871 | 576.294 | 43.665 | 532.629 |
| headless-mixed_int8-A-r2 | 541 | 1341.420 | 35.144 | 1306.276 |

三个长停顿均主要位于帧间间隙，且发生在前一帧每 30 帧日志输出之后。源码在该区间包含日志输出、循环管理和系统调度等；现有日志只能定位区间，不能证明根因。A、B、C 均出现过一次，不能据三轮样本判断某个入口改善或恶化了长尾稳定性。

全部 20250 个保留帧均有有效完成帧龄；各运行保留窗口内采集时间戳均严格递增，但这不代表没有丢弃相机帧。三次停顿的原始行、分项和所有其他异常值都保存在 CSV/summary 中。

tegrastats 记录的 GPU 温度范围为 38.75–59.75°C，CPU 频率样本为 729–2188 MHz，说明运行环境确有动态变化。这些采样覆盖初始化和处理全程，不仅是稳态；没有可用 GPU MHz/完整调度追踪，也没有关闭桌面后台应用，不能以温度范围排除频率或系统负载影响。

### 对架构的判断与使用建议

1. **可以保留这次公共模块提取。** A→B 在三类场景中的平均吞吐分别为 +1.03%、+3.15%、+2.30%，没有观察到平均性能损失；FP16 的小差异仍属于本测试难以可靠归因的范围。
2. **性能回归优先用无显示场景和相同 engine。** 这里仅重新编译应用，无需重建 engine。用旧 engine 可避免量化/build 差异混入架构比较。
3. **配置入口的实时预览有可见成本。** 本次混合 INT8 为 24.129 FPS，对比兼容入口 25.983 FPS，下降 7.14%；应按需要选择预览分辨率。本文没有修改目标提交来优化这一点。
4. **本次不能证明精度不变、长期稳定或多模型扩展效率。** 未测试 ResNet、多个 engine 同时加载、编码输出、FP16 显示和长时间耐久运行。若需要量化微小的抽象层开销，应另做固定输入、固定频率、相同显示画布的控制实验；真实摄像头结果保留为实际使用体验证据。


## 复现与证据

测试脚本：[benchmark_camera_architecture.py](../../scripts/benchmark_camera_architecture.py)。统计脚本：[analyze_camera_architecture.py](../../scripts/analyze_camera_architecture.py)，仅依赖 Python 标准库。

从仓库根目录执行，先将两份固定提交导出到新的缓存目录，再编译程序：

```bash
mkdir -p cache/camera-architecture-5b736f9-20261022/{before,after}
git archive 282645d1bd684df4130409f20ce377a21b4266b0 -o cache/camera-architecture-5b736f9-20261022/before.tar
git archive 5b736f9f308fbb115f73c234b642aba4cafdcf85 -o cache/camera-architecture-5b736f9-20261022/after.tar
tar -xf cache/camera-architecture-5b736f9-20261022/before.tar -C cache/camera-architecture-5b736f9-20261022/before
tar -xf cache/camera-architecture-5b736f9-20261022/after.tar -C cache/camera-architecture-5b736f9-20261022/after
cmake -S cache/camera-architecture-5b736f9-20261022/before/tensorrt-yolo -B cache/camera-architecture-5b736f9-20261022/build-before -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build cache/camera-architecture-5b736f9-20261022/build-before --target yolo_video_cuda benchmark_yolo_cuda -j4
cmake -S cache/camera-architecture-5b736f9-20261022/after -B cache/camera-architecture-5b736f9-20261022/build-after -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DEDGE_BUILD_RESNET=OFF
cmake --build cache/camera-architecture-5b736f9-20261022/build-after --target edge_video yolo_video_cuda benchmark_yolo_cuda -j4
python3 scripts/benchmark_camera_architecture.py \
  --output outputs/camera-architecture-5b736f9-retest \
  --build-before cache/camera-architecture-5b736f9-20261022/build-before \
  --build-after cache/camera-architecture-5b736f9-20261022/build-after
python3 scripts/analyze_camera_architecture.py outputs/camera-architecture-5b736f9-retest
```

使用不同的 `--output` 保存复测，脚本拒绝覆盖已有结果。显示测试需要能访问桌面和 GPU/摄像头；仅测无显示可加 `--suite headless`。完整测试约需 20 分钟。

原始证据保存在仓库忽略的 `outputs/camera-architecture-5b736f9-20261022/`，不会随普通 Git 提交分发：

- [metadata.json](../../outputs/camera-architecture-5b736f9-20261022/metadata.json)：设备、工具版本、engine/二进制哈希及开始/结束时间。
- [provenance.json](../../outputs/camera-architecture-5b736f9-20261022/provenance.json)：源码归档、关键算法文件、编译参数的核验。
- [schedule.json](../../outputs/camera-architecture-5b736f9-20261022/schedule.json) 与 [runs.json](../../outputs/camera-architecture-5b736f9-20261022/runs.json)：实际顺序、完整命令、返回码与 CSV 哈希。
- [summary.json](../../outputs/camera-architecture-5b736f9-20261022/summary.json)：可重算的逐轮与分组统计。
- 同目录下每次运行的 `.csv`、`.log`、`.tegrastats.log`、配置入口的 `.yaml`，以及两份构建日志和测试脚本快照。

独立的 3 次、每次 60 帧显示冒烟测试位于 `outputs/camera-architecture-5b736f9-smoke-20261022/`，不纳入本文正式结果。
