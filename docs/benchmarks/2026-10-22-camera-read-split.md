# 720p 摄像头读取阶段拆分实测

## 计时定义

`read_ms` 保留总读取耗时，新增 `grab_ms`、`retrieve_ms`；使用同一组连续时钟边界，所以两项之和等于总计（CSV 舍入误差除外）。三个字段在第一帧均为 0：该帧已在初始化时读取并验证。

本机 OpenCV 4.5.4，上游同版本 V4L2 实现中，`grabFrame()` 调用 `read_frame_v4l2()` 取出缓冲，未就绪时通过 `select()` 等待；`retrieveFrame()` 调用颜色转换，MJPG 分支通过 `imdecode()` 解码，并归还缓冲。API 还会处理输出复制等操作。源码依据：[OpenCV 4.5.4 cap_v4l.cpp](https://github.com/opencv/opencv/blob/4.5.4/modules/videoio/src/cap_v4l.cpp)。这是对匹配版本上游源码的核对，不代表已审计本机发行包的全部补丁。

因此 `grab_ms` 是“等待与取缓冲阶段”，`retrieve_ms` 是“解码与取图阶段”；都是 host wall time，可能包含线程被调度出去的时间，不能分别称为纯等待时间、纯解码 CPU 时间。FFmpeg 等其他后端也可能在 grab 中完成解码。

## 对照方法

真实 USB 摄像头，1280×720、MJPG、30 FPS、2 个采集缓冲；同一 mixed INT8 engine，CUDA Graph 开启，无显示、无视频写盘。保留 MAXN/DVFS，不锁频、不修改曝光。每组 900 帧，丢弃开始 150 帧，只统计随后 750 帧，保留所有离群点。3 轮 C/D 顺序为 CD、DC、CD，共 6 组；这是读取拆分专项无显示复测，并非重跑之前的有显示测试。

- C：串行基线父版本 `5b736f9`，只添加读取拆分计时，独立导出和构建，原历史基线未修改。
- D：双缓冲父版本 `3476fb2` 加本次读取拆分计时。
- 原始数据：`outputs/camera-read-split-20261022/`。`metadata.json` 的 commit 表示父版本；须结合 `instrumentation-provenance.json`、两份 instrumentation patch 和 binary SHA256 识别实际被测代码。
- 串行补丁：[camera-read-split-serial.patch](assets/camera-read-split-serial.patch)。独立源码和构建位于 `cache/camera-read-split-20261022/`。

## 复现

已有上述构建目录时，用新的输出目录运行（脚本拒绝覆盖）：

```bash
python3 scripts/benchmark_camera_architecture.py \
  --output outputs/camera-read-split-repeat \
  --build-before cache/camera-architecture-5b736f9-20261022/build-before \
  --build-after cache/camera-read-split-20261022/build-serial \
  --build-double build-shared --double-commit 3476fb2 \
  --comparison double --suite headless \
  --width 1280 --height 720 --frames 900 --warmup 150 --rounds 3
python3 scripts/analyze_camera_architecture.py outputs/camera-read-split-repeat
```

`--double-commit` 应填写实际父版本，并另存未提交计时补丁；不要把父版本描述为未经修改的被测源码。正常使用不需要改 YAML，重新构建 `edge_video` 后，配置的 `timing_csv` 会自动包含新增列。分析器兼容旧 CSV。

## 实测结果

采集时间（UTC）：2026-10-22T09:27:25.167304+00:00 至 2026-10-22T09:30:54.502777+00:00。6 组成功，共 5400 帧；稳态 4500 帧、4494 个相邻完成间隔。分析器校验每份 CSV 的 SHA256、帧数、帧号、有限数值，以及全量读取拆分非负、求和、首帧为零。

| 指标 | 串行 C | 双缓冲 D |
|---|---:|---:|
| 平均 read_ms | 19.003 | 30.850 |
| 平均 grab_ms | 10.270 | 22.206 |
| 平均 retrieve_ms | 8.732 | 8.643 |
| 三轮平均 FPS | 29.954 | 30.007 |
| 平均 CPU 几何准备 + infer（ms） | 14.367 | 13.618 |
| 平均完成时帧龄（ms） | 56.435 | 55.632 |
| 完成间隔 p95（ms） | 39.436 | 39.291 |
| 完成间隔 p99（ms） | 42.093 | 42.270 |
| 完成间隔 max（ms） | 90.270 | 50.663 |
| 完成间隔 > 50 ms 次数 | 5 | 1 |
| 完成间隔 > 100 ms 次数 | 0 | 0 |
| 完成间隔 > 200 ms 次数 | 0 | 0 |

每轮读取阶段均值（ms）：

| 轮次 | C grab | D grab | C retrieve | D retrieve |
|---|---:|---:|---:|---:|
| 1 | 10.466 | 22.329 | 8.747 | 8.504 |
| 2 | 10.163 | 21.856 | 8.764 | 8.924 |
| 3 | 10.182 | 22.434 | 8.685 | 8.501 |

## 如何解读

双缓冲的 read 平均增加 11.847 ms，其中 grab 增加 11.936 ms，retrieve 减少 0.089 ms。三轮 grab 均增加，新增耗时主要在等待/取缓冲阶段，没有观察到解码/取图明显变慢。结合 30 FPS 输入节拍和 V4L2 职责，这支持“更早请求下一帧，所以等待下一帧更久”的解释；仅靠 API wall time 仍不能把内核等待与调度开销完全分离。

串行流程要等本帧推理结束才开始读取下一帧；双缓冲在本帧推理期间已经开始下一帧 grab。原来晚调用读取时已经流逝的时间，现在变成生产线程 grab 内的等待，与消费线程推理重叠。等待更长不意味着 CPU 多算了这些时间，也不意味着相机变慢。

CPU 转置和缩放属于生产线程 prepare，本次平均 2.441 ms，仍然在双缓冲流水线内。但要先获得下一帧才能执行；输入每约 33.3 ms 才来一帧，本帧 infer 平均约 11.176 ms，很多时候 GPU 工作结束后下一帧才到达，因此主要重叠的是取帧等待，不是下一帧转置缩放。

双缓冲改变跨帧调度，理论上可把受计算限制的稳态间隔由阶段耗时之和缩短到较慢阶段的耗时；它不会自动缩短同一帧各阶段的执行耗时之和。本次两版仍约 30 FPS（29.954 对 30.007，约 +0.18%），不能据此声称实质吞吐提升。CPU 几何准备与 infer 合计的小幅变化也不能单独归因于流水线：DVFS、场景与系统负载未固定，且串行 infer 内包含 CPU 几何，必须使用可比总口径。

本次所有稳态完成间隔均小于 100 ms，未复现历史秒级尾延迟。C 最大 90.270 ms、D 最大 50.663 ms；短时无显示采样不能证明尾延迟已修复，也不能替代有显示路径的验证。帧龄是驱动时间戳到消费完成的诊断量，不是曝光到屏幕延迟。

## 验证

C++ CTest 10/10 通过，覆盖文件 EOF、帧顺序/所有权、准备与推理异常、双缓冲重叠及 CSV 分解；Python 4/4 通过，包含无效拆分数据拒绝与历史 CSV 兼容。读取 API 失败未另建可注入的采集抽象，保留既有错误传播路径。
