# 2026-10-14 CUDA pipeline measurements

Release, GCC 9.4.0, CUDA 11.4.315, OpenCV 4.5.4, SM87, Jetson Linux R35.4.1。
每种模式 50 warmup / 200 iterations；同一 engine、test.jpg、同步单帧路径。
三种模式串行执行；GPU correctness/可视化测试在计时前完成。本任务未同时运行其他 GPU 操作。
没有锁时钟、没有改变功耗模式或采集温度/频率时序，也没有核实系统其他任务负载。
TensorRT 在各模式初始化均提示 engine 跨设备型号使用；本轮保留原 engine 以便同一文件对比，没有重建。

| Mode | E2E mean ms | P50 ms | P95 ms | FPS |
|---|---:|---:|---:|---:|
| CPU preprocess/postprocess | 21.315 | 21.045 | 26.358 | 46.915 |
| CUDA preprocess/postprocess | 12.208 | 11.176 | 14.532 | 81.915 |
| CUDA + TensorRT Graph | 10.516 | 9.752 | 12.775 | 95.089 |

本轮 CPU→CUDA+Graph 观测加速约 2.03 倍，平均延迟降低约 50.7%。
这是单图、未控频环境的观测，不是保证 100 FPS，也不是各优化因果贡献的独立证明。

FP16 与 FP32 均通过 test_cuda_yolo：预处理及 raw 输出逐字节一致，Graph 切换/变更输入帧正确，
检测结构及所有框值与 CPU 完全相同。CPU CTest 和原有 7 项 Python tests 通过。
Graph FP16 可视化输出保存到 /tmp/yolo-cuda-graph.jpg（临时产物，不作为版本库资产）。

## SHA-256

- `engines/yolov8m_fp16.plan`: `5426f5c14084d42f5092e4079069a7d64d640de5a075bb18dd1452c3f7461b7a`
- `engines/yolov8m_fp32_notf32.plan`: `43616ff78f663fc161b1b270d1714a448736f596037cc24680bd768477bc6499`
- `data/images/test.jpg`: `719d2dc2bd0970328c3f618daaa45b07036bf56c8dc0e9b8a5b587ecdb1bfbf6`

## cpu

```text
[TensorRT] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[TensorRT] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
Engine: tensorrt-yolo/engines/yolov8m_fp16.plan
Mode: cpu
Warmup: 50
Iterations: 200
GPU name/stats depend on current clocks and load; no frequency changes applied.
Stage (ms)                       Mean         P50         P95
Preprocess CPU                  2.881       2.677       3.962
H2D event                       0.859       0.832       0.992
Inference event                14.058      13.883      17.207
D2H event                       1.346       1.133       3.502
Postprocess CPU                 1.935       1.886       2.248
E2E wall                       21.315      21.045      26.358
FPS (1000/mean E2E): 46.915
E2E includes staging, copies, GPU work, waits and per-frame allocations.
Excludes decode, initialization, warmup, validation, logging and drawing.
Upload/download event spans include host staging or count round-trip gaps.
Graph covers TensorRT only; GPU transfers BGR + selected detections, not raw tensors.
Final detections match CPU reference exactly.

Detections: 1

Detection 0
candidate : 8191
class     : person
confidence: 0.819824
model box : [286.375000, 65.875000, 499.375000, 538.875000]
business  : [146.375000, 65.875000, 359.375000, 538.875000]
```

## cuda

```text
[TensorRT] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[TensorRT] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
Engine: tensorrt-yolo/engines/yolov8m_fp16.plan
Mode: cuda
Warmup: 50
Iterations: 200
GPU name/stats depend on current clocks and load; no frequency changes applied.
Stage (ms)                       Mean         P50         P95
BGR upload span                 0.193       0.194       0.226
Preprocess event                0.511       0.533       0.642
Inference event                10.890       9.967      13.374
Postprocess event               0.205       0.194       0.199
Results download span           0.372       0.140       2.300
E2E wall                       12.208      11.176      14.532
FPS (1000/mean E2E): 81.915
E2E includes staging, copies, GPU work, waits and per-frame allocations.
Excludes decode, initialization, warmup, validation, logging and drawing.
Upload/download event spans include host staging or count round-trip gaps.
Graph covers TensorRT only; GPU transfers BGR + selected detections, not raw tensors.
Final detections match CPU reference exactly.

Detections: 1

Detection 0
candidate : 8191
class     : person
confidence: 0.819824
model box : [286.375000, 65.875000, 499.375000, 538.875000]
business  : [146.375000, 65.875000, 359.375000, 538.875000]
```

## graph

```text
[TensorRT] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[TensorRT] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
Engine: tensorrt-yolo/engines/yolov8m_fp16.plan
Mode: graph
Warmup: 50
Iterations: 200
GPU name/stats depend on current clocks and load; no frequency changes applied.
Stage (ms)                       Mean         P50         P95
BGR upload span                 0.128       0.125       0.159
Preprocess event                0.306       0.317       0.576
Inference event                 9.614       8.604      11.641
Postprocess event               0.182       0.181       0.194
Results download span           0.269       0.098       0.735
E2E wall                       10.516       9.752      12.775
FPS (1000/mean E2E): 95.089
E2E includes staging, copies, GPU work, waits and per-frame allocations.
Excludes decode, initialization, warmup, validation, logging and drawing.
Upload/download event spans include host staging or count round-trip gaps.
Graph covers TensorRT only; GPU transfers BGR + selected detections, not raw tensors.
Final detections match CPU reference exactly.

Detections: 1

Detection 0
candidate : 8191
class     : person
confidence: 0.819824
model box : [286.375000, 65.875000, 499.375000, 538.875000]
business  : [146.375000, 65.875000, 359.375000, 538.875000]
```

构建、测试、测量边界和实现限制见 [CUDA_PIPELINE.md](CUDA_PIPELINE.md)。
