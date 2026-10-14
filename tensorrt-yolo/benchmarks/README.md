# 常驻 Runtime 与首版 Benchmark

从仓库根目录构建（必须使用 Release，避免把调试构建开销当成部署性能）：

```bash
cmake -S tensorrt-yolo -B tensorrt-yolo/build -DCMAKE_BUILD_TYPE=Release
cmake --build tensorrt-yolo/build -j4
ctest --test-dir tensorrt-yolo/build --output-on-failure
```

## 接口

`TrtRunner` 在构造时加载 engine、创建 runtime/context、分配两个 device buffer、创建 stream 和四个 CUDA events。析构释放资源；构造失败也会清理已创建的 CUDA 资源。仅支持固定形状、一个输入和一个输出、FP32 linear device I/O；engine 内部可使用 FP16。每个 Runner 同步执行，不能被多个线程并发调用。

```cpp
cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
TrtRunner runner(enginePath);
for (...) {
    auto input = preprocess(image);       // 不读取文件、不修改 image
    TrtTimings timing;
    auto output = runner.infer(input, &timing); // 不需要计时时省略第二个参数
    auto detections = postprocess(output, false); // 关闭候选数量日志
}
```

`inferRaw()` 保留为单次调用兼容接口，每次仍创建 Runner，不用于 benchmark。独立 `yolo_infer_raw`、`yolo_preprocess`、`yolo_postprocess_raw` 保持可用。

组合程序保留图片可视化接口：

```bash
tensorrt-yolo/build/yolo_infer \
  tensorrt-yolo/engines/yolov8m_fp16.plan \
  tensorrt-yolo/data/images/test.jpg /tmp/yolo-result.jpg
```

## 测量

下面两个命令串行运行；运行时不要同时启动其他 GPU 测试：

```bash
tensorrt-yolo/build/benchmark_yolo \
  tensorrt-yolo/engines/yolov8m_fp16.plan \
  tensorrt-yolo/data/images/test.jpg

tensorrt-yolo/build/benchmark_yolo \
  tensorrt-yolo/engines/yolov8m_fp32_notf32.plan \
  tensorrt-yolo/data/images/test.jpg
```

- 解码图片、初始化 Runner、一次正确性参考推理均在预热和计时前完成。
- 固定预热 50 次、测量 200 次；同一张图片，每次执行预处理、推理和后处理。最后一次 raw 输出必须与本进程参考推理完全相同，否则返回错误。
- Preprocess、Postprocess、Runner CPU wall 和 E2E 使用 `steady_clock`。
- H2D、Inference、D2H 使用同一 stream 上四个持久 CUDA events；所有 CUDA 调用检查返回值。事件在每帧重复记录，不重复创建。
- Mean 为算术均值；P50/P95 使用 nearest-rank（排序后第 ceil(p*N) 个值）；FPS = 1000 / E2E mean(ms)，表示当前串行处理延迟对应的速率。
- E2E 从预处理开始到后处理结束，包含 host vector 分配、enqueue、同步及 event 计时开销；不含图片解码、engine 初始化、打印、绘制/保存图片、结果比较和循环末尾最终 vector 析构。
- 首版使用 pageable host vector，未做 pinned memory、buffer 复用或流水线优化。CUDA event 测的是 stream 上事件之间的时间段，可能包含 host 提交导致的空闲；不是纯 kernel 时间或纯 DMA 时间。各 GPU 段之和不必等于 Runner CPU wall，CPU/GPU 混合分段也不必严格相加为 E2E。
- FP32/FP16 标签来自所选 engine；FP16 engine 的输入输出仍是 FP32，不按 I/O dtype 推断内部精度。

## 回归验证

CPU CTest 验证 Mat/path 一致、原图不变、非连续 ROI 和非法输入。现有 Python 测试：

```bash
/home/ssd/anaconda3/envs/py38/bin/python -m unittest discover -s tensorrt-yolo/tests -v
```

GPU 测试需要本机 engine 和**重构前**保存的 raw baseline，显式运行（不放入默认 CPU CTest）：

```bash
tensorrt-yolo/build/test_trt_runner <engine.plan> <business_image> <before_raw.bin>
```

它在同一 Runner 上交替执行带计时/不带计时的推理共 5 次，每次与旧版 baseline 做逐字节比较，并检查错误输入拒绝后仍能正常推理。

2026-10-13 本机 FP16、FP32 均通过上述 GPU 回归。FP16 检测仍为 candidate=8191、confidence=0.819824、business box=[146.375,65.875,359.375,538.875]。预处理结果与冻结的 input.bin 逐字节一致。实际测量报告另见同目录基线文件；尚未做 Nsight 定位或定向优化。

## CUDA v3

已新增 CUDA 预处理/后处理与 TensorRT CUDA Graph 路径。构建、测试和三路径 benchmark 见 [CUDA_PIPELINE.md](CUDA_PIPELINE.md)，本轮实测见 [2026-10-14 report](2026-10-14-cuda-pipeline.md)。
