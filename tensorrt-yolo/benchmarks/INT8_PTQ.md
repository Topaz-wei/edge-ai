# YOLO INT8 PTQ

适用环境：当前 Jetson 的 TensorRT 8.5，使用 `IInt8EntropyCalibrator2` 对无 Q/DQ 的 FP32 ONNX 做隐式 PTQ。启用 INT8 和 FP16 fallback；最终层精度由 TensorRT 选择，不保证所有层都是 INT8。输入/输出仍为 FP32，现有 CUDA pipeline 和 CUDA Graph 可直接使用。

本次只实现代码、编译和 CPU 测试；没有运行真实校准、生成 INT8 engine 或测量 INT8 性能。

## 数据和预处理

`calibration/` 已包含从业务数据目录复制的 472 张图片，保留 `10/images`、`czq/images`、`zbx/images` 结构，来源与哈希见 `calibration/manifest.json`。程序递归读取并按路径排序，batch size 固定为 1。

输入必须是宽 360、高 640 的业务图，不能提前转换为 640×640。校准直接复用 `CudaYoloOps::upload/preprocess`：BGR 上传、headrest mask、重新计算 mean 后的 window mask、左右各 140 的 114 padding、RGB CHW FP32 /255。每批等待预处理完成后，再将 device pointer 交给 TensorRT。

## 编译程序和测试（不会构建 engine）

从 `tensorrt-yolo/` 执行：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DYOLO_BUILD_CUDA_PIPELINE=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

新增依赖为 OpenSSL Crypto（用于缓存 SHA256 校验）。关闭 CUDA pipeline 时仍能编译 FP32/FP16 builder，但 INT8 校准不可用。

可另外执行 GPU 单元测试；使用合成图片检查校准 tensor 与 CPU 预处理逐元素完全一致，不会调用 engine builder：

```bash
build/test_calibrator /tmp/yolo-calibrator-gpu-test gpu
```

## 手动校准和构建 engine

以下命令由你准备好后执行：

```bash
build/yolo_build \
  models/yolov8m.onnx \
  engines/yolov8m_int8.plan \
  int8 \
  calibration \
  cache/yolov8m_int8.cache
```

首次执行会校准全部图片，再构建 engine。成功后保存 `.cache`、`.cache.meta` 和 `.plan`。省略最后一个参数时，缓存路径默认为 `<output.plan>.cache`。输出 plan 已存在时程序拒绝覆盖，请选择新文件名。

缓存元数据绑定 ONNX、图片路径及内容、CUDA 预处理源码、TensorRT/OpenCV 版本和校准配置；数据或模型变化、缓存损坏或缺少 `.meta` 时重新校准。复用时保留 cache 与 meta 两个文件。输入图片错误或校准回调失败会使构建报错，不会把部分数据校准当作成功。

## 构建后的验证和比较

保持功耗模式、时钟、温度及后台负载一致，分别运行：

```bash
build/benchmark_yolo_cuda engines/yolov8m_fp16_maxn.plan data/images/test.jpg graph
build/benchmark_yolo_cuda engines/yolov8m_int8.plan data/images/test.jpg graph
```

这里的 “Final detections match CPU reference exactly” 比较的是**同一 engine 的 CPU 与 CUDA 后处理**，不能据此认定 INT8 与 FP16 精度相同。分别记录 detection、分阶段耗时、E2E mean/P50/P95 和 FPS，不预设加速幅度。

仅测 TensorRT，并检查层的实际精度/格式：

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=engines/yolov8m_int8.plan \
  --useCudaGraph --useSpinWait --noDataTransfers \
  --dumpLayerInfo --exportLayerInfo=engines/yolov8m_int8.layers.json
```

FP16 使用相同参数，替换 engine 和导出文件名。trtexec 数据不包含业务预处理、后处理和传输，不能直接当作 E2E 数据。

当前 test.jpg 也在校准数据中，单图结果仅用于流程检查。mAP/召回率需要独立的、有标注的验证集；当前没有完成 INT8 精度评估。
