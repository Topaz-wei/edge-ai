# 视频文件与摄像头推理

`yolo_video_cuda` 是独立的视频入口，复用已有 `CudaYoloPipeline`，不改变单图程序、engine、遮罩或检测阈值。默认 CUDA Graph 开启，检测口径与已有 CUDA pipeline 相同。

## 1. 构建

在 `tensorrt-yolo/` 下执行：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DYOLO_BUILD_VIDEO=ON
cmake --build build --target yolo_video_cuda -j4
```

视频入口需要 OpenCV `videoio`、`highgui` 及现有 `core/imgproc/imgcodecs`。没有视频开发依赖的机器可以用 `-DYOLO_BUILD_VIDEO=OFF` 保留原有构建。视频目标还要求 `YOLO_BUILD_CUDA_PIPELINE=ON`。

## 2. 命令与输入约定

```text
yolo_video_cuda ENGINE SOURCE business|camera
    [--output result.mp4]
    [--no-display]
    [--no-graph]
    [--max-frames N]
    [--capture-width W --capture-height H]
    [--capture-fps FPS] [--capture-fourcc MJPG]
```

| 参数 | 含义 |
|---|---|
| `ENGINE` | 已构建的固定 640 YOLO engine |
| `SOURCE` | 视频路径；纯数字如 `0` 或 `/dev/video*`、`/dev/v4l/*` 表示摄像头 |
| `business` | 每帧已是 BGR uint8、宽 360、高 640，直接输入 |
| `camera` | 按历史规则转置 H/W，然后等比 resize 到高度 640，结果宽度必须为 360 |
| `--output` | 保存带检测框和耗时文字的 MP4，父目录自动创建；拒绝覆盖已有文件 |
| `--no-display` | 不调用窗口 API，适合 SSH/无桌面运行 |
| `--no-graph` | 使用常规 TensorRT enqueue，仍复用 engine/context/buffer |
| `--max-frames N` | 处理 N 帧后正常退出并关闭输出文件；N 必须是正整数 |

数字文件名请写成 `./0`，否则会被解释为摄像头编号。不要把 `camera` 理解为“自动识别任意相机方向”。

历史变换是 **transpose，不是旋转 90°**。1280×720、1920×1080 等 16:9 横屏尺寸在数学上可以转成 360×640，但仅有比例正确并不保证摄像头方向、视野、裁剪和遮罩位置正确。程序只能检查格式与尺寸，实际几何语义仍需拍摄后确认。

## 3. 使用示例

### 已经转换过的业务视频

```bash
build/yolo_video_cuda \
  debug/trial007_sub_model22/yolov8m_int8.plan \
  data/videos/test.mp4 business \
  --output outputs/test_mixed.mp4 \
  --no-display
```

### 原始摄像头方向的视频

```bash
build/yolo_video_cuda \
  debug/trial007_sub_model22/yolov8m_int8.plan \
  data/videos/raw_camera.mp4 camera \
  --output outputs/raw_camera_mixed.mp4 \
  --no-display --max-frames 300
```

### 未来接入摄像头

```bash
build/yolo_video_cuda \
  debug/trial007_sub_model22/yolov8m_int8.plan \
  0 camera
```

程序默认沿用设备设置；可通过 `--capture-width/--capture-height` 成对指定分辨率，通过 `--capture-fps` 和 `--capture-fourcc` 指定帧率和格式。摄像头使用 V4L2 后端并尝试将缓冲数设为 1；仍不保证没有设备或驱动内部延迟。有桌面时按 `q` 或 Esc 退出；无桌面时用 `--max-frames` 做有限时长测试。默认显示需要正常的桌面/显示服务，SSH 环境请主动加 `--no-display`。

只有 `--no-display` 而没有 `--output` 时，程序只推理和计时，不创建可视化副本，也不绘图。

## 4. 源码执行路径

| 文件 | 职责 |
|---|---|
| `include/video_input.hpp` | 命令行解析、数值检查、camera→business 转换和安全 FPS 计算 |
| `apps/yolo_video_cuda.cpp` | 打开源、建立持久 pipeline、处理帧、绘图、编码和计时 |
| `tests/test_video_input.cpp` | CPU 上验证坐标变换、非法输入、命令行边界和 FPS 退化情况 |

```text
检查输出路径 → 打开 VideoCapture → 读取并检查首帧
  → 创建一次 CudaYoloPipeline → 可选打开 VideoWriter
  → 循环：读取 → 几何转换 → infer → 可选绘图/写出/显示 → 更新计时
  → 释放 writer，输出处理帧数和 EMA
```

首帧格式检查先于 GPU 对象创建，因此无视频、空视频、几何不符等错误会尽早报告。`pipeline.infer()` 不接收画过框的图；绘图使用 `business.clone()`。按退出键的当前帧也计入已处理帧数。

视频输入的 `read()` 返回 false 时，OpenCV 通用接口不能可靠区分正常 EOF 与所有解码错误，所以日志明确记录“EOF or decoder failure”。摄像头中途读取失败按错误返回，不自动重连。

## 5. 时间与输出视频的含义

- **Pipeline EMA**：CPU 单调时钟包围 `pipeline.infer()` 的墙钟时间，包含上传、GPU 预处理、TensorRT、GPU 后处理和结果下载。不是单独 TensorRT kernel 时间。
- **Full loop EMA**：包含读取/解码、几何转换、pipeline，以及启用的绘图、编码或窗口操作。
- EMA 平滑系数为 0.1；显示的 FPS 是 `1000 / EMA毫秒`，不是整段平均 FPS。
- 首帧读取和初始化在循环外，首个 loop 不包含它们；首轮推理仍计入，没有 benchmark 式 warmup。
- 叠加在当前画面上的 loop 指标使用前一帧更新后的 EMA，标记为 `prev`，因为当前帧写出/显示完成前无法知道其完整耗时。

文件按最快处理速度读取，不按原始时间戳节流。输出采用源 FPS；FPS 非有限或 ≤1 时回退为 30。输出编码固定 `mp4v`，不包含音频，也不保留可变帧率时间戳。VideoWriter 缺少逐帧编码成功的可靠状态返回，因此短片测试还会重新解码文件检查完整性。

这不是低延迟多线程摄像头架构，没有捕获队列管理、主动丢旧帧、断线重连或 NVDEC/NVMM 路径。真实相机延迟、缓冲、色彩及编码支持需接入后验证。

## 6. 当前验证及以后调试顺序

CPU 测试检查 business 原样通过、转置像素一致性、1280×720 area resize 一致性、尺寸/类型/参数拒绝和 FPS fallback。可以运行：

```bash
cmake --build build --target test_video_input -j2
ctest --test-dir build --output-on-failure
```

无录制文件时可以生成移动色块的短合成视频来测试读写和连续推理。合成视频仅证明执行流程可运行，不证明人物检测、框抖动或真实流稳定性。

有真实视频后，先无窗口处理少量帧并检查输出：方向、遮罩区域、框边界是否正确；再看 person 分数在 0.7 附近是否闪烁、移动时框是否抖动。最后连接摄像头，确认实际采集尺寸、帧率和长时间运行行为。

本次合成视频 smoke 记录位于本地 `debug/video-smoke-v1/`（不进入 Git）：

- `business_graph.log` / `.mp4`：Graph 开启，6 帧全部处理；输出重新解码为 6 帧、360×640、15 FPS。
- `camera_nograph.log` / `.mp4`：转置输入、Graph 关闭、限制 3 帧；输出重新解码为 3 帧、360×640、15 FPS。
- `verification.json`：上述检查及非法输入/已有输出保护记录。

测试使用 `debug/trial007_sub_model22/yolov8m_int8.plan`，没有重新构建 engine。短片包含首轮开销，日志中的 EMA 不能用于比较 Graph 性能。真实摄像头和桌面窗口尚未测试。


## 7. LRCP F1080P 实机采集配置（2026-10-20）

此次 USB 设备对应 `/dev/video8`，`/dev/video9` 为伴随节点。推荐使用稳定设备路径：

```text
/dev/v4l/by-id/usb-LRCP_F1080P_LRCP_F1080P_000000000001-video-index0
```

设备原设置为 800×600 YUYV/15 FPS，不符合历史 camera 几何。已确认 1280×720 MJPG/30 FPS 可采集，设置请求均被接受；同分辨率 YUYV 最高只支持 10 FPS。指定格式时先设置 MJPG，再设置尺寸和 FPS。

以下命令用于验证**历史模式采集链路**，不是普通办公室横屏图像的正确检测配置：

```bash
build/yolo_video_cuda \
  debug/trial007_sub_model22/yolov8m_int8.plan \
  /dev/v4l/by-id/usb-LRCP_F1080P_LRCP_F1080P_000000000001-video-index0 camera \
  --capture-width 1280 --capture-height 720 \
  --capture-fps 30 --capture-fourcc MJPG \
  --no-display --max-frames 60 \
  --output outputs/camera_historical_check.mp4
```

本次实际画面为办公室横屏场景；历史转置会使人物横倒，车内遮罩也不符合此场景。正式检测前必须决定是保持历史车内安装方式，还是另外实现不转置、不施加车内遮罩的普通摄像头预处理。不得通过强行拉伸或默默改变现有 CUDA 预处理来掩盖差异。

本地诊断材料保存在 `debug/camera-debug-v1/`，包含原始帧、历史转换帧、采集配置和运行日志；不会提交摄像头画面到 Git。
