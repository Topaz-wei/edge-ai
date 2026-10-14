# CUDA preprocessing → TensorRT → CUDA postprocessing

本版支持固定宽 360、高 640 的 BGR uint8 业务图，engine 必须是单输入 `[1,3,640,640]`、单输出 `[1,84,8400]`，FP32 linear device I/O。内部 FP16 使用现有 FP16 engine；不会根据文件名转换精度或重新构建 engine。

## 构建与运行

从仓库根目录运行，Orin 使用 SM87；其他 GPU 按实际架构更改参数：

```bash
cmake -S tensorrt-yolo -B tensorrt-yolo/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build tensorrt-yolo/build -j4

# CUDA pipeline，默认启用 TensorRT CUDA Graph
./tensorrt-yolo/build/yolo_infer_cuda \
  tensorrt-yolo/engines/yolov8m_fp16.plan \
  tensorrt-yolo/data/images/test.jpg /tmp/yolo-cuda.jpg

# 关闭 Graph，其他处理相同
./tensorrt-yolo/build/yolo_infer_cuda \
  tensorrt-yolo/engines/yolov8m_fp16.plan \
  tensorrt-yolo/data/images/test.jpg /tmp/yolo-cuda-no-graph.jpg --no-graph
```

原有 `yolo_infer`、`benchmark_yolo` 和独立 raw/preprocess/postprocess 程序保持可用。没有 nvcc 时可配置 `-DYOLO_BUILD_CUDA_PIPELINE=OFF` 仅构建原路径。

摄像头可直接复用同一个 pipeline（单线程，调用同步返回）：

```cpp
CudaYoloPipeline pipeline(enginePath, true);
for (...) {
    cv::Mat businessImage = ...; // 360x640 CV_8UC3，可以是有 stride 的 ROI
    auto detections = pipeline.infer(businessImage);
}
```

## 数据路径与实现

- 初始化：Runner 持有 runtime/context/device input/output/stream；CUDA ops 分配图片、reduction sums、排序 scratch、候选/结果和 pinned host staging。每帧无 cudaMalloc/cudaFree、无 event 创建、无 Graph capture。
- mask 边界：初始化时用 OpenCV fillPoly 光栅化两张固定 mask，上传一次；这是复现像素边界的静态配置，每帧不调用 CPU mask 操作。
- 上传：将调用者 BGR 图逐行复制到持久 pinned staging，再 H2D 上传 691,200 bytes。支持非连续 ROI，不修改原图。
- 预处理：GPU 整数求和（单通道总和不超过 58,752,000）→ double 均值并截断 uint8 → headrest mask → 重新求和 → window mask。BGR 通道各自求均值等价于先交换为 RGB 再交换回来；最后一个 kernel 融合左右 140 padding、BGR→RGB、HWC→CHW 和 float32 `/255`，直接写 Runner device input。
- 推理：device input → TensorRT → device output，两个 FP32 张量不落回 CPU。
- Graph：先用已初始化输入做一次普通 enqueue 并同步，随后捕获 **TensorRT enqueue 部分**并实例化。Graph 开启时逐帧 replay。捕获失败会报告错误，不会静默退回普通 enqueue。
- 后处理：GPU 并行遍历 8,400 个候选，选 80 类最高分，再按 person + `score >= 0.7` 筛选。CUB radix sort 使用置信度降序、candidate 索引升序，排序 scratch 一次分配。CPU 参考排序也明确了此同分规则。
- NMS：GPU 上执行 exact greedy NMS（`IoU > 0.7` 才抑制），然后坐标还原/裁剪。当前单线程 NMS 针对本样本仅 9 个有效候选，保留准确语义；不采用近似并行 NMS，不暗中限为 300 个框。支持最多 8,400 个候选/结果，候选密集时该串行 O(N²) 部分可能成为瓶颈。
- 回传：先下载 count 并同步，再只下载 count 个 Detection。每条当前为 44 bytes；单检测样本每帧 D2H 为 4+44 bytes。使用两次同步获得可变长结果。不是零拷贝，未接入 NVMM/EGL 摄像头输入。

`TrtRunner` device API 的地址由 Runner 所有，不能释放；在其 stream 上提交工作。`CudaYoloOps` 低层 API 供测试/组合使用，调用者必须在下一次 upload 复用 pinned staging 前同步上一帧。通常直接使用已管理同步的 `CudaYoloPipeline`。

## 一致性测试

```bash
ctest --test-dir tensorrt-yolo/build --output-on-failure
/home/ssd/anaconda3/envs/py38/bin/python -m unittest discover -s tensorrt-yolo/tests -v

# 以下需要真实 GPU 访问，显式运行，不纳入默认 CPU CTest
./tensorrt-yolo/build/test_cuda_yolo \
  tensorrt-yolo/engines/yolov8m_fp16.plan tensorrt-yolo/data/images/test.jpg
./tensorrt-yolo/build/test_cuda_yolo \
  tensorrt-yolo/engines/yolov8m_fp32_notf32.plan tensorrt-yolo/data/images/test.jpg
```

覆盖：真实图、随机图、常量/零图、非连续 ROI 的逐字节 preprocessing 对比；原图不变、错误几何拒绝；空检测、同分排序、类别 argmax、confidence/IoU 边界、坐标裁剪、512 个未抑制候选；更换帧的 Graph on/off raw 逐字节比较；整个 pipeline 的 timed/untimed 多次复用。使用当前有限数值的 YOLO decoded raw 输出契约。

## 三路径 benchmark

分别运行下面命令，避免并发运行任何 GPU 测试：

```bash
./tensorrt-yolo/build/benchmark_yolo_cuda tensorrt-yolo/engines/yolov8m_fp16.plan tensorrt-yolo/data/images/test.jpg cpu
./tensorrt-yolo/build/benchmark_yolo_cuda tensorrt-yolo/engines/yolov8m_fp16.plan tensorrt-yolo/data/images/test.jpg cuda
./tensorrt-yolo/build/benchmark_yolo_cuda tensorrt-yolo/engines/yolov8m_fp16.plan tensorrt-yolo/data/images/test.jpg graph
```

每个模式先独立运行 CPU 参考并销毁该 Runner，然后构造选定路径，预热 50 次、测量 200 次。预热后和测量末尾的完整 detection 均与 CPU 参考比较。P50/P95 使用 nearest rank；FPS = 1000 / E2E mean(ms)。

E2E 使用 steady_clock，计入 pinned staging CPU copy、上传、预处理、推理、后处理、count/结果下载、同步和每帧内存分配；不包含 imread/JPEG、初始化/capture、输出打印/绘图、参考比较。返回结果 vector 的最终析构位于 E2E 计时外。CPU 模式的临时输入/raw 输出析构位于返回结果前，计入本版统一 benchmark 的 E2E；因此它与旧 benchmark_yolo 的边界略有区别。

GPU 阶段使用同一 stream 上持久 CUDA events，开启/关闭 Graph 均带同样分段计时。事件区间可能包含 CPU 提交造成的 GPU 空闲。上传区间包含 pinned staging 等待；结果下载区间包含 count 回传的主机往返及结果 vector 构造，不能解读为纯 DMA 时间。没有测量零开销的纯 kernel 时间，也没有用 Nsight 证明所有差异来源。

本轮结果见 [2026-10-14 CUDA report](2026-10-14-cuda-pipeline.md)。未锁时钟、未改变功耗设置；TensorRT 提示现有 engine 跨设备型号使用。CPU/CUDA/Graph 的顺序测量也可能受时钟/温度/系统负载漂移影响，不能把 CUDA 预处理在 Graph 模式下的全部变快归因于 Graph（它并未捕获预处理）。

## Nsight Systems / NVTX 准备

Nsight Systems 用于检查整帧的 CPU 提交、CUDA API、GPU kernels、memcpy 和同步时间线；Nsight Compute 用于分析单个 kernel。本机 `nsys --version` 返回 2023.2.4，已经安装，无需重复安装。

CUDA pipeline 已加入如下嵌套 NVTX ranges（包括 warmup 中的调用）：

```text
YOLO Frame
  BGR Upload
  CUDA Preprocess
  TensorRT Inference
  CUDA Postprocess
  Results Download
    Submit Count D2H
    Wait Count Ready
    Submit Boxes D2H
    Wait Boxes Ready
    Build Host Detections
```

`TensorRT Inference` 在 Graph 开启时包围 graph launch，关闭时包围 enqueueV3。上传包含 pinned staging copy；下载包含 count/检测框回传及等待。NVTX 范围表示 CPU 提交/等待的作用域，不是 GPU kernel 执行时长：GPU 实际执行应结合 CUDA stream timeline 查看。作用域对象析构时自动 pop，异常路径不会留下未关闭标记。

使用 `<nvtx3/nvToolsExt.h>` 的 NVTX3 header-only API。CMake 从已找到的 CUDA Toolkit include 路径中查找头文件，通过 `${CMAKE_DL_LIBS}` 链接 Linux 动态加载依赖，不依赖旧版 `libnvToolsExt`。原先无路径提示的 `find_library(nvToolsExt)` 返回 NOTFOUND，正是此次配置失败原因。

重新配置和编译：

```bash
cmake -S tensorrt-yolo -B tensorrt-yolo/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build tensorrt-yolo/build -j4
```

已有缓存中的 `NVTOOLSEXT_LIBRARY-NOTFOUND` 不再被使用，无需删除整个 build 目录。非标准安装可通过 `-DNVTX3_INCLUDE_DIR=/path/to/include` 指定包含 `nvtx3/nvToolsExt.h` 的目录。

下载细分标记中，`Wait Count Ready` 等待同一 stream 上尚未完成的预处理、推理、后处理及 count 复制，不能全部视为 D2H 开销。`Wait Boxes Ready` 在 count 同步完成之后，主要等待检测框回传。无检测框时跳过两个 Boxes 标记。`Build Host Detections` 包含返回 vector 的构造。所有标记使用同一个 RAII 封装，异常时也自动关闭；CPU 提交/等待范围不是 GPU 实际 memcpy 时长，需与 CUDA timeline 对照。重新采集报告后才能看到新增标记，已有报告不会改变。
