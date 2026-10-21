# edge-ai

Jetson 上的模型量化、TensorRT 推理与视觉验证工作区。

## 模块职责

| 目录 | 职责 |
| --- | --- |
| `trt-runtime/` | 与模型无关的 engine 加载、Tensor 信息、buffer/stream、推理和 CUDA Graph |
| `video-pipeline/` | 配置解析、视频采集、顺序模型调度、绘图、录像和计时 |
| `configs/` | 部署组合，路径相对于配置文件目录解析 |
| `tensorrt-yolo/` | YOLO 量化、固定业务预处理、检测后处理、适配器与验证 |
| `tensorrt-resnet/` | ResNet 量化与推理；固定输入路径已复用公共 runtime |
| `cmake/` | 公共依赖发现 |
| `datasets/` | 共享数据；模型目录保存样例或引用，避免复制大型数据集 |

依赖方向为：视频应用 → 模型适配器 → 公共 runtime。视频核心通过接口调用适配器，
不包含 YOLO 算法。公共 runtime 不依赖 OpenCV。
当前只有 YOLO 提供已验证的视频适配器；ResNet 的现有随机 Tensor 参考不能代表真实视频分类。

## 构建

```bash
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release -DEDGE_BUILD_RESNET=ON
cmake --build build-shared -j4
ctest --test-dir build-shared --output-on-failure
```

以上只编译程序，不构建或更改 TensorRT engine。
模型目录原有独立 CMake 入口保留。只构建 runtime：

```bash
cmake -S . -B build-runtime-only \
  -DEDGE_BUILD_YOLO=OFF -DEDGE_BUILD_RESNET=OFF -DEDGE_BUILD_VIDEO=OFF
cmake --build build-runtime-only -j4
```

关闭 `EDGE_BUILD_VIDEO` 可排除视频/GUI 库；YOLO 的 CUDA 路径由
`YOLO_BUILD_CUDA_PIPELINE` 控制，视频适配需要开启 CUDA 路径。

## 视频验证

```bash
build-shared/tensorrt-yolo/edge_video configs/yolo-video.yaml
```

示例读取摄像头 0，处理 300 帧，不显示窗口。使用前按实际来源设置 `source` 与
`input_stage`；原始相机输入采用 YOLO 历史转置规则，不能自动猜测方向。
离线视频及多模型配置见 [video-pipeline](video-pipeline/README.md)。
旧 `yolo_video_cuda` 命令仍可用，保持 business 画布、参数与 CSV 列。

## 模型目录约定

- `apps/`：可执行程序入口；`include/src`：可复用接口与实现；`tests`：验证。
- `configs/scripts`：模型设置与准备/构建/评测工具；`docs`：说明。
- `models`：源权重/ONNX；`engines`：部署引擎；`calibration`：校准清单、配置、说明。
- `data`：小样例或数据集引用；`benchmarks`：可复现精度/性能报告。
- `build/cache/logs/debug/profiles`：按需生成的产物，不强制创建空目录；profiles 指性能分析产物。

engine 仍保存在模型原路径，不因本次提取而重建。每个模型的校准预处理和量化策略
属于模型自身，公共 runtime 不决定这些行为。

新增模型：实现 `edge::video::IModel`、验证 engine I/O 和预后处理、注册 factory，
为结果提供原帧坐标与真实标签，再添加 YAML 实例。多个实例按顺序处理同一帧，
支持加载不同 engine；检测后裁剪的级联与动态 `.so` 插件尚未实现。

[回归记录](tensorrt-yolo/benchmarks/shared-runtime/README.md)
