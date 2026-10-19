# 新校准划分 INT8 baseline 与 benchmark — 2026-10-16

## 结论

新引擎构建成功，345张校准图片的缓存哈希集合与划分清单完全相符，没有 validation 图片。
引擎为 INT8 PTQ、允许自动 FP16 fallback、FP32 I/O、TF32关闭，没有手工 FP16 层约束。
三种引擎各3轮 CUDA Graph benchmark 均通过同引擎 CPU/CUDA 检测一致性检查。
但固定调试图上新 INT8 person 分数只有0.6674805，仍低于0.7，检测数为0；不能替换 FP16 baseline。

| 引擎 | 推理均值 ms | E2E均值 ms | FPS | 最高person分数 | 检测数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 maxn | 11.809 | 13.261 | 75.41 | 0.8198242 | 1 |
| 旧 INT8（472张） | 8.621 | 9.883 | 101.19 | 0.6538086 | 0 |
| 新 INT8（345张） | 8.697 | 10.004 | 99.96 | 0.6674805 | 0 |

三轮均值为每轮200帧均值的算术平均，FPS=1000/平均E2E。未将各轮P95平均当作总体P95。
相对于 FP16，新 INT8 本次平均推理耗时下降约26.4%，E2E下降约24.6%；检测工作量不同，不能只据E2E判断收益。
新旧 INT8 的速度差异小且运行有波动，不据此宣称新校准改变了性能。

## 产物与构建来源

以下路径相对于 tensorrt-yolo：

- Engine：`engines/int8-baseline-v1/yolov8m_int8.plan`（26.4863 MiB）
- Calibration cache：`cache/int8-baseline-v1/yolov8m_int8.cache`，另有 `.meta`
- ONNX：`models/yolov8m.onnx`
- Calibration：`data/splits/person_temporal_v1/calibration`（345张）
- 验证记录：`build-verification.json`；产物哈希：`artifacts.json`

实验当时，为保留工作区的 model.15–22 修改，曾将提交 `62b46b6` 的无手工约束 builder 单独编译。临时源码、目标文件、可执行文件及改造前备份现已按用户要求删除；历史来源与哈希保留在 `provenance.json`，编译记录保留在 `compile.log`。当前正式入口为 `apps/build_engine.cpp` → `build/yolo_build`。
历史构建命令（当时已成功执行；下列临时程序现已删除，仅用于记录）：

```bash
./benchmarks/int8-split-2026-10-16/yolo_build_baseline \
  models/yolov8m.onnx engines/int8-baseline-v1/yolov8m_int8.plan int8 \
  data/splits/person_temporal_v1/calibration \
  cache/int8-baseline-v1/yolov8m_int8.cache
```

构建日志 `build.log` 保留了 TensorRT 警告：一个 Constant tensor 缺少 scale/zero-point，会回退至非 INT8；83个 weights 出现 subnormal FP16 值。构建成功不代表数值精度达标。本次没有加入 timing cache。

## Benchmark 方法和环境

使用现有 `build/benchmark_yolo_cuda <engine> data/images/test.jpg graph`。
每轮预热50次、计时200次；顺序为 FP16/旧INT8/新INT8，新INT8/旧INT8/FP16，旧INT8/FP16/新INT8。
Graph只覆盖TensorRT。E2E包含预处理、传输、推理、后处理、同步及逐帧分配，排除解码、初始化、预热和日志。

功耗查询报告MAXN，但EMC参数读取报权限错误；`jetson_clocks --show` 因需要root退出1。未切换功耗模式或锁频，不能宣称锁频条件下的受控测试。
设备状态保存在 `power_mode.log`、`clocks.log`、`tegrastats.log`、`processes.log`，各命令时间及返回码见 `commands.json`。

| 引擎 | 轮次 | 推理均值 ms | E2E均值 ms | E2E P50 ms | E2E P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | 1 | 11.776 | 13.251 | 12.691 | 16.421 |
| fp16 | 2 | 11.910 | 13.352 | 13.488 | 16.886 |
| fp16 | 3 | 11.741 | 13.180 | 12.864 | 16.733 |
| int8_old | 1 | 8.602 | 9.866 | 9.423 | 13.070 |
| int8_old | 2 | 8.690 | 9.912 | 9.514 | 13.167 |
| int8_old | 3 | 8.570 | 9.870 | 9.321 | 13.732 |
| int8_new | 1 | 9.116 | 10.454 | 10.566 | 13.785 |
| int8_new | 2 | 8.464 | 9.760 | 9.330 | 13.225 |
| int8_new | 3 | 8.510 | 9.798 | 9.401 | 13.178 |

## 固定输入检查

三个引擎均使用 `data/reference/input.bin` 重新推理，输出保存为本目录的 `fp16_output.bin`、`int8_old_output.bin`、`int8_new_output.bin`；全部为有限值。后处理日志 `postprocess_*.log` 记录相同阈值下的结果。
旧INT8输出与用户指定的 `data/outputs/tensorrt_int8_output.bin` 逐元素完全一致，确认历史基线对应关系。

性能对照使用 `engines/yolov8m_fp16_maxn.plan`，其raw输出与保存的 `data/outputs/tensorrt_fp16_output.bin` 并非完全一致，最大绝对差为3.90625；最高person分数均为0.8198242。报告中的 `bbox_iou_at_fp16_best_candidate` 对照本轮maxn输出；`bbox_iou_8191_vs_saved_fp16` 另对照用户冻结输出。没有覆盖用户原始参考文件。

这是单张 calibration 图片 test.jpg 的回归与性能采样，尚未运行115张 validation、未计算mAP、未做精度层搜索。
后续应对 validation 导出 FP32/FP16/INT8 输出并评价一致性，再开展混合精度搜索；mAP需真实标注。
