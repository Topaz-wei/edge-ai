# INT8 engine 验证 — 2026-10-15

使用已有 engine，未重新构建、未修改置信度阈值或源代码。

## 结论

INT8 engine 加载、推理、CUDA Graph 和同 engine CPU/CUDA 路径检测一致性检查通过。但 test.jpg 检测发生回退：FP16 为 1 个 person，INT8 为 0 个。因此当前 INT8 engine 不宜替换 FP16 baseline。

使用 yolo_preprocess 从 test.jpg 重新生成完全相同的输入，yolo_infer_raw 分别导出两份原始输出，均为有限值（无 NaN/Inf）。最高 person 候选均为 8191：

| 项目 | FP16 maxn | INT8 |
|---|---:|---:|
| person confidence | 0.81982421875 | 0.65380859375 |
| 阈值 0.7 以上 person 候选（NMS 前） | 9 | 0 |
| 最终检测数 | 1 | 0 |
| xywh（模型坐标） | [393,302.5,213,473.25] | [392.4375,327.625,213.875,527.25] |

漏检发生在置信度过滤阶段，不是 CUDA NMS 丢框。坐标也存在漂移，不能仅靠降低阈值认定量化精度恢复。

## 同次会话性能采样

命令：`build/benchmark_yolo_cuda <engine> data/images/test.jpg graph`，每个 engine warmup 50 次、计时 200 次；先 INT8 后 FP16，未锁定或改变功耗模式/时钟，未采集温度及并发负载。数据波动明显，以下仅为本次观测，不能作为受控性能结论。

| ms，除 FPS 外 | FP16 maxn | INT8 |
|---|---:|---:|
| Upload mean | 0.185 | 0.189 |
| Preprocess mean | 0.482 | 0.465 |
| Inference mean | 11.076 | 8.116 |
| Postprocess mean | 0.195 | 0.148 |
| Download mean | 0.535 | 0.355 |
| E2E mean | 12.498 | 9.298 |
| E2E P50 | 12.178 | 8.639 |
| E2E P95 | 15.647 | 12.621 |
| FPS | 80.012 | 107.547 |

INT8 没有检测输出，后处理/下载工作量也与 FP16 不同。E2E 排除图片解码、初始化、warmup，包含 staging、传输、GPU 执行、同步及每帧分配。Graph 仅覆盖 TensorRT。

## 复现与下一步

```bash
# 从 tensorrt-yolo 目录执行
build/benchmark_yolo_cuda engines/yolov8m_int8.plan data/images/test.jpg graph
build/benchmark_yolo_cuda engines/yolov8m_fp16_maxn.plan data/images/test.jpg graph
build/yolo_preprocess data/images/test.jpg /tmp/yolo-test-input.bin
build/yolo_infer_raw engines/yolov8m_int8.plan /tmp/yolo-test-input.bin /tmp/yolo-int8-output.bin
```

本目录保存重新生成的 input.bin、两份 output.bin、raw-comparison.json 和 artifacts.json（engine/图片 SHA256）。

尚未检查 engine 内部逐层精度，尚不能确定具体敏感层或校准问题。下一步应检查量化层分布，在独立验证集评估精度，再决定校准策略或敏感层保留 FP16。test.jpg 属于校准集，本次不构成独立精度或 mAP 评估。
