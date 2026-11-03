<!-- 本文记录 FP16-enabled engine 的开发集双基线验证与实际层格式，不代表 GT 精度或性能测试。 -->

# RTMPose-Hand FP16 开发集验证

本阶段只开启 TensorRT FP16，允许优化器选择必要的 FP32 fallback，仍关闭 TF32。原始 ONNX、reference、profile 1/2/4、55 个 development 样本、公共 Runner、C++ 预处理及解码均保持不变。未运行 INT8 或最终 258 张 validation。

首次 1 GiB/trtexec 实验已执行，精度门槛未通过。NPY batch=1 对 ORT 的关键点平均/P95/最大偏移为 0.33016/1/24.54588 px，输入差为 0，有效掩码翻转为 0。原始失败报告保留在 `validation/trt-fp16/report.json`。后续公共 C++ builder、2 GiB 新实验见 [专项记录](cpp-builder-2g.md)。

## 配置与边界

[FP16 配置](../configs/rtmpose_hand_trt_fp16.json) 独立于冻结的 reference 和 FP32 配置。参考 SHA-256 为 `3036459e1f6bc230c9059f34ea260070928669f54f9b1bfef3e031279aef0aec`。

首次实验使用 trtexec 和原始动态 ONNX，MIN/OPT/MAX 分别为 1/2/4，workspace=1024 MiB；通过 `--fp16 --noTF32` 构建，不设置强制逐层精度，也不启用 INT8。Engine 的外部 I/O 仍为 FP32，因此可直接复用公共 runtime 的既有边界。

现有 C++ 可执行文件名 `rtmpose_fp32` 保持不变；它没有把计算精度硬编码为 FP32，实际计算由传入的序列化 engine 决定。本次直接传入 `engines/rtmpose_hand_fp16.engine`，不复制或修改旧 C++ 实现。

## 验证矩阵与门槛

两条路径均覆盖全部 55 个开发样本：

```text
A：冻结 NPY → 公共 Runner → TRT FP16-enabled → SimCC
B：原图+bbox → 既有 C++ preprocess → 同一 engine → C++ decode
```

请求 batch 为 1/2/3/4/6，6 按 4+2 分块，尾批不丢弃。每种路径、每个请求 batch 均比较：

- FP16 vs 已冻结 ORT FP32 单样本输出：部署总偏差。
- FP16 vs 同请求 batch、同输入路径的 TRT FP32：开启 FP16 后的额外偏差。
- FP16 batched vs FP16 单样本：动态 batch 一致性。

全部关键点参与比较，不筛掉低分点。对于同一份 FP16 SimCC，再用既有 Python 解码与 C++ 解码交叉检查，以分离推理误差与解码实现误差。

以下是读取任何 FP16 输出前确定的 development 工程门槛，不是 PCK/NME 或 GT 精度标准。对上述三种对照使用相同门槛，失败时保留失败结果，不回调 reference 或事后放宽阈值。

| 指标 | 门槛 |
| --- | ---: |
| 输入张量最大绝对差 | 0 |
| 每个 SimCC 输出 MAE | ≤0.001 |
| 每个 SimCC 输出最大绝对差 | ≤0.05 |
| 输入坐标平均 L2 偏移 | ≤0.25 px |
| 输入坐标 P95 L2 偏移 | ≤0.5 px |
| 输入坐标最大 L2 偏移 | ≤2 px |
| 分数最大绝对差 | ≤0.02 |
| 有效掩码翻转数 | 0 |
| 同一 SimCC 的 C++/Python 解码最大绝对差 | ≤1e-9 |

原图空间的 mean/P95/max 位移、变化关键点数和有效关键点总数也保存，不用原图尺寸差异掩盖输入空间偏移。

## 层精度记录

构建时保存 `layers.json`。由于动态 shape 可能让这个文件的 `Format/Datatype` 显示 `N/A`，另由 [inspect_fp16_engine.py](../scripts/inspect_fp16_engine.py) 在实际 `[2,3,256,256]` context 上导出 `layers-context.json`，并关联 engine SHA-256。

[verify_trt_fp16.py](../scripts/verify_trt_fp16.py) 根据 inspector 可见的层输入/输出格式，分别记录 FP16、FP32、两者混合和未知，并列出 reformat 名称及所有层的格式/tactic。**这是实际 tensor 格式统计，不能用它推断所有融合算子的内部累加精度。** 不把开启 `--fp16` 写成“全模型纯 FP16”，未知信息保持未知。

## 追溯与测试

`prepare` 将以下内容写入本地 `input-lock.json`：原 reference/模型、旧实现源码及二进制、55 个样本/原图、ORT raw、FP32 engine/完成记录和本次 FP16 配置的 SHA-256。重跑准备步骤发现锁不同会拒绝覆盖。

FP16 比较先将报告标为 `incomplete`；只有两种精度、两条路径的完成记录、engine/cases 身份及所需原始文件哈希全部匹配后才比较。FP32 raw 保持只读，不重新生成或修改其基线。

新增 [test_fp16_validation.py](../tests/test_fp16_validation.py) 覆盖低分点参与统计、非有限值拒绝、门槛失败、未知层精度不猜测、旧 engine 身份/缺失证据/篡改输出拒绝。

## 后续实验入口

以下脚本已切换为公共 C++ builder 的 2 GiB 新实验，不会覆盖上述首次结果。
在仓库根目录执行；构建与 GPU 运行需 Jetson 设备权限：

```bash
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/tests/test_fp16_validation.py
bash tensorrt-rtmpose-hand/scripts/build_fp16_engine.sh
bash tensorrt-rtmpose-hand/scripts/run_fp16_validation.sh
```

运行脚本使用系统 `/usr/bin/python3` 的 TensorRT binding 导出 inspector，使用既有 py38 环境执行 NumPy 比较。不要用 `python -O` 禁用断言。

首次实验原始产物位于 `validation/trt-fp16/`，包括构建日志、两份 inspector、清单、锁文件、两条路径的 raw/decoded/完成记录、比较报告。模型与原始产物按既有忽略规则保留本地。此阶段不进行正式 latency/throughput benchmark，也不声称 FP16 相对 FP32 的速度提升。
