<!-- 本文记录 FP16 分段激活诊断，检查点误差不等同于单层敏感性。 -->
# FP16 分段 Activation Tensor 诊断

本实验针对 FP16 精度退化，暂停 INT8 调层。使用 55 个 development 样本，batch=1；
公共 C++ Builder、Workspace 2 GiB、TF32 OFF，推理 profile 仍为 1/2/4。
未使用最终 validation，也不修改 reference、原始 ONNX 或原始 FP32/FP16 engine。

## 诊断方式

派生 ONNX 仅新增中间输出声明，node 和 initializer 的序列化内容与原模型逐项相同。
16 个检查点覆盖 backbone 分段、Head 入口、归一化、GAU 矩阵计算、残差和 SimCC。
按 tensor 名称、shape 对齐，不能使用融合后的 engine 层序号对齐。

| Tensor | ONNX 节点 | batch=1 形状 |
| --- | --- | --- |
| 413 | Concat_38 | 1×96×64×64 |
| 486 | Concat_95 | 1×192×32×32 |
| 559 | Concat_152 | 1×384×16×16 |
| 624 | Mul_204 | 1×768×8×8 |
| 625 | Conv_205 | 1×21×8×8 |
| 644 | Div_219 | 1×21×64 |
| 647 | MatMul_221 | 1×21×256 |
| 658 | Div_228 | 1×21×256 |
| 663 | Mul_232 | 1×21×1152 |
| 677 | MatMul_241 | 1×21×21 |
| 681 | Mul_245 | 1×21×21 |
| 682 | MatMul_246 | 1×21×512 |
| 685 | MatMul_248 | 1×21×256 |
| 687 | Add_250 | 1×21×256 |
| simcc_x/y | MatMul_251/252 | 1×21×512 |

所有诊断 I/O 使用 FP32 容器；FP16 engine 内部仍由 TensorRT 选择精度。
标记额外输出可能改变融合、计算精度或 tactic，因此必须同时比较诊断 engine 与原 engine 的最终输出。
本表遵循拓扑顺序，但相邻检查点不一定是直接串行关系；报告的 `checkpoint_inputs` 保留最近上游探针连接。

## 指标与异常规则

逐样本、逐检查点使用 FP64 计算 MAE、RMSE、Max Abs、Relative L2、Cosine、Norm Ratio。
同时记录两侧范数、min/max、零值比例和非有限元素数量。
误差汇总 median/P95/max；Cosine 重点观察 median/P5/min。最终 SimCC 额外按 21 个关键点分别统计。

Relative L2 = norm(actual-reference)/norm(reference)；Norm Ratio = norm(actual)/norm(reference)。
Cosine = dot(reference,actual)/(norm(reference)×norm(actual))。
零范数时相关指标为 null，并区分双方全零、仅 reference 全零、仅 actual 全零。
非有限元素不静默忽略，标记对应比较不可定义；不通过筛除低置信度点美化最终输出误差。

Cosine 对整体缩放不敏感；例如 actual=2×reference 时 Cosine 仍为 1，因此不能单独使用。
探针误差包含上游累计误差，不直接证明该算子敏感；候选层需通过移除探针后的局部 FP32 回退实验验证。

## 来源与复现

- [准备脚本](../scripts/prepare_activation_probe.py)：冻结源模型、样本、派生模型、C++ 实现与可执行文件哈希；ORT batch1/2 核验输出形状。
- [C++ 诊断入口](../apps/activation_probe.cpp)：复用公共 builder/runtime，构建和采集相互独立。
- [采集封装](../scripts/capture_activation_probe.py)：运行前后核验 engine/config，记录每份 raw 文件哈希，拒绝覆盖。
- [指标函数](../scripts/activation_metrics.py)与[分析脚本](../scripts/analyze_activation_probe.py)：指标、扰动、图表、已知异常点复现。
- 原始结果保存在 `validation/fp16-activations/`，派生模型、engine 和 raw 不纳入 Git。

首次准备后，依次执行 FP32 和 FP16 的 `rtmpose_probe build-fp32|build-fp16 config.json engine unused`。
正式采集和分析：

```bash
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/scripts/capture_activation_probe.py fp32
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/scripts/capture_activation_probe.py fp16
MPLCONFIGDIR=/tmp/rtmpose-mpl OPENBLAS_NUM_THREADS=1 \
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/scripts/analyze_activation_probe.py
```

采集目录必须不存在；已有完整采集可重复运行分析。`fp32/`、`fp16/` 保留首轮采集；正式分析使用 `fp32-verified/`、`fp16-verified/`。


## 首轮16检查点结果

全部55样本完成 ORT、TRT FP32、TRT FP16 对照，无非有限激活。
ORT 派生输出相对冻结 reference 的最终 SimCC max abs 为 2.08616e-6。
TRT FP32 诊断重建相对原 FP32 engine 的 SimCC max abs 为1.66893e-6，关键点坐标完全一致。

FP16 诊断重建相对原 FP16 engine：SimCC X/Y MAE 0.001074/0.001110，关键点最大变化7.51665px。
这是新增输出和重新选择 tactic 的共同影响，不能全部归因于探针。
原 engine 与首轮诊断 Inspector 有163个同名层，其中72个 tactic 字段不同。
因此本轮排名只用于选择细查区间，不能直接视为正式 engine 的敏感层定位。

三个大偏移点均复现（误差相对各自 FP32，单位为输入像素）：

| 样本 / 点（从0编号） | 原 engine | 首轮诊断 |
| --- | ---: | ---: |
| F0012-D000 / 3 | 23.58495 | 24.58150 |
| F0010-D001 / 4 | 19.50000 | 19.55761 |
| F0031-D000 / 0 | 18.00694 | 18.00694 |

关键检查点统计如下，Relative L2 为55个独立样本的中位数；Cosine取最小值。
Cosine 越接近1越相似，但不代表 argmax 一致，也不代表真实GT精度。

| 节点 | Relative L2 median | Cosine min |
| --- | ---: | ---: |
| Concat_38 | 0.005151 | 0.999972484 |
| Concat_95 | 0.004020 | 0.999982631 |
| Concat_152 | 0.004014 | 0.999989117 |
| Mul_204 | 0.005834 | 0.999968230 |
| Conv_205 | 0.007598 | 0.999960696 |
| Div_219 | 0.007252 | 0.999960860 |
| MatMul_221 | 0.008495 | 0.999947418 |
| Div_228 | 0.008561 | 0.999946884 |
| Mul_232 | 0.010009 | 0.999928595 |
| MatMul_241 | 0.002339 | 0.999996141 |
| Mul_245 | 0.004955 | 0.999977535 |
| MatMul_246 | 0.003700 | 0.999988663 |
| MatMul_248 | 0.011983 | 0.999867318 |
| Add_250 | 0.011718 | 0.999885149 |
| MatMul_251 | 0.010583 | 0.999893612 |
| MatMul_252 | 0.011209 | 0.999890341 |

[逐样本 Cosine 热力图](../validation/fp16-activations/cosine-heatmap.svg) ·
[误差曲线](../validation/fp16-activations/error-profile.svg) ·
[完整逐样本 CSV](../validation/fp16-activations/per-sample.csv) ·
[逐关键点 SimCC 指标](../validation/fp16-activations/simcc-keypoints.csv)。

热力图颜色范围很窄，最低 Cosine 仍约0.999867，不能将深色直接解释为严重精度失败。
F0012-D000 的第3点，SimCC Y Cosine 仍约0.999950，但诊断坐标误差24.58px，说明近似等高峰的 argmax 仍可发生远距离跳变。

## 缩小范围的依据

首轮中 `MatMul_246` 到 `MatMul_248` 之间误差增加，但它们并非直接相连：

```text
663 → Split → 664（门控） ─────────┐
            → 665（值分支）       │
681 + 665 → MatMul_246 → 682 ─────┤
                                ↓
                         Mul_247 → 683
                                ↓
                  MatMul_248（权重871）→ 685
                                ↓
                    与647的缩放残差相加 → 687
```

第二轮只导出683、685及SimCC，区分门控后的累计误差和投影执行的额外误差。
派生ONNX计算节点与权重仍不变，ORT最终输出相对原模型变化为0。
同时用固定FP32权重在FP64中做线性拆分：

```text
total      = Y16 - Y32
propagated = X16 × W32 - X32 × W32
execution  = (Y16 - X16 × W32) - (Y32 - X32 × W32)
total      = propagated + execution
```

其中 execution 包含权重转换、计算与 tactic 的差异，不能解释为仅权重误差。
两个分量可能抵消或同向，范数比不是可相加的误差百分比。


## 第二轮结果与当前判断

两个中间检查点的FP32/FP16诊断 engine 均已完成，55个样本正式采集与哈希核验通过，无非有限激活。

| 检查点 | Relative L2 median | Cosine min |
| --- | ---: | ---: |
| Mul_247 | 0.012215 | 0.999859222 |
| MatMul_248 | 0.011823 | 0.999855892 |
| MatMul_251 | 0.010262 | 0.999895052 |
| MatMul_252 | 0.010629 | 0.999867786 |

`MatMul_248` 输入683的相对L2中位数已达到约1.22%，输出685约1.18%。
这修正了首轮只看投影前后远端检查点时可能产生的误判：高误差在进入投影之前已经存在。

线性拆分中，execution/total 的范数比 min/median/P95/max 为
0.08446 / 0.11805 / 0.13520 / 0.14488。
多数输出误差与上游输入传播一致，不能把这组比例解释为“只有11.8%的误差来自本层”，因为误差向量可能相互抵消。
在三个异常样本上，传播分量与总误差范数也接近：

| 样本 | 总误差L2 | 传播分量L2 | 执行残差L2 |
| --- | ---: | ---: | ---: |
| F0010-D001 | 0.074596 | 0.074411 | 0.007229 |
| F0012-D000 | 0.106966 | 0.106444 | 0.013224 |
| F0031-D000 | 0.107644 | 0.106684 | 0.014731 |

三个原大偏移点继续复现：

| 样本 / 点 | 原 engine 偏移 / px | 第二轮偏移 / px |
| --- | ---: | ---: |
| F0012-D000 / 3 | 23.58495 | 23.52127 |
| F0010-D001 / 4 | 19.50000 | 19.55761 |
| F0031-D000 / 0 | 18.00694 | 18.50000 |

但缩小探针**没有消除诊断重建差异**：相对历史FP16 engine，最终坐标最大变化仍为7.51665px，
平均变化约0.3234px，P95为1px。不能据此宣称已经精确重现原engine内部计算，更不能把某层判为已确认根因。
本轮停止继续扩展探针，保留两轮结果作为诊断证据。

当前更合理的候选区间是 `Mul_232 → Split_233 → Mul_247` 的门控分支及其输入，
而不是直接将 `MatMul_248` 单独列为首要回退对象。还需区分门控算子自身误差与其上游累计误差。
执行残差相对较小也不能完全排除该层对近似等高峰 argmax 的影响；局部回退仍需用最终输出验证。
下一步实验前，应先考虑无探针重建对照或进一步约束 tactic，以分离重建变化与诊断改动，再开展局部FP32回退。
本轮未开始强制回退实验，也未改变验收门槛。

[第二轮 Cosine 图](../validation/fp16-activations-narrow/cosine-heatmap.svg) ·
[第二轮误差曲线](../validation/fp16-activations-narrow/error-profile.svg) ·
[第二轮逐样本 CSV](../validation/fp16-activations-narrow/per-sample.csv) ·
[线性误差拆分 JSON](../validation/fp16-activations-narrow/projection-decomposition.json)。

第二轮准备入口为 [prepare_narrow_activation_probe.py](../scripts/prepare_narrow_activation_probe.py)。
采集/分析使用相同工具，并设置 `RTMPOSE_PROBE_DIR` 为 `validation/fp16-activations-narrow` 的绝对路径；
[analyze_projection_error.py](../scripts/analyze_projection_error.py) 生成投影拆分结果。
两轮完整统计和来源摘要见[结果快照](../benchmarks/fp16-activations.json)。

工程验证：C++诊断入口编译通过，CTest 13项通过；Python统计与采集身份测试通过。
独立审阅提出的采集身份绑定、非有限输出中断问题已修复并通过回归测试。
这些工程测试不等于模型精度验收；当前仍处于诊断阶段。

## 原 engine 的关键点人工对照

为便于判断大偏移的实际表现，另提供[FP32/FP16 关键点对照说明](fp16-keypoint-visual-review.md)及
[本地交互评审页](../validation/fp16-keypoint-review/index.html)。该页面使用正式 engine 的原始结果，
不使用本诊断实验的 probe engine 输出。
