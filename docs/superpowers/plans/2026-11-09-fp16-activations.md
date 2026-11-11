# FP16 分段激活诊断

用户已批准：先比较中间激活，包括 Cosine，再依据证据筛选敏感层。

- 固定 55 development、batch1、公共 C++ builder、2 GiB、TF32 OFF、profile1/2/4。
- 原 ONNX/engine/reference 不改。派生 ONNX 只增加具名中间输出；计算节点和权重序列化内容必须相同。
- 首轮分段检查 backbone、归一化、GAU、输出；FP32/FP16/ORT 三方对照。
- 每 tensor/样本输出 FP64 MAE/RMSE/max/relativeL2/cosine/normRatio、范围、零值和非有限统计。
- 明确零范数规则，不丢弃非有限值。汇总逐样本分位数，不拼接样本。
- 先量化诊断 engine 对最终 SimCC 的扰动，确认原三个大偏移样本是否仍复现。
- 若扰动使问题消失，缩小探针范围；不能将当前探针排名直接当作敏感层结论。
- 本轮不自动开始局部 FP32 回退，不运行最终 validation 或性能，不提交 Git。

执行选择：派生 ONNX + 独立 C++ 诊断入口复用公共 builder/runtime，避免修改已冻结公共代码。

进展：16 个探针，ORT 55 样本最终 SimCC 扰动 max=2.08616e-6，node/initializer 完全一致。
独立代码审阅发现采集身份绑定与非有限最终输出两项重要问题，已补充 Python 采集封装和安全比较器。
测试先以缺失接口失败，再通过：test_activation_capture 覆盖 engine 身份变化拒绝及非有限 SimCC 报告。
采用额外 verified 目录重新采集，保留初次输出，不改变正在构建的 C++ 二进制或已冻结 manifest。
原 engine SHA 额外对照已提交的 benchmark；ORT 原始 NPZ 额外对照历史输入锁。

首轮完成：FP32 探针坐标扰动0；FP16 探针相对原 engine 最大坐标扰动7.51665px。
三个原异常点仍复现，但不能把16探针的局部排名直接归因为正式engine敏感层。
按批准方案缩小到 MatMul_248 输入683/输出685（另保留SimCC），独立构建两种精度；ORT最终扰动0。
补充固定FP32权重下的线性误差拆分，分离输入误差传播与执行残差，避免把累积误差误判为本层问题。

解释限制：诊断 engine 相对历史 engine 的变化包含新增输出和重新构建时 tactic 选择变化，不能全部归因为探针本身。
首轮 Inspector 中共有163个同名层，72个 tactic 字段不同；仅作为构建差异证据，不把层名相同等同于计算实现相同。

完成记录：两轮诊断engine与55开发样本均已完成，含FP32/FP16/ORT、逐点SimCC和两次重建扰动比较。
缩小后683相对L2中位数0.0122146，685为0.0118233；执行残差/总误差范数比中位数0.118051。
三个原异常点仍复现，但FP16重建差异最大仍7.51665px，不能宣称已确认正式engine敏感层。
已按计划缩小一次探针；本轮停止继续增加诊断构建，下一步需控制重建差异并验证门控分支候选。
验证：CTest13/13，Python31/31；采集身份、零范数、非有限、线性拆分测试通过。
最终报告：tensorrt-rtmpose-hand/docs/fp16-activation-analysis.md。
未运行最终validation、强制FP32回退或性能；未提交Git。
