# RTMPose INT8 校准器与完整性检查

按用户已明确的第一项工作执行：校准器实现、数据完整性检查和独立配置，不构建INT8 engine。
使用 executing-plans 顺序实施，当前工作区保留未提交的性能实验。

1. C++ CPU audit：固定352校准/55开发清单、无重复/交集，校准manifest split/review/hash，NPY shape/dtype/finite，冻结模型/reference及预处理实现身份。
2. 模型专用EntropyCalibrator2：batch1、按名字绑定input、直接H2D不预处理、noexcept捕获并外部重抛、完整消费验证、cache/meta严格匹配和哈希校验。
3. 独立rtmpose_int8入口：复用现有公共builder；INT8 ON，FP16 fallback OFF，TF32 OFF，2048MiB，执行profile1/2/4、校准1/1/1。已有FP32/FP16工具代码不改，避免历史构建身份失效。
4. CPU错误输入/cache测试，GPU只做352个tensor供给与回读一致性测试，不做校准计算，不生成正式cache/engine。
5. 更新v2状态及报告。FP16两轮未通过，根因分析暂缓；INT8为独立实验。最终validation、mixed precision、性能测试均不在本轮。

执行记录：公共builder接口已满足要求，未修改。新增独立模型入口，避免旧builder binary/source身份变化。
CPU测试从缺失实现失败到通过；新增清单负面测试发现OpenCV无法解析coverage null，改为冻结coverage和四项来源SHA后通过。
CUDA runtime版本查询在沙箱无GPU访问时失败，已移至GPU操作入口；CPU audit只记录headers版本，不绕过设备权限。
独立评审提出cache实现身份、冻结划分和配置路径三个问题，均已修正并复核通过。
GPU供给352/352成功，回读差0；未生成正式cache或INT8 engine。
