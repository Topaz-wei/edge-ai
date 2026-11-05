# RTMPose INT8首轮构建和开发集验证

用户授权完成剩余构建与验证；按executing-plans执行，不提交、不调优。

- 冻结既有低精度门槛（与FP16相同），RMSE和argmax shift新增统计但不临时增加门槛。
- 复用已核验的C++校准器与公共builder；352固定NPY，Entropy2，校准batch1，推理profile1/2/4，2GiB，FP16/TF32 OFF。
- 新建cache/meta/engine与完整日志，核验cache身份及SHA。构建成功不等于精度通过。
- context batch2 inspector：记录INT8/FP32/混合tensor格式，检查无FP16与FP32 I/O；保留缺失scale/权重转换提示。
- 复用既有C++验证器，两路径55样本，batch1/2/3/4及6→4+2。对比ORT、同C++ builder/2GiB TRT FP32及INT8 single。
- 指标包括SimCC MAE/RMSE/max、所有点的坐标/score/valid-mask及argmax shift；保存原始数据和失败报告。
- 本轮不做最终validation、混合精度或性能；按实际结果提出下一步。

## 执行记录（2026-11-05）

- 构建完成：352 次校准 batch，正式 cache/meta/engine 已生成，SHA 和准备身份核验通过。
- Inspector 完成：136 层，94 INT8、34 FP32、8 混合，无 FP16。
- 开发集验证完成：55 样本，双路径、全部 batch；数值验收失败，完整结果保留。
- 新增统计与旧元数据读取测试：20/20 Python；现有 CTest 13/13。
- 最终只读审阅完成，未发现重要正确性问题。
- Ruling: 保留本轮旧 FileStorage 元数据原文并严格兼容读取，不改写已经构建的身份记录；原因是有序张量和设置未丢失且准备快照完全匹配。代价是旧格式的外部读取者仍需兼容，下次升级生产格式必须使用新版身份，不能冒充本轮 cache。
- 不提交 Git；不执行最终 validation、INT8 性能或混合精度实验。
