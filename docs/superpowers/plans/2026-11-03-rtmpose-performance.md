# 同公共 C++ builder、2 GiB 的 FP32/FP16 性能对照

用户授权直接实施。使用 executing-plans 在当前工作区执行，不提交。

1. 复用未改动的公共 builder/可执行文件，复制配置仅改变 selected_precision=fp32，TF32关闭、workspace2048 MiB、profile1/2/4。
2. 新 FP32 engine 独立命名并保存配置/构建日志/SHA；与ORT进行55个开发样本两路径和batch一致性核验，保留旧基线。
3. 增加模型专用 C++ benchmark：NPY输入路径和内存原图+bbox全链，所有图像与文件读取在计时之外。固定batch1/2/3/4，每组warmup100、iterations1000，3轮交替顺序，保留每次样本时间。
4. CUDA event统计H2D/inference/D2H；steady_clock统计preprocess/decode/host E2E，不把host时间当纯GPU时间。测量包含现有Runner事件和输出分配开销；shape在固定batch组开始设置。
5. 保存电源/时钟/软件/engine/代码身份；尽可能MAXN+jetson_clocks，任何无法满足项如实报告。性能比较不等于FP16精度通过，不运行INT8或最终validation。
6. Python聚合每轮与整体P50/P95、batch吞吐、加速比；测试统计/缺失样本拒绝；代码审阅和报告。

两条路径使用相同固定开发样本顺序；一个batch可能来自多张图，不能直接称为同一张双手图片延迟。

## 实际执行入口

从仓库根目录运行（首次实验；现有产物不可覆盖）：

```bash
mkdir -p tensorrt-rtmpose-hand/validation/cpp-2g-performance
cmake -S . -B build-shared -DEDGE_BUILD_RTMPOSE=ON
cmake --build build-shared --target rtmpose_build rtmpose_benchmark -j4
build-shared/tensorrt-rtmpose-hand/rtmpose_build \
  tensorrt-rtmpose-hand \
  tensorrt-rtmpose-hand/configs/rtmpose_hand_builder_fp32_2g.json \
  tensorrt-rtmpose-hand/engines/rtmpose_hand_fp32_cpp_2g.engine \
  > tensorrt-rtmpose-hand/validation/cpp-2g-performance/build-fp32.log 2>&1
for mode in npy image; do
  build-shared/tensorrt-rtmpose-hand/rtmpose_fp32 \
    tensorrt-rtmpose-hand/engines/rtmpose_hand_fp32_cpp_2g.engine \
    tensorrt-rtmpose-hand/validation/trt-fp32/cases.json \
    "tensorrt-rtmpose-hand/validation/cpp-2g-performance/fp32/$mode" "$mode"
done
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/scripts/verify_cpp_fp32_2g.py
/usr/bin/python3 tensorrt-rtmpose-hand/scripts/run_cpp_performance.py
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/scripts/summarize_performance.py
```

构建前额外保存 `build-inputs.json`：FP32 配置完整内容及 SHA、原 FP16 构建 provenance、原 FP16 engine SHA。
本轮已在源代码变更前核验与原 FP16 相同的 builder executable/source；新增 CMake 仅用于 benchmark target，builder/Runner/reference不改。

执行记录：FP32新engine成功，55样本两路径全部batch旧FP32门槛通过（SimCC max 4.74453e-5，坐标max 0.5px）。
benchmark短测48条通过；统计测试先观察缺少实现失败，随后3项通过。完整CTest12项、Python13项通过。
首次采样启动在写provenance之前因相对__file__路径报错，修为resolve后启动；该失败没有产生正式采样。
用户手动锁频后开始正式采样，读到GPU min/max/cur均930750000Hz，CPU各核2188800kHz。
独立reviewer多次因模型容量不可用失败，本轮暂以逐项自查代替，不声称已完成独立代码评审。


完成记录：正式采样48000条，3轮×4batch×2模式×2精度×1000。warmup每组100。
完整性检查、输入/engine/provenance hash校验通过；13项Python测试、12项CTest通过。
性能结果：NPY推理P50加速2.31–3.11倍，image全链P50加速1.67–1.87倍；FP16精度仍未通过。
用户时钟设置保留，不由工具恢复或修改。原始日志/CSV/telemetry和新FP32 raw输出全部本地保留。
报告 docs/cpp-2g-performance.md，快照 benchmarks/cpp-2g-performance.json；未提交。
