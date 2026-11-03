<!-- 本文比较同公共 C++ builder、2 GiB workspace 下的 FP32/FP16 性能。 -->
# RTMPose-Hand 同构建条件性能对照

本实验新增独立 FP32 engine，与既有第二轮 FP16 engine 比较。两者使用同一个公共 C++ builder 可执行文件与源代码，workspace=2048 MiB，profile MIN/OPT/MAX=1/2/4，TF32关闭。FP32禁用FP16，FP16允许FP32 fallback。

新 FP32 engine 的 SHA：`4a926a062a05307923d3d39548d29bdba51dac9caac72e26f9c9653980af324d`。
FP16 engine 的 SHA：`18a95f59220a896918a085119f2c3916ac69ab609d2cf0ff9491fdf0adc3f57a`。
旧 FP32/trtexec/1 GiB 基线与失败实验不覆盖。新 FP32 相对 ORT 的 SimCC 最大误差为 4.74453e-5，关键点最大偏移为 0.5 px，55个development样本、两路径及全部batch通过旧FP32门槛。

**FP16精度验收仍未通过，本报告不推荐其作为已验收部署版本。性能优势与精度可接受性分别报告。**

## 测量条件与口径

- Jetson AGX Orin，JetPack/L4T R35.4.1；TensorRT版本、源码、配置、binary和engine哈希见结果快照。
- 用户手动执行 `sudo jetson_clocks`，MAXN；CPU 2.1888 GHz，GPU 930.75 MHz，EMC 3.199 GHz。工具不改变用户的时钟设置。
- batch=1/2/3/4；每组预热100次，采样1000次，三轮交替FP32/FP16顺序，总计48000条原始测量。
- OpenCV线程数固定为1。CUDA Graph关闭；使用现有同步 `Runner::inferNamed()`、FP32 linear I/O、持久设备buffer。
- NPY路径：预先加载并打包输入，计时覆盖H2D、推理、D2H及解码。
- image路径：图像预先读入内存，计时覆盖CPU affine/RGB/normalization/打包、H2D、推理、D2H、解码。
- 文件读取、engine/context初始化、固定batch组的shape设置、预热和写盘不计入测量。各组完成后才写CSV。
- CUDA event记录stream上H2D/inference/D2H区间；它不是所有GPU kernel的纯执行时间之和，可能包含提交间隙。CPU预处理、解码、E2E为steady_clock墙钟时间。
- E2E包含实际Runner输出分配、事件记录和同步等开销，到解码完成结束；不包含返回后的结果检查、对象释放或外层循环开销。
- 每组在同一55样本清单上循环，batch可跨不同原图。batch=2的时间不能直接称为一张双手图的业务延迟。
- 全部离群值保留。汇总P50/P95直接对原始样本求分位数，不平均各轮P95。另保留每轮统计观察漂移。
- 吞吐是batch数除以相应测量均值的估算值，不能作为包含采集、detector、磁盘I/O的持续视频FPS。

## 结果

完整48000条采样已通过身份与完整性检查，三轮各组均保留1000次测量。Release编译参数为 `-O3 -DNDEBUG -std=gnu++17`。结果快照：[cpp-2g-performance.json](../benchmarks/cpp-2g-performance.json)。

### NPY路径推理区间（CUDA event）

每项为一个batch的毫秒数；加速比使用P50。

| Batch | FP32 P50/P95 ms | FP16 P50/P95 ms | P50加速比 |
| --- | ---: | ---: | ---: |
| 1 | 6.476 / 10.177 | 2.801 / 4.655 | 2.31× |
| 2 | 9.624 / 13.460 | 3.366 / 5.238 | 2.86× |
| 3 | 13.470 / 17.894 | 4.373 / 6.299 | 3.08× |
| 4 | 15.757 / 19.962 | 5.070 / 8.047 | 3.11× |

### 内存原图→预处理→推理→解码端到端

| Batch | FP32 P50/P95 ms | FP16 P50/P95 ms | P50加速比 |
| --- | ---: | ---: | ---: |
| 1 | 9.765 / 13.520 | 5.227 / 8.900 | 1.87× |
| 2 | 14.445 / 18.786 | 8.144 / 11.986 | 1.77× |
| 3 | 20.720 / 25.126 | 11.508 / 15.377 | 1.80× |
| 4 | 26.149 / 30.988 | 15.615 / 18.904 | 1.67× |

### 性能解释与限制

1. FP16推理P50比FP32快约2.31–3.11倍，但包含CPU预处理的端到端加速为1.67–1.87倍。推理加速不能等比例转换为全链加速。
2. FP16 image路径的CPU预处理均值占E2E均值约37%（batch1）至53%（batch4）。预处理P50约2.17/4.29/6.43/8.59 ms，随手数近似线性增加；解码P50约0.07–0.29 ms。
3. FP16 image路径按E2E均值换算约171/228/245/247 hands/s（batch1/2/3/4）。batch3→4吞吐收益有限，而单批等待时间增加；这只是本同步CPU预处理链的结果，不是所有pipeline的最优batch结论。
4. 尾延迟和轮次漂移不可忽略：例如FP32 batch4 NPY推理各轮P50约14.736/17.711/16.991 ms；FP16约5.060/5.105/5.108 ms。最高单次host E2E为58.431 ms，未剔除。加速比是这三轮的描述统计，不是置信区间或无噪声硬件上限。
5. GPU采样前后min/max/cur均930.75 MHz；623条遥测记录中GPU温度38.125–46.312°C，CPU温度42.843–46.843°C。tegrastats未输出GPU/EMC实时频率数值，因此不能宣称已逐秒证明GPU/EMC完全无频率变化；EMC锁频依据用户提供的jetson_clocks输出。
6. FP16在新的同构建条件FP32对照下仍存在明显关键点偏移，性能收益不能替代精度验收。本轮不调整precision constraints、不做INT8，也不使用最终validation。
7. 采样初段曾并行执行短时CPU构建检查与测试，未删除对应样本；本次为开发工作站环境实测，并未宣称系统独占。最终性能验收应在独占环境复测。
8. 计时统计、实验身份与代码已自查；独立reviewer多次因模型容量不可用而未能完成评审。


## 复现入口与产物

- 公共构建：[builder.cpp](../../trt-runtime/src/builder.cpp)。
- 模型入口：[build_engine.cpp](../apps/build_engine.cpp)，新配置：[rtmpose_hand_builder_fp32_2g.json](../configs/rtmpose_hand_builder_fp32_2g.json)。
- C++采样：[benchmark.cpp](../apps/benchmark.cpp)。
- 数值核验：[verify_cpp_fp32_2g.py](../scripts/verify_cpp_fp32_2g.py)。
- 性能编排：[run_cpp_performance.py](../scripts/run_cpp_performance.py)。
- 统计：[summarize_performance.py](../scripts/summarize_performance.py)。

本地目录 `validation/cpp-2g-performance/` 保存 FP32 构建日志、两路径raw输出、数值核验报告、benchmark原始CSV、运行日志、逐秒tegrastats、采样前后sysfs、电源信息及完成记录。脚本拒绝覆盖已有正式采样。

```bash
cmake --build build-shared --target rtmpose_benchmark -j4
/usr/bin/python3 tensorrt-rtmpose-hand/scripts/run_cpp_performance.py
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/scripts/summarize_performance.py
```

这两个性能入口要求独立FP32 engine及 `consistency.json` 已存在且核验通过；首次GPU构建、原始数值运行命令见本轮计划记录。当前JSON配置仅改变selected_precision，其他字段完全一致，汇总时再次检查。
