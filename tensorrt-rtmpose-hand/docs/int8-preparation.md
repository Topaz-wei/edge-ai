<!-- 本文记录INT8校准器实现和数据供给核验，不代表实际校准或engine构建已完成。 -->
# RTMPose-Hand INT8 PTQ 准备

本文保留准备阶段历史状态。后续已完成首轮校准、构建和开发集验证，精度未通过，见[INT8 基线报告](trt-int8-baseline.md)。

FP16两轮实验均已完成但精度验收未通过，失败结果保留，根因分析暂缓。
INT8是独立实验，不依赖FP16先通过。本轮完成校准器实现和数据完整性检查，尚未执行TensorRT校准计算。

## 配置与公共接口

公共 [builder.cpp](../../trt-runtime/src/builder.cpp) 已支持 `setInt8Calibrator()` 和
`setCalibrationProfile()`，本轮没有修改公共builder或Runner。
模型新入口为 [build_int8_engine.cpp](../apps/build_int8_engine.cpp)，不修改旧FP32/FP16入口。

[独立配置](../configs/rtmpose_hand_builder_int8.json) 固定：

| 参数 | 设置 |
| --- | --- |
| 算法 | IInt8EntropyCalibrator2 |
| INT8 / FP16 / TF32 | ON / OFF / OFF |
| Workspace | 2048 MiB |
| 推理MIN/OPT/MAX | [1,3,256,256] / [2,3,256,256] / [4,3,256,256] |
| 校准MIN/OPT/MAX | 均为[1,3,256,256] |
| 输入输出 | FP32 linear |
| 校准入口 | calibration/calibration_tensor_order.txt |

配置中的cache路径实际生效；meta路径必须为cache路径加 `.meta.json`，算法和顺序文件字段必须符合固定契约。
公共API默认允许INT8的FP16 fallback，模型入口明确设置 `int8Fp16Fallback=false`。

## 已执行的数据检查

- 固定352个校准条目、55个开发条目，无重复或交集。
- 固定已审核的coverage SHA及manifest、split、两份order摘要，拒绝修改顺序、混入开发数据或改写冻结划分。
- 每个校准条目必须为approved/keep且split=calibration；路径位于calibration_set/tensors，不允许重复真实路径或逃逸。
- 352个NPY全部核验SHA、little-endian FP32、C连续[1,3,256,256]、精确payload长度和有限值。
- 不递归扫描calibration目录，不读取validation原图，不对NPY重新归一化、换通道或裁剪。

数据入口实际上从 `F0047-D000.npy` 开始，以固定order为准；不能使用说明示例中的F0001等开发样本作为校准顺序。
本机OpenCV 4.5.4不支持coverage JSON中的null，故使用已审核版本的固定SHA，而不改写coverage或另行推断字段。

## 校准器行为与cache

[calibrator.cpp](../src/calibrator.cpp) 按顺序提供一个FP32张量，校准batch=1。
GPU输入缓冲复用，使用同步H2D，返回时数据已就绪；绑定名必须是 `input`。
所有noexcept回调异常保存为exception_ptr，由构建入口重新抛出；不能把读取失败当作正常数据结束。

正式cache预计写入：

```text
calibration/rtmpose_hand_int8.cache
calibration/rtmpose_hand_int8.cache.meta.json
```

Meta包含模型/reference、手框manifest、split、清单顺序和每个tensor SHA、C++/Python预处理和张量生成实现SHA、
校准器/入口/公共builder/可执行文件/配置身份、TensorRT版本、CUDA头文件与实际runtime版本、算法及profile/精度策略。
实际CUDA runtime版本在GPU阶段读取，CPU audit不初始化驱动。

只接受完整匹配的cache与meta，缺失一半、数据篡改或身份变化时明确失败，不静默复用或覆盖。
cache先在内存中接收，构建成功后检查完整消费352个输入（或有效cache复用），再保存。
engine先写入 `.building` 路径，校准状态检查成功后才发布正式engine；失败残留不是可验收engine。
cache两文件若写入中断，下次会拒绝不完整或不匹配记录；当前入口按串行、独立输出路径使用。

## 实测结果

[准备结果快照](../benchmarks/int8-preparation.json)：

- C++数据audit通过。
- GPU供给回读 **352/352通过，max abs=0**。
- TensorRT版本整数8502，CUDA runtime版本11040。
- 未生成正式校准cache，未构建INT8 engine，未进行Inspector、开发集INT8推理或性能测试。

供给测试调用与TensorRT相同的getBatch接口，但不等于已经验证TensorRT实际校准算法和INT8构建过程；这属于下一步。
CPU测试覆盖错误绑定、回调异常传播、未完成消费拒绝、匹配cache接受/篡改拒绝，以及重复清单、开发集混入和顺序变化拒绝。

## 使用方法

从仓库根目录执行；输出已存在时拒绝覆盖。

```bash
cmake --build build-shared --target rtmpose_int8 test_hand_calibrator -j4
build-shared/tensorrt-rtmpose-hand/rtmpose_int8 audit \
  tensorrt-rtmpose-hand /tmp/rtmpose-calibration-audit.json
build-shared/tensorrt-rtmpose-hand/rtmpose_int8 check-inputs \
  tensorrt-rtmpose-hand /tmp/rtmpose-calibration-gpu-check.json
```

下一步实际构建入口（本轮未执行）：

```bash
build-shared/tensorrt-rtmpose-hand/rtmpose_int8 build \
  tensorrt-rtmpose-hand \
  tensorrt-rtmpose-hand/engines/rtmpose_hand_int8.engine
```

构建后仍须保存完整日志、检查Inspector实际INT8/FP32格式及FP16=0，并验证55个开发样本，不能把成功产出engine当成精度通过。
