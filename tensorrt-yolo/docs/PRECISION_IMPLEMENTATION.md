# 混合精度搜索实现解析

命令操作见 [PRECISION_SEARCH.md](PRECISION_SEARCH.md)。本文解释调用关系和算法。

## 1. 文件职责

| 文件 | 职责 |
|---|---|
| `include/precision_options.hpp` | 解析 builder 参数、读取精确层名、检查文件别名和临时文件 |
| `apps/build_engine.cpp` | 解析 ONNX、应用精度约束、配置校准和 timing cache、构建 engine |
| `scripts/precision_check.py` | 固定输入的输出比较与 PASS/FAIL 判定 |
| `scripts/precision_search.py` | 调用 C++ builder，以删除 FP16 约束的方式精筛 |
| `scripts/polygraphy_precision.py` | 校验输入，生成新实验目录、命令、配置和 run.sh |
| `scripts/polygraphy_supervise.py` | 开始运行前复核输入，保存日志，遇到基础设施错误终止 |
| `scripts/polygraphy_config.py` | 给 Polygraphy 提供 TRT 配置、缓存和实际约束名单 |
| `scripts/polygraphy_entry.py` | 适配固定版本 CLI，在迭代边界实施构建次数预算 |
| `benchmarks/int8-split-2026-10-16/run_benchmark.py` | 历史基线实验的多轮推理/计时记录，独立于精度搜索 |

## 2. 通用 builder 的执行路径

`main()` 捕获异常，调用 `buildEngine()`。后者依次解析参数、创建 TensorRT 对象、解析 ONNX、遍历网络层、设置精度/缓存、构建和保存。

### 参数与文件保护

旧的位置参数仍可使用：模型、engine、模式，以及 INT8 的校准目录和可选 cache。新增的命名参数是 `--fp16-layers`、`--timing-cache`、`--dump-layers` 和 `--inspect-only`。

`readNames()` 读取每行一个精确名称，去除首尾空白，忽略空行及以 `#` 开头的行，拒绝重复名称。它不展开 `model.9~22`，也不做模糊匹配。

`parse()` 不仅检查显式路径，还检查隐式 `.meta`、`.tmp` 等文件。弱规范化路径与已存在文件的等价检查用于发现符号链接或硬链接别名，避免写输出时覆盖输入。

### 哪些层可以约束 FP16

`isFp16ComputeLayer()` 允许卷积、激活、逐元素运算、Softmax、Pooling、Resize，并检查输出是否属于浮点执行 tensor，排除 shape tensor。网络层总数与可约束层数不是同一概念。

遍历时对精确命中的层调用 `setPrecision(kHALF)`；有约束时打开 `kOBEY_PRECISION_CONSTRAINTS`。未知名称、不支持的类型或重名都会报错。没有设置输出类型强制转换；网络边界仍使用 FP32。

移除某层的 FP16 约束只表示恢复 TensorRT 的选择权，**不保证该层一定以 INT8 执行**。

### 两种 cache

| Cache | 保存内容 | 本实现用途 |
|---|---|---|
| calibration cache | INT8 激活量化所需校准信息 | 避免重复遍历校准图片 |
| timing cache | tactic profiling 信息 | 减少重复构建的测量开销 |

C++ builder 先设置构建配置，再关联 timing cache。构建成功后序列化更新的 cache，通过临时文件保存；engine 也先写临时文件再重命名。不能假定 cache 让所有构建耗时相同，也不能假定独立构建逐位一致。

## 3. 单图判定器如何工作

`precision_check.evaluate(reference, candidate, ...)` 的参考是冻结的 FP16 输出，不是人工框。

1. 校验输出形状、有限值以及阈值范围。
2. 从参考输出选出类别为 person、置信度至少 0.7 的候选索引。
3. 在相同索引比较两份输出：平均分数误差不超过 0.03、平均框 IoU 至少 0.85，且候选类别仍为 person。
4. 两份输出分别按 NMS IoU 0.7 做检测后处理。
5. 检测数量须一致，并存在满足 IoU 至少 0.85、分数误差不超过 0.05 的一对一匹配。

`match_detections()` 构造二分图，使用可重新分配的匹配过程，而非只匹配第一个可用框；这样不会因早期选择占用另一个框的唯一匹配而误判。

原始 bbox/class 误差统计写入报告，但不直接作为最大误差门槛。退出码 0 表示通过，1 表示精度不通过，2 表示输入或运行错误。第三种情况不能作为普通 FAIL 继续缩小层集合。

## 4. Polygraphy 粗搜索的调用链

```text
polygraphy_precision.py
  → experiment.json / command.json / run.sh
  → polygraphy_supervise.py
  → polygraphy_entry.py
  → Polygraphy debug precision
  → polygraphy_config.load_config()
  → engine → precision_check.py → PASS/FAIL
```

准备与实际运行可能相隔很久，所以 supervisor 在执行 `run.sh` 时重新检查哈希和 calibration 来源。它独占创建 `search.log`，避免重复运行污染产物，并管理整个子进程组的退出。

配置回调把 Polygraphy 选择的网络范围过滤成可约束的计算层，再导出 `fp16_layers.txt`。`CacheOnlyCalibrator` 只接受已有 cache；若 TensorRT 要求重新校准，立即报错，防止意外只拿调试图片校准。

`_KEEP` 保留 calibrator/cache 的 Python 所有者，避免原生 TensorRT 仍使用对象时其被回收。CLI 适配器在完整迭代结束后计数，预算耗尽返回 3；结束时恢复被包装的方法并清理引用。

当前固定版本使用 bisect/linear。不能把旧版文档中的 worst-first 当作此安装版已有能力。bisect 返回的是搜索到的范围边界，不保证非连续最小集合，也不保证精度对范围大小单调。

## 5. C++ 精筛为何还要重建 seed

`prune(seed, check, max_trials)` 先对完整 seed 调一次 `check()`。这一次会真正经过 C++ builder，验证部署构建器是否也通过。

随后以约一半的组宽删除约束：删除成功就接受更小的集合并重新尝试；一轮没有成功则缩小组宽；组宽为 1 且所有删除都失败时停止。到达预算立即返回当前 best，并标记 `budget_exhausted=True`。

例如预算 12 用完只说明运行了 12 次检查，不能声称每个剩余层都必不可少。即使单层删除均失败，也只是该贪心路径下的局部结果。

`check()` 为每次尝试建立 `trial-NNN`，保存名单、构建日志、engine、推理输出和判定报告。每次通过就立即更新 `best.json` 和顶层名单。脚本不支持在原目录断点恢复；新目录从已有最佳名单开始仍会先重建验证。

## 6. 测试阅读建议

- `test_precision_options.cpp`：合法参数、未知参数、别名、重复名单。
- `test_precision_check.py`：输出门槛、检测匹配、错误报告路径保护。
- `test_precision_search.py`：预算、删除策略以及 supervisor 的错误处理。

先理解这些测试中的反例，再修改阈值或搜索逻辑，可以避免把程序错误与量化误差混在一起。
