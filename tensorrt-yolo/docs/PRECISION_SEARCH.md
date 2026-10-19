# 单图 INT8 精度搜索

本流程只针对 `data/reference/input.bin` 和冻结的 `data/outputs/tensorrt_fp16_output.bin`，不读取 validation。
这是一套调试工具；通过单图检查不代表数据集精度恢复。现有 baseline 引擎、参考输出、校准缓存不会被搜索覆盖。

## 环境

- TensorRT 8.5.2.2，Python `/usr/bin/python3`（3.8）。不要用默认 Python3.13 运行 TensorRT。
- Polygraphy 0.49.9 私有安装于 `.deps/polygraphy`；其 precision 工具要求 TensorRT>=8.5。
- 已通过本机导入、CLI help，以及小模型实际 Polygraphy INT8+FP16构建和 timing cache 保存测试。
- `colored` 未安装只影响日志颜色，无需安装。未改动系统 Python 包。

安装/查看帮助（从 tensorrt-yolo 执行）：

```bash
/usr/bin/python3 -m pip install --no-deps --target .deps/polygraphy -r scripts/requirements-precision.txt
PYTHONPATH=.deps/polygraphy /usr/bin/python3 .deps/polygraphy/bin/polygraphy debug precision --help
```

该版本只有 bisect/linear，没有 worst-first。不要使用 `python -m polygraphy`，该版本没有 `__main__`。

## 构建器

保持原来的位置参数，新增可选参数：

```bash
build/yolo_build models/yolov8m.onnx engines/debug-v1/yolov8m_int8.plan int8 \
  data/splits/person_temporal_v1/calibration cache/int8-baseline-v1/yolov8m_int8.cache \
  --fp16-layers debug/selected_layers.txt --timing-cache cache/debug-v1/timing.cache
```

不传 `--fp16-layers` 表示无手工约束的 INT8 PTQ baseline，仍允许 TensorRT 自动 FP16 fallback。
名单每行一个解析后的 TensorRT network layer name；支持空行、以#开头的注释；不支持通配符。
未知名、重复名、不适用层、输入输出路径冲突均报错；不静默跳过。
计算精度约束不等同于输出类型约束，本实现不调用 setOutputType，FP32模型输入输出保持不变。
当前 eligible 范围为 convolution/activation/elementwise/softmax/pooling/resize，且输出必须是执行张量、非shape tensor并含浮点输出。

仅导出层清单，不校准或构建：

```bash
build/yolo_build models/yolov8m.onnx /tmp/unused.plan int8 \
  --dump-layers debug/network.tsv --inspect-only
```

实际模型371层，当前规则筛出263层。已有导出：`debug/precision-smoke/network.tsv`。
已生成历史范围的精确名单：`debug/precision-smoke/model9_22.txt` 和 `model15_22.txt`。
原硬编码15–22源码备份及临时baseline构建程序已清理；当前统一使用 `apps/build_engine.cpp` 编译生成的 `build/yolo_build`。

Timing cache加载使用ignoreMismatch=false，成功构建后写回；不支持同一路径的并发构建，不保证命中或构建结果确定性。
相同设备、CUDA/TRT、配置家族内复用；它与 calibration cache 完全不同。不要把跨版本缓存混在一起。

## 单图检查

```bash
/usr/bin/python3 scripts/precision_check.py \
  --engine engines/int8-baseline-v1/yolov8m_int8.plan \
  --input data/reference/input.bin --reference data/outputs/tensorrt_fp16_output.bin \
  --report debug/my-check/metrics.json
```

也可用 `--output path.bin` 直接分析已有raw输出。退出码0=通过、1=精度失败、2=输入/运行错误。
报告及派生output/log路径必须是新路径，避免误覆盖参考文件。

默认gate（可配置）：

- 参考中person为argmax且分数>=0.7的候选，平均person分数误差<=0.03，平均IoU>=0.85，候选最高分类仍为person。
- 先80类argmax再筛person，NMS IoU=0.7，最多300个检测。
- 最终检测一对一匹配，IoU>=0.85、分数误差<=0.05；不允许新增或遗漏person检测。
- 空参考候选、错误shape、NaN/Inf均报错；raw MAE/RMSE/P95/P99只记录，不作为硬阈值。

已有单图结果：冻结FP16自检PASS，新INT8 FAIL（高置信候选平均分数误差约0.14659，平均IoU约0.87207，且漏检）。

## Polygraphy 粗定位

已准备但没有启动完整YOLO搜索：`debug/polygraphy-v1/run.sh`。

```bash
bash debug/polygraphy-v1/run.sh
```

需要新的参数或新实验时：

```bash
/usr/bin/python3 scripts/polygraphy_precision.py \
  --work-dir debug/polygraphy-v2 --mode bisect --direction reverse --max-builds 12
# 检查新目录的 experiment.json、command.json 后执行其 run.sh；或上条命令加 --run。
```

默认按网络后缀粗定位，FP16提升精度。自定义config回调将Polygraphy选择限制到C++相同eligible规则，导出本轮的精确名单。
首次开始前校验模型/校准cache哈希、345张图片集合及预处理源码指纹；cache-only校准器拒绝重新校准，不会用单张调试图片或随机数据校准。
工作目录保留：`search.log`、`timing.cache`、`replay.json`、`experiment.json`。
每轮engine、`fp16_layers.txt`、checker JSON、raw输出和推理日志移动到 `artifacts/good` 或 `artifacts/bad`，文件带迭代后缀。
找到通过的集合后应选择对应good名单交给C++重建确认；Polygraphy与C++构建不能假定逐位相同。
构建预算最多12次（可调整），达到预算退出3，保留已有结果；退出0本身也不代表找到通过集合，必须查看good artifacts/checker结果。
不提供无检查断点续跑；重新执行旧run目录会拒绝覆盖search.log。需要继续时新建目录，或在确认配置完全一致后自行使用Polygraphy replay能力。

bisect依赖近似单调性，不保证找到最小集合，更不保证性能最优。基础设施错误由统一supervisor中断，避免当作精度失败继续缩减。
第一次构建没有timing cache，可能耗时很长；完整搜索可能需要数小时。不要同时跑benchmark或另一搜索。

## C++ 构建器驱动的精筛

```bash
/usr/bin/python3 scripts/precision_search.py --work-dir debug/refine-v1 \
  --seed-layers debug/precision-smoke/model9_22.txt --max-builds 12
```

不传seed则从全部eligible层开始。初始集合必须实际构建且通过checker；失败即停止，不宣称“没有解”。
后续分组删除，再逐层删除，每个候选都真实重建和比较；不依据非单调误差作未经验证的推断。
每次运行使用自己的timing.cache、校准cache副本；每个trial下保存engine、名单、构建/推理日志和metrics。
通过时实时保存 `best.json` 和 `fp16_layers.txt`，预算耗尽仍保留最后已验证的通过结果。
该算法只返回经验证可行的删减结果，不保证全局最少层数，也不自动优化延迟；最终候选仍需benchmark。

## 验证记录

- `debug/precision-smoke/integration.log`：真实网络导出、小模型INT8/FP16约束、timing cache复用、非法层拒绝、真实INT8 checker、Polygraphy构建。
- `debug/precision-smoke/cli-final.log`：完整Polygraphy CLI两轮构建/检查均PASS，timing cache和good artifacts归档成功，预算耗尽正常退出3。
- Python20项、CTest3项通过。Python全套测试使用 `/usr/bin/python3 -m unittest discover -s tests -v`。
- C++测试使用 `ctest --test-dir build --output-on-failure`。
- 未进行validation评估；未声称已获得YOLO最优层名单或最终mixed engine。
