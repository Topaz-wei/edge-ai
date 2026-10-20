# 批量验证与指标计算实现

实现由 `scripts/validate_reviewed_engines.py` 和 `apps/infer_batch.cpp` 组成。历史完整结果保存在 [compare-v2 报告](../validation/compare-v2/README.md)。

## 1. 从 Python 到 C++ 的数据流

Python 的 `main()` 完成以下阶段：

1. 要求输出目录不存在，检查 Ultralytics 8.3.173、115 张全部 reviewed。
2. 根据初始预标注记录校验原图 SHA256，读取人工标签并还原成像素 xyxy。
3. 保存标签快照、审核记录及 manifest；记录 engine、脚本、预处理、标签、原图的哈希。
4. 为所有图片生成相同业务预处理下的 FP32 input.bin。
5. 对每个 engine，生成交替输入/输出路径的清单，调用批量 C++ 程序。
6. 对原始输出进行后处理，保存预测，再计算汇总、逐图及分组指标。

脚本目前是固定数据集评估器，不是任意数据集的通用框架：图片形状为 `(640,360,3)`、数量为 115、类别为 person。现有输入校验使用 `assert`，不要用 `python -O` 启动，否则会关闭这些检查。

## 2. 为什么要有 infer_batch.cpp

`yolo_infer_raw` 是一次性接口，每次调用都创建 `TrtRunner`。逐图启动它会反复反序列化 engine。

`yolo_infer_batch` 将 `TrtRunner runner(argv[1])` 放在循环外，复用同一个 engine/context。循环内依然调用已有的 `loadFloatFile()`、`runner.infer()` 和 `saveFloatBinary()`，并不引入另一套推理实现。

清单格式为：

```text
/absolute/path/image_001.input.bin
/absolute/path/image_001.output.bin
/absolute/path/image_002.input.bin
/absolute/path/image_002.output.bin
```

输入和输出路径交替，一行一个，可以包含普通空格，但不能包含换行。当前程序按成对 `getline()` 读取；若最后只有孤立输入行会停止而不是报错。Python 负责生成完整成对清单。本次整理没有改变这个已有行为。

当时中断的逐图运行产生了 63 份输出，它们与批量运行对应输出逐字节一致。该旧目录已按用户要求删除，核对记录保存在 `verification.json`。

## 3. 为什么需要标签快照

`labels_snapshot/` 保存本次运行实际读取的标签，`review_snapshot.csv` 保存当时的审核状态。它们与工作目录中以后可能继续编辑的标签分离。

只保存模型预测而不保存当时的标签，未来就不能可靠复算指标。当前 Git 用独立提交保存 snapshot，再用另一个提交保存代码和实验结果。

`manifest.json` 中的路径描述当时机器上的来源，哈希描述文件内容。换机器后路径可能不存在，但不应修改历史 manifest 来伪装成新运行。

## 4. 后处理及坐标系

输入 `[1,3,640,640]`，输出 `[1,84,8400]`。复用已有 `postprocess()`：80 类 argmax 后筛选 person、做 NMS，再把 letterbox 坐标还原到原图并截到边界。

AP 的预测置信度下限是 0.001，NMS IoU 是 0.7，每图最多保留 300 个预测。这是为了获得覆盖更多阈值的 PR 曲线，不能先只保留 0.7 以上预测再期待完整 AP。

## 5. match()：如何判定框匹配

`match(gt, boxes)` 把人工框和预测框转换成 float32 tensor，调用固定版本的 `box_iou()` 与 `BaseValidator.match_predictions()`。

IoU 是交集面积除以并集面积。IoU 矩阵的行对应人工目标，列对应预测框。使用 0.50、0.55、…、0.95 共十个阈值，各自进行一对一匹配，返回形状为“预测数 × 10”的布尔表。

调用的是 Ultralytics 默认匹配路径，`use_scipy=False`；它按满足阈值的 IoU 匹配并去重。不要把它解释为任意实现都完全相同的 COCO 官方评估器。

## 6. 两套指标为什么要分开算

### 固定工作点：Precision 与 Recall

选出 confidence ≥0.7 的预测，**重新匹配**人工框，在 IoU ≥0.5 下计数：

```text
TP = 成功匹配的预测数
FP = 未匹配的预测数
FN = 未匹配的人工目标数
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

必须重新匹配，而不是直接过滤低阈值的匹配表：否则一个后来被移除的低分框可能已占用人工目标，影响高分框的匹配结果。

代码对分母为零采用 `max(...,1)`，因此无预测时 Precision 为 0。

### 跨置信度阈值：AP 与 mAP

汇总所有图片的低阈值预测及十列匹配表，调用 Ultralytics `ap_per_class()`。函数按 confidence 排序，累积 TP/FP，构建 PR 曲线并计算 AP。

- `map50`：IoU=0.5 下的 AP。
- `map50_95`：十个 IoU 阈值的 AP 均值。
- 当前只有一个 person 类，所以 mAP 等于该类 AP。
- `precision_at_best_f1`、`recall_at_best_f1` 是库在最佳 F1 工作点返回的额外数值，与固定 0.7 的 `precision_07`、`recall_07` 不同。

`ap_per_class()` 返回元组，代码索引 5 为 AP 矩阵，2/3 为最佳 F1 的 P/R。这是依赖固定库版本的重要原因之一。

## 7. 结果怎么阅读

| 文件 | 适合回答的问题 |
|---|---|
| `summary.json` | 两个 engine 的总体差异是什么？ |
| `metrics.json` | 一个 engine 在不同 IoU、不同人物组表现如何？ |
| `per_image.csv` | 置信度 0.7 下，哪张图有误检或漏检？ |
| `predictions.json` | 具体预测框和分数是什么，如何离线复算？ |
| `PR_curve.png` | Precision 与 Recall 随工作点变化的关系如何？ |
| `P_curve.png`、`R_curve.png`、`F1_curve.png` | 置信度阈值变化有什么影响？ |
| `inference.log` | 哪些输入输出被批量 runner 处理？ |
| `manifest.json` | engine、标签、代码和评估参数来自哪里？ |

分组 AP 是在每个人物子集内独立计算，不能用四个分组 AP 的简单平均代替整体 AP。尤其 person_1 只有 3 张，不能把其结果视作充分统计证据。

## 8. 当前实验结果及边界

| Engine | P@0.7 | R@0.7 | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| trial-007 | 100.00% | 93.04% | 99.49% | 92.33% |
| trial007_sub_model22 | 100.00% | 93.04% | 99.49% | 91.99% |

两者 TP/FP/FN 都是 107/0/8，但漏检图片并不完全相同。203 两者都漏检。

这是同人物时间段留出、只有四个人、没有无人负样本的验证。标注最初来自模型提议再经人工审核，且按用户决定的目标范围计算。结果可以比较这两个 engine 在当前标注集上的表现，不能证明新人物泛化能力、全天候误报率或推理速度。

## 9. 构建与重新运行

在 `tensorrt-yolo/` 下：

```bash
cmake -S . -B build
cmake --build build --target yolo_infer_batch -j2

/home/ssd/anaconda3/envs/py38/bin/python scripts/validate_reviewed_engines.py \
  --output validation/compare-new \
  --batch-runner build/yolo_infer_batch \
  --engine trial007=debug/refine-v1/trial-007/yolov8m_int8.plan \
  --engine trial007_sub_model22=debug/trial007_sub_model22/yolov8m_int8.plan
```

必须使用新输出目录。原图、engine 和审核包是本地依赖，Git 中的实验记录并不包含这些大文件。评估脚本使用当前审核包，若要重现历史实验，需核对其标签是否与快照相同。
