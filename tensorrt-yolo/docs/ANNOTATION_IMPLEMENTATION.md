# Validation 预标注与审核包实现

入口：`scripts/prepare_validation_annotations.py`。它只生成辅助审核材料，不计算模型的最终检测精度。

## 1. 输入与模型

默认图片来自 `data/splits/person_temporal_v1/validation`，权重来自 `models/yolov8m.pt`。脚本要求新输出目录，拒绝覆盖已有审核标签。

读取 split 的 `manifest.csv`，用图片数量和 SHA256 集合检查输入是否正是既定 validation 集合。然后在 Ultralytics 8.3.173 环境加载原始 PyTorch 模型，使用 FP32、关闭 TF32，并设置确定性的 cuDNN 配置。

这一步复用 `prepare_yolo_reference.prepare_reference()` 和 `run_pytorch_reference.select_raw_output()`，保证预处理、固定输入及原始输出形状与已有参考流程一致。没有另走 Ultralytics 的图片预测器或自动下载模型。

## 2. 每张图的处理顺序

```text
原始业务图片 360×640
  → 遮罩、letterbox、RGB、FP32 /255
  → 原始 PyTorch 输出 [1,84,8400]
  → person 候选筛选及 NMS
  → 还原至业务图坐标
  → labels + proposals + preview + review.csv
```

默认预标注置信度 0.25、NMS IoU 0.7。它们是辅助标注参数，不是 validation 固定工作点的 0.7 置信度。

`postprocess()` 先在 80 类中取最大分数类别，再保留 person，因此它不同于“无论其他类得分多少，都只读取 person 分数”。还原后的框会被截到原图边界，退化框被丢弃并计数。

## 3. YOLO 标签坐标

`yolo_line(box, width, height)` 接收原图像素坐标 `(x1,y1,x2,y2)`，输出：

```text
0 center_x center_y width height
```

其中：

```text
center_x = (x1 + x2) / (2 × 图宽)
center_y = (y1 + y2) / (2 × 图高)
width    = (x2 - x1) / 图宽
height   = (y2 - y1) / 图高
```

第一个数字 0 是 person 类别。其余四个是归一化坐标，写入八位小数。没有 confidence 字段；每行对应一个人。

标签属于原始 360×640 图片坐标，而不是 letterbox 后的 640×640，也不是顶部加了标题的预览图坐标。`test_preannotations.py` 检查坐标归一化、非法边界和非有限数值。

## 4. 审核包为什么有多种文件

| 文件/目录 | 用途 | 人工审核时如何处理 |
|---|---|---|
| `images/` | 原图副本 | 不在原图上永久绘框后替换它 |
| `labels/` | 可编辑的 YOLO 标签 | 改框、删框、补框都更新这里 |
| `proposals.json` | 初始模型提议、分数、来源哈希 | 作为来源记录保留，不覆盖人工结果 |
| `review.csv` | 逐图审核状态及备注 | 确认后改为 reviewed，不确定填 needs_review |
| `previews/` | 原图和遮罩图的带框对照 | 静态辅助材料，不会自动跟随标签刷新 |
| `contact_sheets/` | 每组最多 16 张缩略预览 | 用来快速浏览，不能替代逐图审核 |
| `classes.txt` | 单一类别 person | 与标签 class_id=0 对应 |

`save_progress()` 在每张图后保存审核清单；全部初始状态为 pending。后续人工修改不是重新跑生成脚本完成的，否则会冒着覆盖审核结果的风险，因此脚本要求全新的输出目录。

## 5. 当前实验的标注口径

生成时 README 建议检查所有可辨认人物，但最终实验采用用户审核后的标签，其中 `person_3/203` 按用户决定只保留前景人物框 `(0,207,360,515)`，不含后排人物。

因此实际指标必须解释为“与本次审核标签的一致性”，不能声称它测量了所有可见人的检测能力。技术文档保留这种差异，既不替用户补标签，也不追改历史结果。

如果以后改变评估范围，应修改整个数据集的一致标注规则、重新审核并生成新的实验；不要只为了某个 engine 得分更高而改标签。
