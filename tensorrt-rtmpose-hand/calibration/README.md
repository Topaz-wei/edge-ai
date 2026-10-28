# RTMPose-Hand 校准与开发数据集

本目录虽然叫 `calibration/`，但同时保存校准集、开发集及两者的审核和追溯文件。**校准裁剪和张量存放在 `calibration_set/`，开发裁剪和张量存放在 `development_set/`，两个集合在目录上分开。** 用户已认可当前划分和全部已检出裁剪，重叠框保留，零检测图片不补框。

```text
calibration/
├── calibration_set/
│   ├── crops/           # 352 张校准裁剪
│   └── tensors/         # 352 个校准张量
├── development_set/
│   ├── crops/           # 55 张开发裁剪
│   └── tensors/         # 55 个开发张量
├── frames/              # 两个集合的标框原图，共 331 张
├── detections/          # 两个集合的原始检测结果
├── contact_sheets/      # 总览图
├── provenance/          # 历史记录
├── calibration_tensor_order.txt
├── development_tensor_order.txt
└── 清单、审核记录及 HTML 报告
```

## 哪些用于 calibration，哪些用于 development

| 集合 | 来源原图数 | 手部裁剪/张量数 | 读取入口 | 用途 |
| --- | ---: | ---: | --- | --- |
| calibration（校准集） | 285 | 352 | [calibration_tensor_order.txt](calibration_tensor_order.txt) | 统计激活分布，确定 INT8 量化范围 |
| development（开发集） | 46 | 55 | [development_tensor_order.txt](development_tensor_order.txt) | 检查 INT8/FP16 一致性、调试量化策略；不参与校准 |
| validation（最终验证集） | 258 | 尚未生成 | [split.json](split.json) 中的两个 `validation_*_image_ids` 字段 | 方案确定后做最终验证；本次未处理这些原图 |

两个 `.txt` 文件每行是一个 `.npy` 张量路径，按行固定读取顺序。路径相对本目录，例如：

```text
calibration_tensor_order.txt → calibration_set/tensors/F0047-D000.npy  # 校准样本
development_tensor_order.txt → development_set/tensors/F0001-D000.npy  # 开发样本
```

做 calibration 时只使用第一份顺序表的 352 个张量，做开发集检查时使用第二份顺序表的 55 个张量。目录区分用途，顺序表固定读取顺序，便于复现；不要递归收集整个 `calibration/` 下的 `.npy`，否则仍会混入开发集。每个张量是一只手，不是一张完整原图；一张原图可能生成多个张量，也可能没有有效检测而不生成张量。

当前来源及编号对应如下。来源均位于模块 `data/raw/comfort_sensing_air_conditioner/` 下：

| 来源相对目录 | 原图编号 | 原图数 | 裁剪数 | 集合 |
| --- | --- | ---: | ---: | --- |
| `camera/calibration/rgb` | `F0001`～`F0046` | 46 | 55 | development |
| `camera/calibration/v2/rgb` | `F0047`～`F0062` | 16 | 26 | calibration |
| `data_collect/10/images` | `F0063`～`F0128` | 66 | 56 | calibration |
| `data_collect/czq/images` | `F0129`～`F0331` | 203 | 270 | calibration |

编号范围用于人工查阅，程序以顺序表及 JSONL 的 `split` 字段为准，不应硬编码编号范围。validation 的 258 张原图包括车内 203 张、补充场景 55 张，与上述集合保持独立。

## 每个文件夹是什么

| 文件夹 | 当前内容 | 是否直接用于 calibration |
| --- | --- | --- |
| `calibration_set/tensors/` | 352 个校准 `.npy`，已完成仿射裁剪和归一化 | **用于校准，按校准顺序表读取** |
| `calibration_set/crops/` | 352 张校准手部裁剪，256×256 PNG，和校准张量一一对应 | 不直接输入校准器；用于查看归一化前的图像 |
| `development_set/tensors/` | 55 个开发 `.npy`，与校准张量格式相同 | 不参与校准；仅用于独立的开发集检查 |
| `development_set/crops/` | 55 张开发手部裁剪，256×256 PNG，和开发张量一一对应 | 不参与校准；用于检查开发集裁剪 |
| `frames/` | 331 张标有检测框和分数的原图预览，包含校准和开发来源，也保留零检测图片 | 不用于校准；图片有绘图覆盖，仅用于检查 |
| `detections/` | 331 个 `.npz`，保存每张原图的 RTMDet 原始 `dets`、`labels`、`resize_ratio`，包括分数低于阈值的检测结果 | 不用于校准；用于复查检测和坐标还原 |
| `contact_sheets/` | 14 张拼图：8 张校准裁剪总览、2 张开发裁剪总览、4 张零检测原图总览 | 不用于校准；用于快速浏览 |
| `provenance/` | 历史导入清单、认可前的清单快照、一次性生成脚本及迁移前哈希 | 不用于校准；用于追溯 |

`contact_sheets/` 中的 `calibration_candidate_*.jpg` 对应校准集，`development_candidate_*.jpg` 对应开发集，`zero_detection_*.jpg` 对应零检测图片。文件名保留生成时的 `candidate` 字样以维持引用稳定，不表示仍未审核；当前认可状态以 `approval.json` 为准。

没有标框的原始图片存放在模块 [data/raw/](../data/raw/)，不在本目录的 `frames/` 中。

## 根目录各文件的作用

| 文件 | 内容及使用方式 |
| --- | --- |
| `calibration_tensor_order.txt` | 352 个校准张量的固定顺序；校准程序的数据入口 |
| `development_tensor_order.txt` | 55 个开发张量的固定顺序；开发集检查的数据入口 |
| `split.json` | 已认可的原图分组：`calibration_image_ids`、`development_image_ids`、`validation_in_car_image_ids`、`validation_supplement_image_ids` |
| `hand_boxes.jsonl` | 每行一只手，共 407 行；记录 `hand_id`、来源原图、`split`、框、检测分数、仿射及逆变换矩阵、裁剪/张量路径、张量 SHA-256 和审核结果 |
| `frames.jsonl` | 每行一张原图，共 331 行；记录 `frame_id`、`source_path`、`split`、尺寸、原图 SHA-256、对应手框编号及零检测/审核状态 |
| `hand_review.csv` | 逐手审核表；当前 407 个手框均为 `keep`，包含保留的重叠框 |
| `frame_review.csv` | 逐帧审核表；287 张有检测的原图为 `keep`，44 张零检测原图为 `skip` |
| `approval.json` | 用户认可的数据数量、划分及处理策略：保留全部已检出裁剪，漏检不补框、重叠框不剔除 |
| `summary.json` | 生成环境、预处理参数、集合数量、模型/来源/当前清单哈希和当前状态 |
| `verification.json` | 迁移后的完整性和集合隔离检查；此前数值检查的位置也记录在其中，不代表量化精度报告 |
| `overlapping_boxes.json` | 5 对 IoU > 0.5 的重叠手框及编号；仅作记录，用户已认可保留 |
| `index.html` | 逐帧报告：标框原图、手部裁剪、集合标签及编号 |
| `overview.html` | 裁剪拼图、零检测原图总览及重叠框记录入口 |
| `spot_check.jpg` | 生成阶段抽样检查的拼图，不覆盖整个数据集，也不决定集合归属 |
| `README.md` | 本目录的说明和使用入口 |

`provenance/` 内的重要文件：

- `import_split.json`：最初导入时的候选划分，当时还未分出开发集，不能作为当前划分使用。
- `review-v1-metadata/`：用户认可前的历史报告、审核表和清单快照，可能仍含 `pending` 等旧状态；不是当前报告入口。
- `migration_binary_sha256.json`：上一次移除 `review-v1/` 时的 1,491 个二进制文件哈希，其中裁剪和张量仍使用混放时的旧路径。
- `before-set-separation/`：本次分目录前的清单、文档及报告快照，仅用于追溯。
- `set-separation-path-map.json`：本次移动的 814 个文件的旧路径到新路径映射，路径均相对本目录。
- `set-separation-binary-sha256.json`：使用当前路径的 1,491 个二进制文件哈希，确认本次分目录没有改变文件内容。
- `build_calibration_review_v1.py`、`check_calibration_review_v1.py`：保留当时源码及哈希的一次性脚本，包含旧目录和旧状态断言，仅供追溯；不要直接重跑覆盖当前状态。

## 如何从一张裁剪找到原图与集合

例如 `F0047-D000`：

1. `F0047` 是原图编号，`D000` 是该原图的原始检测结果索引，不是左手/右手标签，也不是关键点编号。
2. 在 `hand_boxes.jsonl` 中查找 `hand_id=F0047-D000`，可看到 `split=calibration`、手框和路径。
3. 裁剪为 `calibration_set/crops/F0047-D000.png`，模型输入为 `calibration_set/tensors/F0047-D000.npy`。
4. 在 `frames.jsonl` 中查找 `frame_id=F0047`，通过 `source_path` 定位没有绘图覆盖的原始图片。

`F0001-D000` 的查找方法相同，但其 `split=development`，文件位于 `development_set/crops/` 和 `development_set/tensors/`，只出现在开发集顺序表中。检测索引可能不连续，因为阈值以下的检测不会生成裁剪。

路径约定：张量顺序表、`tensor_path`、`crop_path`、`frame_preview` 和 `raw_detection_path` 相对本目录；`frames.jsonl` 的 `source_path` 相对模块根目录 `tensorrt-rtmpose-hand/`。`image_id` 是保留来源项目及目录的追溯标识，不应直接当成本目录下的文件路径。

## 模型输入与当前状态

每个 `.npy` 保存一个 FP32 数组，形状为 `[1,3,256,256]`，即 batch=1、3 个通道、256×256。文件含 NumPy 格式头部，应使用 NumPy 读取或兼容的 NPY 读取器，不能当作无头部的原始浮点二进制文件。数组已完成当前预处理，读取后**不再归一化、不再除以 255、不再做通道交换或图像裁剪**。

当前预处理采用本地 rtmlib 0.0.15：OpenCV BGR 输入、不反转通道；手框扩展 1.25 倍并调整为正方形；双线性仿射、越界零填充；均值 `(123.675,116.28,103.53)`，标准差 `(58.395,57.12,57.375)`。裁剪 PNG 用于查看归一化前的图像，不能代替已准备的张量直接送入模型。

当前用户认可的是数据划分、手框和裁剪内容。`split.json` 的 `frozen=true` 只确认这一范围；预处理参考验证尚未完成，未生成 TensorRT INT8 校准缓存。后续仍需验证真实样本参考输出，再执行 INT8 构建。预处理参数改变时必须重建张量和相关哈希。

44 张零检测图片已按用户意见跳过，不补框，也不将其当作无手真值；它们没有对应的张量。相似裁剪和重叠框继续保留。两个集合都没有标准 21 点人工真值，开发集主要用于推理后端之间的一致性检查，不应将结果称为有真值的真实精度。

## 查看报告与保存资产

用本机浏览器打开 [index.html](index.html) 或 [overview.html](overview.html)。页面为静态报告，点击不会修改审核状态。若复制到其他电脑查看，应携带图片子目录，不能只复制 HTML。

当前划分以本目录的 `split.json` 为准；模块 `data/manifest.jsonl` 是导入时清单，其中的候选集合字段不代表当前划分。图片、检测和张量在上次整理中仅移动，没有重新生成。数据资产按仓库忽略规则保存在本机，克隆 Git 仓库不会自动获得它们。

更完整的准备过程和追溯约束见 [数据集说明](../docs/calibration-dataset.md)。
