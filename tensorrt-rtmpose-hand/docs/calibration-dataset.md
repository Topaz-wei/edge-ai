# RTMPose-Hand 校准数据集

用户已认可当前手部裁剪和数据划分：保留全部已检出裁剪，包括重叠框；零检测图片不补框。文件直接存放在 `calibration/`，不再使用 `review-v1/` 包装目录。数据划分及手框已确认，预处理参考验证尚未完成，未生成 INT8 cache。

## 数据划分

| 集合 | 来源原图数 | 已认可裁剪数 | 用途 |
| --- | ---: | ---: | --- |
| calibration | 285 | 352 | 统计激活分布、确定 INT8 量化范围 |
| development | 46 | 55 | 独立检查量化误差和调整方案，不参与 calibration |
| validation 车内 | 203 | 尚未生成 | 方案确定后做最终验证 |
| validation 补充场景 | 55 | 尚未生成 | 方案确定后做跨场景验证 |

开发集来自完整的 `comfort_sensing_air_conditioner/camera/calibration/rgb` 目录，占校准来源原图的 13.9%；未随机拆散目录内近似帧。现有感知哈希候选对在划分间无交叉，但不据此声称人物或采集场次完全独立。源图归属记录于 `split.json`。

331 张校准/开发来源原图中有 44 张没有分数 > 0.3 的手框，记录为 `skip`，不生成手部张量，也不补框；不把它们认定为无手真值。407 个已检出手框全部记录为 `keep`，其中包括 5 对 IoU > 0.5 的重叠框。开发集的 55 个裁剪保持独立。

## 查看与使用

- [逐帧报告](../calibration/index.html)：原图标框和对应 256×256 裁剪，保留原来的 F/D 编号。
- [总览报告](../calibration/overview.html)：校准集、开发集裁剪总览，以及零检测图片和重叠框记录。
- [逐帧审核记录](../calibration/frame_review.csv)、[逐手审核记录](../calibration/hand_review.csv)：已按用户意见更新，无需重复审核。
- [认可记录](../calibration/approval.json)：认可范围、漏检和重叠框处理策略。

HTML 为静态本地报告，可直接用浏览器打开。绿色/橙色框仅表示生成时的自动提示，不表示拒绝。原图坐标使用像素 xyxy（左、上、右、下），手框编号未因目录整理而改变。

`calibration_tensor_order.txt` 包含 352 个校准张量，位于 `calibration_set/tensors/`；`development_tensor_order.txt` 包含 55 个开发集张量，位于 `development_set/tensors/`。顺序表中的路径相对 `calibration/`，例如 `calibration_set/tensors/F0047-D000.npy`。两个集合的裁剪和张量分别存放，使用时仍按各自顺序表读取，以固定顺序并便于复现。

## 目录与追溯

```text
calibration/
├── frames/                       # 标框原图
├── calibration_set/
│   ├── crops/                    # 352 张校准裁剪
│   └── tensors/                  # 352 个 FP32 NCHW 校准输入
├── development_set/
│   ├── crops/                    # 55 张开发裁剪
│   └── tensors/                  # 55 个 FP32 NCHW 开发输入
├── detections/                   # 全部原始检测输出，包括阈值以下结果
├── contact_sheets/               # 裁剪和零检测图片总览
├── provenance/                   # 历史候选清单、生成脚本与迁移哈希
├── index.html
├── overview.html
├── split.json                    # 已认可的原图分组，包括 validation 归属
├── approval.json
├── frames.jsonl
├── hand_boxes.jsonl               # 框、分数、仿射矩阵、张量路径及哈希
├── frame_review.csv
├── hand_review.csv
├── calibration_tensor_order.txt
├── development_tensor_order.txt
├── summary.json
└── verification.json
```

原始图片仍位于模块的 `data/raw/`，`frames.jsonl` 中的 `source_path` 相对模块根目录；报告、裁剪、张量及检测结果路径相对 `calibration/`。此次图片和张量仅移动位置，没有重新生成或改变二进制内容。

旧的导入划分存于 `provenance/import_split.json`，认可前的报告与清单存于 `provenance/review-v1-metadata/`。`data/manifest.jsonl` 继续作为导入时的原始清单，其中的候选集合字段不代表当前划分；当前以 `calibration/split.json` 为准。历史 Python 脚本保留原始路径和哈希，仅用于追溯，不能直接重跑来覆盖当前认可状态。

本次按集合分目录前的清单与报告留在 `provenance/before-set-separation/`；旧路径到新路径的对应关系记录于 `provenance/set-separation-path-map.json`，当前路径的二进制哈希记录于 `provenance/set-separation-binary-sha256.json`。数据归属、手框编号及样本顺序未改变。

## 处理参数与当前边界

RTMDet 使用本地 ONNX、CPUExecutionProvider，FP32，2 个 intra-op 线程和 1 个 inter-op 线程。候选框阈值为 `score > 0.3`，保留模型内 NMS 的结果，不追加第二轮 NMS；零检测图片没有整图手框 fallback。

预处理复用本地 rtmlib 0.0.15。OpenCV 以 BGR 读取，不反转通道；RTMPose 均值 `(123.675,116.28,103.53)`、标准差 `(58.395,57.12,57.375)`，手框扩展 1.25 倍、宽高比调整为 1:1、双线性仿射、越界零填充，输出 FP32 `[1,3,256,256]`。

这些张量与本地参考预处理一致，但历史训练/导出配置尚未完全核实。`frozen=true` 仅确认当前原图分组及手框，不代表预处理或量化精度已验收。预处理参数如有改变，应重新生成张量及哈希，并使已有校准缓存失效。

原始生成检查已确认全部裁剪与张量匹配、数值有限、来源和张量哈希正确。迁移检查另记录于 `verification.json`，确认文件内容未变、报告链接有效、352/55 个张量顺序无交叉，validation 未参与校准。

数据资产沿用 Git 忽略规则保存在本机；复制数据集时应包含 `calibration/` 以及需要追溯的 `data/`、`models/`。本次没有提交 Git。
