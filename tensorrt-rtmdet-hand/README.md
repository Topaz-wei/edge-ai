<!-- 本文说明 RTMDet-Hand 初始资产来源、原图划分与尚未完成的参考契约。 -->
# RTMDet-Hand

本模块准备手部检测模型的部署与量化资产。当前已完成模型导入、原图分组和完整性检查；
尚未冻结检测模型预后处理，未生成校准张量、calibration cache 或 TensorRT engine。

## 模型

入口：[models/rtmdet_hand.onnx](models/rtmdet_hand.onnx)，通过相对软链接引用
`../tensorrt-rtmpose-hand/models/rtmdet_hand.onnx`，不复制权重。
原始文件来自 vh3 项目：

```text
/home/ssd/code/vh3/src/py_algorithm/comfort_sensing_air_conditioner/data/weights/rtmdet_hand.onnx
```

SHA-256：`568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2`。
上游官方模型身份尚未核验，不将 RTMPose 的官方来源证明套用于此模型。

| Tensor | 数据类型 | ONNX 声明形状 |
| --- | --- | --- |
| input | FP32 | `[1,3,320,320]` |
| dets | FP32 | `[1,num_detections,5]` |
| labels | INT64 | `[1,num_detections]` |

ONNX checker 与 ORT CPU batch=1 零输入推理已通过，输出均有限。
这只证明模型可执行，不代表真实图像预处理正确。输出第二维是符号维，
后续需核验 TensorRT 的实际输出维度与 INT64 labels；当前公共 runtime 的
FP32 I/O 边界不能直接当作已支持本模型。完整记录见 [模型清单](models/manifest.json)。

## 数据集

沿用 RTMPose 已冻结的**原图采集组划分**，不使用手部裁剪、bbox 或姿态张量作检测输入。
一张原图是一个样本，包含此前检测器未检出手部的图片；零检测不等于无手GT。

| 集合 | 原图数 | 来源（comfort 表示 comfort_sensing_air_conditioner） |
| --- | ---: | --- |
| calibration | 285 | comfort/camera/calibration/v2/rgb：16；data_collect/10/images：66；data_collect/czq/images：203 |
| development | 46 | comfort/camera/calibration/rgb：46 |
| validation | 258 | comfort/data_collect/zbx/images：203；comfort/camera/calibration/v1/rgb：25；Thermal_Comfort_Algorithm：30 |

Thermal 的30张来自 camera/calibration 下 rgb（10）、v1_H/rgb（4）、v2_H/rgb（7）、
v3_H/rgb（7）、v1_H/t（2）。完整来源以 [datasets/split.json](datasets/split.json) 为准。
这保证原图ID和文件哈希不跨集合重复，不额外承诺人物或采集场次完全独立。

```text
datasets/
├── calibration_set/
│   ├── images/             # 平铺软链接：CAL0001.jpg ～ CAL0285.jpg
│   ├── manifest.jsonl      # 原始来源、当前路径、尺寸及SHA-256
│   └── image_order.txt     # 固定原图读取顺序
├── development_set/        # images/DEV0001.jpg ～ DEV0046.jpg
├── validation_set/         # images/VAL0001.jpg ～ VAL0258.jpg
├── split.json
└── verification.json
```

编号按冻结原图顺序生成，manifest 保留 sample_id、原始 image_id、文件名、采集组及来源哈希。
只改变链接名称和布局，不改变集合、内容或顺序；保留来源图片的文件扩展名。

顺序表中的路径相对 `tensorrt-rtmdet-hand/` 根目录。校准阶段只读取 calibration_set/image_order.txt，
不要递归读取整个 datasets。三组共589张，均可解码，589个文件哈希唯一。
目前没有人工检测框GT，历史RTMDet预测不能作为独立真实标注。
validation本轮只做导入/完整性检查，未运行模型或据其调参。

模型和图片软链接依赖仓库现有 RTMPose 本地资产；单独复制本目录不能获得完整二进制数据。
链接及大型资产不提交Git，元数据和准备脚本可提交。换机器须先恢复来源资产，再运行：

```bash
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmdet-hand/scripts/prepare_assets.py
```

脚本核对导入哈希、集合隔离与图片解码；已有链接必须指向预期来源，否则拒绝覆盖。
重复运行会重建清单和检查报告，不修改来源文件。

## 下一步

核验并冻结 RTMDet 专用 resize/padding、颜色顺序、归一化、输出解码与筛选规则，
建立真实图像 ORT reference，随后从285张校准原图生成正式输入张量。
RTMPose 的RGB/256×256/affine规则不能直接套用到RTMDet。
