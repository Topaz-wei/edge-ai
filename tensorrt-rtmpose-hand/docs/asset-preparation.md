# RTMPose-Hand 模型与候选原图准备记录

2026-10-28：完成部署设计中的模型与原图导入、追溯清单和初步核验。当前资产是候选数据，尚未冻结正式校准/开发/验证划分，也未生成手框、手部输入张量或 engine。

当前数据已获用户认可并整理至 `calibration/`，详见 [校准数据集说明](calibration-dataset.md)。本文保留初次导入时的检查记录；当前划分以 `calibration/split.json` 为准。

## 导入结果

所有路径以 `tensorrt-rtmpose-hand/` 为基准；源绝对路径仅用于追溯。

| 资产 | 目标位置 | 数量 |
| --- | --- | ---: |
| RTMPose-Hand | `models/rtmpose_hand.onnx` | 1 |
| RTMDet-Hand | `models/rtmdet_hand.onnx` | 1 |
| 校准候选原图 | `data/raw/`，保留项目与来源目录结构 | 331 |
| 车内验证候选原图 | 同上，`data_collect/zbx/images` | 203 |
| 补充验证候选原图 | 同上，comfort v1 与 Thermal RGB | 55 |

两个模型同时核对了 edge-ai 归档副本与 vh3 原始来源，SHA-256 均与设计文档一致。只复制所需两个 ONNX；未复制历史 engine。

589 张原图共 51,775,384 字节，全部可完整解码，589 个 SHA-256 均不同。czq、10、zbx 共 472 张图片同时检查了文件名集合和逐文件哈希，确认 YOLO 目录中的副本与 vh3 一致。本次从 vh3 导入原图，没有导入相机标定 JSON、红外图或其他模型数据。

## 清单与检查产物

- [模型清单](../models/manifest.json)：来源、目标相对路径、大小与 SHA-256。
- [图片清单](../data/manifest.jsonl)：589 行，记录尺寸、哈希、采集组、候选场景和候选集合。
- [导入时的候选划分](../calibration/provenance/import_split.json)：`frozen=false`，校准候选 331、车内验证候选 203、补充验证候选 55；开发集尚未划出。不可将这份文件直接用于 INT8 校准。
- [近似重复候选](../data/near_duplicate_candidates.json)：灰度 32×32 DCT 感知哈希，取左上 8×8 并排除 DC，共 63 位；汉明距离 ≤ 6 的候选对共 3,911 对，跨候选集合为 0 对。
- [来源抽样图](../data/source_contact_sheet.jpg)：各目录首张、四分位、中间及末张的抽样拼图。抽样场景与设计大体一致，尚未逐张审核。
- [环境与模型实跑记录](../models/environment_check.json)：版本、模型 I/O 与已安装 rtmlib 全部 Python 源文件的 SHA-256。

感知哈希相近不能直接等同于重复手势；车内背景相同会增加候选对。没有跨集合候选对也不能证明不存在泄漏。人物重叠与采集场次缺乏已核验元信息，不能声称独立人物测试。后续应审核相似图片连通组，整组划分或剔除，再从校准候选中留出约 10% 的开发组。

## 环境与模型检查

本次使用 `/home/ssd/anaconda3/envs/py38/bin/python`：Python 3.8.20、NumPy 1.24.4、ONNX 1.17.0、ONNX Runtime 1.19.2、OpenCV 4.13.0、rtmlib 0.0.15。系统 `/usr/bin/python3` 导入 ONNX Runtime 时存在 NumPy ABI 不匹配；默认 `python3` 为另一套环境。本次未安装或升级任何依赖。

设备为 Jetson AGX Orin Developer Kit，L4T 35.4.1，CUDA SDK 11.4.19；已安装 `libnvinfer8` 包版本 `8.5.2-1+cuda11.4`。本次没有构建 engine 或验证 TensorRT 实际链接库。

两个 ONNX 均通过 `onnx.checker.check_model`，opset 均为 11。使用 CPUExecutionProvider、2 个推理线程，以 FP32 零张量进行形状冒烟检查：

| 模型 | 输入 | 实跑输出 |
| --- | --- | --- |
| RTMPose-Hand | `input: [1,3,256,256]`，源模型 batch 动态 | `simcc_x`、`simcc_y` 均为 FP32 `[1,21,512]` |
| RTMDet-Hand | `input: [1,3,320,320]` | `dets: [1,41,5]` FP32，`labels: [1,41]` INT64 |

全部输出有限。RTMDet 的 41 是本次零输入得到的数量，不是固定检测数。零输入仅检查可执行性与形状，不用于校准，不代表真实图片精度验证。

## 本地 rtmlib 核查与后续约束

已发现本地安装的 rtmlib 0.0.15，源码位于 `/home/ssd/anaconda3/envs/py38/lib/python3.8/site-packages/rtmlib/`，源码哈希已记录，但尚不能证明它与 vh3 历史运行环境完全一致，也尚未冻结为部署参考。

对当前安装源码的检查结果：

- RTMPose：框格式 xyxy，扩展系数 1.25，调整宽高比后进行仿射，`INTER_LINEAR`，默认零边界填充；部署时须显式指定 256×256。
- 均值为 `(123.675,116.28,103.53)`，标准差为 `(58.395,57.12,57.375)`；所查看的预处理与 BaseTool 推理路径没有 BGR/RGB 通道转换。后续需结合实际读图入口与模型配置核实通道语义，不能只凭均值猜测。
- SimCC split ratio 为 2.0；解码取 argmax，当前源码分数为 X/Y 峰值的算术平均，分数 ≤ 0 时位置置为 -1。不能替换为其他版本的峰值最小值或额外 softmax。
- RTMDet 默认均值为 `(103.53,116.28,123.675)`，标准差为 `(57.375,57.12,58.395)`，等比例缩放并在右侧/底部填充 114；部署时须显式指定 320×320。

下一阶段需要用显式本地模型路径生成手框、审核异常与漏检，核实预处理和解码配置，冻结重复组与开发集，之后再生成逐手 FP32 输入与参考输出。当前没有可用于校准的正式手部 batch 数。

## 复查与存储

本次导入及检查命令留存在本地 `data/provenance/`，便于复查：

```bash
python3 tensorrt-rtmpose-hand/data/provenance/prepare_rtmpose_assets.py
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/data/provenance/check_rtmpose_models.py
```

以上为首次导入的历史命令，当前划分已获认可，不应直接重跑覆盖当前状态。导入脚本依赖本机源路径、Python 3.10+、Pillow 和 NumPy；相同目标按哈希校验后跳过，目标同名异内容或来源数量改变时报错。模型检查脚本重新生成环境检查记录。跨机器使用已导入资产时依赖清单中的目标相对路径，不依赖原始来源路径。

导入时 edge-ai HEAD 为 `09a85e76f606a003e595bc01b77bae78a2a0b3b9`，工作区初始无修改；vh3 HEAD 为 `c66280a8f2deb1be4b9fb0d64b0104de1572d280`。资产内容以清单哈希为准，不以仓库 commit 代替二进制版本。

二进制资产及生成清单沿用现有 Git 忽略规则，仅保存在本地。搬运模块时需一并携带 `models/`、`data/`、`calibration/split.json`；只克隆 Git 仓库不会获得这些文件。本次未提交 Git。
