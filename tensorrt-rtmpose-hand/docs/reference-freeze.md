# RTMPose-Hand Reference Freeze 核验记录

2026-10-29：技术契约已冻结，关键点可视化已通过用户人工评审。本次只使用 development 做模型推理；最终 validation 的 258 张原图未参与推理，未构建 TensorRT engine 或 INT8 cache。

配套学习文档：[Reference Freeze 流程与原理详解](reference-freeze-walkthrough.md)，包含完整处理链、公式、示例与代码出处。

## 当前入口

- [冻结配置](../configs/rtmpose_hand_reference.json)：输入输出、颜色、几何、归一化、SimCC 解码及源码哈希。
- [核验结果快照](../configs/rtmpose_hand_reference.verification.json)：配置/模型/脚本哈希、环境和实际检查结果。
- [本地人工评审页面](../validation/reference-freeze/index.html)：10 个按输入几何、亮度、边缘、双手和重叠框选择的开发样本，包含旧 BGR 与 RGB 输出对照及原图坐标叠图。
- `validation/reference-freeze/raw/`：55 份开发样本原始 SimCC NPZ；`decoded/`：对应关键点、分数、有效掩码、输入/原图坐标与配置哈希。

评审页面和逐样本输出保留在本地，不提交图片及大型数据。直接用浏览器打开 HTML 即可，须保留同目录的 `visualization/`。

## 颜色依据与数据修正

[官方 ONNX SDK 模型包](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip) 内的 `end2end.onnx` 与本地模型 SHA-256 完全一致：`39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29`。同包 `pipeline.json` 的 Normalize 明确 `to_rgb=true`；[MMPose v1.1.0 训练配置](https://github.com/open-mmlab/mmpose/blob/v1.1.0/projects/rtmpose/rtmpose/hand_2d_keypoint/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py) 也明确 `bgr_to_rgb=True`。

包内部分 SDK 尺寸元数据写着 192×256，与实际 ONNX 冲突；本次采用实际图及训练配置共同确认的 256×256。颜色判断来自模型来源证据，不通过比较置信度挑选颜色。原始 ONNX 直接接卷积，未内置通道交换或归一化。

本地 rtmlib 0.0.15 的 RTMPose 预处理没有自动交换通道，旧路径直接输入 OpenCV BGR。现在明确先转 RGB，再调用锁定的几何/归一化实现；407 个旧张量全部重建。每个重建输入均与“原图转 RGB 后重新仿射”逐元素相等。原图、手框、裁剪、划分、样本编号与顺序表不变。

旧张量及元数据备份位于 `calibration/provenance/before-rgb-reference/`；逐文件旧/新哈希位于 `calibration/provenance/rgb-reference-migration.json`。现行 `hand_boxes.jsonl`、`summary.json` 和 `coverage.json` 已更新。历史导入及分目录报告中的 BGR 和旧张量哈希只表示当时状态。

## 冻结的关键规则

| 项目 | 契约 |
| --- | --- |
| 输入 | `input`，FP32 NCHW `[N,3,256,256]`，仅 N 动态 |
| 输出 | 按名字读取 `simcc_x`、`simcc_y`，均为 FP32 `[N,21,512]` |
| 颜色与归一化 | RGB；`(pixel - mean) / std`，不除以 255；mean/std 见配置 |
| 张量计算 | 与本地 NumPy 参考一致，归一化计算为 float64，最后转连续 FP32 NCHW |
| ROI | 原图 xyxy，不预先裁框；中心、1.25 倍扩展、正方形、无旋转 |
| 仿射 | `INTER_LINEAR`，越界零填充 |
| 坐标解码 | SimCC argmax，平局取第一个；坐标除以 2，不加 softmax/DARK/翻转测试 |
| 分数 | `(x_peak + y_peak) / 2`，不设额外置信度筛选阈值 |
| 无效点 | score ≤ 0 时，SimCC 坐标先设 -1，除以 2 后为 -0.5；仍进行原图坐标变换，消费者必须用有效掩码，不能以原图坐标是否 -1 判断 |
| 原图还原 | `input_xy / [256,256] * scale + center - scale/2`，不裁边或镜像 |

空手框列表应直接返回空结果，禁止调用 rtmlib 的整图兜底。非法框应拒绝；这是后续适配器需要落实的接口规则。现成 NPY 已完成全部预处理，读取后不要再次换通道、归一化或裁剪。

## 验证范围

核验覆盖 ONNX checker、输入输出形状及 dtype、模型/rtmlib 源码哈希、407 个真实输入的预处理一致性，以及全部 55 个开发样本的 ORT FP32 推理和解码。零分、负分、并列峰值及平均分规则用构造数据补充检查。

参考原图坐标与保存的逆仿射矩阵交叉核验，最大绝对坐标分量差为 `3.90075147151947e-05` 像素，低于 `1e-3` 容差。ORT 的 batch 1/2/3/4 使用前 N 个开发样本与各自单独推理比较，本环境下 SimCC 与解码坐标差均为 0；这不是 TensorRT 动态 batch 验证。

代理已查看 10 个代表样本的裁剪对照拼图及边缘样本原图叠图，未见整体镜像或系统性几何偏移。困难、遮挡样本仍可能存在局部预测误差；用户已明确确认本次关键点可视化评审通过。这里没有人工 GT，不报告 PCK/NME 或真实精度，也不以 RGB/BGR 输出差异证明哪个更准确。

## 复现

在仓库根目录运行，使用已有 py38 环境及本地模型、数据、官方元数据侧文件：

```bash
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/scripts/verify_reference_config.py
/home/ssd/anaconda3/envs/py38/bin/python tensorrt-rtmpose-hand/calibration/provenance/generate_coverage.py
```

前者写本地 `validation/reference-freeze/report.json` 和样本输出，不修改模型或数据集；检查失败会报错退出，请勿使用 `python -O` 禁用断言。已冻结配置要求全部现用张量匹配 RGB 契约。核验脚本重新运行不会自动把人工/代理视觉评审标为完成；受版本控制的结果快照保留本次检查证据。

官方来源侧文件在 `models/reference_provenance/`：解压上述 SDK 包的 `deploy.json`、`pipeline.json`、`detail.json`，保存来源身份记录 `archive_identity.json`，以及配置中链接的 `training_config_v1.1.0.py`。校验哈希均在冻结配置中，缺失或不匹配时核验失败，不自动下载替换。

`scripts/rebuild_reference_tensors.py` 是本次一次性迁移脚本，先备份和生成候选文件，再替换现用张量；备份目录存在时会拒绝重跑。不要为重复执行而删除备份。常规复核只运行上面的核验与覆盖度脚本。

人工评审状态保存在核验结果快照中，并关联冻结配置 SHA-256。配置内部的 `human_visual_review` 保留冻结时的待评审状态，避免仅因评审状态变化而修改契约哈希及全部数据追溯记录；当前状态以核验结果快照为准。
