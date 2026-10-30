#!/usr/bin/env bash
# 按冻结输入契约构建动态 batch FP32 engine；关闭 TF32，不启用 FP16/INT8。
set -euo pipefail
module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '%s  %s\n' \
    '3036459e1f6bc230c9059f34ea260070928669f54f9b1bfef3e031279aef0aec' "$module_dir/configs/rtmpose_hand_reference.json" \
    '39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29' "$module_dir/models/rtmpose_hand.onnx" | sha256sum --check
mkdir -p "$module_dir/engines" "$module_dir/validation/trt-fp32"
/usr/src/tensorrt/bin/trtexec \
    --onnx="$module_dir/models/rtmpose_hand.onnx" \
    --saveEngine="$module_dir/engines/rtmpose_hand_fp32.engine" \
    --minShapes=input:1x3x256x256 \
    --optShapes=input:2x3x256x256 \
    --maxShapes=input:4x3x256x256 \
    --noTF32 \
    --memPoolSize=workspace:1024 \
    --buildOnly \
    --profilingVerbosity=detailed \
    --exportLayerInfo="$module_dir/validation/trt-fp32/layers.json" \
    2>&1 | tee "$module_dir/validation/trt-fp32/build.log"
