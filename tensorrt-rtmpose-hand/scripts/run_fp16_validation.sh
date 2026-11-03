#!/usr/bin/env bash
# 本脚本执行 2 GiB C++ builder 新实验；保持旧 engine、基线和全部验收门槛。
set -euo pipefail
module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$module_dir/.." && pwd)"
output_dir="$module_dir/validation/trt-fp16-cpp-2g"
engine="$module_dir/engines/rtmpose_hand_fp16_cpp_2g.engine"
python_reference=/home/ssd/anaconda3/envs/py38/bin/python
"$python_reference" "$module_dir/scripts/verify_trt_fp16.py" prepare --experiment cpp-2g
/usr/bin/python3 "$module_dir/scripts/inspect_fp16_engine.py" \
    --engine "$engine" --output "$output_dir/layers-context.json" \
    2>&1 | tee "$output_dir/inspector.log"
for mode in npy image; do
    "$repo_dir/build-shared/tensorrt-rtmpose-hand/rtmpose_fp32" \
        "$engine" "$output_dir/cases.json" "$output_dir/$mode" "$mode" \
        2>&1 | tee "$output_dir/$mode.log"
done
"$python_reference" "$module_dir/scripts/verify_trt_fp16.py" compare --experiment cpp-2g \
    2>&1 | tee "$output_dir/compare.log"
