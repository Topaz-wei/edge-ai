#!/usr/bin/env bash
# 本脚本只编排公共 C++ builder 并记录构建身份；历史 trtexec engine 不覆盖。
set -euo pipefail
module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$module_dir/.." && pwd)"
output_dir="$module_dir/validation/trt-fp16-cpp-2g"
engine="$module_dir/engines/rtmpose_hand_fp16_cpp_2g.engine"
if [[ -e "$engine" || -e "$output_dir/build-provenance.json" ]]; then
    echo 'Existing experiment; choose a new experiment directory instead of overwriting.' >&2
    exit 1
fi
mkdir -p "$output_dir"
python3 "$module_dir/scripts/record_builder_provenance.py"
"$repo_dir/build-shared/tensorrt-rtmpose-hand/rtmpose_build" \
    "$module_dir" "$module_dir/configs/rtmpose_hand_builder.json" "$engine" \
    2>&1 | tee "$output_dir/build.log"
python3 "$module_dir/scripts/record_builder_provenance.py" --complete
