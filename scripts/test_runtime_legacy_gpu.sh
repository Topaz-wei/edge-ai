#!/usr/bin/env bash
# 本脚本通过改造后的公共 Runner 验证旧模型与静态 CUDA Graph，不重建 engine。
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p logs/runtime-vnext
build-shared/tensorrt-resnet/test_shared_runner \
    tensorrt-resnet/engines/resnet18_cpp_fp16.plan \
    tensorrt-resnet/data/input.bin \
    tensorrt-yolo/cache/resnet-baseline/output.bin \
    2>&1 | tee logs/runtime-vnext/resnet.log
build-shared/tensorrt-yolo/test_trt_runner \
    tensorrt-yolo/engines/yolov8m_fp16.plan \
    tensorrt-yolo/data/images/test.jpg \
    tensorrt-yolo/cache/shared-runtime-baseline/fp16.bin \
    2>&1 | tee logs/runtime-vnext/yolo.log
build-shared/tensorrt-yolo/test_cuda_yolo \
    tensorrt-yolo/engines/yolov8m_fp16.plan \
    tensorrt-yolo/data/images/test.jpg \
    2>&1 | tee logs/runtime-vnext/yolo-graph.log
