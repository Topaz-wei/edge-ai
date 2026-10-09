#!/usr/bin/env bash

set -e

WARMUP=100
ITERS=1000

echo "===== FP32 ====="
./build/trt_benchmark \
    engines/resnet18_cpp_fp32.plan \
    1 ${WARMUP} ${ITERS}

echo
echo "===== FP16 ====="
./build/trt_benchmark \
    engines/resnet18_cpp_fp16.plan \
    1 ${WARMUP} ${ITERS}

echo
echo "===== INT8 ====="
./build/trt_benchmark \
    engines/resnet18_int8_detailed.plan \
    1 ${WARMUP} ${ITERS}