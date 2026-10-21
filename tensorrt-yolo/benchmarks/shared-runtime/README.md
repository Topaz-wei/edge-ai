# Shared runtime regression

Baseline source: `5969315` (product code inherited from `282645d`).
Capture: `python3 scripts/regression_shared_runtime.py capture --build tensorrt-yolo/build-baseline --output tensorrt-yolo/cache/shared-runtime-baseline`.
Original engines: `engines/yolov8m_fp16.plan` and `debug/trial007_sub_model22/yolov8m_int8.plan`.
The local manifest records engine/input hashes and commands. It includes fixed-image preprocessing,
raw output, Graph/non-Graph detections and PNGs, 115 validation tensors per engine and Graph benchmarks.
Baseline capture never builds engines. GPU commands require device access outside the filesystem sandbox.

ResNet's original fixed inference was compiled separately with only an output dump added under
`cache/resnet-baseline/`. Its engine was `resnet18_cpp_fp16.plan`; input was the existing `data/input.bin`.
The ResNet export uses a synthetic random input; no meaningful video classifier is inferred from this fixture.

Compare after a new capture with:
`python3 scripts/regression_shared_runtime.py compare --baseline tensorrt-yolo/cache/shared-runtime-baseline --output NEW_CAPTURE`.
Do not overwrite baseline directories. A byte difference is a failed regression, not a tolerance adjustment.

## Final results (2026-10-21)

Implementation is on `dev`; existing engine binaries were never rebuilt or moved.
Final SHA-256:

- FP16: `5426f5c14084d42f5092e4079069a7d64d640de5a075bb18dd1452c3f7461b7a`
- Mixed INT8: `8353b32b5c9c03654418caab042758ea78045b505a32326c84987ab19c87b203`

**Numerical regression:** all 241 captured artifacts compare byte-for-byte equal.
This includes C++ preprocessing, raw inference on one fixed image plus 115 reviewed inputs per engine,
Graph/non-Graph single-image detections and rendered PNGs. `test_cuda_yolo` additionally checks real/random/ROI/
constant inputs, postprocessing edge cases, changing frames and Graph/non-Graph equality.
ResNet's fixed FP16 engine produces byte-identical output in three repeated shared-runner calls;
its existing CLI still reports Top-1 601, matching its PyTorch fixture.

The original pinned evaluator recomputed these metrics from the frozen old/new raw outputs:

| Engine | Images | Precision @0.7 | Recall @0.7 | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16, before = after | 115 | 1.000000 | 0.939130 | 0.995000 | 0.994846 |
| Mixed INT8, before = after | 115 | 1.000000 | 0.930435 | 0.994913 | 0.919871 |

Recompute with `/home/ssd/anaconda3/envs/py38/bin/python scripts/evaluate_shared_regression.py`.
These are the existing reviewed labels and evaluation scope, not a new independent accuracy study;
see `validation/compare-v2/README.md` for annotation/negative-sample limitations.

**Video verification:** `test_video_model ENGINE IMAGE` compares adapter detections/classes/scores/coordinates
against the validated CUDA pipeline for business/camera and Graph on/off, and verifies input preservation.
`regression_video_pipeline.py --output NEW_DIR` separately smoke-tests old/new entry points on six-frame
lossless clips, recording, CSV, frame limits, and two independent engine instances from a different cwd.
Its video smoke test does not claim encoded-frame numerical equivalence: timing overlays naturally differ.
Final mixed-engine smoke artifacts: `cache/shared-video-mixed-final/` (business loads INT8 + FP16; camera loads two independent INT8 instances). Earlier runs: `cache/shared-video-final/` and `cache/shared-video-integration/`.
Physical camera operation and desktop window display were not rerun; offline source/geometry paths were tested.

**Build/test coverage:** root YOLO+ResNet build and 9 CPU CTests passed; runtime-only build/1 CTest,
YOLO without CUDA/video build/5 CTests, standalone YOLO/ResNet builds, existing Python 22 tests,
and regression utility 2 tests passed. GPU tests are explicit commands, not implicit CTest requirements.
An existing dynamic ResNet engine was correctly rejected with `Invalid tensor dimension`.
Review fixes have regression tests for integer-overflow/embedded-apostrophe configuration inputs,
first-frame geometry rejection before output creation, and optional metadata hash/environment mismatches.

### Performance observations

Same existing engines and benchmark source, 50 warmup + 200 timed iterations per run, no clock/power changes.
Three alternating old/new rounds produced the following median of per-run mean end-to-end latency:

| Engine | Old (ms) | New (ms) | Observed difference |
| --- | ---: | ---: | ---: |
| FP16 | 9.344 | 10.088 | +7.96% |
| Mixed INT8 | 9.041 | 9.132 | +1.01% |

A later FP16 new/old/old/new sequence measured new 12.166/12.203 ms and old 12.012/11.629 ms
(about +3.1% between averages). This also demonstrates substantial between-session variation.
Initial power query reported MAXN; one EMC sysfs read was denied and clocks were not locked.
Do not claim a speedup or proven performance parity. A controlled-clock performance investigation remains
separate follow-up if this difference matters; this extraction's demonstrated guarantee is numerical equivalence
on the tested inputs. Raw timing logs remain in `cache/shared-runtime-after/perf-*.log` and `reverse-*.log`.

### Current boundaries

- Runtime supports fixed positive shapes, one input/output and FP32 linear device I/O.
- ResNet reuses the runtime but has no fabricated image/video adapter; its fixture is synthetic.
- Video supports sequential independent model instances; only YOLO is currently registered.
- Existing CLI and business presentation remain; the config entry renders in original source coordinates.
- Engine audit is initialization-only. Optional `ENGINE.meta.yaml` is checked strictly; missing historical
  metadata is not invented. See the shared video README for its schema.
