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
