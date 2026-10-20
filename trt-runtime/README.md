# TensorRT runtime

`edge_trt_runtime` is a C++17 library depending only on TensorRT runtime and CUDA runtime.
`edge::trt::Runner` owns one engine/context/stream and persistent I/O buffers; instances are not thread-safe.
Supported contract: fixed positive shapes, one input and one output, FP32 linear device I/O.
Internal FP16/INT8 computation is supported when engine I/O meets this contract.
Dynamic shapes, multiple outputs, host bindings and non-FP32 I/O are rejected, not converted.

Include `edge/trt/runner.hpp`. `infer(vector<float>, timings)` is synchronous;
`enqueueDevice()` uses the runner stream without host copies/synchronization.
Device buffers and the stream remain valid only during the runner lifetime.
`inputInfo()`/`outputInfo()` describe actual engine tensors; model validation belongs to the caller.
`prepareGraph()` captures inference only. No builder/parser, image processing or model-specific shapes live here.

Standalone: `cmake -S trt-runtime -B trt-runtime/build && cmake --build trt-runtime/build`.
No command here builds an engine. Engine/runtime compatibility is checked during deserialization.
