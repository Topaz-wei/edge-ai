# CUDA pipeline implementation plan

Approved scope: fixed 360x640 BGR business images, exact sequential mean masks,
fixed640 FP32 I/O with an existing FP16 engine, CUDA postprocessing and optional
TensorRT CUDA Graph. Keep CPU reference targets and existing engine-builder edits.

1. Extend TrtRunner with device-buffer/stream access and explicit graph preparation;
   no allocation or graph capture in the frame loop. Abort on capture failure.
2. GPU preprocessing: copy uint8 pixels through persistent pinned staging; precompute
   exact OpenCV polygon masks once; GPU integer reductions and sequential masks;
   fuse padding, RGB/CHW and normalization. No per-frame CPU image processing.
3. GPU postprocessing: per-anchor class argmax, confidence filter, CUB descending
   radix sort with candidate-index tie break, exact greedy NMS and box restoration.
   No arbitrary candidate/max_det truncation. Return count plus selected detections.
4. Add a persistent CUDA pipeline and separate infer/benchmark targets, retaining CPU
   programs. CUDA events measure GPU stages; steady_clock measures complete host-image
   to host-detections latency. Test graph off/on, reuse and changing input frames.
5. Tests compare all preprocessing elements and decoded results to CPU, including
   ties, class selection, threshold boundaries, zero/many detections and ROI strides.
6. Release build, CPU tests, real GPU FP16/FP32 regression, then serial CPU/CUDA/Graph
   50-warmup/200-iteration measurements. Report limitations and actual numbers.

Data movement: upload 691200 bytes BGR per frame; keep 4915200-byte input tensor and
2822400-byte raw output on device. Download a count and selected Detection records.
This is not zero-copy. Graph capture covers TensorRT only, with stable addresses.
