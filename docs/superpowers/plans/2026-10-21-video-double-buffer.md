# Video double-buffer implementation plan

Goal: overlap CPU preparation of frame N+1 with GPU inference of frame N.
Design: the user approved prepare/infer separation, two owning frame slots, and YOLO-owned preprocessing in this conversation. Implement in the current checkout; no additional approval or publishing is needed.

- [x] Add owning PreparedInput and PreparedFrame interfaces; prepare is CPU-only and may run concurrently with infer on the same model. Each prepared payload owns its per-frame data; infer owns GPU state. Tests cover ordered dispatch and original-frame preservation.
- [x] Move toBusinessFrame into yolo_video_preprocess.hpp/.cpp, retain exact transpose/INTER_AREA behavior and adapt both video entry points and GPU comparison test.
- [x] Add a two-slot producer/consumer with ordered EOF/errors, backpressure, cancellation and joining. Test actual overlap with a condition-variable handshake, slot ownership, EOF and exceptions.
- [x] Integrate capture, transforms and model.prepare in the producer; retain inference/render/display on the main thread. Stop at max_frames without speculative extra reads. Keep first-frame validation before opening outputs.
- [x] Document prepare_ms/queue_wait_ms and latency vs throughput; build CPU core and full project, run CTest and available GPU regression.

Review focus: slow consumers cannot cause overwrite; producer exceptions drain earlier frames before propagating; consumer exceptions cancel a producer waiting for space; short videos and max_frames preserve exact counts; model payloads retain source geometry/id/timestamp. OpenCV blocking reads cannot be interrupted portably, so shutdown can wait for an in-progress read.

## Initial implementation verification record

- Baseline core CTest: 3/3 passed.
- After prepare/infer API migration, the overlap handshake failed with “Next-frame preparation did not overlap inference”; after double-buffer integration it passed.
- Final full build (YOLO + ResNet) and CTest: 10/10 passed; standalone core: 4/4; no-video build: 5/5.
- GPU adapter test: FP16, Graph on/off, camera/business, exact detection equality and input preservation passed.
- GPU video integration: existing regression_video_pipeline.py using /usr/bin/python3, 10 old/new/config runs, recording and CSV, two-model configs, both geometries and Graph settings passed. Artifacts: /tmp/edge-double-buffer-regression-20261022.
- GPU commands required sandbox escalation because Jetson device access was blocked. Default python3 lacks cv2; the system interpreter has it. No dependencies were installed.
- Independent code review found no important defects; its timing-documentation suggestion is incorporated into video-pipeline/README.md.
- These initial regression clips verify behavior, not warmed-up throughput; live-camera benchmarking was performed separately after implementation validation.
