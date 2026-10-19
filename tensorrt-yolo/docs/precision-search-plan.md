# Single-image precision tooling plan

Goal: automate fixed-input INT8 precision debugging, without validation work.
Architecture: C++ builder owns construction; Python checker/search own candidate selection and logs; Polygraphy is an optional coarse localization frontend.

User-authorized scope: timing cache, exact FP16 layer list, fixed input.bin and existing FP16 output reference, optional Polygraphy. Preserve existing artifacts. Work in place; archive user's hardcoded 15–22 builder before replacing; no automatic commits.

- [x] Builder: retain positional CLI; add --timing-cache, --fp16-layers, --dump-layers, --inspect-only. Reject unknown/ineligible/duplicate names. Preserve FP32 boundary I/O. CPU parser tests before implementation.
- [x] Checker: score error <=0.03 and mean IoU >=0.85 on reference person candidates >=0.7; one-to-one NMS detection matching, IoU>=0.85 and score error<=0.05, no extra/missing person detections. Reject bad size/nonfinite/empty reference. Configurable CLI; test before implementation.
- [x] Search: same C++ builder, bounded build count, baseline/full-set or explicit seed, deletion refinement with actual checks (no monotonic assumption), all logs/metrics/lists saved. Optional Polygraphy CLI launcher with the same checker and cache-only calibration.
- [x] Verify compile/CPU suite, actual network export and GPU checker, pinned Polygraphy imports/help. Record what was and was not GPU-built; avoid launching an unbounded multi-hour search.

Defaults are single-image debugging gates, not deployment acceptance criteria. Cached calibration is from345 images; never calibrate using the one debug image. Engine/cache use simple names inside experiment directories.

## Completion evidence

- 20 Python tests on /usr/bin/python3 and3 CTest tests passed. Default Python3.13 cannot run existing cv2 tests; use3.8.
- GPU smoke: real network371/eligible263, tiny model INT8+FP16 constraint, cache reuse, unknown-layer rejection, real baseline checker FAIL as expected, Polygraphy API and full two-iteration CLI.
- Full CLI stopped cleanly at budget with both passing artifacts preserved; logs at debug/precision-smoke/cli-final.log.
- Independent review found/fixed report alias and implicit temporary-path overwrite, and unified prepared/immediate supervisors. Re-review found no important remaining issues.
- Complete YOLO search is prepared in debug/polygraphy-v1/run.sh, not started. No validation work.

## Post-completion cleanup

At user request, removed the temporary historical baseline builder source/object/executable and the pre-edit source backup. Historical logs/provenance and model artifacts remain. The sole maintained builder is apps/build_engine.cpp.
