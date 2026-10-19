# Validation comparison

115 reviewed images, 115 person boxes. Both engines use the same frozen labels and preprocessed input binaries.

| Engine | Precision @0.7 | Recall @0.7 | mAP50 | mAP50-95 | TP / FP / FN |
|---|---:|---:|---:|---:|---|
| trial007 | 100.0000% | 93.0435% | 99.4913% | 92.3268% | 107 / 0 / 8 |
| trial007_sub_model22 | 100.0000% | 93.0435% | 99.4913% | 91.9871% | 107 / 0 / 8 |

## Metric definitions

- Precision/Recall: confidence >=0.7, one-to-one matching at IoU >=0.5; matching is recomputed after filtering.
- AP: confidence >=0.001; NMS IoU 0.7; maximum 300 detections/image; IoU 0.50 to 0.95 in 0.05 steps.
- Uses installed Ultralytics 8.3.173 matching and AP integration. Single person class: mAP equals person AP. This is not a COCO 80-class evaluation or pycocotools COCOeval.
- Existing business postprocessing: argmax across 80 classes, then retain person; fixed masks and centered letterbox. FP32 raw engine I/O. No image augmentation.
- Additional best-F1 Precision/Recall in metrics.json are distinct from fixed-confidence metrics in this table.

## Scope and limitations

Scores apply to the reviewed annotation scope. In particular, person_3/203 retains only the foreground person by user decision; the rear passenger is not annotated. Thus these results must not be described as exhaustive all-person detection accuracy.
Labels were initialized by the FP32 PyTorch model and reviewed by the user; they are not independent expert annotations. Same-person temporal holdout, only four people, no negative images. Precision=100% here does not establish general false-alarm performance.
No latency benchmark was run. This evaluation cannot establish performance gains from removing model22 constraints.

## Artifacts

- summary.json: aggregate and per-person-group metrics for both engines.
- manifest.json: settings, engine hashes, image/label hashes and evaluator provenance.
- labels_snapshot/, review_snapshot.csv: exact reviewed labels used in this run.
- inputs/: common input binaries.
- Each engine folder: metrics.json, per_image.csv, predictions.json, raw/*.bin, inference.log, PR/P/R/F1 plots.
- ../compare-v2.log: full evaluation output.
- ../tools/infer_batch.cpp and infer_batch: batch adapter using existing libtrt_runner.a; same runner as yolo_infer_raw, loaded once per engine.
- The interrupted compare-v1 run and its log were deleted after verification. All 63 completed one-shot outputs had been checked byte-for-byte identical to batch results.

## Missed images at confidence 0.7

trial007: person_2/57, person_3/163, person_3/165, person_3/166, person_3/167, person_3/168, person_3/169, person_3/203

trial007_sub_model22: person_2/57, person_3/163, person_3/164, person_3/165, person_3/167, person_3/168, person_3/169, person_3/203

## Reproduce after checkout

The batch adapter source is maintained at `apps/infer_batch.cpp`; build with
`cmake --build build --target yolo_infer_batch` after CMake configuration.
Run from the project root using the pinned Python environment:

```bash
/home/ssd/anaconda3/envs/py38/bin/python scripts/validate_reviewed_engines.py \
  --output validation/compare-new \
  --batch-runner build/yolo_infer_batch \
  --engine trial007=debug/refine-v1/trial-007/yolov8m_int8.plan \
  --engine trial007_sub_model22=debug/trial007_sub_model22/yolov8m_int8.plan
```

Models, engine binaries, source images and the reviewed annotation package must
be available locally; they are not included in this experiment commit. Restore
the recorded label snapshot if current labels have changed. Manifest paths and
hashes describe the actual historical run, before the adapter was added to CMake.
Input/raw tensors and the interrupted compare-v1 run are intentionally excluded
from Git; prediction JSON and labels are retained for metric recomputation.
