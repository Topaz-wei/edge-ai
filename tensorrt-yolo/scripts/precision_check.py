#!/usr/bin/env python3
"""Single-image YOLO checker. Exit: 0 pass, 1 accuracy failure, 2 infrastructure/input error."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from analyze_int8_output import load_output, bbox_iou_xywh, error_stats, SHAPE


def detections(output, confidence, nms_iou):
    classes = np.argmax(output[0, 4:], axis=0)
    scores = np.max(output[0, 4:], axis=0)
    indices = np.flatnonzero((classes == 0) & (scores >= confidence))
    indices = sorted(indices.tolist(), key=lambda i: (-float(scores[i]), i))
    kept = []
    for i in indices:
        box = output[0, :4, i].tolist()
        if all(bbox_iou_xywh(box, item[2]) <= nms_iou for item in kept):
            kept.append((0, float(scores[i]), box))
            if len(kept) == 300:
                break
    return kept


def match_detections(reference, candidate, min_iou, max_score_error):
    if len(reference) != len(candidate):
        return False
    # Build a bipartite graph of acceptable matches. Reassignment prevents
    # an early greedy match from hiding another valid one-to-one assignment.
    edges = [
        [
            j
            for j, b in enumerate(candidate)
            if a[0] == b[0]
            and abs(a[1] - b[1]) <= max_score_error
            and bbox_iou_xywh(a[2], b[2]) >= min_iou
        ]
        for a in reference
    ]
    owner = {}

    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or visit(owner[j], seen):
                owner[j] = i
                return True
        return False

    return all(visit(i, set()) for i in range(len(reference)))


def evaluate(
    reference,
    candidate,
    confidence=0.7,
    score_error=0.03,
    mean_iou=0.85,
    detection_iou=0.85,
    detection_score_error=0.05,
    nms_iou=0.7,
):
    for value in (
        confidence,
        score_error,
        mean_iou,
        detection_iou,
        detection_score_error,
        nms_iou,
    ):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Thresholds must be finite and in [0,1]")
    for a in (reference, candidate):
        if a.shape != SHAPE or not np.isfinite(a).all():
            raise ValueError("Expected finite output [1,84,8400]")

    # Candidate IDs come from the reference: compare the same output slots
    # even if quantization changes their scores or their winning classes.
    classes = np.argmax(reference[0, 4:], axis=0)
    ids = np.flatnonzero((classes == 0) & (reference[0, 4] >= confidence))
    if not len(ids):
        raise ValueError("Reference has no high-confidence person candidates")
    score = float(np.mean(np.abs(reference[0, 4, ids] - candidate[0, 4, ids])))
    ious = [bbox_iou_xywh(reference[0, :4, i], candidate[0, :4, i]) for i in ids]

    ref_det = detections(reference, confidence, nms_iou)
    test_det = detections(candidate, confidence, nms_iou)
    matched = match_detections(ref_det, test_det, detection_iou, detection_score_error)

    # Raw tensor errors below are diagnostics; these task-level gates decide PASS.
    gates = {
        "candidate_score": score <= score_error,
        "candidate_iou": float(np.mean(ious)) >= mean_iou,
        "detections": matched,
        "candidate_classes": bool(
            np.all(np.argmax(candidate[0, 4:, ids], axis=-1) == 0)
        ),
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "reference_candidates": ids.tolist(),
        "mean_score_error": score,
        "mean_iou": float(np.mean(ious)),
        "min_iou": float(min(ious)),
        "reference_detections": ref_det,
        "candidate_detections": test_det,
        "bbox_error": error_stats(reference[0, :4], candidate[0, :4]),
        "class_error": error_stats(reference[0, 4:], candidate[0, 4:]),
        "thresholds": dict(
            confidence=confidence,
            score_error=score_error,
            mean_iou=mean_iou,
            detection_iou=detection_iou,
            detection_score_error=detection_score_error,
            nms_iou=nms_iou,
        ),
    }


def add_gate_args(parser):
    for key, default in [
        ("confidence", 0.7),
        ("score-error", 0.03),
        ("mean-iou", 0.85),
        ("detection-iou", 0.85),
        ("detection-score-error", 0.05),
        ("nms-iou", 0.7),
    ]:
        parser.add_argument("--" + key, type=float, default=default)


def gate_kwargs(args):
    return {
        key: getattr(args, key)
        for key in (
            "confidence",
            "score_error",
            "mean_iou",
            "detection_iou",
            "detection_score_error",
            "nms_iou",
        )
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--output", type=Path)
    source.add_argument("--engine", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--input", type=Path)
    p.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build/yolo_infer_raw",
    )
    p.add_argument("--report", type=Path, required=True)
    add_gate_args(p)
    args = p.parse_args()
    # Do not write even an error report until all destination aliases are checked.
    safe_report = False
    try:
        sources = [
            x.resolve()
            for x in (args.reference, args.output, args.input, args.engine)
            if x is not None
        ]
        targets = [
            args.report,
            args.report.with_suffix(".output.bin"),
            args.report.with_suffix(".infer.log"),
        ]
        if len({x.resolve() for x in targets}) != len(targets):
            raise ValueError("Report suffix aliases generated output/log")
        for target in targets:
            if target.exists() or target.is_symlink() or target.resolve() in sources:
                raise ValueError(
                    "Report/output/log path exists or aliases an input: " + str(target)
                )
        safe_report = True
        args.report.parent.mkdir(parents=True, exist_ok=True)
        reference = load_output(str(args.reference))
        if args.engine:
            if args.input is None or args.input.stat().st_size != 4915200:
                raise ValueError("Need FP32 [1,3,640,640] input.bin")
            if not np.isfinite(np.fromfile(args.input, dtype="<f4")).all():
                raise ValueError("Input contains nonfinite values")
            output = args.report.with_suffix(".output.bin")
            with args.report.with_suffix(".infer.log").open("w") as log:
                subprocess.run(
                    [
                        str(args.runner.resolve()),
                        str(args.engine.resolve()),
                        str(args.input.resolve()),
                        str(output.resolve()),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        else:
            output = args.output
        result = evaluate(reference, load_output(str(output)), **gate_kwargs(args))
        result.update(
            reference=str(args.reference.resolve()), output=str(output.resolve())
        )
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            "PASS" if result["pass"] else "FAIL",
            "score_error=",
            result["mean_score_error"],
            "IoU=",
            result["mean_iou"],
        )
        return 0 if result["pass"] else 1
    except Exception as exc:
        if safe_report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps({"error": str(exc), "pass": False}, indent=2) + "\n"
            )
        print("ERROR:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
