#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

SHAPE = (1, 84, 8400)
NUM_CANDIDATES = 8400


def load_output(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype='<f4')
    expected = int(np.prod(SHAPE))
    if data.size != expected:
        raise ValueError(
            f'{path}: expected {expected} float32 values for shape {SHAPE}, got {data.size}'
        )
    data = data.reshape(SHAPE)
    if not np.isfinite(data).all():
        bad = data.size - int(np.isfinite(data).sum())
        raise ValueError(f'{path}: contains {bad} non-finite values')
    return data


def split_output(output: np.ndarray):
    bbox = output[0, :4, :]      # [4,8400], xywh in model coordinates
    cls = output[0, 4:, :]       # [80,8400]
    best_class = np.argmax(cls, axis=0)
    best_score = np.max(cls, axis=0)
    person_score = cls[0]
    return bbox, cls, best_class, best_score, person_score


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = [float(v) for v in box]
    return np.array([
        x - w / 2.0,
        y - h / 2.0,
        x + w / 2.0,
        y + h / 2.0,
    ], dtype=np.float64)


def bbox_iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    a = xywh_to_xyxy(a)
    b = xywh_to_xyxy(b)

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0.0 else 0.0


def error_stats(reference: np.ndarray, test: np.ndarray):
    diff = test - reference
    abs_diff = np.abs(diff)
    return {
        'max': float(abs_diff.max()),
        'mae': float(abs_diff.mean()),
        'rmse': float(np.sqrt(np.mean(diff * diff))),
        'p95': float(np.percentile(abs_diff, 95)),
        'p99': float(np.percentile(abs_diff, 99)),
    }


def print_error_stats(label: str, reference: np.ndarray, test: np.ndarray):
    ref_bbox, ref_cls, *_ = split_output(reference)
    tst_bbox, tst_cls, *_ = split_output(test)

    bbox = error_stats(ref_bbox, tst_bbox)
    cls = error_stats(ref_cls, tst_cls)

    print(f'\n=== Raw error: {label} vs FP16 ===')
    print(
        'BBox  '
        f"max={bbox['max']:.7g}  MAE={bbox['mae']:.7g}  "
        f"RMSE={bbox['rmse']:.7g}  P95={bbox['p95']:.7g}  P99={bbox['p99']:.7g}"
    )
    print(
        'Cls   '
        f"max={cls['max']:.7g}  MAE={cls['mae']:.7g}  "
        f"RMSE={cls['rmse']:.7g}  P95={cls['p95']:.7g}  P99={cls['p99']:.7g}"
    )


def print_candidate(label: str, output: np.ndarray, candidate: int, fp16_bbox: np.ndarray):
    bbox, _, best_class, best_score, person = split_output(output)
    print(
        f'{label:<14} '
        f'class={int(best_class[candidate]):2d}  '
        f'best={best_score[candidate]:.7f}  '
        f'person={person[candidate]:.7f}  '
        f'IoU={bbox_iou_xywh(fp16_bbox[:, candidate], bbox[:, candidate]):.6f}  '
        f'bbox={bbox[:, candidate]}'
    )


def print_threshold_counts(outputs: dict, thresholds):
    parsed = {name: split_output(output) for name, output in outputs.items()}

    print('\n=== Person candidate counts (argmax class == 0) ===')
    print('threshold' + ''.join(f'{name:>14}' for name in outputs))

    for threshold in thresholds:
        row = f'{threshold:>9.2f}'
        for name in outputs:
            _, _, best_class, best_score, _ = parsed[name]
            count = int(np.sum((best_class == 0) & (best_score >= threshold)))
            row += f'{count:>14d}'
        print(row)


def print_high_conf_reference_set(outputs: dict, threshold: float):
    fp16 = outputs['FP16']
    fp16_bbox, _, fp16_class, fp16_score, _ = split_output(fp16)

    candidates = np.flatnonzero((fp16_class == 0) & (fp16_score >= threshold))

    print(
        f'\n=== FP16 person reference set @ {threshold:.2f} '
        f'({len(candidates)} candidates) ==='
    )

    if len(candidates) == 0:
        print('No FP16 reference candidates at this threshold.')
        return

    parsed = {name: split_output(output) for name, output in outputs.items()}

    score_errors = {name: [] for name in outputs if name != 'FP16'}
    ious = {name: [] for name in outputs if name != 'FP16'}

    for candidate in candidates:
        print(
            f'\ncandidate {candidate}: '
            f'FP16 score={fp16_score[candidate]:.7f} '
            f'bbox={fp16_bbox[:, candidate]}'
        )

        for name in outputs:
            if name == 'FP16':
                continue

            bbox, _, best_class, best_score, person = parsed[name]
            score_err = abs(float(person[candidate] - fp16_score[candidate]))
            iou = bbox_iou_xywh(fp16_bbox[:, candidate], bbox[:, candidate])
            score_errors[name].append(score_err)
            ious[name].append(iou)

            print(
                f'  {name:<12} '
                f'class={int(best_class[candidate]):2d} '
                f'best={best_score[candidate]:.7f} '
                f'person={person[candidate]:.7f} '
                f'score_err={score_err:.7f} '
                f'IoU={iou:.6f}'
            )

    print('\n--- High-confidence reference summary ---')
    for name in score_errors:
        print(
            f'{name:<12} '
            f'mean_score_abs_err={np.mean(score_errors[name]):.7f}  '
            f'mean_IoU={np.mean(ious[name]):.6f}  '
            f'min_IoU={np.min(ious[name]):.6f}'
        )


def print_top_person(label: str, output: np.ndarray, topk: int):
    _, _, best_class, best_score, person = split_output(output)
    top = np.argsort(person)[-topk:][::-1]

    print(f'\n=== {label}: top {topk} person scores ===')
    for i in top:
        print(
            f'{int(i):4d}  person={person[i]:.7f}  '
            f'best_class={int(best_class[i]):2d}  '
            f'best_score={best_score[i]:.7f}'
        )


def print_recovery(fp16: np.ndarray, int8: np.ndarray, mixed: np.ndarray, candidate: int):
    fp16_bbox, _, _, _, fp16_person = split_output(fp16)
    int8_bbox, _, _, _, int8_person = split_output(int8)
    mix_bbox, _, _, _, mix_person = split_output(mixed)

    base_score_error = abs(float(int8_person[candidate] - fp16_person[candidate]))
    mixed_score_error = abs(float(mix_person[candidate] - fp16_person[candidate]))

    if base_score_error > 0.0:
        score_recovery = (1.0 - mixed_score_error / base_score_error) * 100.0
    else:
        score_recovery = float('nan')

    int8_iou = bbox_iou_xywh(fp16_bbox[:, candidate], int8_bbox[:, candidate])
    mixed_iou = bbox_iou_xywh(fp16_bbox[:, candidate], mix_bbox[:, candidate])

    print('\n=== Mixed-precision recovery ===')
    print(f'candidate               : {candidate}')
    print(f'FP16 person score       : {fp16_person[candidate]:.7f}')
    print(f'INT8 person score       : {int8_person[candidate]:.7f}')
    print(f'Mixed person score      : {mix_person[candidate]:.7f}')
    print(f'INT8 score abs error    : {base_score_error:.7f}')
    print(f'Mixed score abs error   : {mixed_score_error:.7f}')
    print(f'Score error recovery    : {score_recovery:.2f}%')
    print(f'INT8 bbox IoU vs FP16   : {int8_iou:.6f}')
    print(f'Mixed bbox IoU vs FP16  : {mixed_iou:.6f}')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Compare YOLOv8 TensorRT raw FP32-boundary outputs for '
            'FP16, INT8 and an optional mixed-precision engine.'
        )
    )
    parser.add_argument(
        '--fp16',
        default='data/outputs/tensorrt_fp16_output.bin',
        help='FP16 TensorRT raw output (.bin, float32 boundary output)',
    )
    parser.add_argument(
        '--int8',
        default='data/outputs/tensorrt_int8_output.bin',
        help='INT8 TensorRT raw output (.bin, float32 boundary output)',
    )
    parser.add_argument(
        '--mixed',
        default=None,
        help='Optional mixed-precision raw output to evaluate',
    )
    parser.add_argument('--candidate', type=int, default=8191)
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[0.1, 0.3, 0.5, 0.6, 0.7],
    )
    parser.add_argument('--reference-threshold', type=float, default=0.7)
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()

    if not 0 <= args.candidate < NUM_CANDIDATES:
        raise ValueError(f'--candidate must be in [0, {NUM_CANDIDATES - 1}]')

    outputs = {
        'FP16': load_output(args.fp16),
        'INT8': load_output(args.int8),
    }
    if args.mixed:
        outputs['MIXED'] = load_output(args.mixed)

    fp16 = outputs['FP16']
    fp16_bbox, _, _, _, _ = split_output(fp16)

    print('=== Inputs ===')
    paths = {'FP16': args.fp16, 'INT8': args.int8, 'MIXED': args.mixed}
    for name in outputs:
        print(f'{name:<6}: {Path(paths[name])}')

    for name, output in outputs.items():
        if name != 'FP16':
            print_error_stats(name, fp16, output)

    print(f'\n=== Candidate {args.candidate} ===')
    for name, output in outputs.items():
        print_candidate(name, output, args.candidate, fp16_bbox)

    print('\n=== Best person candidate ===')
    for name, output in outputs.items():
        _, _, _, _, person = split_output(output)
        idx = int(np.argmax(person))
        print(f'{name:<14} candidate={idx:4d}  person={person[idx]:.7f}')

    print_threshold_counts(outputs, args.thresholds)
    print_high_conf_reference_set(outputs, args.reference_threshold)

    for name, output in outputs.items():
        print_top_person(name, output, args.topk)

    if 'MIXED' in outputs:
        print_recovery(outputs['FP16'], outputs['INT8'], outputs['MIXED'], args.candidate)


if __name__ == '__main__':
    main()
