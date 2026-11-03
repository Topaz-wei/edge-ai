# 本脚本只读已有开发集 SimCC，分析 FP16 峰切换；不构建 engine、不修改解码或验收门槛。
import csv
import json
from pathlib import Path

import numpy as np

from verify_trt_fp16 import (
    ROOT, digest, read_json, write_json, load_batch, required_artifacts,
    validate_completion, validate_build_completion,
)


def axis_metrics(
    reference,
    actual
):
    if reference.ndim != 1 or reference.size < 2 or reference.shape != actual.shape:
        raise ValueError('Expected matching one-dimensional SimCC arrays')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite SimCC')
    result = {}
    # stable 排序使并列峰优先取最小 bin，与冻结 argmax 规则一致。
    for name, values in [('reference', reference), ('actual', actual)]:
        order = np.argsort(
            -values,
            kind='stable'
        )
        first, second = int(order[0]), int(order[1])
        result.update({
            name + '_top1_bin': first,
            name + '_top2_bin': second,
            name + '_top1_value': float(values[first]),
            name + '_top2_value': float(values[second]),
            name + '_margin': float(values[first]) - float(values[second]),
        })
    first = result['reference_top1_bin']
    second = result['actual_top1_bin']
    result.update({
        'shift_bins': second - first,
        'shift_input_px': (second - first) / 2.0,
        'reference_value_at_actual_winner': float(reference[second]),
        'actual_value_at_reference_winner': float(actual[first]),
        'reference_winner_gap': float(reference[first]) - float(reference[second]),
        'actual_winner_gap': float(actual[second]) - float(actual[first]),
        'max_abs': float(np.max(np.abs(actual - reference))),
    })
    return result


def plot_curves(
    curves,
    output
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7)
    )
    for column, name in enumerate(['x', 'y']):
        values = curves[name]
        winners = [int(np.argmax(value)) for value in values.values()]
        for row in range(2):
            axis = axes[row, column]
            for label, distribution in values.items():
                axis.plot(
                    distribution,
                    label=label,
                    linewidth=1.3
                )
            if row == 1:
                start = max(0, min(winners) - 12)
                end = min(511, max(winners) + 12)
                axis.set_xlim(start, end)
                window = np.array([value[start:end + 1] for value in values.values()])
                axis.set_ylim(float(window.min()) - .0005, float(window.max()) + .0005)
            axis.set_title(name.upper() + (' full distribution' if row == 0 else ' peak region'))
            axis.set_xlabel('SimCC bin (1 bin = 0.5 input pixel)')
            axis.set_ylabel('Raw score (no softmax)')
            axis.grid(alpha=.2)
    axes[0, 0].legend()
    figure.suptitle(curves['sample_id'] + ' / keypoint ' + str(curves['keypoint_id']) + ' (zero-based)')
    figure.tight_layout()
    figure.savefig(output / 'worst-curves.svg')
    figure.savefig(
        output / 'worst-curves.png',
        dpi=150
    )
    plt.close(figure)


def main():
    current = ROOT / 'validation/trt-fp16-cpp-2g'
    fp32 = ROOT / 'validation/trt-fp32'
    output = ROOT / 'validation/simcc-margin'
    output.mkdir(
        parents=True,
        exist_ok=True
    )
    report = read_json(current / 'report.json')
    assert digest(ROOT / report['config']['engine']) == report['engine_sha256']
    validate_build_completion(
        current,
        ROOT / report['config']['engine']
    )
    cases = read_json(current / 'cases.json')
    samples = cases['samples']
    assert len(samples) == 55
    assert digest(current / 'cases.json') == report['cases_sha256'] == digest(fp32 / 'cases.json')
    lock = read_json(current / 'input-lock.json')
    for path, expected in lock['assets_sha256'].items():
        assert digest(Path(path)) == expected, 'Frozen source changed: ' + path
    for folder, sha in [(current, report['engine_sha256']), (fp32, report['fp32_engine_sha256'])]:
        validate_completion(
            folder / 'npy',
            sha,
            report['cases_sha256'],
            'npy',
            len(samples),
            required_artifacts(samples)
        )
    actual, _ = load_batch(
        current / 'npy',
        samples,
        1
    )
    baseline, _ = load_batch(
        fp32 / 'npy',
        samples,
        1
    )
    ort = [[], []]
    for sample in samples:
        with np.load(
            ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz'),
            allow_pickle=False
        ) as data:
            for index, name in enumerate(['simcc_x', 'simcc_y']):
                ort[index].append(data[name])
    ort = [np.concatenate(axis) for axis in ort]
    records = []
    summary = {
        'scope': '55 development hands, 21 points each, NPY batch=1; no confidence filtering',
        'keypoint_indexing': 'zero-based; no semantic labels inferred',
        'source_report_sha256': digest(current / 'report.json'),
        'engine_sha256': report['engine_sha256'],
        'fp32_engine_sha256': report['fp32_engine_sha256'],
        'analysis_script_sha256': digest(Path(__file__)),
        'comparisons': {},
    }
    for name, expected in [('ort_fp32', ort), ('trt_fp32_1g', baseline)]:
        points = []
        for index, sample in enumerate(samples):
            for keypoint in range(21):
                axes = {}
                for axis, label in enumerate(['x', 'y']):
                    axes[label] = axis_metrics(
                        expected[axis][index, keypoint],
                        actual[axis][index, keypoint]
                    )
                    records.append({
                        'baseline': name, 'sample_id': sample['id'],
                        'keypoint_id': keypoint, 'axis': label, **axes[label],
                    })
                distance = float(np.hypot(
                    axes['x']['shift_input_px'],
                    axes['y']['shift_input_px']
                ))
                points.append({
                    'sample_id': sample['id'], 'keypoint_id': keypoint,
                    'distance_input_px': distance, 'axes': axes,
                })
        points.sort(
            key=lambda value: value['distance_input_px'],
            reverse=True
        )
        distances = np.array([point['distance_input_px'] for point in points])
        summary['comparisons'][name] = {
            'points': len(points),
            'changed': int(np.count_nonzero(distances)),
            'over_2px': int(np.count_nonzero(distances > 2)),
            'over_5px': int(np.count_nonzero(distances > 5)),
            'over_10px': int(np.count_nonzero(distances > 10)),
            'max_l2_px': float(distances.max()),
            'top_20': points[:20],
        }
    with (output / 'all-keypoint-axis-margins.csv').open(
        'w',
        newline=''
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)
    # 保存最大偏移点的完整分布，避免只凭两个峰值描述形状。
    worst = summary['comparisons']['ort_fp32']['top_20'][0]
    index = next(i for i, sample in enumerate(samples) if sample['id'] == worst['sample_id'])
    keypoint = worst['keypoint_id']
    curves = {'sample_id': worst['sample_id'], 'keypoint_id': keypoint}
    for axis, label in enumerate(['x', 'y']):
        curves[label] = {
            'ort_fp32': ort[axis][index, keypoint].tolist(),
            'trt_fp32_1g': baseline[axis][index, keypoint].tolist(),
            'trt_fp16_2g': actual[axis][index, keypoint].tolist(),
        }
    write_json(
        output / 'worst-curves.json',
        curves
    )
    plot_curves(
        curves,
        output
    )
    summary['csv_sha256'] = digest(output / 'all-keypoint-axis-margins.csv')
    write_json(
        output / 'summary.json',
        summary
    )
    print(json.dumps(
        {name: {key: value for key, value in result.items() if key != 'top_20'}
         for name, result in summary['comparisons'].items()},
        indent=2
    ))
    print(json.dumps(
        worst,
        indent=2
    ))


if __name__ == '__main__':
    main()
