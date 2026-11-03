# 聚合原始延迟样本；保留全部离群值，检查每轮完整性，区分 CUDA event 与 host wall time。
import csv
from collections import defaultdict
from pathlib import Path
import json
import re

import numpy as np

from verify_trt_fp32 import ROOT, digest, write_json

COLUMNS = ['preprocess_ms', 'h2d_ms', 'inference_ms', 'd2h_ms', 'decode_ms', 'e2e_ms']


def summarize_values(values):
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError('Expected nonempty finite nonnegative timings')
    return {
        'count': int(data.size),
        'mean_ms': float(data.mean()),
        'p50_ms': float(np.percentile(data, 50)),
        'p95_ms': float(np.percentile(data, 95)),
        'min_ms': float(data.min()),
        'max_ms': float(data.max()),
    }


def validate_iterations(values, count):
    if sorted(values) != list(range(count)):
        raise ValueError('Missing or duplicate benchmark iterations')


def main():
    directory = ROOT / 'validation/cpp-2g-performance'
    raw = directory / 'timings.csv'
    groups = defaultdict(list)
    pooled = defaultdict(list)
    with raw.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        key = (int(row['round']), int(row['batch']), row['mode'], row['precision'])
        groups[key].append(row)
        pooled[key[1:]].append(row)
        precision_index = {'fp32': 0, 'fp16': 1}[row['precision']]
        assert int(row['order']) == (precision_index - int(row['round'])) % 2
        assert int(row['start_sample']) == int(row['iteration']) % 55
    expected = {(r, b, mode, precision) for r in range(3) for b in [1, 2, 3, 4]
                for mode in ['npy', 'image'] for precision in ['fp32', 'fp16']}
    assert set(groups) == expected
    per_round = []
    aggregate = []
    for key, values in sorted(groups.items()):
        validate_iterations([int(row['iteration']) for row in values], 1000)
        per_round.append({
            'round': key[0], 'batch': key[1], 'mode': key[2], 'precision': key[3],
            'metrics': {column: summarize_values([float(row[column]) for row in values])
                        for column in COLUMNS},
        })
    for key, values in sorted(pooled.items()):
        metrics = {column: summarize_values([float(row[column]) for row in values])
                   for column in COLUMNS}
        aggregate.append({
            'batch': key[0], 'mode': key[1], 'precision': key[2], 'metrics': metrics,
            'hands_per_second_from_mean_e2e': key[0] * 1000 / metrics['e2e_ms']['mean_ms'],
            'hands_per_second_from_mean_event_inference': key[0] * 1000 / metrics['inference_ms']['mean_ms'],
        })
    speeds = []
    for batch in [1, 2, 3, 4]:
        for mode in ['npy', 'image']:
            pair = {row['precision']: row['metrics'] for row in aggregate
                    if row['batch'] == batch and row['mode'] == mode}
            speeds.append({
                'batch': batch, 'mode': mode,
                'p50_event_inference_speedup': pair['fp32']['inference_ms']['p50_ms'] / pair['fp16']['inference_ms']['p50_ms'],
                'p50_e2e_speedup': pair['fp32']['e2e_ms']['p50_ms'] / pair['fp16']['e2e_ms']['p50_ms'],
            })
    provenance = json.loads((directory / 'benchmark-inputs.json').read_text())
    for relative, expected_sha in provenance['files_sha256'].items():
        assert digest(ROOT.parent / relative) == expected_sha, relative
    completion = json.loads((directory / 'benchmark-complete.json').read_text())
    assert completion['timings_sha256'] == digest(raw)
    assert completion['inputs_sha256'] == digest(directory / 'benchmark-inputs.json')
    for name, field in [
        ('environment-before.json', 'environment_before_sha256'),
        ('environment-after.json', 'environment_after_sha256'),
        ('tegrastats.log', 'telemetry_sha256'),
        ('benchmark.log', 'log_sha256'),
    ]:
        assert digest(directory / name) == completion[field], name
    assert digest(directory / 'tracked-changes.patch') == provenance['tracked_patch_sha256']
    configurations = [json.loads((ROOT / 'configs' / name).read_text()) for name in
                      ['rtmpose_hand_builder_fp32_2g.json', 'rtmpose_hand_builder.json']]
    assert configurations[0].pop('selected_precision') == 'fp32'
    assert configurations[1].pop('selected_precision') == 'fp16'
    assert configurations[0] == configurations[1]
    environment = {
        name: json.loads((directory / ('environment-' + name + '.json')).read_text())
        for name in ['before', 'after']
    }
    telemetry = (directory / 'tegrastats.log').read_text()
    temperatures = {}
    for name in ['CPU', 'GPU']:
        values = [float(value) for value in re.findall(name + r'@([0-9.]+)C', telemetry)]
        if values:
            temperatures[name] = {'min_c': min(values), 'max_c': max(values), 'samples': len(values)}
    report = {
        'status': 'performance_measurement_complete',
        'summary_script_sha256': digest(Path(__file__).resolve()),
        'compiler_configuration': 'Release, -O3 -DNDEBUG, gnu++17',
        'fp16_accuracy_accepted': False,
        'warmup_per_group': 100, 'iterations_per_group': 1000, 'rounds': 3,
        'raw_samples': len(rows), 'raw_sha256': digest(raw),
        'provenance': provenance,
        'environment': environment, 'temperature_ranges': temperatures,
        'completion': completion,
        'consistency': json.loads((directory / 'consistency.json').read_text()),
        'per_round': per_round, 'aggregate': aggregate, 'speedups': speeds,
        'timing_scope': 'CUDA event stream intervals for H2D/inference/D2H; steady_clock for preprocessing/decode/E2E; image loading, engine load, shape setup and file writes excluded. E2E includes instrumentation and synchronous Runner allocations.',
        'dataset_scope': '55 development hands; batch can span different source images; no detector, validation or calibration execution',
        'power_state_source': 'User supplied jetson_clocks --show: MAXN, CPU 2188800 kHz, GPU 930750000 Hz, EMC 3199000000 Hz, override=1',
    }
    write_json(directory / 'performance.json', report)
    write_json(ROOT / 'benchmarks/cpp-2g-performance.json', report)
    print(json.dumps(speeds, indent=2))


if __name__ == '__main__':
    main()
