# 分段激活误差分析：核验来源、逐样本统计并检查探针对正式 engine 最终输出的扰动。
import csv
import json
import os
from pathlib import Path
import numpy as np
import onnx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from activation_metrics import metrics, safe_final_comparison
from capture_activation_probe import verify_capture
from verify_trt_fp16 import digest, validate_completion, required_artifacts, load_batch, compare_raw, decode

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('RTMPOSE_PROBE_DIR', str(ROOT / 'validation/fp16-activations'))).resolve()


def write_json(
    path,
    value
):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_output(
    mode,
    sample,
    output
):
    shape = [1 if dim == -1 else dim for dim in output['shape']]
    data = np.fromfile(OUT / (mode if mode == 'ort' else mode + '-verified') / (sample['id'] + '_' + output['file'] + '.bin'), dtype=np.float32)
    if data.size != np.prod(shape):
        raise ValueError('Unexpected activation size')
    return data.reshape(shape)


def main():
    write_json(OUT / 'report.json', {'status': 'incomplete'})
    config = json.loads((OUT / 'config.json').read_text())
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, sha in manifest['sha256'].items():
        assert digest(Path(name)) == sha, name
    samples = config['samples']
    outputs = config['outputs']
    graph = onnx.load(str(ROOT / 'models/rtmpose_hand.onnx')).graph
    producers = {name: node for node in graph.node for name in node.output}
    selected_names = {item['name'] for item in outputs}
    checkpoint_inputs = {}
    for output in outputs:
        pending = list(producers[output['name']].input)
        seen = set()
        nearest = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in selected_names or name == 'input':
                nearest.add(name)
            elif name in producers:
                pending.extend(producers[name].input)
        checkpoint_inputs[output['name']] = sorted(nearest)
    assert len(samples) == 55
    historical_lock = json.loads((ROOT / 'validation/trt-fp16-cpp-2g/input-lock.json').read_text())['assets_sha256']
    for sample in samples:
        reference_path = ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz')
        assert digest(reference_path) == historical_lock[str(reference_path)]
    for mode in ['ort', 'fp32', 'fp16']:
        folder = OUT / (mode if mode == 'ort' else mode + '-verified')
        if mode != 'ort':
            record = verify_capture(folder)
            assert record['mode'] == mode
            assert record['assets_sha256'][str(OUT / (mode + '.engine'))] == digest(OUT / (mode + '.engine'))
        assert (folder / 'complete.txt').read_text().strip() == '55'
        assert len(list(folder.glob('*.bin'))) == len(samples) * len(outputs)
    cases_path = ROOT / 'validation/trt-fp32/cases.json'
    assert json.loads(cases_path.read_text())['samples'] == samples
    baseline_folders = {'fp32': ROOT / 'validation/cpp-2g-performance/fp32/npy',
                        'fp16': ROOT / 'validation/trt-fp16-cpp-2g/npy'}
    approved = {
        'fp32': json.loads((ROOT / 'benchmarks/cpp-2g-performance.json').read_text())['consistency']['fp32_engine_sha256'],
        'fp16': json.loads((ROOT / 'benchmarks/fp16-cpp-2g.json').read_text())['engine_sha256'],
    }
    baseline_raw = {}
    for mode, folder in baseline_folders.items():
        engine = ROOT / ('engines/rtmpose_hand_' + mode + '_cpp_2g.engine')
        assert digest(engine) == approved[mode], 'Original approved engine changed'
        validate_completion(folder, digest(engine), digest(cases_path), 'npy', 55, required_artifacts(samples))
        baseline_raw[mode], _ = load_batch(folder, samples, 1)
    rows = []
    slices = []
    final = {mode: [[], []] for mode in ['ort', 'fp32', 'fp16']}
    for sample in samples:
        for output in outputs:
            data = {mode: read_output(mode, sample, output) for mode in ['ort', 'fp32', 'fp16']}
            for ref, actual in [('fp32', 'fp16'), ('ort', 'fp32'), ('ort', 'fp16')]:
                row = {'sample_id': sample['id'], 'tensor': output['name'], 'node': output['node'], 'comparison': actual + '_vs_' + ref}
                row.update(metrics(data[ref], data[actual]))
                rows.append(row)
            if output['name'] in ['simcc_x', 'simcc_y']:
                axis = ['simcc_x', 'simcc_y'].index(output['name'])
                for mode in final:
                    final[mode][axis].append(data[mode])
                for point in range(21):
                    row = {'sample_id': sample['id'], 'tensor': output['name'], 'keypoint': point}
                    row.update(metrics(data['fp32'][0, point], data['fp16'][0, point]))
                    slices.append(row)
    for name, records in [('per-sample.csv', rows), ('simcc-keypoints.csv', slices)]:
        with (OUT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summaries = []
    for output in outputs:
        for comparison in ['fp16_vs_fp32', 'fp32_vs_ort', 'fp16_vs_ort']:
            subset = [row for row in rows if row['tensor'] == output['name'] and row['comparison'] == comparison]
            summary = {'tensor': output['name'], 'node': output['node'], 'comparison': comparison,
                       'nonfinite_samples': sum(bool(row['reference_nonfinite'] or row['actual_nonfinite']) for row in subset)}
            for key in ['mae', 'rmse', 'max_abs', 'relative_l2', 'cosine', 'norm_ratio']:
                defined = [row for row in subset if row[key] is not None]
                values = np.array([row[key] for row in defined])
                summary[key] = None if not defined else {
                    'min': float(values.min()), 'p5': float(np.percentile(values, 5)),
                    'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
                    'max': float(values.max()), 'undefined_count': len(subset) - len(defined),
                    'worst_sample': min(defined, key=lambda row: row[key])['sample_id'] if key == 'cosine' else max(defined, key=lambda row: abs(row[key] - 1.) if key == 'norm_ratio' else row[key])['sample_id']}
            summaries.append(summary)
    final = {mode: [np.concatenate(axis) for axis in pair] for mode, pair in final.items()}
    perturbation = {mode: safe_final_comparison(final[mode], baseline_raw[mode], samples) for mode in ['fp32', 'fp16']}
    end_comparison = safe_final_comparison(final['fp16'], final['fp32'], samples)
    original_comparison = compare_raw(baseline_raw['fp16'], baseline_raw['fp32'], samples)
    selected = []
    original_coords = [decode(*baseline_raw[mode], samples)[0] for mode in ['fp32', 'fp16']]
    probe_finite = all(np.isfinite(value).all() for mode in ['fp32', 'fp16'] for value in final[mode])
    probe_coords = [decode(*final[mode], samples)[0] for mode in ['fp32', 'fp16']] if probe_finite else None
    for sample_id, point in [('F0012-D000', 3), ('F0010-D001', 4), ('F0031-D000', 0)]:
        index = next(index for index, sample in enumerate(samples) if sample['id'] == sample_id)
        selected.append({'sample_id': sample_id, 'keypoint': point,
                         'original_fp32_xy': original_coords[0][index, point].tolist(),
                         'original_fp16_xy': original_coords[1][index, point].tolist(),
                         'probe_fp32_xy': probe_coords[0][index, point].tolist() if probe_coords is not None else None,
                         'probe_fp16_xy': probe_coords[1][index, point].tolist() if probe_coords is not None else None,
                         'original_error_px': float(np.linalg.norm(original_coords[1][index, point] - original_coords[0][index, point])),
                         'probe_error_px': float(np.linalg.norm(probe_coords[1][index, point] - probe_coords[0][index, point])) if probe_coords is not None else None})
    # 每格是独立样本的 Cosine，不将不同样本拼接成向量。
    heatmap = np.array([[next(row['cosine'] for row in rows if row['sample_id'] == sample['id'] and row['tensor'] == output['name'] and row['comparison'] == 'fp16_vs_fp32') for sample in samples] for output in outputs], dtype=float)
    figure, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    picture = axis.imshow(heatmap, aspect='auto', interpolation='nearest', cmap='viridis')
    axis.set_yticks(range(len(outputs)), [output['node'] for output in outputs])
    axis.set_xticks(range(len(samples)), [sample['id'] for sample in samples], rotation=90, fontsize=6)
    axis.set_title('Diagnostic TRT FP16 vs FP32: per-sample activation cosine')
    figure.colorbar(picture, ax=axis, label='Cosine similarity')
    figure.savefig(OUT / 'cosine-heatmap.png', dpi=150)
    figure.savefig(OUT / 'cosine-heatmap.svg')
    plt.close(figure)
    entries = [value for value in summaries if value['comparison'] == 'fp16_vs_fp32']
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for key, axis in [('relative_l2', axes[0]), ('rmse', axes[1])]:
        for percentile in ['median', 'p95']:
            axis.plot(range(len(entries)), [entry[key][percentile] if entry[key] else np.nan for entry in entries], marker='o', label=percentile)
        axis.set_yscale('log')
        axis.set_ylabel(key)
        axis.set_xticks(range(len(outputs)), [output['node'] for output in outputs], rotation=45, ha='right')
        axis.legend()
    figure.suptitle('Topological checkpoints: branches/residuals are not one serial path')
    figure.savefig(OUT / 'error-profile.png', dpi=150)
    figure.savefig(OUT / 'error-profile.svg')
    plt.close(figure)
    raw_hashes = {str(path.relative_to(OUT)): digest(path) for mode in ['fp32', 'fp16'] for path in (OUT / (mode + '-verified')).glob('*.bin')}
    write_json(OUT / 'raw-sha256.json', raw_hashes)
    report = {'status': 'diagnostic_only', 'summaries': summaries, 'checkpoint_inputs': checkpoint_inputs, 'perturbation': perturbation,
              'probe_fp16_vs_fp32': end_comparison, 'original_fp16_vs_fp32': original_comparison,
              'known_outliers': selected, 'manifest_sha256': digest(OUT / 'manifest.json'),
              'raw_manifest_sha256': digest(OUT / 'raw-sha256.json'),
              'capture_sha256': {mode: digest(OUT / (mode + '-verified') / 'capture.json') for mode in ['fp32', 'fp16']},
              'script_sha256': digest(Path(__file__).resolve()), 'metrics_sha256': digest(ROOT / 'scripts/activation_metrics.py'),
              'engine_sha256': {mode: digest(OUT / (mode + '.engine')) for mode in ['fp32', 'fp16']},
              'limits': 'Cumulative activation errors; probe outputs can alter fusion/tactics. Not sensitive-layer proof.'}
    write_json(OUT / 'report.json', report)
    print(json.dumps({'perturbation': perturbation, 'known_outliers': selected}, indent=2))


if __name__ == '__main__':
    main()
