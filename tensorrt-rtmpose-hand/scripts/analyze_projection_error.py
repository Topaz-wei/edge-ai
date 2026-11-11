# 使用固定 FP32 权重，在 FP64 中拆分 MatMul_248 的上游误差传播与执行残差；不修改模型。
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from capture_activation_probe import verify_capture, digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/fp16-activations-narrow'


def decompose(
    input32,
    input16,
    output32,
    output16,
    weight
):
    arrays = [value.astype(np.float64) for value in [input32, input16, output32, output16, weight]]
    if not all(np.isfinite(value).all() for value in arrays):
        return {'status': 'nonfinite'}
    x32, x16, y32, y16, w = arrays
    ideal32 = x32 @ w
    ideal16 = x16 @ w
    if ideal32.shape != y32.shape or ideal16.shape != y16.shape:
        raise ValueError('Projection shape mismatch')
    propagated = ideal16 - ideal32
    execution = (y16 - ideal16) - (y32 - ideal32)
    total = y16 - y32
    norm = float(np.linalg.norm(total))
    return {
        'status': 'finite',
        'total_l2': norm,
        'propagated_l2': float(np.linalg.norm(propagated)),
        'execution_residual_l2': float(np.linalg.norm(execution)),
        'execution_to_total_norm_ratio': float(np.linalg.norm(execution) / norm) if norm else None,
        'fp32_implementation_residual_l2': float(np.linalg.norm(y32 - ideal32)),
        'closure_max_abs': float(np.max(np.abs(total - propagated - execution))),
    }


def main():
    for mode in ['fp32', 'fp16']:
        verify_capture(OUT / (mode + '-verified'))
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, sha in manifest['sha256'].items():
        assert digest(Path(name)) == sha
    config = json.loads((OUT / 'config.json').read_text())
    model = onnx.load(str(ROOT / 'models/rtmpose_hand.onnx'))
    node = next(node for node in model.graph.node if node.name == 'MatMul_248')
    assert list(node.input) == ['683', '871'] and list(node.output) == ['685']
    weight = numpy_helper.to_array(next(value for value in model.graph.initializer if value.name == '871'))
    assert weight.shape == (512, 256)
    outputs = {value['name']: value for value in config['outputs']}
    records = []
    for sample in config['samples']:
        values = []
        for name, mode in [('683', 'fp32'), ('683', 'fp16'), ('685', 'fp32'), ('685', 'fp16')]:
            output = outputs[name]
            path = OUT / (mode + '-verified') / (sample['id'] + '_' + output['file'] + '.bin')
            values.append(np.fromfile(path, dtype=np.float32).reshape([1] + output['shape'][1:]))
        record = {'sample_id': sample['id']}
        record.update(decompose(*values, weight))
        records.append(record)
    defined = [row['execution_to_total_norm_ratio'] for row in records if row['status'] == 'finite' and row['execution_to_total_norm_ratio'] is not None]
    report = {'node': 'MatMul_248', 'samples': records,
              'execution_to_total_norm_ratio_min_median_p95_max': np.percentile(defined, [0, 50, 95, 100]).tolist() if defined else None,
              'interpretation': 'total = propagated input error through fixed FP32 weights + execution residual difference. Norm ratios are NOT additive percentages; residual includes weight rounding, compute, and tactics.',
              'script_sha256': digest(Path(__file__).resolve()),
              'manifest_sha256': digest(OUT / 'manifest.json')}
    (OUT / 'projection-decomposition.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(report['execution_to_total_norm_ratio_min_median_p95_max'])


if __name__ == '__main__':
    main()
