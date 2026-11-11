# 首轮探针扰动较大后，缩小到 MatMul_248 的输入/输出；保留原诊断数据和冻结来源。
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from prepare_activation_probe import ROOT, digest

OUT = ROOT / 'validation/fp16-activations-narrow'
PARENT = ROOT / 'validation/fp16-activations'


def main():
    parent = json.loads((PARENT / 'manifest.json').read_text())
    for name, sha in parent['sha256'].items():
        assert digest(Path(name)) == sha, name
    OUT.mkdir(parents=True, exist_ok=False)
    original = onnx.load(str(ROOT / 'models/rtmpose_hand.onnx'))
    model = onnx.load(str(ROOT / 'models/rtmpose_hand.onnx'))
    specs = [('683', 'Mul_247', 'Mul', [-1, 21, 512]),
             ('685', 'MatMul_248', 'MatMul', [-1, 21, 256]),
             ('simcc_x', 'MatMul_251', 'MatMul', [-1, 21, 512]),
             ('simcc_y', 'MatMul_252', 'MatMul', [-1, 21, 512])]
    outputs = []
    names = []
    for index, (name, node, op, shape) in enumerate(specs):
        names.append(name)
        if name not in {value.name for value in model.graph.output}:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, ['batch'] + shape[1:]))
        outputs.append({'name': name, 'node': node, 'op': op, 'shape': shape, 'file': 'probe_%02d' % index})
    assert [value.SerializeToString() for value in original.graph.node] == [value.SerializeToString() for value in model.graph.node]
    assert [value.SerializeToString() for value in original.graph.initializer] == [value.SerializeToString() for value in model.graph.initializer]
    onnx.checker.check_model(model)
    target = OUT / 'probe.onnx'
    onnx.save(model, str(target))
    samples = json.loads((PARENT / 'config.json').read_text())['samples']
    config = {'onnx': str(target), 'outputs': outputs, 'samples': samples}
    (OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(target), sess_options=options, providers=['CPUExecutionProvider'])
    first = np.load(samples[0]['tensor'], allow_pickle=False)
    double = session.run(names, {'input': np.concatenate([first, first])})
    for output, value in zip(outputs, double):
        assert value.shape == tuple([2] + output['shape'][1:])
    raw = OUT / 'ort'
    raw.mkdir()
    max_error = 0.
    for sample in samples:
        values = session.run(names, {'input': np.load(sample['tensor'], allow_pickle=False)})
        with np.load(ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz'), allow_pickle=False) as frozen:
            for name in ['simcc_x', 'simcc_y']:
                max_error = max(max_error, float(np.max(np.abs(values[names.index(name)] - frozen[name]))))
        for output, value in zip(outputs, values):
            assert value.shape == tuple([1] + output['shape'][1:]) and np.isfinite(value).all()
            value.tofile(raw / (sample['id'] + '_' + output['file'] + '.bin'))
    (raw / 'complete.txt').write_text('55\n')
    locked = dict(parent['sha256'])
    for path in [target, OUT / 'config.json', Path(__file__).resolve(), PARENT / 'manifest.json'] + list(raw.glob('*.bin')):
        locked[str(path)] = digest(path)
    (OUT / 'manifest.json').write_text(json.dumps({'sha256': locked, 'ort_probe_vs_original_simcc_max_abs': max_error,
        'scope': 'Narrow MatMul_248 input/output; parent raw retained; 55 dev, batch1, 2GiB, profile1/2/4, TF32off'}, indent=2) + '\n')
    print('Narrow probe prepared; ORT final perturbation:', max_error)


if __name__ == '__main__':
    main()
