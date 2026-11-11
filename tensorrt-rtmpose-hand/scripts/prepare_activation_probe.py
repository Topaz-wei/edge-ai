# 为批准的分段诊断生成只增加输出声明的 ONNX，验证动态形状并冻结开发集与源文件身份。
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/fp16-activations'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'models/rtmpose_hand.onnx'
    assert digest(source) == '39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29'
    model = onnx.load(str(source))
    names = ['413', '486', '559', '624', '625', '644', '647', '658', '663', '677', '681', '682', '685', '687', 'simcc_x', 'simcc_y']
    producers = {name: node for node in model.graph.node for name in node.output}
    cases = json.loads((ROOT / 'validation/trt-fp32/cases.json').read_text())['samples']
    assert len(cases) == 55
    old_lock = json.loads((ROOT / 'validation/trt-fp16-cpp-2g/input-lock.json').read_text())['assets_sha256']
    locked = {}
    for sample in cases:
        tensor = Path(sample['tensor'])
        assert digest(tensor) == old_lock[str(tensor)]
        locked[str(tensor)] = digest(tensor)
    original_outputs = {value.name for value in model.graph.output}
    for name in names:
        assert name in producers
        if name not in original_outputs:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    first = np.load(cases[0]['tensor'], allow_pickle=False)
    one = session.run(names, {'input': first})
    two = session.run(names, {'input': np.concatenate([first, first])})
    outputs = []
    for index, (name, value, doubled) in enumerate(zip(names, one, two)):
        assert value.dtype == np.float32 and value.shape[0] == 1
        assert doubled.shape == (2,) + value.shape[1:]
        shape = ['batch'] + list(value.shape[1:])
        output = next(item for item in model.graph.output if item.name == name)
        output.CopyFrom(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
        outputs.append({'name': name, 'node': producers[name].name, 'op': producers[name].op_type,
                        'shape': [-1] + list(value.shape[1:]), 'file': 'probe_%02d' % index})
    original = onnx.load(str(source))
    assert [v.SerializeToString() for v in original.graph.node] == [v.SerializeToString() for v in model.graph.node]
    assert [v.SerializeToString() for v in original.graph.initializer] == [v.SerializeToString() for v in model.graph.initializer]
    onnx.checker.check_model(model)
    target = OUT / 'probe.onnx'
    onnx.save(model, str(target))
    config = {'onnx': str(target), 'outputs': outputs, 'samples': cases}
    (OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    session = ort.InferenceSession(str(target), sess_options=options, providers=['CPUExecutionProvider'])
    raw = OUT / 'ort'
    raw.mkdir()
    max_final_error = 0.
    for sample in cases:
        values = session.run(names, {'input': np.load(sample['tensor'], allow_pickle=False)})
        with np.load(ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz'), allow_pickle=False) as reference:
            for name in ['simcc_x', 'simcc_y']:
                max_final_error = max(max_final_error, float(np.max(np.abs(values[names.index(name)] - reference[name]))))
        for output, value in zip(outputs, values):
            assert np.isfinite(value).all()
            value.tofile(raw / (sample['id'] + '_' + output['file'] + '.bin'))
    (raw / 'complete.txt').write_text('55\n')
    paths = [source, target, OUT / 'config.json', ROOT / 'configs/rtmpose_hand_reference.json',
             ROOT.parent / 'build-shared/tensorrt-rtmpose-hand/rtmpose_probe',
             ROOT / 'apps/activation_probe.cpp', Path(__file__).resolve(),
             ROOT.parent / 'trt-runtime/src/builder.cpp', ROOT.parent / 'trt-runtime/src/runner.cpp']
    locked.update({str(path): digest(path) for path in paths})
    locked.update({str(path): digest(path) for path in raw.glob('*.bin')})
    (OUT / 'manifest.json').write_text(json.dumps({'sha256': locked, 'ort_probe_vs_original_simcc_max_abs': max_final_error,
        'scope': '55 development; batch1; profile1/2/4; 2GiB; TF32off; nodes/weights unchanged'}, indent=2) + '\n')
    print('Probe outputs:', len(outputs), 'ORT final perturbation:', max_final_error)


if __name__ == '__main__':
    main()
