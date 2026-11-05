# 本脚本核验首轮INT8的构建身份、实际层格式及55个开发样本；不修改reference或门槛。
import argparse
from collections import Counter
import csv
from pathlib import Path
import shutil

import numpy as np

from verify_trt_fp16 import (
    ROOT, digest, read_json, write_json, required_artifacts, validate_completion,
    load_batch, compare_raw, gate_failures, decode, BATCHES,
)

OUTPUT = ROOT / 'validation/trt-int8'
FP32 = ROOT / 'validation/cpp-2g-performance/fp32'
ENGINE = ROOT / 'engines/rtmpose_hand_int8.engine'
CONFIG = ROOT / 'configs/rtmpose_hand_trt_int8.json'


def axis_statistics(
    actual,
    expected
):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Mismatched or nonfinite SimCC')
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    shift = np.argmax(actual, axis=-1) - np.argmax(expected, axis=-1)
    return {
        'rmse': float(np.sqrt(np.mean(delta * delta))),
        'argmax_changed_count': int(np.count_nonzero(shift)),
        'argmax_shift_mean_abs_bins': float(np.mean(np.abs(shift))),
        'argmax_shift_p95_abs_bins': float(np.percentile(np.abs(shift), 95)),
        'argmax_shift_max_abs_bins': int(np.max(np.abs(shift))),
    }


def layer_summary(information):
    counts = Counter()
    details = []
    producers = {}
    for layer in information['Layers']:
        formats = [item.get('Format/Datatype', '') for item in layer.get('Inputs', []) + layer.get('Outputs', [])]
        text = ' '.join(formats).lower()
        types = [name for name in ['int8', 'fp16', 'fp32'] if name in text]
        category = types[0] if len(types) == 1 else ('mixed_' + '_'.join(types) if types else 'non_float_or_unknown')
        counts[category] += 1
        detail = {'name': layer['Name'], 'type': layer.get('LayerType', ''), 'io_precision': category,
                  'formats': formats, 'tactic': layer.get('TacticName', layer.get('TacticValue')),
                  'inputs': [value.get('Name', '') for value in layer.get('Inputs', [])],
                  'outputs': [value.get('Name', '') for value in layer.get('Outputs', [])]}
        details.append(detail)
        for name in detail['outputs']:
            producers[name] = detail
    branches = {}
    for output in ['simcc_x', 'simcc_y']:
        pending = [output]
        found = []
        seen = set()
        for _ in range(4):
            next_inputs = []
            for name in pending:
                layer = producers.get(name)
                if layer and layer['name'] not in seen:
                    seen.add(layer['name'])
                    found.append(layer)
                    next_inputs.extend(layer['inputs'])
            pending = next_inputs
        branches[output] = found
    return {
        'layer_count': len(details), 'io_precision_counts': dict(counts),
        'fp16_tensor_layers': sum('fp16' in value['io_precision'] for value in details),
        'layers': details, 'output_branches': branches,
        'convolution_layers': [value for value in details if 'conv' in value['type'].lower()],
        'interpretation': 'Layer tensor formats only, not fused operator accumulator precision; branches trace up to four producer hops.',
    }


def prepare():
    # 锁定模型/数据/参考与门槛，不读取INT8输出来调整判定。
    initial = read_json(OUTPUT / 'build-inputs.json')
    frozen = dict(initial['assets_sha256'])
    for path, checksum in frozen.items():
        assert digest(Path(path)) == checksum, path
    config = read_json(CONFIG)
    assert config['acceptance'] == read_json(ROOT / 'configs/rtmpose_hand_trt_fp16_cpp_2g.json')['acceptance']
    baseline = read_json(ROOT / 'benchmarks/cpp-2g-performance.json')
    assert baseline['consistency']['status'] == 'passed'
    cases_path = ROOT / 'validation/trt-fp32/cases.json'
    assert digest(cases_path) == baseline['consistency']['cases_sha256']
    cases = read_json(cases_path)
    assert len(cases['samples']) == 55
    order = (ROOT / 'calibration/development_tensor_order.txt').read_text().splitlines()
    assert [str(Path(sample['tensor']).relative_to(ROOT / 'calibration')) for sample in cases['samples']] == order
    fp32_baseline = read_json(ROOT / 'benchmarks/fp32-baseline.json')
    for name, checksum in fp32_baseline['source_sha256'].items():
        path = ROOT.parent / name
        assert digest(path) == checksum, name
        frozen[str(path)] = checksum
    executable = ROOT.parent / 'build-shared/tensorrt-rtmpose-hand/rtmpose_fp32'
    assert digest(executable) == fp32_baseline['executable_sha256']
    paths = [cases_path, executable, ROOT / 'benchmarks/cpp-2g-performance.json',
             ROOT / 'engines/rtmpose_hand_fp32_cpp_2g.engine', OUTPUT / 'build-inputs.json']
    for sample in cases['samples']:
        paths += [Path(sample['tensor']), Path(sample['image']), ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz')]
    # 对照原已验证FP16输入锁中的样本和ORT输出摘要，防止当前数据替换。
    previous = read_json(ROOT / 'validation/trt-fp16-cpp-2g/input-lock.json')['assets_sha256']
    for path in paths:
        checksum = digest(path)
        if str(path) in previous:
            assert checksum == previous[str(path)], str(path)
        frozen[str(path)] = checksum
    for mode in ['npy', 'image']:
        path = FP32 / mode / 'complete.json'
        frozen[str(path)] = digest(path)
        validate_completion(
            FP32 / mode,
            baseline['consistency']['fp32_engine_sha256'],
            digest(cases_path),
            mode,
            55,
            required_artifacts(cases['samples'])
        )
    lock = {'assets_sha256': frozen, 'acceptance': config['acceptance']}
    lock_path = OUTPUT / 'input-lock.json'
    if lock_path.exists():
        assert read_json(lock_path) == lock, 'Input lock changed'
    else:
        write_json(lock_path, lock)
    shutil.copyfile(cases_path, OUTPUT / 'cases.json')
    print('Locked 55 development samples, ORT/matched-FP32 references and unchanged gates.')


def ordered_tensors(identity):
    # 旧版OpenCV FileStorage将以 '[' 开头的字符串解释为序列；保留原始证据。
    if 'ordered_tensors' in identity:
        return identity['ordered_tensors']
    legacy = identity['calibration_shape']
    assert legacy[:12] == [
        'inference_profile', '1/2/4', 'workspace_mib', 2048,
        'fp16', 0, 'tf32', 0, 'input_name', 'input', 'input_dtype', 'FP32'
    ]
    assert len(legacy) == 14 and legacy[12] == 'ordered_tensors'
    return legacy[13]


def build_completion():
    for path, checksum in read_json(OUTPUT / 'build-inputs.json')['assets_sha256'].items():
        assert digest(Path(path)) == checksum, path
    cache = ROOT / 'calibration/rtmpose_hand_int8.cache'
    meta_path = ROOT / 'calibration/rtmpose_hand_int8.cache.meta.json'
    meta = read_json(meta_path)
    expected = read_json(ROOT / 'benchmarks/int8-preparation.json')['identity']
    assert meta['identity'] == expected, 'Cache identity differs from verified preparation'
    assert meta['cache_sha256'] == digest(cache), 'Cache payload changed'
    tensors = ordered_tensors(meta['identity'])
    assert len(tensors) == 352
    for tensor in tensors:
        assert digest(ROOT / 'calibration' / tensor['path']) == tensor['sha256']
    log = (OUTPUT / 'build.log').read_text()
    assert 'INT8 engine saved:' in log and '; sha256=' + digest(ENGINE) in log
    completed = {
        'engine_sha256': digest(ENGINE), 'cache_sha256': digest(cache),
        'cache_meta_sha256': digest(meta_path), 'build_log_sha256': digest(OUTPUT / 'build.log'),
        'build_inputs_sha256': digest(OUTPUT / 'build-inputs.json'),
        'warnings': [line for line in log.splitlines() if any(word in line.lower() for word in
                     ['warning', 'missing', 'scale', 'subnormal', 'int64', 'workspace memory'])],
    }
    path = OUTPUT / 'build-complete.json'
    if path.exists():
        assert read_json(path) == completed
    else:
        write_json(path, completed)
    return completed


def compare():
    write_json(OUTPUT / 'report.json', {'status': 'incomplete'})
    build = build_completion()
    lock = read_json(OUTPUT / 'input-lock.json')
    for path, checksum in lock['assets_sha256'].items():
        assert digest(Path(path)) == checksum, path
    limits = read_json(CONFIG)['acceptance']
    assert lock['acceptance'] == limits
    assert digest(OUTPUT / 'cases.json') == digest(ROOT / 'validation/trt-fp32/cases.json')
    samples = read_json(OUTPUT / 'cases.json')['samples']
    for mode in ['npy', 'image']:
        for folder, engine in [(OUTPUT, ENGINE), (FP32, ROOT / 'engines/rtmpose_hand_fp32_cpp_2g.engine')]:
            validate_completion(folder / mode, digest(engine), digest(OUTPUT / 'cases.json'), mode, 55, required_artifacts(samples))
    inspector = read_json(OUTPUT / 'layers-context.json')
    assert inspector['engine_sha256'] == build['engine_sha256'] and inspector['input_shape'] == [2, 3, 256, 256]
    assert {value['name'] for value in inspector['tensors']} == {'input', 'simcc_x', 'simcc_y'}
    assert all(value['dtype'] == 'DataType.FLOAT' for value in inspector['tensors'])
    layers = layer_summary(inspector['inspector'])
    write_json(OUTPUT / 'layer-summary.json', layers)
    failures = []
    if layers['fp16_tensor_layers']:
        failures.append('Unexpected FP16 layer tensor formats')
    if not any('int8' in category for category in layers['io_precision_counts']):
        failures.append('No observed INT8 layer tensor formats')
    ort = [[], []]
    for sample in samples:
        with np.load(ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz'), allow_pickle=False) as data:
            for index, name in enumerate(['simcc_x', 'simcc_y']):
                ort[index].append(data[name])
    ort = [np.concatenate(values) for values in ort]
    checks = []
    shifts = []
    for mode in ['npy', 'image']:
        input_error = 0.
        for sample in samples:
            actual = np.fromfile(OUTPUT / mode / (sample['id'] + '_input.bin'), dtype=np.float32)
            expected = np.load(sample['tensor'], allow_pickle=False).ravel()
            assert actual.shape == expected.shape and np.isfinite(actual).all()
            input_error = max(input_error, float(np.max(np.abs(actual - expected))))
        if input_error > limits['input_tensor_max_abs']:
            failures.append(mode + ' input mismatch')
        single, _ = load_batch(OUTPUT / mode, samples, 1)
        for batch in BATCHES:
            raw, cpp = load_batch(OUTPUT / mode, samples, batch)
            fp32, _ = load_batch(FP32 / mode, samples, batch)
            python = decode(*raw, samples)
            packed = np.concatenate([python[0], python[1], python[2][..., None], python[3][..., None]], axis=-1)
            decode_error = float(np.max(np.abs(cpp - packed)))
            if decode_error > limits['cpp_decode_max_abs']:
                failures.append('%s b%d decode' % (mode, batch))
            comparisons = {}
            for name, expected in [('ort_fp32', ort), ('trt_fp32_cpp_2g', fp32), ('int8_single', single)]:
                metric = compare_raw(raw, expected, samples)
                for axis, label in enumerate(['simcc_x', 'simcc_y']):
                    metric['simcc'][label].update(axis_statistics(raw[axis], expected[axis]))
                    actual_bins = np.argmax(raw[axis], axis=-1)
                    expected_bins = np.argmax(expected[axis], axis=-1)
                    for index, sample in enumerate(samples):
                        for keypoint in range(21):
                            shifts.append({'mode': mode, 'batch': batch, 'baseline': name,
                                           'sample_id': sample['id'], 'keypoint_id': keypoint, 'axis': label,
                                           'expected_bin': int(expected_bins[index, keypoint]),
                                           'actual_bin': int(actual_bins[index, keypoint]),
                                           'shift_bins': int(actual_bins[index, keypoint] - expected_bins[index, keypoint])})
                comparisons[name] = metric
                failures += ['%s b%d vs %s: %s' % (mode, batch, name, key) for key in gate_failures(metric, limits)]
            checks.append({'mode': mode, 'batch': batch, 'sample_count': 55, 'input_max_abs': input_error,
                           'cpp_decode_max_abs': decode_error, 'comparisons': comparisons})
    with (OUTPUT / 'argmax-shifts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(shifts[0]))
        writer.writeheader()
        writer.writerows(shifts)
    report = {
        'status': 'failed' if failures else 'passed', 'failures': failures,
        'build': build, 'config': read_json(CONFIG), 'config_sha256': digest(CONFIG),
        'checks': checks, 'layer_summary': layers,
        'input_lock_sha256': digest(OUTPUT / 'input-lock.json'),
        'comparison_script_sha256': digest(Path(__file__).resolve()),
        'inspector_sha256': digest(OUTPUT / 'layers-context.json'),
        'argmax_shifts_sha256': digest(OUTPUT / 'argmax-shifts.csv'),
        'fp32_reference_engine_sha256': digest(ROOT / 'engines/rtmpose_hand_fp32_cpp_2g.engine'),
        'completion_sha256': {mode: digest(OUTPUT / mode / 'complete.json') for mode in ['npy', 'image']},
        'calibration_tensors': 352, 'development_tensors': 55, 'final_validation_images_used': 0,
        'scope': 'First INT8 baseline; FP16 disabled; development consistency only; no performance or mixed precision tuning',
    }
    write_json(OUTPUT / 'report.json', report)
    print('INT8 development status:', report['status'])
    print('Layer I/O precision:', layers['io_precision_counts'])
    print('Gate failures:', len(failures))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'build-completion', 'compare'])
    arguments = parser.parse_args()
    if arguments.action == 'prepare':
        prepare()
    elif arguments.action == 'build-completion':
        build_completion()
    else:
        compare()
