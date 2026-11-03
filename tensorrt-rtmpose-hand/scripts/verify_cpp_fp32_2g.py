# 验证新的同构建条件 FP32 对照，并记录 FP16 相对它的偏差；不改旧基线。
import numpy as np
from verify_trt_fp16 import (
    ROOT, digest, read_json, write_json, required_artifacts, validate_completion,
    validate_build_completion, load_batch, compare_raw, decode, BATCHES,
)


def main():
    directory = ROOT / 'validation/cpp-2g-performance'
    fp32 = directory / 'fp32'
    fp16 = ROOT / 'validation/trt-fp16-cpp-2g'
    cases_path = ROOT / 'validation/trt-fp32/cases.json'
    samples = read_json(cases_path)['samples']
    assert len(samples) == 55
    engine32 = ROOT / 'engines/rtmpose_hand_fp32_cpp_2g.engine'
    engine16 = ROOT / 'engines/rtmpose_hand_fp16_cpp_2g.engine'
    validate_build_completion(fp16, engine16)
    snapshot = read_json(ROOT / 'benchmarks/fp16-cpp-2g.json')
    lock = read_json(fp16 / 'input-lock.json')
    # 新 benchmark 增加了 CMake 目标；模型、参考、数据和执行器源码仍逐项核验。
    for path, checksum in lock['assets_sha256'].items():
        if path.endswith('/CMakeLists.txt'):
            continue
        assert digest(__import__('pathlib').Path(path)) == checksum, path
    initial = read_json(directory / 'build-inputs.json')
    assert digest(ROOT / 'configs/rtmpose_hand_builder_fp32_2g.json') == initial['config_sha256']
    log = (directory / 'build-fp32.log').read_text()
    assert '; sha256=' + digest(engine32) in log and 'precision=fp32; TF32=off; workspace_mib=2048' in log
    gates = read_json(ROOT / 'validation/trt-fp32/acceptance.json')
    ort = [[], []]
    for sample in samples:
        with np.load(ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz'), allow_pickle=False) as data:
            for index, name in enumerate(['simcc_x', 'simcc_y']):
                ort[index].append(data[name])
    ort = [np.concatenate(values) for values in ort]
    checks = []
    failures = []
    for mode in ['npy', 'image']:
        for folder, engine in [(fp32, engine32), (fp16, engine16)]:
            validate_completion(folder / mode, digest(engine), digest(cases_path), mode, 55, required_artifacts(samples))
        input_error = 0.
        for sample in samples:
            raw = np.fromfile(fp32 / mode / (sample['id'] + '_input.bin'), dtype=np.float32)
            expected = np.load(sample['tensor'], allow_pickle=False).ravel()
            assert raw.shape == expected.shape and np.isfinite(raw).all()
            input_error = max(input_error, float(np.max(np.abs(raw - expected))))
        if input_error != 0:
            failures.append(mode + ' input mismatch')
        single, _ = load_batch(fp32 / mode, samples, 1)
        for batch in BATCHES:
            raw, decoded = load_batch(fp32 / mode, samples, batch)
            low, _ = load_batch(fp16 / mode, samples, batch)
            metric = compare_raw(raw, ort, samples)
            batch_error = max(float(np.max(np.abs(raw[i] - single[i]))) for i in range(2))
            python = decode(*raw, samples)
            packed = np.concatenate([python[0], python[1], python[2][..., None], python[3][..., None]], axis=-1)
            decode_error = float(np.max(np.abs(packed - decoded)))
            for key in ['input_coordinate_mean_l2_px', 'input_coordinate_p95_l2_px', 'input_coordinate_max_l2_px', 'score_max_abs', 'valid_mask_flip_count']:
                if metric[key] > gates[key]:
                    failures.append('%s b%d %s' % (mode, batch, key))
            if max(value['max_abs'] for value in metric['simcc'].values()) > gates['simcc_max_abs']:
                failures.append('%s b%d simcc' % (mode, batch))
            if batch_error > gates['simcc_max_abs'] or decode_error > gates['cpp_decode_source_max_abs_px']:
                failures.append('%s b%d batch/decode' % (mode, batch))
            checks.append({'mode': mode, 'batch': batch, 'fp32_vs_ort': metric,
                           'input_max_abs': input_error, 'cpp_decode_max_abs': decode_error,
                           'batch_vs_single_max_abs': batch_error,
                           'fp16_vs_matched_fp32': compare_raw(low, raw, samples)})
    report = {'status': 'passed' if not failures else 'failed', 'failures': failures,
              'fp32_engine_sha256': digest(engine32), 'fp16_engine_sha256': digest(engine16),
              'cases_sha256': digest(cases_path), 'gates': gates, 'checks': checks,
              'build_log_sha256': digest(directory / 'build-fp32.log'),
              'historical_fp16_accuracy_status': snapshot['status']}
    write_json(directory / 'consistency.json', report)
    print(report['status'], failures)
    assert not failures, failures


if __name__ == '__main__':
    main()
