# 本脚本导出已冻结开发样本清单，并比较公共 Runner 的 FP32 输出与 ORT 基线。
# 不读取最终 validation 图像，不修改 reference，不依赖置信度过滤误差。
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(
    __file__
).resolve().parents[1]
OUTPUT = ROOT / 'validation/trt-fp32'
REFERENCE_SHA = '3036459e1f6bc230c9059f34ea260070928669f54f9b1bfef3e031279aef0aec'


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read_json(path):
    return json.loads(
        path.read_text()
    )


def write_json(
    path,
    value
):
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ) + '\n'
    )


def rows(path):
    return [json.loads(
        line
    ) for line in path.read_text().splitlines()]


def prepare():
    assert digest(
        ROOT / 'configs/rtmpose_hand_reference.json'
    ) == REFERENCE_SHA
    hands = rows(
        ROOT / 'calibration/hand_boxes.jsonl'
    )
    by_tensor = {row['tensor_path']: row for row in hands}
    frames = {row['frame_id']: row for row in rows(
        ROOT / 'calibration/frames.jsonl'
    )}
    order = (ROOT / 'calibration/development_tensor_order.txt').read_text().splitlines()
    samples = []
    for relative in order:
        hand = by_tensor[relative]
        assert hand['split'] == 'development'
        tensor = ROOT / 'calibration' / relative
        assert digest(
            tensor
        ) == hand['tensor_sha256']
        frame = frames[hand['frame_id']]
        samples.append(
            {'id': hand['hand_id'], 'tensor': str(
                tensor
            ),
                        'image': str(
                ROOT / frame['source_path']
            ), 'bbox': hand['bbox_xyxy']}
        )
    assert len(
        samples
    ) == 55
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )
    write_json(
        OUTPUT / 'cases.json',
        {'samples': samples, 'reference_sha256': REFERENCE_SHA}
    )
    # 阈值在读取 TRT 结果之前确定，不能为了通过测试而回调参考契约。
    write_json(
        OUTPUT / 'acceptance.json',
        {
        'input_tensor_max_abs': 0.0,
        'simcc_max_abs': 1e-4,
        'score_max_abs': 1e-4,
        'input_coordinate_mean_l2_px': 0.1,
        'input_coordinate_p95_l2_px': 0.5,
        'input_coordinate_max_l2_px': 1.0,
        'valid_mask_flip_count': 0,
        'cpp_decode_source_max_abs_px': 1e-9,
        'scope': 'development FP32 engineering gates, not GT accuracy or final validation'
    }
    )
    print(
        'Prepared 55 frozen development samples and acceptance gates.'
    )


def decode(
    x,
    y,
    samples
):
    positions = np.stack(
        [x.argmax(
            -1
        ), y.argmax(
            -1
        )],
        axis=-1
    ).astype(
        np.float32
    )
    scores = np.float32(
        0.5
    ) * (x.max(
        -1
    ) + y.max(
        -1
    ))
    positions[scores <= 0] = -1
    positions /= 2
    source = []
    for index, sample in enumerate(
        samples
    ):
        x1, y1, x2, y2 = sample['bbox']
        center = np.asarray(
            [(x1 + x2) / 2, (y1 + y2) / 2]
        )
        scale = 1.25 * max(
            x2 - x1,
            y2 - y1
        )
        source.append(
            positions[index] / np.array(
                [256, 256]
            ) * scale + center - scale / 2
        )
    return positions, np.asarray(
        source
    ), scores, scores > 0


def chunks(
    total,
    requested
):
    start = 0
    while start < total:
        count = min(
            4,
            requested - start % requested,
            total - start
        )
        yield start, count
        start += count


def check_inputs(
    mode,
    samples
):
    maximum = 0.0
    for sample in samples:
        expected = np.load(
            sample['tensor'],
            allow_pickle=False
        ).ravel()
        actual = np.fromfile(
            OUTPUT / mode / (sample['id'] + '_input.bin'),
            dtype=np.float32
        )
        assert actual.shape == expected.shape and np.isfinite(
            actual
        ).all()
        maximum = max(
            maximum,
            float(
                np.abs(
                    actual - expected
                ).max()
            )
        )
    return maximum


def compare():
    write_json(
        OUTPUT / 'report.json',
        {'status': 'incomplete', 'reason': 'verification in progress'}
    )
    assert digest(
        ROOT / 'configs/rtmpose_hand_reference.json'
    ) == REFERENCE_SHA
    samples = read_json(
        OUTPUT / 'cases.json'
    )['samples']
    limits = read_json(
        OUTPUT / 'acceptance.json'
    )
    reference_report = read_json(
        ROOT / 'configs/rtmpose_hand_reference.verification.json'
    )
    assert reference_report['reference_config_sha256'] == REFERENCE_SHA
    reference = []
    for sample in samples:
        path = ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz')
        with np.load(
            path,
            allow_pickle=False
        ) as data:
            reference.append(
                (data['simcc_x'], data['simcc_y'])
            )
    golden = [np.concatenate(
        [item[axis] for item in reference]
    ) for axis in (0, 1)]
    golden_decoded = decode(
        *golden,
        samples
    )
    checks = []
    failures = []
    for mode in ['npy', 'image']:
        complete = read_json(
            OUTPUT / mode / 'complete.json'
        )
        assert complete['engine_sha256'] == digest(
            ROOT / 'engines/rtmpose_hand_fp32.engine'
        )
        assert complete['cases_sha256'] == digest(
            OUTPUT / 'cases.json'
        )
        assert complete['mode'] == mode and complete['samples'] == len(
            samples
        )
        for artifact in complete['files']:
            assert digest(
                OUTPUT / mode / artifact['path']
            ) == artifact['sha256']
        input_error = check_inputs(
            mode,
            samples
        )
        if input_error > limits['input_tensor_max_abs']:
            failures.append(
                mode + ': input mismatch'
            )
        single = None
        for batch in [1, 2, 3, 4, 6]:
            outputs = [[], []]
            cpp = []
            for start, count in chunks(
                len(
                    samples
                ),
                batch
            ):
                prefix = OUTPUT / mode / ('b%d_%d' % (batch, start))
                for axis, name in enumerate(
                    ['simcc_x', 'simcc_y']
                ):
                    raw = np.fromfile(
                        str(
                            prefix
                        ) + '_' + name + '.bin',
                        dtype=np.float32
                    )
                    assert raw.size == count * 21 * 512 and np.isfinite(
                        raw
                    ).all()
                    outputs[axis].append(
                        raw.reshape(
                            count,
                            21,
                            512
                        )
                    )
                values = np.fromfile(
                    str(
                        prefix
                    ) + '_decoded.bin',
                    dtype=np.float64
                )
                assert values.size == count * 21 * 6 and np.isfinite(
                    values
                ).all()
                cpp.append(
                    values.reshape(
                        count,
                        21,
                        6
                    )
                )
            raw = [np.concatenate(
                parts
            ) for parts in outputs]
            cpp = np.concatenate(
                cpp
            )
            python = decode(
                *raw,
                samples
            )
            expected_cpp = np.concatenate(
                [python[0], python[1], python[2][..., None], python[3][..., None]],
                axis=-1
            )
            decode_error = float(
                np.abs(
                    cpp - expected_cpp
                ).max()
            )
            if batch == 1:
                single = raw
            distances = np.linalg.norm(
                python[0] - golden_decoded[0],
                axis=-1
            )
            source_distances = np.linalg.norm(
                python[1] - golden_decoded[1],
                axis=-1
            )
            entry = {
                'mode': mode, 'requested_batch': batch, 'samples': len(
                    samples
                ),
                'input_tensor_max_abs': input_error,
                'simcc': {name: {'max_abs': float(
                    np.abs(
                        raw[axis] - golden[axis]
                    ).max()
                ),
                                'mae': float(
                                    np.abs(
                                        raw[axis] - golden[axis]
                                    ).mean()
                                )}
                          for axis, name in enumerate(
                              ['simcc_x', 'simcc_y']
                          )},
                'batch_vs_single_max_abs': max(
                    (float(
                        np.abs(
                            raw[axis] - single[axis]
                        ).max()
                    ) for axis in (0, 1))
                ),
                'input_coordinate_mean_l2_px': float(
                    distances.mean()
                ),
                'input_coordinate_p95_l2_px': float(
                    np.percentile(
                        distances,
                        95
                    )
                ),
                'input_coordinate_max_l2_px': float(
                    distances.max()
                ),
                'source_coordinate_max_l2_px': float(
                    source_distances.max()
                ),
                'score_max_abs': float(
                    np.abs(
                        python[2] - golden_decoded[2]
                    ).max()
                ),
                'valid_mask_flip_count': int(
                    np.count_nonzero(
                        python[3] != golden_decoded[3]
                    )
                ),
                'cpp_decode_source_max_abs_px': decode_error
            }
            checks.append(
                entry
            )
            for key in ['input_coordinate_mean_l2_px', 'input_coordinate_p95_l2_px',
                        'input_coordinate_max_l2_px', 'score_max_abs', 'valid_mask_flip_count',
                        'cpp_decode_source_max_abs_px']:
                if entry[key] > limits[key]:
                    failures.append(
                        '%s batch%d %s' % (mode, batch, key)
                    )
            if entry['batch_vs_single_max_abs'] > limits['simcc_max_abs']:
                failures.append(
                    '%s batch%d vs single' % (mode, batch)
                )
            if max(
                (value['max_abs'] for value in entry['simcc'].values())
            ) > limits['simcc_max_abs']:
                failures.append(
                    '%s batch%d simcc' % (mode, batch)
                )
    report = {'status': 'passed' if not failures else 'failed', 'failures': failures,
              'reference_sha256': REFERENCE_SHA,
              'model_sha256': digest(
                  ROOT / 'models/rtmpose_hand.onnx'
              ),
              'engine_sha256': digest(
                  ROOT / 'engines/rtmpose_hand_fp32.engine'
              ),
              'cases_sha256': digest(
                  OUTPUT / 'cases.json'
              ),
              'profile': {'min': [1, 3, 256, 256], 'opt': [2, 3, 256, 256], 'max': [4, 3, 256, 256]},
              'precision': 'FP32; TF32/FP16/INT8 disabled', 'final_validation_used': 0,
              'acceptance': limits, 'checks': checks,
              'runtime': read_json(
                  OUTPUT / 'npy/runtime.json'
              )}
    report['source_sha256'] = {
        relative: digest(ROOT.parent / relative)
        for relative in [
            'trt-runtime/include/edge/trt/runner.hpp',
            'trt-runtime/src/runner.cpp',
            'trt-runtime/src/tensor_io.cpp',
            'tensorrt-rtmpose-hand/include/rtmpose/reference.hpp',
            'tensorrt-rtmpose-hand/src/reference.cpp',
            'tensorrt-rtmpose-hand/apps/verify_fp32.cpp',
            'tensorrt-rtmpose-hand/scripts/verify_trt_fp32.py'
        ]
    }
    report['executable_sha256'] = digest(
        ROOT.parent / 'build-shared/tensorrt-rtmpose-hand/rtmpose_fp32'
    )
    report['completion_sha256'] = {
        mode: digest(OUTPUT / mode / 'complete.json')
        for mode in ['npy', 'image']
    }
    report['build_log_sha256'] = digest(OUTPUT / 'build.log')
    report['environment'] = {
        'device': 'Jetson AGX Orin',
        'tensorrt': '8.5.2',
        'cuda_runtime': '11.4',
        'cpp_opencv': report['runtime']['opencv'],
        'python_reference_environment': reference_report['environment']
    }
    write_json(
        OUTPUT / 'report.json',
        report
    )
    print(
        json.dumps(
            {'status': report['status'], 'failures': failures,
                      'worst_simcc_abs': max(
                (v['max_abs'] for c in checks for v in c['simcc'].values())
            ),
                      'worst_input_coordinate_px': max(
                (c['input_coordinate_max_l2_px'] for c in checks)
            ),
                      'input_max_abs': max(
                (c['input_tensor_max_abs'] for c in checks)
            )},
            indent=2
        )
    )
    assert not failures, failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action',
        choices=['prepare', 'compare', 'inputs']
    )
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'compare':
        compare()
    else:
        samples = read_json(
            OUTPUT / 'cases.json'
        )['samples']
        error = check_inputs(
            'image',
            samples
        )
        print(
            'C++ vs frozen Python input max abs:',
            error
        )
        assert error == 0
