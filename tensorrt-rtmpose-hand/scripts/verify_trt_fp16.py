# 本脚本只读复用冻结 reference 和 FP32 产物，验证 FP16 的两条输入路径及双基线。
# 固定门槛来自独立配置；全部关键点参与，失败时保留真实报告，不调整 reference。
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from verify_trt_fp32 import ROOT, REFERENCE_SHA, chunks, decode, digest, read_json, rows, write_json

OUTPUT = ROOT / 'validation/trt-fp16'
FP32 = ROOT / 'validation/trt-fp32'
CONFIG = ROOT / 'configs/rtmpose_hand_trt_fp16.json'
ENGINE = ROOT / 'engines/rtmpose_hand_fp16.engine'
BATCHES = [1, 2, 3, 4, 6]


def validate_build_completion(
    directory,
    engine
):
    completion = read_json(directory / 'build-complete.json')
    assert completion['engine_sha256'] == digest(engine), 'Engine differs from completed build'
    assert completion['provenance_sha256'] == digest(directory / 'build-provenance.json')
    assert completion['log_sha256'] == digest(directory / 'build.log')
    return completion


def validate_completion(
    directory,
    engine_sha,
    cases_sha,
    mode,
    count,
    required_files
):
    completion = read_json(
        directory / 'complete.json'
    )
    assert completion['engine_sha256'] == engine_sha, 'stale engine identity'
    assert completion['cases_sha256'] == cases_sha, 'stale cases identity'
    assert completion['mode'] == mode and completion['samples'] == count
    files = {item['path']: item['sha256'] for item in completion['files']}
    assert required_files <= files.keys(), 'missing artifact hash'
    for name, checksum in files.items():
        relative = Path(
            name
        )
        assert not relative.is_absolute() and '..' not in relative.parts
        assert digest(
            directory / relative
        ) == checksum, 'artifact hash mismatch: ' + name
    return completion


def required_artifacts(samples):
    names = {'runtime.json'}
    names.update(
        (sample['id'] + '_input.bin' for sample in samples)
    )
    for batch in BATCHES:
        for start, _ in chunks(
            len(
                samples
            ),
            batch
        ):
            prefix = 'b%d_%d_' % (batch, start)
            names.update(
                (prefix + suffix for suffix in ['simcc_x.bin', 'simcc_y.bin', 'decoded.bin'])
            )
    return names


def load_batch(
    folder,
    samples,
    batch
):
    outputs = [[], []]
    decoded = []
    for start, count in chunks(
        len(
            samples
        ),
        batch
    ):
        prefix = folder / ('b%d_%d_' % (batch, start))
        for axis, name in enumerate(
            ['simcc_x', 'simcc_y']
        ):
            value = np.fromfile(
                str(
                    prefix
                ) + name + '.bin',
                dtype=np.float32
            )
            assert value.size == count * 21 * 512 and np.isfinite(
                value
            ).all()
            outputs[axis].append(
                value.reshape(
                    count,
                    21,
                    512
                )
            )
        value = np.fromfile(
            str(
                prefix
            ) + 'decoded.bin',
            dtype=np.float64
        )
        assert value.size == count * 21 * 6 and np.isfinite(
            value
        ).all()
        decoded.append(
            value.reshape(
                count,
                21,
                6
            )
        )
    return [np.concatenate(
        values
    ) for values in outputs], np.concatenate(
        decoded
    )


def compare_raw(
    actual,
    expected,
    samples
):
    for value in list(
        actual
    ) + list(
        expected
    ):
        assert value.shape == (len(
            samples
        ), 21, 512) and np.isfinite(
            value
        ).all()
    current = decode(
        *actual,
        samples
    )
    baseline = decode(
        *expected,
        samples
    )
    distances = np.linalg.norm(
        current[0] - baseline[0],
        axis=-1
    )
    source_distances = np.linalg.norm(
        current[1] - baseline[1],
        axis=-1
    )
    return {
        'simcc': {
            name: {'mae': float(
                np.abs(
                    actual[axis] - expected[axis]
                ).mean()
            ),
                   'max_abs': float(
                       np.abs(
                           actual[axis] - expected[axis]
                       ).max()
                   )}
            for axis, name in enumerate(
                ['simcc_x', 'simcc_y']
            )
        },
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
        'source_coordinate_mean_l2_px': float(
            source_distances.mean()
        ),
        'source_coordinate_p95_l2_px': float(
            np.percentile(
                source_distances,
                95
            )
        ),
        'source_coordinate_max_l2_px': float(
            source_distances.max()
        ),
        'changed_keypoint_count': int(
            np.count_nonzero(
                distances
            )
        ),
        'score_max_abs': float(
            np.abs(
                current[2] - baseline[2]
            ).max()
        ),
        'valid_mask_flip_count': int(
            np.count_nonzero(
                current[3] != baseline[3]
            )
        ),
        'actual_valid_count': int(
            current[3].sum()
        ),
        'reference_valid_count': int(
            baseline[3].sum()
        )
    }


def gate_failures(
    metrics,
    limits
):
    failures = []
    for name, values in metrics['simcc'].items():
        for key in ['mae', 'max_abs']:
            if values[key] > limits['simcc_' + key]:
                failures.append(
                    name + '_' + key
                )
    for key in ['input_coordinate_mean_l2_px', 'input_coordinate_p95_l2_px',
                'input_coordinate_max_l2_px', 'score_max_abs', 'valid_mask_flip_count']:
        if metrics[key] > limits[key]:
            failures.append(
                key
            )
    return failures


def summarize_layers(inspector):
    details = []
    counts = Counter()
    reformats = []
    for layer in inspector['Layers']:
        formats = [item.get(
            'Format/Datatype',
            'unknown'
        )
                   for item in layer.get(
                       'Inputs',
                       []
                   ) + layer.get(
                       'Outputs',
                       []
                   )]
        text = ' '.join(
            formats
        ).lower()
        half = 'fp16' in text or 'half' in text
        single = 'fp32' in text or 'float' in text
        if half and single:
            category = 'mixed_fp16_fp32'
        elif half:
            category = 'fp16'
        elif single:
            category = 'fp32'
        else:
            category = 'unknown'
        counts[category] += 1
        if layer.get(
            'LayerType',
            ''
        ).lower() == 'reformat' or 'reformat' in layer['Name'].lower():
            reformats.append(
                layer['Name']
            )
        details.append(
            {'name': layer['Name'], 'type': layer.get(
                'LayerType'
            ),
                        'io_precision': category, 'formats': formats,
                        'tactic': layer.get(
                'TacticName',
                layer.get(
                    'TacticValue'
                )
            )}
        )
    return {'layer_count': len(
        details
    ), 'io_precision_counts': dict(
        counts
    ),
            'reformat_layers': reformats, 'layers': details,
            'interpretation': 'Classified from inspector layer tensor formats, not a claim about accumulator precision or every fused operator.'}


def prepare():
    assert digest(
        ROOT / 'configs/rtmpose_hand_reference.json'
    ) == REFERENCE_SHA
    config = read_json(
        CONFIG
    )
    assert config['reference_sha256'] == REFERENCE_SHA
    assert digest(
        ROOT / 'models/rtmpose_hand.onnx'
    ) == config['model_sha256']
    baseline = read_json(
        ROOT / 'benchmarks/fp32-baseline.json'
    )
    assert baseline['status'] == 'passed'
    frozen = {}
    for relative, checksum in baseline['source_sha256'].items():
        assert digest(
            ROOT.parent / relative
        ) == checksum, 'FP32 implementation changed'
        frozen[str(
            ROOT.parent / relative
        )] = checksum
    executable = ROOT.parent / 'build-shared/tensorrt-rtmpose-hand/rtmpose_fp32'
    assert digest(
        executable
    ) == baseline['executable_sha256'], 'C++ executable changed'
    cases = read_json(
        FP32 / 'cases.json'
    )
    assert cases['reference_sha256'] == REFERENCE_SHA and len(
        cases['samples']
    ) == 55
    hands = {row['hand_id']: row for row in rows(
        ROOT / 'calibration/hand_boxes.jsonl'
    )}
    frames = {row['frame_id']: row for row in rows(
        ROOT / 'calibration/frames.jsonl'
    )}
    order = (ROOT / 'calibration/development_tensor_order.txt').read_text().splitlines()
    assert [hands[sample['id']]['tensor_path'] for sample in cases['samples']] == order
    for sample in cases['samples']:
        hand = hands[sample['id']]
        frame = frames[hand['frame_id']]
        assert hand['split'] == 'development' and sample['bbox'] == hand['bbox_xyxy']
        assert Path(
            sample['tensor']
        ) == ROOT / 'calibration' / hand['tensor_path']
        assert Path(
            sample['image']
        ) == ROOT / frame['source_path']
        assert digest(
            Path(
                sample['tensor']
            )
        ) == hand['tensor_sha256']
        assert digest(
            Path(
                sample['image']
            )
        ) == frame['source_sha256']
        frozen[sample['tensor']] = hand['tensor_sha256']
        frozen[sample['image']] = frame['source_sha256']
        golden = ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz')
        frozen[str(
            golden
        )] = digest(
            golden
        )
    for mode in ['npy', 'image']:
        assert digest(
            FP32 / mode / 'complete.json'
        ) == baseline['completion_sha256'][mode]
    assets = [CONFIG, executable, ROOT / 'configs/rtmpose_hand_reference.json',
              ROOT / 'models/rtmpose_hand.onnx', ROOT / 'engines/rtmpose_hand_fp32.engine',
              ROOT / 'benchmarks/fp32-baseline.json', FP32 / 'cases.json']
    if CONFIG.name == 'rtmpose_hand_trt_fp16_cpp_2g.json':
        build_config = ROOT / config['build_config']
        assert read_json(build_config)['workspace_mib'] == config['workspace_mib'] == 2048
        assert read_json(build_config)['selected_precision'] == 'fp16'
        provenance_path = OUTPUT / 'build-provenance.json'
        provenance = read_json(provenance_path)
        assert provenance['workspace_mib'] == 2048 and provenance['precision'] == 'fp16'
        for relative, checksum in provenance['files_sha256'].items():
            path = ROOT.parent / relative
            assert digest(path) == checksum, 'Builder source/artifact changed: ' + relative
            frozen[str(path)] = checksum
        validate_build_completion(
            OUTPUT,
            ENGINE
        )
        assets += [build_config, provenance_path, OUTPUT / 'build-complete.json', ENGINE]
    assets += [FP32 / mode / 'complete.json' for mode in ['npy', 'image']]
    for path in assets:
        frozen[str(
            path
        )] = digest(
            path
        )
    assert frozen[str(
        ROOT / 'engines/rtmpose_hand_fp32.engine'
    )] == baseline['engine_sha256']
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )
    lock_path = OUTPUT / 'input-lock.json'
    lock = {'assets_sha256': frozen, 'acceptance': config['acceptance']}
    if lock_path.exists():
        assert read_json(
            lock_path
        ) == lock, 'Existing FP16 input lock differs; do not silently overwrite'
    else:
        write_json(
            lock_path,
            lock
        )
    shutil.copyfile(
        FP32 / 'cases.json',
        OUTPUT / 'cases.json'
    )
    write_json(
        OUTPUT / 'report.json',
        {'status': 'incomplete', 'reason': 'awaiting FP16 evaluation'}
    )
    print(
        'Locked unchanged implementation, 55 development samples, ORT/FP32 baseline and FP16 gates.'
    )


def compare():
    write_json(
        OUTPUT / 'report.json',
        {'status': 'incomplete', 'reason': 'comparison in progress'}
    )
    lock = read_json(
        OUTPUT / 'input-lock.json'
    )
    for path, checksum in lock['assets_sha256'].items():
        assert digest(
            Path(
                path
            )
        ) == checksum, 'Frozen input changed: ' + path
    assert lock['acceptance'] == read_json(CONFIG)['acceptance'], 'acceptance differs from frozen config'
    assert digest(
        OUTPUT / 'cases.json'
    ) == digest(
        FP32 / 'cases.json'
    )
    samples = read_json(
        OUTPUT / 'cases.json'
    )['samples']
    limits = lock['acceptance']
    required = required_artifacts(
        samples
    )
    engine_sha = digest(
        ENGINE
    )
    if CONFIG.name == 'rtmpose_hand_trt_fp16_cpp_2g.json':
        validate_build_completion(
            OUTPUT,
            ENGINE
        )
    for root, sha in [(OUTPUT, engine_sha), (FP32, digest(
        ROOT / 'engines/rtmpose_hand_fp32.engine'
    ))]:
        for mode in ['npy', 'image']:
            validate_completion(
                root / mode,
                sha,
                digest(
                    root / 'cases.json'
                ),
                mode,
                len(
                    samples
                ),
                required
            )
    inspector = read_json(
        OUTPUT / 'layers-context.json'
    )
    assert inspector['engine_sha256'] == engine_sha and inspector['input_shape'] == [2, 3, 256, 256]
    summary = summarize_layers(
        inspector['inspector']
    )
    write_json(
        OUTPUT / 'layer-summary.json',
        summary
    )
    golden = [[], []]
    for sample in samples:
        with np.load(
            ROOT / 'validation/reference-freeze/raw' / (sample['id'] + '.npz'),
            allow_pickle=False
        ) as data:
            for axis, name in enumerate(
                ['simcc_x', 'simcc_y']
            ):
                golden[axis].append(
                    data[name]
                )
    golden = [np.concatenate(
        parts
    ) for parts in golden]
    checks = []
    failures = []
    for mode in ['npy', 'image']:
        max_input_error = 0.0
        for sample in samples:
            actual = np.fromfile(
                OUTPUT / mode / (sample['id'] + '_input.bin'),
                dtype=np.float32
            )
            expected = np.load(
                sample['tensor'],
                allow_pickle=False
            ).ravel()
            assert actual.shape == expected.shape and np.isfinite(
                actual
            ).all()
            max_input_error = max(
                max_input_error,
                float(
                    np.abs(
                        actual - expected
                    ).max()
                )
            )
        if max_input_error > limits['input_tensor_max_abs']:
            failures.append(
                mode + ': input mismatch'
            )
        single, _ = load_batch(
            OUTPUT / mode,
            samples,
            1
        )
        for batch in BATCHES:
            raw, cpp = load_batch(
                OUTPUT / mode,
                samples,
                batch
            )
            fp32, _ = load_batch(
                FP32 / mode,
                samples,
                batch
            )
            python = decode(
                *raw,
                samples
            )
            python_packed = np.concatenate(
                [python[0], python[1], python[2][..., None], python[3][..., None]],
                axis=-1
            )
            decode_error = float(
                np.abs(
                    cpp - python_packed
                ).max()
            )
            comparisons = {'ort_fp32': compare_raw(
                raw,
                golden,
                samples
            ),
                           'trt_fp32_same_batch': compare_raw(
                               raw,
                               fp32,
                               samples
                           ),
                           'fp16_single': compare_raw(
                               raw,
                               single,
                               samples
                           )}
            for target, metrics in comparisons.items():
                failures.extend(
                    ('%s b%d vs %s: %s' % (mode, batch, target, item)
                                for item in gate_failures(
                        metrics,
                        limits
                    ))
                )
            if decode_error > limits['cpp_decode_max_abs']:
                failures.append(
                    '%s b%d: C++ decode mismatch' % (mode, batch)
                )
            checks.append(
                {'mode': mode, 'requested_batch': batch, 'sample_count': len(
                    samples
                ),
                           'input_tensor_max_abs': max_input_error, 'cpp_decode_max_abs': decode_error,
                           'comparisons': comparisons}
            )
    report = {
        'status': 'passed' if not failures else 'failed', 'failures': failures,
        'reference_sha256': REFERENCE_SHA, 'engine_sha256': engine_sha,
        'fp32_engine_sha256': digest(
            ROOT / 'engines/rtmpose_hand_fp32.engine'
        ),
        'config': read_json(
            CONFIG
        ), 'config_sha256': digest(
            CONFIG
        ),
        'cases_sha256': digest(
            OUTPUT / 'cases.json'
        ), 'input_lock_sha256': digest(
            OUTPUT / 'input-lock.json'
        ),
        'executable_sha256': digest(
            ROOT.parent / 'build-shared/tensorrt-rtmpose-hand/rtmpose_fp32'
        ),
        'comparison_script_sha256': digest(
            Path(
                __file__
            )
        ),
        'completion_sha256': {mode: digest(
            OUTPUT / mode / 'complete.json'
        ) for mode in ['npy', 'image']},
        'build_log_sha256': digest(
            OUTPUT / 'build.log'
        ),
        'context_inspector_sha256': digest(
            OUTPUT / 'layers-context.json'
        ),
        'checks': checks, 'layer_summary': summary,
        'final_validation_images_used': 0, 'calibration_tensors_used': 0,
        'scope': 'Development consistency only, no GT accuracy or performance claim.'
    }
    write_json(
        OUTPUT / 'report.json',
        report
    )
    print(
        json.dumps(
            {'status': report['status'], 'failures': failures,
                      'layer_io_precision_counts': summary['io_precision_counts']},
            indent=2
        )
    )
    assert not failures, failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action',
        choices=['prepare', 'compare']
    )
    parser.add_argument(
        '--experiment',
        choices=['legacy', 'cpp-2g'],
        default='legacy'
    )
    arguments = parser.parse_args()
    if arguments.experiment == 'cpp-2g':
        OUTPUT = ROOT / 'validation/trt-fp16-cpp-2g'
        CONFIG = ROOT / 'configs/rtmpose_hand_trt_fp16_cpp_2g.json'
        ENGINE = ROOT / 'engines/rtmpose_hand_fp16_cpp_2g.engine'
    if arguments.action == 'prepare':
        prepare()
    else:
        compare()
