# 本脚本核验 RTMPose-Hand 参考契约，使用锁定的 rtmlib 源码和开发集真实输入。
# 只生成核验结果；不修改校准样本、划分或原始模型，不读取最终 validation 图片。
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import rtmlib
from rtmlib.tools.pose_estimation.rtmpose import RTMPose
from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum

ROOT = Path(
    __file__
).resolve().parents[1]
CONFIG = ROOT / 'configs/rtmpose_hand_reference.json'
OUTPUT = ROOT / 'validation/reference-freeze'


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(
            encoding='utf-8'
        )
    )


def write_json(path, value):
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ) + '\n',
        encoding='utf-8'
    )


def load_rows(path):
    return [json.loads(
        line
    ) for line in path.read_text().splitlines()]


def to_tensor(normalized):
    return np.ascontiguousarray(
        normalized.transpose(
            2,
            0,
            1
        )[None],
        dtype=np.float32
    )


def draw_pose(image, points, valid):
    canvas = image.copy()
    colors = [(0, 130, 255), (255, 80, 180), (255, 200, 0), (100, 255, 100), (220, 180, 255)]
    for finger in range(
        5
    ):
        chain = [0] + list(
            range(
                1 + finger * 4,
                5 + finger * 4
            )
        )
        for first, second in zip(
            chain[:-1],
            chain[1:]
        ):
            if valid[first] and valid[second]:
                cv2.line(
                    canvas,
                    tuple(
                        np.rint(
                            points[first]
                        ).astype(
                            int
                        )
                    ),
                    tuple(
                        np.rint(
                            points[second]
                        ).astype(
                            int
                        )
                    ),
                    colors[finger],
                    2,
                    cv2.LINE_AA
                )
    for index, point in enumerate(
        points
    ):
        if valid[index]:
            position = tuple(
                np.rint(
                    point
                ).astype(
                    int
                )
            )
            cv2.circle(
                canvas,
                position,
                2,
                (0, 255, 255),
                -1
            )
    return canvas


def verify_decoder_contract(pose, config):
    # 单独构造平均值、并列最大值、零分和负分情况，避免真实数据只覆盖正常点。
    x = np.zeros(
        (1, 21, 512),
        dtype=np.float32
    )
    y = x.copy()
    x[0, 0, 20] = 0.8
    y[0, 0, 40] = 0.2
    x[0, 1, [5, 9]] = 0.6
    y[0, 1, [7, 10]] = 0.4
    x[0, 2] = -1.0
    y[0, 2] = -1.0
    positions, scores = get_simcc_maximum(
        x,
        y
    )
    np.testing.assert_allclose(
        scores[0, :3],
        [0.5, 0.5, -1.0],
        rtol=0,
        atol=1e-7
    )
    np.testing.assert_array_equal(
        positions[0, :4],
        [[20, 40], [5, 7], [-1, -1], [-1, -1]]
    )
    center = np.array(
        [100.0, 150.0]
    )
    scale = np.array(
        [200.0, 200.0]
    )
    source, _ = RTMPose.postprocess(
        pose,
        [x, y],
        center,
        scale,
        simcc_split_ratio=config['simcc']['split_ratio']
    )
    np.testing.assert_allclose(
        source[0, 2],
        [-0.390625, 49.609375],
        rtol=0,
        atol=1e-8
    )
    return {'average_peak_score': True, 'argmax_first_tie': True,
            'zero_and_negative_invalid': True, 'invalid_source_coordinate_not_minus_one': True}


def main():
    config = read_json(
        CONFIG
    )
    prep = config['preprocessing']
    assert prep['source_color_order'] == 'BGR' and prep['color_order'] == 'RGB'
    assert prep['bbox_scale_factor'] == 1.25 and prep['input_size_wh'] == [256, 256]
    assert config['simcc']['split_ratio'] == 2.0
    assert config['simcc']['score_decode_rule'] == '0.5 * (max(simcc_x) + max(simcc_y))'
    package = Path(
        rtmlib.__file__
    ).parent
    for relative, checksum in config['reference_implementation']['source_hashes'].items():
        assert digest(
            package / relative
        ) == checksum, relative
    official = ROOT / 'models/reference_provenance'
    identity = read_json(
        official / 'archive_identity.json'
    )
    assert identity['exact_match_to_local']
    assert identity['onnx_sha256'] == config['model']['sha256']
    for name, checksum in config['color_evidence']['sdk_metadata_sha256'].items():
        assert digest(
            official / name
        ) == checksum
    assert digest(
        official / 'training_config_v1.1.0.py'
    ) == config['color_evidence']['training_config_sha256']
    assert 'bgr_to_rgb=True' in (official / 'training_config_v1.1.0.py').read_text()
    pipeline = read_json(
        official / 'pipeline.json'
    )
    transforms = pipeline['pipeline']['tasks'][0]['transforms']
    normalize = next(
        (
        item for item in transforms if item['type'] == 'Normalize'
    )
    )
    assert normalize['to_rgb'] and normalize['mean'] == prep['mean'] and normalize['std'] == prep['std']
    model_path = ROOT / config['model']['path']
    assert digest(
        model_path
    ) == config['model']['sha256']
    model = onnx.load(
        str(
            model_path
        )
    )
    onnx.checker.check_model(
        model
    )
    assert model.graph.node[0].op_type == 'Conv' and model.graph.node[0].input[0] == 'input'
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(
            model_path
        ),
        sess_options=options,
        providers=['CPUExecutionProvider']
    )
    assert len(
        session.get_inputs()
    ) == 1
    assert session.get_inputs()[0].name == config['model']['input_name']
    assert session.get_inputs()[0].shape[1:] == [3, 256, 256]
    assert session.get_inputs()[0].type == 'tensor(float)'
    assert [item.name for item in session.get_outputs()] == config['model']['output_names']
    pose = SimpleNamespace(
        model_input_size=tuple(
            prep['input_size_wh']
        ),
        mean=np.array(
            prep['mean']
        ),
        std=np.array(
            prep['std']
        )
    )
    for folder in ['raw', 'decoded', 'visualization']:
        (OUTPUT / folder).mkdir(
            parents=True,
            exist_ok=True
        )
    decoder_checks = verify_decoder_contract(
        pose,
        config
    )
    calibration = ROOT / 'calibration'
    hands = load_rows(
        calibration / 'hand_boxes.jsonl'
    )
    frames = {row['frame_id']: row for row in load_rows(
        calibration / 'frames.jsonl'
    )}
    coverage = read_json(
        calibration / 'coverage.json'
    )
    split = read_json(
        calibration / 'split.json'
    )
    assert len(
        hands
    ) == 407
    assert set(
        (frames[row['frame_id']]['image_id'] for row in hands)
    ).isdisjoint(
        split['validation_in_car_image_ids'] + split['validation_supplement_image_ids']
    )
    # 选择由输入几何和亮度决定的展示样本，不依据预测效果筛选。
    metrics = [row for row in coverage['sample_metrics'] if row['split'] == 'development']
    selected = []
    for field, largest in [('bbox_max_side_px', False), ('bbox_max_side_px', True),
                           ('crop_mean_luminance_0_255', False), ('excluded_border_pixel_fraction', True)]:
        sample = sorted(
            metrics,
            key=lambda row: row[field],
            reverse=largest
        )[0]['hand_id']
        if sample not in selected:
            selected.append(
                sample
            )
    for pair in read_json(
        calibration / 'overlapping_boxes.json'
    ):
        if frames[pair['frame_id']]['split'] == 'development':
            selected.extend(
                (item for item in [pair['hand_a'], pair['hand_b']] if item not in selected)
            )
    multi = next(
        (row for row in frames.values() if row['split'] == 'development' and len(
            row['hand_ids']
        ) >= 2)
    )
    selected.extend(
        (item for item in multi['hand_ids'] if item not in selected)
    )
    for row in metrics:
        if len(
            selected
        ) >= 10:
            break
        if row['hand_id'] not in selected:
            selected.append(
                row['hand_id']
            )
    selected = selected[:10]
    source_images = {}
    rgb_tensors = {}
    legacy_tensors = {}
    geometry = {}
    current_matches = 0
    crop_checks = 0
    max_affine_error = 0.0
    for hand in hands:
        frame = frames[hand['frame_id']]
        if hand['frame_id'] not in source_images:
            path = ROOT / frame['source_path']
            assert digest(
                path
            ) == frame['source_sha256']
            source_images[hand['frame_id']] = cv2.imread(
                str(
                    path
                ),
                cv2.IMREAD_COLOR
            )
        bgr = source_images[hand['frame_id']]
        assert bgr is not None
        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )
        normalized, center, scale = RTMPose.preprocess(
            pose,
            rgb,
            hand['bbox_xyxy']
        )
        tensor = to_tensor(
            normalized
        )
        assert tensor.shape == (1, 3, 256, 256) and np.isfinite(
            tensor
        ).all()
        np.testing.assert_allclose(
            center,
            hand['center_xy'],
            rtol=0,
            atol=1e-9
        )
        np.testing.assert_allclose(
            scale,
            hand['adjusted_scale_wh'],
            rtol=0,
            atol=1e-9
        )
        crop = cv2.imread(
            str(
                calibration / hand['crop_path']
            ),
            cv2.IMREAD_COLOR
        )
        assert crop is not None
        crop_rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )
        crop_tensor = to_tensor(
            (crop_rgb - pose.mean) / pose.std
        )
        np.testing.assert_array_equal(
            tensor,
            crop_tensor
        )
        crop_checks += 1
        existing_path = calibration / hand['tensor_path']
        assert digest(
            existing_path
        ) == hand['tensor_sha256']
        existing = np.load(
            existing_path,
            allow_pickle=False
        )
        current_matches += int(
            np.array_equal(
                existing,
                tensor
            )
        )
        if hand['split'] == 'development':
            rgb_tensors[hand['hand_id']] = tensor
            legacy_tensors[hand['hand_id']] = to_tensor(
                (crop - pose.mean) / pose.std
            )
            geometry[hand['hand_id']] = (center, scale)
    assert len(
        rgb_tensors
    ) == 55
    if config['frozen']:
        assert current_matches == len(
            hands
        ), 'Frozen config must match every stored tensor'
    outputs = {}
    keypoints = {}
    comparison_displacements = []
    selected_pages = {}
    for hand in hands:
        identifier = hand['hand_id']
        if identifier not in rgb_tensors:
            continue
        raw = session.run(
            config['model']['output_names'],
            {config['model']['input_name']: rgb_tensors[identifier]}
        )
        assert all(
            (value.shape == (1, 21, 512) and value.dtype == np.float32 and np.isfinite(
                value
            ).all() for value in raw)
        )
        outputs[identifier] = raw
        positions, scores = get_simcc_maximum(
            *raw
        )
        input_xy = positions / config['simcc']['split_ratio']
        center, scale = geometry[identifier]
        source_xy, source_scores = RTMPose.postprocess(
            pose,
            raw,
            center,
            scale,
            simcc_split_ratio=config['simcc']['split_ratio']
        )
        np.testing.assert_array_equal(
            scores,
            source_scores
        )
        independent = input_xy / np.array(
            prep['input_size_wh']
        ) * scale + center - scale / 2
        np.testing.assert_array_equal(
            source_xy,
            independent
        )
        affine = np.asarray(
            hand['input_to_source_affine']
        )
        homogeneous = np.concatenate(
            [input_xy[0], np.ones(
                (21, 1)
            )],
            axis=1
        )
        restored = homogeneous @ affine.T
        error = float(
            np.abs(
                restored - source_xy[0]
            ).max()
        )
        max_affine_error = max(
            max_affine_error,
            error
        )
        assert error < 1e-3
        valid = scores[0] > 0
        keypoints[identifier] = input_xy
        np.savez_compressed(
            OUTPUT / 'raw' / (identifier + '.npz'),
            simcc_x=raw[0],
            simcc_y=raw[1]
        )
        write_json(
            OUTPUT / 'decoded' / (identifier + '.json'),
            {'hand_id': identifier, 'frame_id': hand['frame_id'],
             'split': 'development', 'input_xy': input_xy[0].tolist(),
             'source_xy': source_xy[0].tolist(), 'scores': scores[0].tolist(),
             'valid_mask': valid.tolist(), 'config_sha256': digest(
                CONFIG
            )}
        )
        legacy_raw = session.run(
            config['model']['output_names'],
            {config['model']['input_name']: legacy_tensors[identifier]}
        )
        legacy_xy, _ = get_simcc_maximum(
            *legacy_raw
        )
        legacy_xy /= config['simcc']['split_ratio']
        displacement = np.linalg.norm(
            input_xy - legacy_xy,
            axis=-1
        )
        comparison_displacements.extend(
            displacement.ravel().tolist()
        )
        if identifier in selected:
            crop = cv2.imread(
                str(
                    calibration / hand['crop_path']
                )
            )
            old_scores = get_simcc_maximum(
                *legacy_raw
            )[1][0]
            old_overlay = draw_pose(
                crop,
                legacy_xy[0],
                old_scores > 0
            )
            new_overlay = draw_pose(
                crop,
                input_xy[0],
                valid
            )
            pair = np.full(
                (290, 512, 3),
                255,
                dtype=np.uint8
            )
            pair[34:, :256] = old_overlay
            pair[34:, 256:] = new_overlay
            cv2.putText(
                pair,
                identifier + ' old BGR',
                (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1
            )
            cv2.putText(
                pair,
                'RGB contract',
                (270, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1
            )
            assert cv2.imwrite(
                str(
                    OUTPUT / 'visualization' / (identifier + '_compare.jpg')
                ),
                pair
            )
            original = draw_pose(
                source_images[hand['frame_id']],
                source_xy[0],
                valid
            )
            assert cv2.imwrite(
                str(
                    OUTPUT / 'visualization' / (identifier + '_source.jpg')
                ),
                original
            )
            selected_pages[identifier] = pair
    # 同一输入在不同 batch 下与单独推理对照，不使用最终 validation。
    batch_checks = []
    dev_ids = list(
        rgb_tensors
    )
    for size in [1, 2, 3, 4]:
        batch_ids = dev_ids[:size]
        packed = np.concatenate(
            [rgb_tensors[item] for item in batch_ids],
            axis=0
        )
        batched = session.run(
            config['model']['output_names'],
            {config['model']['input_name']: packed}
        )
        max_raw_error = 0.0
        for output_index, value in enumerate(
            batched
        ):
            assert value.shape == (size, 21, 512) and np.isfinite(
                value
            ).all()
            expected = np.concatenate(
                [outputs[item][output_index] for item in batch_ids],
                axis=0
            )
            max_raw_error = max(
                max_raw_error,
                float(
                    np.abs(
                        value - expected
                    ).max()
                )
            )
            np.testing.assert_allclose(
                value,
                expected,
                rtol=1e-4,
                atol=1e-5
            )
        batched_xy, _ = get_simcc_maximum(
            *batched
        )
        expected_xy = np.concatenate(
            [keypoints[item] for item in batch_ids],
            axis=0
        )
        max_coordinate_error = float(
            np.linalg.norm(
                batched_xy / 2.0 - expected_xy,
                axis=-1
            ).max()
        )
        batch_checks.append(
            {'batch': size, 'hand_ids': batch_ids, 'max_simcc_absolute_error': max_raw_error,
                             'max_input_coordinate_error_px': max_coordinate_error}
        )
    sheet = np.vstack(
        [selected_pages[item] for item in selected]
    )
    assert cv2.imwrite(
        str(
            OUTPUT / 'visualization/selected_comparison.jpg'
        ),
        sheet
    )
    cards = ''.join(
        ('<section><h2>%s</h2><img src="visualization/%s_compare.jpg"><img style="max-height:640px" src="visualization/%s_source.jpg"></section>' % (item, item, item) for item in selected)
    )
    (OUTPUT / 'index.html').write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RTMPose reference 检查</title><style>body{font:16px sans-serif;margin:24px}section{border-top:1px solid #ccc;padding:20px}img{max-width:100%;vertical-align:top}</style><h1>开发集真实样本 reference 检查</h1><p>左：旧 BGR 张量输出；中：RGB 契约输出；右：RGB 输出还原至原图。10 个展示样本按输入尺寸、边缘填充、亮度、双手及重叠框选择，不根据预测效果筛选。全部正分关键点都绘制，无置信度筛选。RGB 选择基于模型来源证据，不基于此对比或置信度。图像仅用于几何检查，不是人工真值精度评估。</p>' + cards + '</html>',
        encoding='utf-8'
    )
    values = np.asarray(
        comparison_displacements
    )
    report = {
        'status': 'passed', 'reference_config_sha256': digest(
            CONFIG
        ),
        'model_sha256': digest(
            model_path
        ), 'verification_script_sha256': digest(
            Path(
                __file__
            )
        ),
        'onnx_check': 'passed', 'input_metadata': {'name': session.get_inputs()[0].name, 'shape': session.get_inputs()[0].shape},
        'onnx_input_directly_feeds_conv': True, 'color_evidence_checks': 'passed',
        'locked_rtmlib_source_hashes': 'passed', 'decoder_synthetic_checks': decoder_checks,
        'rgb_preprocess_recomputed_count': len(
            hands
        ), 'rgb_from_source_equals_rgb_from_saved_crop_count': crop_checks,
        'current_dataset_tensors_matching_rgb_contract': current_matches,
        'development_real_inference_count': len(
            rgb_tensors
        ), 'final_validation_images_used': 0,
        'outputs': {'simcc_x': [1, 21, 512], 'simcc_y': [1, 21, 512]},
        'all_real_outputs_finite': True, 'max_inverse_affine_vs_reference_source_error_px': max_affine_error,
        'ort_batch_checks': batch_checks, 'selected_visualization_hand_ids': selected,
        'rgb_vs_legacy_bgr_input_coordinate_displacement_px': {
            'mean': float(
                values.mean()
            ), 'p95': float(
                np.percentile(
                    values,
                    95
                )
            ), 'max': float(
                values.max()
            ),
            'interpretation': '仅说明通道变更影响，不是 RGB 更准确的量化证明，也不是 GT 精度。'},
        'environment': {'python': __import__(
            'platform'
        ).python_version(), 'numpy': np.__version__,
                        'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'opencv': cv2.__version__,
                        'provider': 'CPUExecutionProvider', 'intra_op_threads': 2, 'inter_op_threads': 1},
        'input_manifest_sha256': {name: digest(
            calibration / name
        ) for name in ['split.json', 'hand_boxes.jsonl', 'frames.jsonl', 'development_tensor_order.txt']},
        'visual_review': 'pending_agent_inspection', 'human_visual_review': 'pending',
        'scope_limit': 'Reference config 核验；不是 TensorRT 一致性、量化精度或完整最终验证。'
    }
    write_json(
        OUTPUT / 'report.json',
        report
    )
    print(
        json.dumps(
            {key: report[key] for key in ['status', 'rgb_preprocess_recomputed_count', 'current_dataset_tensors_matching_rgb_contract', 'development_real_inference_count', 'max_inverse_affine_vs_reference_source_error_px', 'ort_batch_checks', 'selected_visualization_hand_ids']},
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == '__main__':
    main()
