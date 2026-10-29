# 本脚本统计已认可的手部裁剪覆盖度，不修改数据划分、样本顺序或审核结果。
# 遮挡、左右手、真实低光和漏检保持未知；位置和亮度指标仅作为显式定义的代理统计。
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

BASE = Path(
    __file__
).resolve().parents[1]
MODULE = BASE.parent
EDGE_MARGIN = 0.05
DARK_MEAN_THRESHOLD = 60.0


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(
            encoding='utf-8'
        )
    )


def read_jsonl(path):
    lines = path.read_text(
        encoding='utf-8'
    ).splitlines()
    return [json.loads(line) for line in lines]


def distribution(values):
    array = np.asarray(
        values,
        dtype=np.float64
    )
    assert array.size > 0 and np.isfinite(array).all()
    percentiles = np.percentile(
        array,
        [10, 50, 90],
        method='linear'
    )
    return {
        'count': int(array.size),
        'min': float(array.min()),
        'p10': float(percentiles[0]),
        'p50': float(percentiles[1]),
        'p90': float(percentiles[2]),
        'max': float(array.max()),
        'mean': float(array.mean())
    }


def unknown_labels(count, labels):
    return {
        'status': 'not_annotated',
        'known_count': 0,
        'unknown_count': count,
        'counts': {label: None for label in labels}
    }


def main():
    inputs = [
        'frames.jsonl', 'hand_boxes.jsonl', 'split.json', 'approval.json',
        'summary.json', 'overlapping_boxes.json',
        'calibration_tensor_order.txt', 'development_tensor_order.txt'
    ]
    input_hashes = {name: sha256(BASE / name) for name in inputs}
    frames = read_jsonl(
        BASE / 'frames.jsonl'
    )
    hands = read_jsonl(
        BASE / 'hand_boxes.jsonl'
    )
    split = read_json(
        BASE / 'split.json'
    )
    overlap_pairs = read_json(
        BASE / 'overlapping_boxes.json'
    )
    frame_by_id = {row['frame_id']: row for row in frames}
    hand_by_id = {row['hand_id']: row for row in hands}
    assert len(frame_by_id) == len(frames) == 331
    assert len(hand_by_id) == len(hands) == 407
    assert split['status'] == 'user_approved'
    validation_ids = set(
        split['validation_in_car_image_ids'] + split['validation_supplement_image_ids']
    )
    assert not validation_ids.intersection(row['image_id'] for row in frames)
    images = {}
    for frame in frames:
        source_path = MODULE / frame['source_path']
        assert sha256(source_path) == frame['source_sha256']
        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_COLOR
        )
        assert image is not None
        assert image.shape[:2] == (frame['height'], frame['width'])
        images[frame['frame_id']] = image
        frame_hands = [row['hand_id'] for row in hands if row['frame_id'] == frame['frame_id']]
        assert frame_hands == frame['hand_ids']
        assert len(frame_hands) == frame['detections_above_threshold']
        assert frame['decision'] == ('keep' if frame_hands else 'skip')

    samples = []
    for hand in hands:
        frame = frame_by_id[hand['frame_id']]
        assert hand['split'] == frame['split']
        assert hand['source_sha256'] == frame['source_sha256']
        assert hand['decision'] == 'keep' and hand['review_status'] == 'approved'
        assert not hand['automatic_rejection_reason']
        tensor_path = BASE / hand['tensor_path']
        assert sha256(tensor_path) == hand['tensor_sha256']
        tensor = np.load(
            tensor_path,
            allow_pickle=False
        )
        assert tensor.shape == (1, 3, 256, 256) and tensor.dtype == np.float32
        assert np.isfinite(tensor).all()
        crop_path = BASE / hand['crop_path']
        crop = cv2.imread(
            str(crop_path),
            cv2.IMREAD_COLOR
        )
        assert crop is not None and crop.shape == (256, 256, 3)
        image = images[hand['frame_id']]
        height, width = image.shape[:2]
        x1, y1, x2, y2 = hand['bbox_xyxy']
        assert np.isfinite([x1, y1, x2, y2]).all()
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        assert bbox_width > 0 and bbox_height > 0
        assert min(x2, width) > max(x1, 0) and min(y2, height) > max(y1, 0)
        near_edge = (
            x1 <= width * EDGE_MARGIN or y1 <= height * EDGE_MARGIN
            or x2 >= width * (1.0 - EDGE_MARGIN)
            or y2 >= height * (1.0 - EDGE_MARGIN)
        )
        boundary_crossing = x1 < 0 or y1 < 0 or x2 > width or y2 > height
        warp = np.asarray(
            hand['source_to_input_affine'],
            dtype=np.float64
        )
        # 与图像相同的线性插值计算有效支持区域，排除含边界零填充的像素。
        support = cv2.warpAffine(
            np.ones((height, width), dtype=np.float32),
            warp,
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        valid = support >= 1.0 - 1e-6
        assert valid.any()
        bgr = crop.astype(
            np.float64
        )
        luminance = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
        mean_luminance = float(
            luminance[valid].mean()
        )
        samples.append({
            'hand_id': hand['hand_id'],
            'frame_id': hand['frame_id'],
            'split': hand['split'],
            'collection_group': frame['collection_group'],
            'tensor_path': hand['tensor_path'],
            'tensor_sha256': hand['tensor_sha256'],
            'crop_path': hand['crop_path'],
            'crop_sha256': sha256(crop_path),
            'bbox_width_px': bbox_width,
            'bbox_height_px': bbox_height,
            'bbox_max_side_px': max(bbox_width, bbox_height),
            'bbox_area_px2': bbox_width * bbox_height,
            'bbox_area_to_image_area_ratio': bbox_width * bbox_height / (width * height),
            'bbox_center_x_normalized': (x1 + x2) / (2.0 * width),
            'bbox_center_y_normalized': (y1 + y2) / (2.0 * height),
            'bbox_near_image_edge': near_edge,
            'bbox_crosses_image_boundary': boundary_crossing,
            'detector_score': hand['detector_score'],
            'valid_crop_pixel_count_for_brightness': int(valid.sum()),
            'excluded_border_pixel_fraction': float(1.0 - valid.mean()),
            'crop_mean_luminance_0_255': mean_luminance,
            'low_brightness_proxy': mean_luminance < DARK_MEAN_THRESHOLD
        })

    reports = {}
    for name in ['calibration', 'development']:
        group_frames = [row for row in frames if row['split'] == name]
        group_samples = [row for row in samples if row['split'] == name]
        group_ids = {row['frame_id'] for row in group_frames}
        assert {row['image_id'] for row in group_frames} == set(split[name + '_image_ids'])
        order_file = name + '_tensor_order.txt'
        order = (BASE / order_file).read_text().splitlines()
        assert order == [row['tensor_path'] for row in group_samples]
        assert len(order) == len(set(order))
        hand_counts = Counter(
            len(row['hand_ids']) for row in group_frames
        )
        source_groups = []
        for source in sorted({row['collection_group'] for row in group_frames}):
            source_frames = [row for row in group_frames if row['collection_group'] == source]
            source_hands = [row for row in group_samples if row['collection_group'] == source]
            source_groups.append({
                'collection_group': source,
                'raw_images': len(source_frames),
                'contributing_raw_images': sum(bool(row['hand_ids']) for row in source_frames),
                'zero_detection_raw_images': sum(not row['hand_ids'] for row in source_frames),
                'hand_tensors': len(source_hands),
                'tensor_fraction': len(source_hands) / len(group_samples)
            })
        metric_fields = [
            'bbox_width_px', 'bbox_height_px', 'bbox_max_side_px', 'bbox_area_px2',
            'bbox_area_to_image_area_ratio', 'bbox_center_x_normalized',
            'bbox_center_y_normalized', 'detector_score',
            'crop_mean_luminance_0_255', 'excluded_border_pixel_fraction'
        ]
        edge_count = sum(row['bbox_near_image_edge'] for row in group_samples)
        dark_count = sum(row['low_brightness_proxy'] for row in group_samples)
        reports[name] = {
            'tensor_order_path': order_file,
            'tensor_order_sha256': input_hashes[order_file],
            'source_raw_images': len(group_frames),
            'contributing_raw_images': sum(bool(row['hand_ids']) for row in group_frames),
            'valid_hand_boxes': len(group_samples),
            'accepted_hand_crops': len(group_samples),
            'final_input_tensors': len(group_samples),
            'tensor_shape': [1, 3, 256, 256],
            'tensor_dtype': 'float32',
            'detected_hands_per_image': {
                'zero': hand_counts[0],
                'one': hand_counts[1],
                'two_or_more': sum(count for number, count in hand_counts.items() if number >= 2),
                'exact_histogram': {str(number): count for number, count in sorted(hand_counts.items())},
                'max': max(hand_counts)
            },
            'source_groups': source_groups,
            'distributions': {
                field: distribution([row[field] for row in group_samples]) for field in metric_fields
            },
            'position_proxy': {
                'near_image_edge_count': edge_count,
                'interior_count': len(group_samples) - edge_count,
                'bbox_crosses_image_boundary_count': sum(row['bbox_crosses_image_boundary'] for row in group_samples)
            },
            'brightness_proxy': {
                'low_brightness_count': dark_count,
                'not_low_brightness_count': len(group_samples) - dark_count
            },
            'occlusion': unknown_labels(len(group_samples), ['unoccluded', 'partially_occluded', 'fully_occluded']),
            'handedness': unknown_labels(len(group_samples), ['left', 'right']),
            'lighting_annotation': unknown_labels(len(group_samples), ['normal', 'low_light']),
            'skip_and_rejection': {
                'zero_detection_images_skipped': hand_counts[0],
                'zero_detection_frame_ids': [row['frame_id'] for row in group_frames if not row['hand_ids']],
                'zero_detection_policy': 'user_approved_skip_without_adding_boxes',
                'confirmed_detector_missed_hands': None,
                'confirmed_detector_missed_hands_status': 'not_annotated',
                'rejected_detected_crops': 0,
                'wrong_hand_boxes_rejected_by_review': 0,
                'invalid_bbox_in_current_manifest': 0,
                'empty_crop_in_current_manifest': 0,
                'damaged_source_images': 0,
                'damaged_crop_images': 0,
                'invalid_or_nonfinite_tensors': 0,
                'rejection_scope': '当前已审核清单；用户保留全部已检出框，不声称检测器没有误检或漏检。'
            },
            'retained_overlap_pairs': [pair for pair in overlap_pairs if pair['frame_id'] in group_ids],
            'retained_overlap_pair_count': sum(pair['frame_id'] in group_ids for pair in overlap_pairs)
        }

    report = {
        'schema_version': 1,
        'status': 'generated_from_user_approved_dataset',
        'scope': 'hand_crop_and_tensor_coverage',
        'primary_split': 'calibration',
        'path_base': 'tensorrt-rtmpose-hand/calibration/',
        'candidate_pool': {
            'original_raw_images': len(frames),
            'calibration_raw_images': len([row for row in frames if row['split'] == 'calibration']),
            'development_raw_images': len([row for row in frames if row['split'] == 'development']),
            'all_accepted_hand_tensors': len(hands),
            'all_zero_detection_images_skipped': sum(not row['hand_ids'] for row in frames)
        },
        'validation': {
            'in_car_raw_images_reserved': len(split['validation_in_car_image_ids']),
            'cross_scene_raw_images_reserved': len(split['validation_supplement_image_ids']),
            'validation_images_analyzed_in_this_report': 0,
            'validation_tensors_in_calibration': 0
        },
        'definitions': {
            'statistics_population': '仅当前已审核保留框，每个 crop/tensor 权重为 1；重叠框与相似帧均按用户意见保留。calibration/development 分开统计。',
            'detected_hands_per_image': 'RTMDet score > 0.3 后保留的框数，分母为该集合全部来源原图；可能包含同一只手的重叠框，不是独立物理手数量。',
            'bbox_geometry': '原图像素 xyxy，扩展/正方形调整前的原始框；不截断越界坐标。width=x2-x1，height=y2-y1。',
            'quantiles': 'NumPy percentile(method=linear)，含最小值、P10/P50/P90、最大值与均值；保留全部离群点。',
            'edge_rule': 'x1<=0.05*W 或 y1<=0.05*H 或 x2>=0.95*W 或 y2>=0.95*H 则 near_image_edge；其余 interior。基于框与边缘带相交，不是手掌中心位置的人工标签。',
            'edge_margin_fraction': EDGE_MARGIN,
            'luminance_formula': 'PNG 裁剪 BGR 像素，float64 Y=0.114*B+0.587*G+0.299*R，范围 0..255；是编码像素的亮度代理，未做线性光转换。',
            'brightness_support_mask': '原图全 1 float32 掩码，使用记录的 source_to_input_affine、INTER_LINEAR、BORDER_CONSTANT=0 映射到 256x256，仅 support>=1-1e-6 的像素计入亮度均值。',
            'low_brightness_rule': '有效区域 mean(Y)<60 为 low_brightness_proxy，否则 not_low_brightness；仅描述裁剪亮度，不代表已确认低光照，也不用于删除样本。',
            'low_brightness_mean_threshold_0_255': DARK_MEAN_THRESHOLD,
            'unknown_semantics': '遮挡、左右手、真实光照和真实漏检没有独立人工标注；已知类别数量为 null 而非 0。',
            'empty_crop': '无法得到 256x256x3 图像或仿射后没有任何完全落在原图内的有效支持像素；不以暗色或低方差自动判为空。',
            'overlap_pairs': '沿用 overlapping_boxes.json 的同图 bbox IoU>0.5 候选对，统计的是对数，不等于应删除的样本数。'
        },
        'provenance': {
            'input_sha256': input_hashes,
            'model_manifest_path': '../models/manifest.json',
            'model_manifest_sha256': sha256(MODULE / 'models/manifest.json'),
            'generator_path': 'provenance/generate_coverage.py',
            'generator_sha256': sha256(Path(__file__)),
            'python': platform.python_version(),
            'python_executable': sys.executable,
            'numpy': np.__version__,
            'opencv': cv2.__version__,
            'all_source_and_tensor_hashes_checked': True,
            'all_source_and_crop_images_decoded': True,
            'all_tensor_shapes_dtypes_and_finiteness_checked': True,
            'dataset_mutated': False
        },
        'calibration': reports['calibration'],
        'development': reports['development'],
        'limitations': [
            '检测框数不是人工标注的真实手数；零检测只表示阈值过滤后无框。',
            '位置和亮度是显式定义的代理指标；遮挡、左右手、真实光照与漏检均未推断。',
            '覆盖度报告不能证明人物/场次独立、业务分布充分代表性或量化精度达标。',
            '预处理契约状态见 summary.json 与 configs/rtmpose_hand_reference.json；覆盖度报告不代表已完成 INT8 校准。'
        ],
        'sample_metrics': samples
    }
    assert {name: sha256(BASE / name) for name in inputs} == input_hashes
    output = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False
    ) + '\n'
    (BASE / 'coverage.json').write_text(
        output,
        encoding='utf-8'
    )
    for name in ['calibration', 'development']:
        section = report[name]
        print(
            json.dumps({
                'split': name,
                'raw_images': section['source_raw_images'],
                'contributing_images': section['contributing_raw_images'],
                'tensors': section['final_input_tensors'],
                'hands_per_image': section['detected_hands_per_image'],
                'bbox_max_side_px': section['distributions']['bbox_max_side_px'],
                'position': section['position_proxy'],
                'brightness_proxy': section['brightness_proxy']
            }, ensure_ascii=False, indent=2)
        )


if __name__ == '__main__':
    main()
