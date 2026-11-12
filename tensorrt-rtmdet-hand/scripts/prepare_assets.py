# 准备 RTMDet-Hand 模型与原图集合；复用冻结来源，不复制大资产或复用姿态模型张量。
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'tensorrt-rtmpose-hand'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def link_asset(target, source):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise ValueError('目标存在且不是预期软链接: ' + str(target))
    else:
        target.symlink_to(os.path.relpath(source, target.parent))


def main():
    split_path = SOURCE / 'calibration/split.json'
    split = json.loads(split_path.read_text())
    imported = {row['image_id']: row for row in
                map(json.loads, (SOURCE / 'data/manifest.jsonl').read_text().splitlines())}
    groups = {
        'calibration': split['calibration_image_ids'],
        'development': split['development_image_ids'],
        'validation': split['validation_in_car_image_ids'] + split['validation_supplement_image_ids'],
    }
    assert [len(ids) for ids in groups.values()] == [285, 46, 258]
    all_ids = [name for ids in groups.values() for name in ids]
    assert len(all_ids) == len(set(all_ids)) == 589
    all_hashes = set()
    report = {}
    for name, ids in groups.items():
        rows = []
        for index, image_id in enumerate(ids, start=1):
            source = SOURCE / 'data/raw' / image_id
            checksum = sha256(source)
            assert checksum == imported[image_id]['sha256'], image_id
            assert checksum not in all_hashes, image_id
            all_hashes.add(checksum)
            image = cv2.imread(str(source))
            assert image is not None and image.ndim == 3, image_id
            prefix = {'calibration': 'CAL', 'development': 'DEV', 'validation': 'VAL'}[name]
            sample_id = f'{prefix}{index:04d}'
            target = ROOT / 'datasets' / (name + '_set') / 'images' / (sample_id + source.suffix.lower())
            link_asset(target, source)
            rows.append({'sample_id': sample_id, 'image_id': image_id,
                         'original_filename': Path(image_id).name,
                         'collection_group': str(Path(image_id).parent),
                         'path': str(target.relative_to(ROOT)),
                         'source_path': str(source.relative_to(ROOT.parent)),
                         'original_source_path': imported[image_id]['source_path'],
                         'sha256': checksum, 'width': image.shape[1], 'height': image.shape[0],
                         'split': name, 'annotation_status': 'no_ground_truth'})
        folder = ROOT / 'datasets' / (name + '_set')
        (folder / 'manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        (folder / 'image_order.txt').write_text(''.join(row['path'] + '\n' for row in rows))
        report[name] = {'raw_images': len(rows), 'source_groups': dict(Counter(str(Path(i).parent) for i in ids)),
                        'manifest_sha256': sha256(folder / 'manifest.jsonl'),
                        'order_sha256': sha256(folder / 'image_order.txt')}
    model = SOURCE / 'models/rtmdet_hand.onnx'
    expected = next(m for m in json.loads((SOURCE / 'models/manifest.json').read_text())['models']
                    if m['target_path'] == 'models/rtmdet_hand.onnx')
    assert sha256(model) == expected['sha256']
    target = ROOT / 'models/rtmdet_hand.onnx'
    link_asset(target, model)
    graph = onnx.load(str(model))
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    inputs = [{'name': v.name, 'shape': v.shape, 'type': v.type} for v in session.get_inputs()]
    outputs = [{'name': v.name, 'shape': v.shape, 'type': v.type} for v in session.get_outputs()]
    assert len(inputs) == 1 and inputs[0]['type'] == 'tensor(float)'
    shape = inputs[0]['shape']
    assert len(shape) == 4 and all(isinstance(n, int) and n > 0 for n in shape[1:])
    smoke = session.run(None, {inputs[0]['name']: np.zeros([1] + shape[1:], dtype=np.float32)})
    assert all(np.isfinite(v).all() for v in smoke)
    save_json(ROOT / 'models/manifest.json', {
        'path': 'models/rtmdet_hand.onnx', 'sha256': sha256(model), 'size_bytes': model.stat().st_size,
        'source_path': str(model.relative_to(ROOT.parent)), 'original_source_path': expected['original_source_path'],
        'upstream_official_identity': 'not_yet_verified', 'inputs': inputs, 'outputs': outputs,
        'opsets': [{'domain': o.domain, 'version': o.version} for o in graph.opset_import],
        'smoke_test': {'provider': 'CPUExecutionProvider', 'input': 'zeros, batch=1',
                       'output_shapes': [list(v.shape) for v in smoke], 'all_finite': True},
        'preprocessing_contract': 'not_frozen'})
    save_json(ROOT / 'datasets/split.json', {'source_split': str(split_path.relative_to(ROOT.parent)),
        'source_split_sha256': sha256(split_path), 'unit': 'raw_image',
        'policy': '沿用既有原图采集组划分，保留零检测图；不使用RTMPose手框或手部张量作为检测输入。',
        'splits': groups})
    save_json(ROOT / 'datasets/verification.json', {'total_images': 589, 'unique_sha256': len(all_hashes),
        'image_id_overlap': 0, 'sha256_overlap': 0, 'all_images_decodable': True,
        'sets': report, 'tensors_generated': False, 'ground_truth_available': False,
        'onnx_version': onnx.__version__, 'onnxruntime_version': ort.__version__,
        'opencv_version': cv2.__version__, 'script_sha256': sha256(Path(__file__))})
    print(json.dumps({'sets': report, 'model_inputs': inputs, 'model_outputs': outputs}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
