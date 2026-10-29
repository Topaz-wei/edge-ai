# 本脚本按已核验的 RGB 契约重建校准/开发张量，保留旧张量与元数据备份。
# 不改变原图、裁剪、手框或顺序；备份目录存在时拒绝重复迁移。
import shutil

import cv2
import numpy as np

from verify_reference_config import CONFIG, ROOT, digest, load_rows, read_json, to_tensor, write_json


def main():
    config = read_json(
        CONFIG
    )
    assert config['frozen']
    base = ROOT / 'calibration'
    backup = base / 'provenance/before-rgb-reference'
    backup.mkdir(
        parents=True,
        exist_ok=False
    )
    rows = load_rows(
        base / 'hand_boxes.jsonl'
    )
    assert len(
        rows
    ) == 407
    # 先完整备份和生成候选张量，再替换当前文件，避免生成失败破坏现有数据。
    metadata = ['hand_boxes.jsonl', 'summary.json', 'approval.json', 'coverage.json',
                'verification.json', 'split.json', 'frames.jsonl',
                'calibration_tensor_order.txt', 'development_tensor_order.txt']
    for name in metadata:
        shutil.copy2(
            base / name,
            backup / name
        )
    prep = config['preprocessing']
    mean = np.asarray(
        prep['mean']
    )
    std = np.asarray(
        prep['std']
    )
    changes = []
    for row in rows:
        path = base / row['tensor_path']
        assert digest(
            path
        ) == row['tensor_sha256']
        saved = backup / row['tensor_path']
        saved.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        shutil.copy2(
            path,
            saved
        )
        crop_path = base / row['crop_path']
        crop = cv2.imread(
            str(
                crop_path
            ),
            cv2.IMREAD_COLOR
        )
        assert crop is not None
        rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )
        tensor = to_tensor(
            (rgb - mean) / std
        )
        assert tensor.shape == (1, 3, 256, 256) and np.isfinite(
            tensor
        ).all()
        candidate = path.with_suffix(
            '.rgb.npy'
        )
        np.save(
            candidate,
            tensor,
            allow_pickle=False
        )
        changes.append(
            {
            'hand_id': row['hand_id'],
            'tensor_path': row['tensor_path'],
            'old_sha256': row['tensor_sha256'],
            'new_sha256': digest(
                candidate
            ),
            'crop_path': row['crop_path'],
            'crop_sha256': digest(
                crop_path
            )
        }
        )
        row['tensor_sha256'] = digest(
            candidate
        )
    for row in rows:
        path = base / row['tensor_path']
        candidate = path.with_suffix(
            '.rgb.npy'
        )
        candidate.replace(
            path
        )
    import json
    manifest = '\n'.join(
        (
        json.dumps(
            row,
            ensure_ascii=False
        ) for row in rows
    )
    ) + '\n'
    (base / 'hand_boxes.jsonl').write_text(
        manifest,
        encoding='utf-8'
    )
    summary = read_json(
        base / 'summary.json'
    )
    summary['preprocessing']['color'] = 'OpenCV BGR -> RGB before affine and normalization'
    summary['preprocessing']['reference'] = '../configs/rtmpose_hand_reference.json'
    summary['preprocessing']['historical_training_config_match'] = 'RGB confirmed by identical-hash official ONNX archive and versioned training config'
    summary['reference_config_sha256'] = digest(
        CONFIG
    )
    summary['preprocessing_reference_validation'] = 'verified_rgb_contract'
    summary['hand_boxes_sha256'] = digest(
        base / 'hand_boxes.jsonl'
    )
    write_json(
        base / 'summary.json',
        summary
    )
    approval = read_json(
        base / 'approval.json'
    )
    approval['preprocessing_reference_validation'] = 'verified_by_script; human_pose_review_pending'
    write_json(
        base / 'approval.json',
        approval
    )
    unchanged = ['split.json', 'frames.jsonl', 'calibration_tensor_order.txt', 'development_tensor_order.txt']
    assert all(
        (digest(
            base / name
        ) == digest(
            backup / name
        ) for name in unchanged)
    )
    write_json(
        base / 'verification.json',
        {
            'status': 'passed',
            'scope': 'RGB tensor migration; see reference report for ORT checks',
            'reference_config_sha256': digest(
            CONFIG
        ),
            'previous_verification': 'provenance/before-rgb-reference/verification.json',
            'rebuilt_tensors': len(
            rows
        ),
            'unchanged_files': unchanged,
            'old_tensors_backed_up': True,
            'crop_content_unchanged': True,
            'user_approval_scope_unchanged': True
        }
    )
    write_json(
        base / 'provenance/rgb-reference-migration.json',
        {'reference_config_sha256': digest(
            CONFIG
        ), 'changes': changes}
    )
    print(
        'Rebuilt 407 RGB tensors; original tensors backed up.'
    )


if __name__ == '__main__':
    main()
