# 为人工修正的开发集手框建立独立复核分支，保留冻结数据及历史误差统计。
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from visualize_precision_comparison import LABELS, COLORS, panel, svg_pose
from verify_trt_fp16 import ROOT, digest, read_json, load_batch, decode

OUT = ROOT / 'validation/corrected-bbox-review'
ENGINES = dict(zip(LABELS, ['rtmpose_hand_fp32_cpp_2g.engine', 'rtmpose_hand_fp16_cpp_2g.engine',
                           'rtmpose_hand_int8.engine', 'rtmpose_hand_int8_fp16.engine']))


def main():
    OUT.mkdir(exist_ok=False)
    original = next(s for s in read_json(ROOT / 'validation/trt-fp32/cases.json')['samples']
                    if s['id'] == 'F0018-D001')
    corrected = dict(original)
    corrected['bbox'] = [102.0, 332.0, 194.0, 418.0]
    cases = OUT / 'cases.json'
    cases.write_text(json.dumps({'samples': [corrected]}, indent=2))
    image = cv2.imread(original['image'])
    # 两行结果使用同一显示区域，避免裁剪大小变化制造视觉偏差。
    origin = np.array([85.0, 320.0])
    side = 125.0
    matrix = np.array([[640 / side, 0, -origin[0] * 640 / side],
                       [0, 640 / side, -origin[1] * 640 / side]])
    display = cv2.warpAffine(image, matrix, (640, 640))
    cv2.imwrite(str(OUT / 'display.png'), display)
    context = image.copy()
    for bbox, color in [(original['bbox'], (0, 100, 255)), (corrected['bbox'], (0, 220, 0))]:
        cv2.rectangle(context, tuple(np.rint(bbox[:2]).astype(int)),
                      tuple(np.rint(bbox[2:]).astype(int)), color, 1)
    cv2.imwrite(str(OUT / 'bbox-context.png'), context)
    folders = {'fp32': 'cpp-2g-performance/fp32', 'fp16': 'trt-fp16-cpp-2g',
               'int8': 'trt-int8', 'int8_fp16': 'trt-int8-fp16'}
    old_samples = read_json(ROOT / 'validation/trt-fp32/cases.json')['samples']
    old_index = next(i for i, s in enumerate(old_samples) if s['id'] == original['id'])
    results = {'original': {}, 'corrected': {}}
    provenance = {'original': original, 'corrected': corrected,
                  'reason': '人工检查原图，补足被旧检测框截掉的拇指及掌根；不是关键点GT。',
                  'source_sha256': digest(Path(original['image'])), 'engines': {}}
    runner = ROOT.parent / 'build-shared/tensorrt-rtmpose-hand/rtmpose_fp32'
    for mode, filename in ENGINES.items():
        engine = ROOT / 'engines' / filename
        subprocess.run([str(runner), str(engine), str(cases), str(OUT / mode), 'image'], check=True)
        raw, cpp = load_batch(OUT / mode, [corrected], 1)
        results['corrected'][mode] = decode(*raw, [corrected])
        assert np.max(np.abs(results['corrected'][mode][1] - cpp[..., 2:4])) < 1e-9
        raw, _ = load_batch(ROOT / 'validation' / folders[mode] / 'npy', old_samples, 1)
        results['original'][mode] = decode(raw[0][old_index:old_index+1], raw[1][old_index:old_index+1], [original])
        provenance['engines'][mode] = digest(engine)
    tensor = np.fromfile(OUT / 'fp32/F0018-D001_input.bin', dtype=np.float32).reshape(1, 3, 256, 256)
    for mode in ENGINES:
        assert np.array_equal(tensor.ravel(), np.fromfile(OUT / mode / 'F0018-D001_input.bin', dtype=np.float32))
    np.save(OUT / 'F0018-D001-corrected.npy', tensor)
    prep = read_json(ROOT / 'configs/rtmpose_hand_reference.json')['preprocessing']
    rgb = tensor[0].transpose(1, 2, 0) * np.array(prep['std']) + np.array(prep['mean'])
    cv2.imwrite(str(OUT / 'corrected-crop.png'), np.clip(np.rint(rgb), 0, 255).astype(np.uint8)[..., ::-1])
    sections = []
    metrics = {}
    for variant, label in [('original', '修正前'), ('corrected', '修正后')]:
        sections.append('<h2>' + label + '</h2><div class="grid">')
        metrics[variant] = {}
        baseline = results[variant]['fp32']
        for mode in LABELS:
            value = results[variant][mode]
            error = np.linalg.norm(value[0] - baseline[0], axis=-1)
            source_error = np.linalg.norm(value[1] - baseline[1], axis=-1)
            metrics[variant][mode] = {'mean_input_px': float(error.mean()), 'max_input_px': float(error.max()),
                                      'max_source_px': float(source_error.max())}
            points = (value[1][0] - origin) * 640 / side
            pose = svg_pose(points, value[3][0], value[2][0], mode, [0, 1, 2, 3, 4])
            sections.append('<figure><figcaption>' + LABELS[mode] +
                            f'<br>相对本行FP32：max {error.max():.3f} 输入px / {source_error.max():.3f} 原图px</figcaption>' +
                            panel('display.png', pose) + '</figure>')
        sections.append('</div>')
    provenance['metrics'] = metrics
    provenance['cases_sha256'] = digest(cases)
    provenance['script_sha256'] = digest(Path(__file__))
    (OUT / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>手框修正对照</title>
<style>body{font:16px system-ui;margin:24px;background:#f0f4f8}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}svg{width:100%}.minor{display:none}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}</style>
<h1>F0018-D001：手框修正前后</h1><p>同一原图、四个既有engine，不更改Reference。两行使用相同显示区域，各自与本行FP32比较。输入px受ROI尺度影响，跨行同时查看原图px；没有GT，偏差减小不等于真实精度提升。</p>
<p>原图：橙色旧框，绿色修正框。F0012-D000原图缺失区域保持不变，缺失位置不作真实精度判断，历史全点一致性统计保留。</p>
<a href="bbox-context.png">查看原图与手框</a> | <a href="corrected-crop.png">修正后256×256输入</a> | <a href="manifest.json">来源与统计</a>
''' + ''.join(sections))
    provenance['outputs_sha256'] = {str(p.relative_to(OUT)): digest(p) for p in OUT.rglob('*') if p.is_file()}
    (OUT / 'manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
