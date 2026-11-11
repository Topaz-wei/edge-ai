# 在既有 FP16 评审样本上增加两版 INT8，复用正式 engine 的冻结输出和统一显示几何。
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from visualize_fp16_outliers import SIZE, COLORS, BGR, svg_pose, panel, raster_pose
from verify_trt_fp16 import ROOT, digest, read_json, validate_completion, required_artifacts, load_batch, decode

OUT = ROOT / 'validation/precision-keypoint-review'
LABELS = {'fp32': 'TRT FP32', 'fp16': 'TRT FP16',
          'int8': 'INT8（FP32 fallback）', 'int8_fp16': 'INT8（FP16 fallback）'}
TITLES = ['TRT FP32', 'TRT FP16', 'INT8 + FP32 fallback', 'INT8 + FP16 fallback']
COLORS.update({'int8': '#dc3bce', 'int8_fp16': '#22cc70'})
BGR.update({'int8': (206, 59, 220), 'int8_fp16': (112, 204, 34)})


def main():
    if OUT.exists():
        raise ValueError('Output exists; preserve the existing review')
    cases = ROOT / 'validation/trt-fp32/cases.json'
    samples = read_json(cases)['samples']
    folders = {'fp32': 'cpp-2g-performance/fp32', 'fp16': 'trt-fp16-cpp-2g',
               'int8': 'trt-int8', 'int8_fp16': 'trt-int8-fp16'}
    engines = {'fp32': 'rtmpose_hand_fp32_cpp_2g.engine', 'fp16': 'rtmpose_hand_fp16_cpp_2g.engine',
               'int8': 'rtmpose_hand_int8.engine', 'int8_fp16': 'rtmpose_hand_int8_fp16.engine'}
    expected = {'fp32': read_json(ROOT / 'benchmarks/cpp-2g-performance.json')['consistency']['fp32_engine_sha256'],
                'fp16': read_json(ROOT / 'benchmarks/fp16-cpp-2g.json')['engine_sha256'],
                'int8': read_json(ROOT / 'benchmarks/int8-baseline.json')['build']['engine_sha256'],
                'int8_fp16': read_json(ROOT / 'benchmarks/int8-fp16-fallback.json')['build']['engine_sha256']}
    locks = read_json(ROOT / 'validation/trt-fp16-cpp-2g/input-lock.json')['assets_sha256']
    decoded = {}
    provenance = {'engine_sha256': expected, 'cases_sha256': digest(cases), 'completion_sha256': {},
                  'images_sha256': {}, 'script_sha256': digest(Path(__file__).resolve()),
                  'drawing_helpers_sha256': digest(ROOT / 'scripts/visualize_fp16_outliers.py'),
                  'scope': 'Four original engines; same 55 development batch1 outputs; FP32 numerical reference, no GT or diagnostic engines.'}
    for mode in LABELS:
        assert digest(ROOT / 'engines' / engines[mode]) == expected[mode]
        folder = ROOT / 'validation' / folders[mode] / 'npy'
        validate_completion(
            folder,
            expected[mode],
            digest(cases),
            'npy',
            55,
            required_artifacts(samples)
        )
        raw, cpp = load_batch(
            folder,
            samples,
            1
        )
        decoded[mode] = decode(*raw, samples)
        assert np.max(np.abs(decoded[mode][1] - cpp[..., 2:4])) < 1e-9
        provenance['completion_sha256'][mode] = digest(folder / 'complete.json')
    errors = {mode: np.linalg.norm(decoded[mode][0] - decoded['fp32'][0], axis=-1) for mode in LABELS}
    source_errors = {mode: np.linalg.norm(decoded[mode][1] - decoded['fp32'][1], axis=-1) for mode in LABELS}
    previous_path = ROOT / 'validation/fp16-keypoint-review/manifest.json'
    previous = read_json(previous_path)
    assert previous['engine_sha256']['fp32'] == expected['fp32'] and previous['engine_sha256']['fp16'] == expected['fp16']
    selected_ids = [entry['sample_id'] for entry in previous['selected']]
    reasons = {name: ['原FP16评审样本'] for name in selected_ids}
    for mode in ['int8', 'int8_fp16']:
        order = sorted(range(55), key=lambda index: (-float(errors[mode][index].max()), samples[index]['id']))[:3]
        for index in order:
            name = samples[index]['id']
            if name not in selected_ids:
                selected_ids.append(name)
                reasons[name] = []
            reasons[name].append(LABELS[mode] + ' 偏移前三')
    OUT.mkdir(parents=True)
    sections = []
    entries = []
    csv_rows = []
    for sample_id in selected_ids:
        index = next(index for index, sample in enumerate(samples) if sample['id'] == sample_id)
        sample = samples[index]
        image_path = Path(sample['image'])
        sha = digest(image_path)
        assert sha == locks[str(image_path)]
        provenance['images_sha256'][str(image_path)] = sha
        image = cv2.imread(str(image_path))
        assert image is not None
        bbox = np.asarray(sample['bbox'])
        side = float(np.max(bbox[2:] - bbox[:2]) * 1.6)
        origin = (bbox[:2] + bbox[2:]) / 2 - side / 2
        scale = SIZE / side
        if sample_id in {entry['sample_id'] for entry in previous['selected']}:
            old = next(entry for entry in previous['selected'] if entry['sample_id'] == sample_id)
            assert np.allclose(origin, old['display_roi_origin'], rtol=0, atol=1e-10)
            assert abs(side - old['display_roi_side']) < 1e-10
        matrix = np.array([[scale, 0, -origin[0] * scale], [0, scale, -origin[1] * scale]])
        crop = cv2.warpAffine(
            image,
            matrix,
            (SIZE, SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        crop_name = sample_id + '-crop.png'
        assert cv2.imwrite(str(OUT / crop_name), crop)
        points = {mode: (decoded[mode][1][index] - origin) * scale for mode in LABELS}
        worst = set(np.argsort(-np.maximum.reduce([errors[mode][index] for mode in LABELS]), kind='stable')[:3].tolist())
        poses = {mode: svg_pose(points[mode], decoded[mode][3][index], decoded[mode][2][index], mode, worst) for mode in LABELS}
        figures = []
        stat_rows = []
        point_rows = []
        overlay_layers = []
        record = {'sample_id': sample_id, 'reasons': reasons[sample_id], 'max_errors': {}}
        sheet = np.full((2 * (SIZE + 44), 2 * SIZE, 3), 25, dtype=np.uint8)
        for number, mode in enumerate(LABELS):
            figures.append(f'<figure><figcaption style="color:{COLORS[mode]}">{LABELS[mode]}</figcaption>{panel(crop_name, poses[mode])}</figure>')
            rendered = raster_pose(crop, points[mode], decoded[mode][3][index], mode, worst)
            top = (number // 2) * (SIZE + 44)
            left = (number % 2) * SIZE
            sheet[top + 44:top + 44 + SIZE, left:left + SIZE] = rendered
            cv2.putText(
                sheet,
                TITLES[number],
                (left + 12, top + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            if mode == 'fp32':
                continue
            max_point = int(np.argmax(errors[mode][index]))
            record['max_errors'][mode] = {'input_px': float(errors[mode][index].max()),
                                         'source_px': float(source_errors[mode][index].max()), 'point_id': max_point}
            stat_rows.append(f'<tr><td>{LABELS[mode]}</td><td>{errors[mode][index].max():.3f}</td><td>{source_errors[mode][index].max():.3f}</td><td>{max_point}</td></tr>')
            lines = []
            for point in range(21):
                if decoded['fp32'][3][index, point] and decoded[mode][3][index, point] and errors[mode][index, point] > 0:
                    a, b = points['fp32'][point], points[mode][point]
                    lines.append(f'<line x1="{a[0]:.4f}" y1="{a[1]:.4f}" x2="{b[0]:.4f}" y2="{b[1]:.4f}" stroke="#ffeb3b" stroke-width="2" stroke-dasharray="5 3"/>')
                row = {'sample_id': sample_id, 'mode': mode, 'point_id': point,
                       'input_l2_px': float(errors[mode][index, point]), 'source_l2_px': float(source_errors[mode][index, point]),
                       'fp32_score': float(decoded['fp32'][2][index, point]), 'actual_score': float(decoded[mode][2][index, point]),
                       'fp32_valid': bool(decoded['fp32'][3][index, point]), 'actual_valid': bool(decoded[mode][3][index, point])}
                csv_rows.append(row)
                point_rows.append(f'<tr><td>{LABELS[mode]}</td><td>{point}</td><td>{row["input_l2_px"]:.3f}</td><td>{row["source_l2_px"]:.3f}</td><td>{row["fp32_score"]:.5f}</td><td>{row["actual_score"]:.5f}</td><td>{row["fp32_valid"]}/{row["actual_valid"]}</td></tr>')
            display = '' if mode == 'int8' else 'display:none'
            overlay_layers.append(f'<g data-compare="{mode}" style="{display}">{poses[mode]}<g class="arrows">{"".join(lines)}</g></g>')
        png = sample_id + '-four-precisions.png'
        assert cv2.imwrite(str(OUT / png), sheet)
        context = image.copy()
        cv2.rectangle(
            context,
            tuple(np.rint(bbox[:2]).astype(int)),
            tuple(np.rint(bbox[2:]).astype(int)),
            (0, 220, 0),
            2
        )
        context_name = sample_id + '-context.jpg'
        assert cv2.imwrite(str(OUT / context_name), context)
        record['comparison_png'] = png
        entries.append(record)
        sections.append(f'''<section id="{sample_id}"><h2>{sample_id}</h2><p>{'；'.join(reasons[sample_id])}</p>
<table><tr><th>相对FP32</th><th>最大输入偏移 px</th><th>最大原图偏移 px</th><th>最大偏移点</th></tr>{''.join(stat_rows)}</table>
<div class="panels">{''.join(figures)}</div><p><a target="_blank" href="{png}">打开 / 保存四精度高清图</a> · <a target="_blank" href="{crop_name}">无标注裁剪</a></p>
<details open><summary>叠加比较：蓝色FP32 + 顶部所选精度，黄色为偏移连线</summary><div class="overlay">{panel(crop_name, poses['fp32'] + ''.join(overlay_layers))}</div></details>
<details><summary>原图位置与21点分数、偏移</summary><img class="context" src="{context_name}" alt="固定手框所在原图"><table><tr><th>精度</th><th>点</th><th>输入偏移</th><th>原图偏移</th><th>FP32 score</th><th>当前score</th><th>valid</th></tr>{''.join(point_rows)}</table></details></section>''')
    nav = ' '.join(f'<a href="#{name}">{name}</a>' for name in selected_ids)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FP32 / FP16 / INT8 关键点对照</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f1f5f9;color:#172d43;margin:0}main{max-width:1800px;margin:auto;padding:24px}section{background:white;padding:24px;margin:24px 0;border-radius:12px;scroll-margin-top:100px}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{font-weight:bold;padding:10px 0}svg{width:100%;display:block}.overlay{max-width:700px;margin:12px auto}.context{max-width:100%;max-height:500px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.controls{position:sticky;top:0;background:#e2edf9f5;padding:12px;z-index:10;display:flex;gap:20px;flex-wrap:wrap}a{color:#006bb0}nav a{display:inline-block;background:white;padding:6px;margin:4px;border-radius:6px}.point-label.minor{display:none}.all-labels .point-label.minor{display:block}.no-arrows .arrows{display:none}summary{cursor:pointer;padding:10px 0;font-weight:bold}select{font:inherit}@media(max-width:1100px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.panels{grid-template-columns:1fr}.controls{position:static}}</style>
<main><h1>FP32 / FP16 / 两版 INT8 关键点对照</h1>
<p>保留原FP16评审的6张图，再补充两版INT8各自最大偏移的前三个样本，去重展示。全部来自正式C++ Builder、2 GiB、batch=1的既有开发集结果，无诊断探针。</p>
<p>蓝：FP32；橙：FP16；紫：INT8（仅允许FP32 fallback）；绿：INT8（允许FP16 fallback）。开启INT8不代表所有层均为INT8，标签表示构建策略。</p>
<p>四个面板使用同一显示裁剪。输入偏移以256×256坐标计，原图偏移以源图像素计；局部放大不改变模型坐标。黑色边缘可能为越界填充。FP32是数值参考，并非人工真值；不额外按置信度筛点。</p><nav>''' + nav + '''</nav>
<div class="controls"><label>叠加比较 <select id="compare"><option value="int8">INT8（FP32 fallback）</option><option value="int8_fp16">INT8（FP16 fallback）</option><option value="fp16">FP16</option></select></label><label><input id="labels" type="checkbox">全部点编号</label><label><input id="arrows" type="checkbox" checked>偏移连线</label><a href="keypoints.csv">下载CSV</a></div>''' + ''.join(sections) + '''<p><a href="manifest.json">数据来源与哈希</a></p></main><script>
document.getElementById('compare').onchange=function(){for(const group of document.querySelectorAll('[data-compare]')){group.style.display=group.dataset.compare===this.value?'':'none';}};
document.getElementById('labels').onchange=function(){document.body.classList.toggle('all-labels',this.checked);};
document.getElementById('arrows').onchange=function(){document.body.classList.toggle('no-arrows',!this.checked);};
</script></html>'''
    (OUT / 'index.html').write_text(page)
    with (OUT / 'keypoints.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    provenance['selected'] = entries
    provenance['previous_review_sha256'] = digest(previous_path)
    provenance['outputs_sha256'] = {path.name: digest(path) for path in OUT.iterdir() if path.is_file()}
    (OUT / 'manifest.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')
    print('Samples:', len(entries), selected_ids)
    print('Review:', OUT / 'index.html')


if __name__ == '__main__':
    main()
