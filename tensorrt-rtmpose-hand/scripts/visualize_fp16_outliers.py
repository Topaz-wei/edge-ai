# 使用正式 FP32/FP16 engine 的既有开发集输出生成关键点人工对照页，不运行推理或改变预处理。
import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np

from verify_trt_fp16 import (
    ROOT, digest, read_json, load_batch, decode, validate_completion, required_artifacts,
)

EDGES = [(chain[i], chain[i + 1]) for finger in range(5)
         for chain in [[0] + list(range(1 + finger * 4, 5 + finger * 4))] for i in range(4)]
COLORS = {'fp32': '#00a6ff', 'fp16': '#ff6400'}
BGR = {'fp32': (255, 166, 0), 'fp16': (0, 100, 255)}
SIZE = 640


def svg_pose(
    points,
    valid,
    scores,
    mode,
    worst
):
    color = COLORS[mode]
    parts = [f'<g class="{mode}" stroke="{color}" fill="{color}">']
    for first, second in EDGES:
        if valid[first] and valid[second]:
            a, b = points[first], points[second]
            parts.append(f'<line x1="{a[0]:.4f}" y1="{a[1]:.4f}" x2="{b[0]:.4f}" y2="{b[1]:.4f}" stroke-width="2.3"/>')
    for index, point in enumerate(points):
        if not valid[index]:
            continue
        x, y = point
        parts.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="3.3" stroke="white" stroke-width="0.8"><title>{mode} 点{index} score={scores[index]:.6f}</title></circle>')
        minor = '' if index in worst else ' minor'
        parts.append(f'<text class="point-label{minor}" x="{x + 5:.4f}" y="{y - 5:.4f}" font-size="14" stroke="black" stroke-width="3" paint-order="stroke">{index}</text>')
    parts.append('</g>')
    return ''.join(parts)


def panel(
    image,
    poses,
    arrows=''
):
    return f'<svg viewBox="0 0 {SIZE} {SIZE}" xmlns="http://www.w3.org/2000/svg"><image href="{image}" width="{SIZE}" height="{SIZE}"/>{poses}{arrows}</svg>'


def raster_pose(
    image,
    points,
    valid,
    mode,
    worst
):
    result = image.copy()
    pixels = np.rint(points).astype(int)
    for first, second in EDGES:
        if valid[first] and valid[second]:
            cv2.line(
                result,
                tuple(pixels[first]),
                tuple(pixels[second]),
                BGR[mode],
                2,
                cv2.LINE_AA
            )
    for index in range(21):
        if valid[index]:
            cv2.circle(
                result,
                tuple(pixels[index]),
                3,
                BGR[mode],
                -1,
                cv2.LINE_AA
            )
            if index in worst:
                cv2.putText(
                    result,
                    str(index),
                    tuple(pixels[index] + [5, -5]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--output', type=Path, default=ROOT / 'validation/fp16-keypoint-review')
    args = parser.parse_args()
    if not 1 <= args.count <= 55:
        raise ValueError('count must be within 1..55')
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    cases = ROOT / 'validation/trt-fp32/cases.json'
    samples = read_json(cases)['samples']
    assert len(samples) == 55
    baseline32 = read_json(ROOT / 'benchmarks/cpp-2g-performance.json')['consistency']
    baseline16 = read_json(ROOT / 'benchmarks/fp16-cpp-2g.json')
    assert digest(cases) == baseline32['cases_sha256'] == baseline16['cases_sha256']
    folders = {'fp32': ROOT / 'validation/cpp-2g-performance/fp32/npy',
               'fp16': ROOT / 'validation/trt-fp16-cpp-2g/npy'}
    identities = {'fp32': baseline32['fp32_engine_sha256'], 'fp16': baseline16['engine_sha256']}
    locks = read_json(ROOT / 'validation/trt-fp16-cpp-2g/input-lock.json')['assets_sha256']
    decoded = {}
    provenance = {'cases_sha256': digest(cases), 'engine_sha256': identities,
                  'script_sha256': digest(Path(__file__).resolve()), 'images_sha256': {},
                  'completion_sha256': {}, 'scope': 'Original C++ builder 2GiB FP32/FP16, batch1, frozen 55 development; no diagnostic engine, no INT8, no GT.'}
    for mode, folder in folders.items():
        engine = ROOT / ('engines/rtmpose_hand_' + mode + '_cpp_2g.engine')
        assert digest(engine) == identities[mode]
        validate_completion(
            folder,
            identities[mode],
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
    input_error = np.linalg.norm(decoded['fp16'][0] - decoded['fp32'][0], axis=-1)
    source_error = np.linalg.norm(decoded['fp16'][1] - decoded['fp32'][1], axis=-1)
    selected = sorted(range(55), key=lambda index: (-float(input_error[index].max()), samples[index]['id']))[:args.count]
    output.mkdir(parents=True)
    sections = []
    records = []
    csv_rows = []
    for rank, index in enumerate(selected, 1):
        sample = samples[index]
        sample_id = sample['id']
        image_path = Path(sample['image'])
        assert digest(image_path) == locks[str(image_path)], str(image_path)
        provenance['images_sha256'][str(image_path)] = digest(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError('Cannot read source image')
        bbox = np.array(sample['bbox'], dtype=np.float64)
        center = (bbox[:2] + bbox[2:]) / 2
        side = float(np.max(bbox[2:] - bbox[:2]) * 1.6)
        assert side > 0
        origin = center - side / 2
        scale = SIZE / side
        transform = np.array([[scale, 0, -origin[0] * scale], [0, scale, -origin[1] * scale]])
        crop = cv2.warpAffine(
            image,
            transform,
            (SIZE, SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        crop_name = sample_id + '-crop.png'
        assert cv2.imwrite(str(output / crop_name), crop)
        points = {mode: (decoded[mode][1][index] - origin) * scale for mode in decoded}
        assert all(np.isfinite(value).all() for value in points.values())
        worst = set(np.argsort(-input_error[index], kind='stable')[:3].tolist())
        groups = {mode: svg_pose(points[mode], decoded[mode][3][index], decoded[mode][2][index], mode, worst) for mode in decoded}
        arrows = ['<g class="arrows" stroke="#ffeb3b" stroke-width="2.5" stroke-dasharray="5 3">']
        overlay = crop.copy()
        for point in range(21):
            if decoded['fp32'][3][index, point] and decoded['fp16'][3][index, point]:
                start, end = points['fp32'][point], points['fp16'][point]
                if input_error[index, point] > 0:
                    arrows.append(f'<line x1="{start[0]:.4f}" y1="{start[1]:.4f}" x2="{end[0]:.4f}" y2="{end[1]:.4f}"><title>点{point} 输入偏移{input_error[index, point]:.4f}px；原图偏移{source_error[index, point]:.4f}px</title></line>')
                    cv2.line(
                        overlay,
                        tuple(np.rint(start).astype(int)),
                        tuple(np.rint(end).astype(int)),
                        (59, 235, 255),
                        2,
                        cv2.LINE_AA
                    )
        arrows.append('</g>')
        raster = []
        for mode in ['fp32', 'fp16']:
            raster.append(raster_pose(crop, points[mode], decoded[mode][3][index], mode, worst))
            overlay = raster_pose(overlay, points[mode], decoded[mode][3][index], mode, worst)
        # 偏移箭头最后绘制，避免被重合骨架遮住；方向为 FP32 → FP16。
        for point in range(21):
            if decoded['fp32'][3][index, point] and decoded['fp16'][3][index, point] and input_error[index, point] > 0:
                start = tuple(np.rint(points['fp32'][point]).astype(int))
                end = tuple(np.rint(points['fp16'][point]).astype(int))
                cv2.arrowedLine(
                    overlay,
                    start,
                    end,
                    (59, 235, 255),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.2
                )
        raster.append(overlay)
        triptych = np.full((SIZE + 45, SIZE * 3, 3), 25, dtype=np.uint8)
        for number, title in enumerate(['TRT FP32', 'TRT FP16', 'Overlay: FP32 blue / FP16 orange']):
            triptych[45:, number * SIZE:(number + 1) * SIZE] = raster[number]
            cv2.putText(
                triptych,
                title,
                (number * SIZE + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        png_name = sample_id + '-comparison.png'
        assert cv2.imwrite(str(output / png_name), triptych)
        context = image.copy()
        cv2.rectangle(
            context,
            tuple(np.rint(bbox[:2]).astype(int)),
            tuple(np.rint(bbox[2:]).astype(int)),
            (0, 220, 0),
            2
        )
        context_name = sample_id + '-context.jpg'
        assert cv2.imwrite(str(output / context_name), context)
        rows = []
        for point in range(21):
            entry = {'sample_id': sample_id, 'point_id': point,
                     'input_l2_px': float(input_error[index, point]), 'source_l2_px': float(source_error[index, point])}
            for mode in ['fp32', 'fp16']:
                entry[mode + '_score'] = float(decoded[mode][2][index, point])
                entry[mode + '_valid'] = bool(decoded[mode][3][index, point])
                entry[mode + '_source_x'] = float(decoded[mode][1][index, point, 0])
                entry[mode + '_source_y'] = float(decoded[mode][1][index, point, 1])
            csv_rows.append(entry)
            rows.append(f'<tr><td>{point}</td><td>{entry["input_l2_px"]:.3f}</td><td>{entry["source_l2_px"]:.3f}</td><td>{entry["fp32_score"]:.6f}</td><td>{entry["fp16_score"]:.6f}</td><td>{entry["fp32_valid"]} / {entry["fp16_valid"]}</td></tr>')
        max_point = int(np.argmax(input_error[index]))
        records.append({'rank': rank, 'sample_id': sample_id, 'max_point_id': max_point,
                        'max_input_l2_px': float(input_error[index].max()), 'max_source_l2_px': float(source_error[index].max()),
                        'bbox': sample['bbox'], 'display_roi_origin': origin.tolist(), 'display_roi_side': side,
                        'display_scale': scale, 'comparison_png': png_name})
        sections.append(f'''<section id="{sample_id}"><h2>#{rank} · {sample_id}</h2>
<p class="metrics">最大输入偏移 <b>{input_error[index].max():.3f} px</b> · 最大原图偏移 <b>{source_error[index].max():.3f} px</b> · 最大偏移点 <b>{max_point}</b></p>
<div class="panels"><figure><figcaption class="blue">TRT FP32</figcaption>{panel(crop_name, groups['fp32'])}</figure>
<figure><figcaption class="orange">TRT FP16</figcaption>{panel(crop_name, groups['fp16'])}</figure>
<figure class="overlay"><figcaption>叠加：蓝 FP32 / 橙 FP16 / 黄 偏移连线</figcaption>{panel(crop_name, groups['fp32'] + groups['fp16'], ''.join(arrows))}</figure></div>
<p><a href="{png_name}" target="_blank">打开 / 保存高清三联图</a> · <a href="{crop_name}" target="_blank">查看无标注裁剪</a></p>
<details><summary>查看原图位置与全部 21 点数值</summary><img class="context" src="{context_name}" alt="原图中的固定手框"><p>绿色为固定手框。关键点编号从0开始；未设置额外置信度筛选。</p>
<table><thead><tr><th>点ID</th><th>输入偏移 px</th><th>原图偏移 px</th><th>FP32 score</th><th>FP16 score</th><th>valid FP32/FP16</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details></section>''')
    navigation = ' '.join(f'<a href="#{row["sample_id"]}">{row["sample_id"]}</a>' for row in records)
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FP32 / FP16 关键点大偏差评审</title>
<style>body{font:16px/1.65 system-ui,sans-serif;background:#f2f5f8;color:#172d43;margin:0}main{max-width:1600px;margin:auto;padding:28px}h1{margin-bottom:8px}section{background:white;padding:22px;margin:24px 0;border-radius:12px;box-shadow:0 2px 10px #0001;scroll-margin-top:100px}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}svg{width:100%;display:block;background:#111}figcaption{font-weight:700;padding:8px 0}.blue{color:#007bbf}.orange{color:#c74c00}a{color:#006baf}nav a{display:inline-block;padding:5px 10px;background:white;border-radius:5px;margin:4px}.controls{position:sticky;top:0;z-index:5;background:#e8f1f9f5;padding:12px;border-bottom:1px solid #bad0e1;display:flex;gap:20px;flex-wrap:wrap}.point-label.minor{display:none}.all-labels .point-label.minor{display:block}.hide32 .overlay .fp32,.hide16 .overlay .fp16,.hide-arrows .overlay .arrows{display:none}.context{max-width:100%;max-height:500px;margin-top:15px}table{border-collapse:collapse;font-size:14px;width:100%}td,th{border-bottom:1px solid #dce4ec;padding:8px;text-align:right}summary{cursor:pointer;font-weight:600}.metrics{background:#eef5fa;padding:10px}@media(max-width:900px){.panels{grid-template-columns:1fr}main{padding:12px}.controls{position:static}}</style></head><body><main>
<h1>FP32 / FP16：大偏差关键点对照</h1>
<p>使用正式 C++ Builder / 2 GiB 的 TRT FP32、FP16 engine 既有结果，batch=1；不使用诊断 engine 或 INT8。FP32 是数值参考，并非人工真值。</p>
<p>从55个开发样本中按最大<strong>输入空间</strong>关键点偏移排序选出前6个。所有面板使用同一原图和同一显示裁剪，局部放大仅用于观察；不重新裁剪模型输入、不修改关键点位置。</p>
<p>输入偏移以256×256模型输入坐标计，原图偏移以原始图片像素计；放大显示后的距离不代表这两个数值。默认标注每图偏移最大的3个点，所有有效关键点都绘制。黑色区域可能是显示裁剪超出原图后的零填充；可展开原图位置确认。</p>
<nav>''' + navigation + '''</nav><div class="controls"><label><input type="checkbox" checked data-class="hide32">叠加显示 FP32</label><label><input type="checkbox" checked data-class="hide16">叠加显示 FP16</label><label><input type="checkbox" checked data-class="hide-arrows">显示偏移连线</label><label><input id="labels" type="checkbox">显示全部点编号</label><a href="keypoints.csv">下载数值 CSV</a></div>''' + ''.join(sections) + '''<p><a href="manifest.json">数据来源与文件哈希</a></p></main><script>for(const input of document.querySelectorAll('[data-class]')){input.onchange=()=>document.body.classList.toggle(input.dataset.class,!input.checked);}document.getElementById('labels').onchange=function(){document.body.classList.toggle('all-labels',this.checked);};</script></body></html>'''
    page = page.replace('排序选出前6个', '排序选出前%d个' % args.count)
    (output / 'index.html').write_text(page)
    with (output / 'keypoints.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    provenance['selected'] = records
    provenance['outputs_sha256'] = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    (output / 'manifest.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(records, indent=2))
    print('Review page:', output / 'index.html')


if __name__ == '__main__':
    main()
