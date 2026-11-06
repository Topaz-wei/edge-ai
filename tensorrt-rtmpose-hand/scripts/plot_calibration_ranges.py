# 读取 TensorRT 8.5 Entropy2 cache，绘制跨 tensor 的 scale/量化范围分布；不伪造激活频数。
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_scales(path):
    lines = path.read_text().splitlines()
    if not lines or lines[0] != 'TRT-8502-EntropyCalibration2':
        raise ValueError('Expected TensorRT 8.5.2 EntropyCalibration2 cache')
    rows = []
    names = set()
    for line in lines[1:]:
        if not line.strip():
            continue
        name, bits = line.rsplit(':', 1)
        bits = bits.strip()
        if len(bits) != 8 or name in names:
            raise ValueError('Invalid or duplicate cache entry: ' + line)
        scale = struct.unpack('!f', bytes.fromhex(bits))[0]
        names.add(name)
        rows.append({'tensor': name, 'hex': bits, 'scale': scale,
                     'symmetric_threshold_127s': 127 * scale,
                     'int8_grid_min': -128 * scale, 'int8_grid_max': 127 * scale})
    if not rows:
        raise ValueError('Empty cache')
    return rows


def render(cache, output):
    rows = read_scales(cache)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'ranges.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    excluded = [row for row in rows if not math.isfinite(row['scale']) or row['scale'] <= 0]
    scales = np.array([row['scale'] for row in rows if math.isfinite(row['scale']) and row['scale'] > 0])
    if scales.size == 0:
        raise ValueError('No finite positive scales to plot')
    thresholds = scales * 127
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(np.log10(scales), bins=30, color='#176b97', edgecolor='white')
    axes[0].set_xlabel('log10(calibration scale)')
    axes[0].set_ylabel('Number of tensors (NOT activation elements)')
    axes[0].set_title('Calibration scale distribution')
    axes[1].hist(np.log10(thresholds), bins=30, color='#ad5a20', edgecolor='white')
    axes[1].set_xlabel('log10(127 × scale)')
    axes[1].set_ylabel('Number of tensors')
    axes[1].set_title('Derived symmetric threshold distribution')
    figure.suptitle(f'{len(rows)} cache entries; {len(excluded)} excluded (nonfinite/nonpositive) | not activation histograms')
    figure.savefig(output / 'range-distribution.png', dpi=160)
    figure.savefig(output / 'range-distribution.svg')
    plt.close(figure)
    summary = {
        'cache': str(cache.resolve()),
        'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
        'tensor_count': len(rows),
        'excluded_entries': [{'tensor': row['tensor'], 'hex': row['hex'], 'scale_text': str(row['scale'])} for row in excluded],
        'scale_min_p10_p50_p90_max': np.percentile(scales, [0, 10, 50, 90, 100]).tolist(),
        'threshold_127s_min_p10_p50_p90_max': np.percentile(thresholds, [0, 10, 50, 90, 100]).tolist(),
        'interpretation': 'Cache IEEE754 scale; derived threshold=127*scale, grid=[-128*scale,127*scale]. Not observed activation min/max or histogram.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    table = ''.join(
        '<tr><td>' + html.escape(row['tensor']) + '</td><td>' + row['hex'] +
        f"</td><td>{row['scale']:.8g}</td><td>{row['symmetric_threshold_127s']:.8g}</td></tr>"
        for row in sorted(rows, key=lambda value: value['scale'], reverse=True)
    )
    page = '''<!doctype html><meta charset="utf-8"><title>Calibration 量化范围</title>
<style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#203040}
img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
input{padding:10px;width:400px}thead{background:#eaf2f8}</style>
<h1>Calibration cache：各 tensor 的量化范围分布</h1>
<p>这不是每层激活值的频数直方图。纵轴统计 tensor 数量；每个 tensor 仅贡献一个 scale。</p>
<p>cache 保存的是选定的量化 scale，不包含校准过程中的 bins/counts。127×scale 是换算的对称阈值，
不是实测 min/max，也不能据此判断离群点比例或 clipping 比例。</p>
<p>INT8 数值网格可表示为 [-128×scale, 127×scale]；下面按 127×scale 展示正向范围。</p>
<img src="range-distribution.png" alt="跨 tensor 的量化 scale 与阈值分布">
<p><a href="ranges.csv">下载全部数据 CSV</a> · <a href="range-distribution.svg">矢量图 SVG</a></p>
<input id="filter" placeholder="按 tensor 名称筛选">
<table><thead><tr><th>Tensor</th><th>Cache hex</th><th>Scale</th><th>127×scale</th></tr></thead><tbody>'''
    page = page.replace('<input id=', f'<p>共 {len(rows)} 个条目；其中 {len(excluded)} 个非有限或非正 scale 不进入直方图，仍保留在表格和 CSV 中。</p><input id=')
    page += table + '''</tbody></table><script>
document.getElementById('filter').oninput=function(){
for(const row of document.querySelectorAll('tbody tr')){
row.hidden=!row.cells[0].textContent.toLowerCase().includes(this.value.toLowerCase());}};
</script>'''
    (output / 'index.html').write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.cache, args.output)
