#!/usr/bin/env python3
"""Validate and summarize benchmark_camera_architecture.py raw evidence."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    def percentile(p):
        x = (len(values) - 1) * p
        lo = int(x)
        return values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (x-lo)
    return {'n': len(values), 'mean': st.mean(values), 'p50': percentile(.5),
            'p95': percentile(.95), 'min': values[0], 'max': values[-1]}


def analyze(out):
    meta = json.loads((out/'metadata.json').read_text())
    runs = json.loads((out/'runs.json').read_text())
    schedule = json.loads((out/'schedule.json').read_text())
    assert meta.get('finished_utc'), 'Run is not finished'
    assert len(runs) == len(schedule), 'Incomplete schedule'
    assert meta['final_engine_hashes'] == {k:v['sha256'] for k,v in meta['engines'].items()}
    groups = defaultdict(lambda: defaultdict(list))
    details = []
    for run, expected in zip(runs, schedule):
        assert [run[k] for k in ('round','mode','precision','variant')] == expected
        assert run['returncode'] == 0 and 'error' not in run
        path = out/(run['name']+'.csv')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run['csv_sha256']
        with path.open() as f:
            all_rows = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        assert len(all_rows) == meta['arguments']['frames']
        assert [r['frame'] for r in all_rows] == list(range(1,len(all_rows)+1))
        assert all(math.isfinite(v) for row in all_rows for v in row.values())
        rows = all_rows[meta['arguments']['warmup']:]
        intervals = [b['end_ms']-a['end_ms'] for a,b in zip(rows,rows[1:])]
        assert min(intervals) > 0
        fps = len(intervals)*1000/sum(intervals)
        metrics = {k:[r[k] for r in rows] for k in ('read_ms','geometry_ms','pipeline_ms','draw_ms','display_ms','total_ms')}
        metrics['geometry_plus_pipeline_ms'] = [r['geometry_ms']+r['pipeline_ms'] for r in rows]
        for k in ('read_age_ms','completion_age_ms'):
            metrics[k] = [r[k] for r in rows if r[k] >= 0]
        metrics['completion_interval_ms'] = intervals
        stats = (out/(run['name']+'.tegrastats.log')).read_text()
        metrics['gpu_temp_C'] = [float(x) for x in re.findall(r'GPU@([\d.]+)C',stats)]
        metrics['cpu_temp_C'] = [float(x) for x in re.findall(r'CPU@([\d.]+)C',stats)]
        metrics['gpu_util_pct'] = [float(x) for x in re.findall(r'GR3D_FREQ (\d+)%',stats)]
        metrics['cpu_clock_MHz'] = [float(x) for x in re.findall(r'\d+%@([\d]+)',stats)]
        key = '/'.join(run[k] for k in ('mode','precision','variant'))
        for k,v in metrics.items(): groups[key][k].extend(v)
        groups[key]['run_fps'].append(fps)
        details.append({'name':run['name'], 'fps':fps,
            'metrics':{k:distribution(v) for k,v in metrics.items()},
            'completion_intervals_gt100ms':sum(x>100 for x in intervals),
            'completion_intervals_gt200ms':sum(x>200 for x in intervals),
            'long_intervals':[{'frame':int(rows[i+1]['frame']),'interval_ms':v,
                'total_ms':rows[i+1]['total_ms'],'read_ms':rows[i+1]['read_ms'],
                'geometry_plus_pipeline_ms':rows[i+1]['geometry_ms']+rows[i+1]['pipeline_ms'],
                'display_ms':rows[i+1]['display_ms']} for i,v in enumerate(intervals) if v>100],
            'invalid_completion_age':len(rows)-len(metrics['completion_age_ms']),
            'capture_timestamp_nonincreasing':sum(b['capture_timestamp_ms']<=a['capture_timestamp_ms'] for a,b in zip(rows,rows[1:]))})
    result = {'method':'Exclude initial warmup only; FPS per run from end timestamp differences; linear percentile on pooled frames; FPS summary is arithmetic mean of runs.',
              'runs':details, 'groups':{key:{k:distribution(v) for k,v in group.items()} for key,group in groups.items()}}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for key,group in sorted(result['groups'].items()):
        print('{} FPS={:.3f} [{:.3f},{:.3f}] compute={:.3f}ms age={:.3f}ms'.format(key,group['run_fps']['mean'],group['run_fps']['min'],group['run_fps']['max'],group['geometry_plus_pipeline_ms']['mean'],group['completion_age_ms'].get('mean',float('nan'))))
    print('Validated {} runs'.format(len(runs)))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    analyze(ap.parse_args().output)
