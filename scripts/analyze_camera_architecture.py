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
            'p95': percentile(.95), 'p99': percentile(.99), 'min': values[0], 'max': values[-1]}


def model_service_ms(row):
    # prepare moved out of infer in the double-buffer version. Comparing infer
    # alone would falsely attribute CPU work removal to faster GPU execution.
    return row['geometry_ms'] + row.get('prepare_ms', 0) + row['pipeline_ms']


def long_interval(before, after):
    consumer_work = sum(after.get(k, 0) for k in
                        ('pipeline_ms', 'draw_ms', 'encode_ms', 'display_ms'))
    return {'frame': int(after['frame']),
            'interval_ms': after['end_ms'] - before['end_ms'],
            'previous_frame': int(before['frame']),
            'after_logged_frame': before['frame'] == 1 or before['frame'] % 30 == 0,
            'start_minus_previous_end_ms': after['start_ms'] - before['end_ms'],
            # Approximate infer start from measured stage durations; small host
            # bookkeeping gaps remain. This is not a CUDA kernel timestamp.
            'estimated_consumer_gap_ms': after['end_ms'] - consumer_work - before['end_ms'],
            'model_service_ms': model_service_ms(after),
            **{k: after[k] for k in ('total_ms', 'read_ms', 'grab_ms', 'retrieve_ms', 'geometry_ms', 'prepare_ms',
               'pipeline_ms', 'draw_ms', 'display_ms', 'queue_wait_ms',
               'read_age_ms', 'completion_age_ms') if k in after}}


def validate_read_timing(row):
    if 'grab_ms' not in row and 'retrieve_ms' not in row:
        return  # Historical CSV has only read_ms.
    fields = ('read_ms', 'grab_ms', 'retrieve_ms')
    if any(k not in row or not math.isfinite(row[k]) or row[k] < 0 for k in fields):
        raise ValueError('Missing or invalid read-stage timing')
    if abs(row['read_ms'] - row['grab_ms'] - row['retrieve_ms']) > 0.00001:
        raise ValueError('Read-stage timings do not sum to read_ms')
    if row['frame'] == 1 and any(row[k] != 0 for k in fields):
        raise ValueError('First setup read must be excluded')


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
        for row in all_rows:
            validate_read_timing(row)
        rows = all_rows[meta['arguments']['warmup']:]
        intervals = [b['end_ms']-a['end_ms'] for a,b in zip(rows,rows[1:])]
        assert min(intervals) > 0
        fps = len(intervals)*1000/sum(intervals)
        metrics = {k:[r[k] for r in rows] for k in ('read_ms','geometry_ms','pipeline_ms','draw_ms','display_ms','total_ms')}
        for k in ('prepare_ms', 'queue_wait_ms', 'grab_ms', 'retrieve_ms'):
            if k in rows[0]: metrics[k] = [r[k] for r in rows]
        metrics['model_service_ms'] = [model_service_ms(r) for r in rows]
        metrics['consumer_work_ms'] = [sum(r.get(k, 0) for k in ('pipeline_ms','draw_ms','encode_ms','display_ms')) for r in rows]
        metrics['overlap_fraction'] = [float(b['start_ms'] < a['end_ms']) for a,b in zip(rows,rows[1:])]
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
            'completion_intervals_gt50ms':sum(x>50 for x in intervals),
            'completion_intervals_gt100ms':sum(x>100 for x in intervals),
            'completion_intervals_gt200ms':sum(x>200 for x in intervals),
            'long_intervals':[long_interval(a,b) for a,b in zip(rows,rows[1:]) if b['end_ms']-a['end_ms']>100],
            'invalid_completion_age':len(rows)-len(metrics['completion_age_ms']),
            'capture_timestamp_nonincreasing':sum(b['capture_timestamp_ms']<=a['capture_timestamp_ms'] for a,b in zip(rows,rows[1:]))})
    result = {'method':'Exclude initial warmup only; FPS per run from end timestamp differences; linear percentile on pooled frames; FPS summary is arithmetic mean of runs.',
              'runs':details, 'groups':{key:{k:distribution(v) for k,v in group.items()} for key,group in groups.items()}}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for key,group in sorted(result['groups'].items()):
        print('{} FPS={:.3f} [{:.3f},{:.3f}] compute={:.3f}ms age={:.3f}ms'.format(key,group['run_fps']['mean'],group['run_fps']['min'],group['run_fps']['max'],group['model_service_ms']['mean'],group['completion_age_ms'].get('mean',float('nan'))))
    print('Validated {} runs'.format(len(runs)))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    analyze(ap.parse_args().output)
