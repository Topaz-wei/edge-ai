#!/usr/bin/env python3
"""Real-camera A/B/C benchmark; runs existing binaries and never builds engines.

A=pre-extraction YOLO CLI, B=target-commit compatible CLI, C=target config CLI.
D=optional double-buffer config CLI. All CSV data and failure records are preserved.
Run the analyzer for results.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMITS = {'before': '282645d1bd684df4130409f20ce377a21b4266b0',
           'after': '5b736f9f308fbb115f73c234b642aba4cafdcf85'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''): h.update(part)
    return h.hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def query(argv):
    try:
        p = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=15)
        return {'argv': argv, 'returncode': p.returncode, 'output': p.stdout}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv': argv, 'error': str(e)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--build-before', type=Path, required=True)
    ap.add_argument('--build-after', type=Path, required=True)
    ap.add_argument('--build-double', type=Path, help='Optional double-buffer root build directory')
    ap.add_argument('--double-commit', help='Source commit for the supplied double-buffer binary')
    ap.add_argument('--source', default='/dev/v4l/by-id/usb-LRCP_F1080P_LRCP_F1080P_000000000001-video-index0')
    ap.add_argument('--frames', type=int, default=900)
    ap.add_argument('--warmup', type=int, default=150)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--suite', choices=['all', 'headless', 'display'], default='all')
    ap.add_argument('--comparison', choices=['architecture', 'double'], default='architecture',
                    help='double: only serial C vs double-buffer D, mixed INT8')
    args = ap.parse_args()
    if args.comparison == 'double' and not args.build_double:
        ap.error('--comparison double requires --build-double')
    if args.build_double and not args.double_commit:
        ap.error('--build-double requires --double-commit for provenance')
    if args.frames <= args.warmup + 1 or args.warmup < 0 or args.rounds <= 0:
        ap.error('Need frames > warmup + 1 and positive rounds')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before, after = args.build_before.resolve(), args.build_after.resolve()
    bins = {'A': before / 'yolo_video_cuda',
            'B': after / 'tensorrt-yolo/yolo_video_cuda',
            'C': after / 'tensorrt-yolo/edge_video'}
    if args.build_double:
        bins['D'] = args.build_double.resolve() / 'tensorrt-yolo/edge_video'
    engines = {'fp16': ROOT / 'tensorrt-yolo/engines/yolov8m_fp16.plan',
               'mixed_int8': ROOT / 'tensorrt-yolo/debug/trial007_sub_model22/yolov8m_int8.plan'}
    if args.comparison == 'double':
        bins = {k: bins[k] for k in ('C', 'D')}
        engines = {'mixed_int8': engines['mixed_int8']}
    occupied = query(['fuser', args.source])
    if occupied.get('returncode') == 0:
        raise RuntimeError('Camera is already in use: ' + occupied['output'])
    metadata = {'commits': dict(COMMITS), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'platform': platform.platform(), 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'engines': {k: {'path': str(p), 'sha256': sha(p)} for k, p in engines.items()},
                'binaries': {k: {'path': str(p), 'sha256': sha(p)} for k, p in bins.items()},
                'script_sha256': sha(Path(__file__).resolve()),
                'display': os.environ.get('DISPLAY'), 'xauthority': os.environ.get('XAUTHORITY'),
                'environment': [query(x) for x in [
                    ['v4l2-ctl', '-d', args.source, '--all', '--list-formats-ext'],
                    ['nvpmodel', '-q'], ['jetson_clocks', '--show'],
                    ['cmake', '--version'], ['g++', '--version'],
                    ['/usr/local/cuda/bin/nvcc', '--version'],
                    ['pkg-config', '--modversion', 'opencv4']]],
                'clock_policy': 'Existing MAXN/DVFS; no power/frequency/exposure changes',
                'warmup_policy': 'Exclude first warmup rows only; retain every subsequent outlier',
                'recording': False}
    if args.build_double:
        metadata['commits']['double'] = args.double_commit
    dump(out / 'metadata.json', metadata)
    # Rotate execution order; three rounds are fully position-balanced for ABC only.
    orders = {'fp16': ['ABC', 'BCA', 'CAB'],
              'mixed_int8': ['CBA', 'BAC', 'ACB'],
              'display': ['BAC', 'ACB', 'CBA']}
    if args.build_double:
        orders = {'fp16': ['ABCD', 'DCBA', 'BCDA', 'ADCB'],
                  'mixed_int8': ['DCBA', 'ABCD', 'CBAD', 'DABC'],
                  'display': ['BCDA', 'ADCB', 'DABC', 'CBAD']}
    schedule = []
    for r in range(args.rounds):
        if args.suite in ('all', 'headless'):
            precisions = ['fp16', 'mixed_int8'] if r % 2 == 0 else ['mixed_int8', 'fp16']
            for precision in precisions:
                schedule.extend((r + 1, 'headless', precision, v) for v in orders[precision][r % len(orders[precision])])
        if args.suite in ('all', 'display'):
            schedule.extend((r + 1, 'display', 'mixed_int8', v) for v in orders['display'][r % len(orders['display'])])
    if args.comparison == 'double':
        schedule = []
        for r in range(args.rounds):
            if args.suite in ('all', 'headless'):
                schedule.extend((r+1, 'headless', 'mixed_int8', v) for v in ('CD' if r%2 == 0 else 'DC'))
            if args.suite in ('all', 'display'):
                schedule.extend((r+1, 'display', 'mixed_int8', v) for v in ('DC' if r%2 == 0 else 'CD'))
    dump(out / 'schedule.json', schedule)
    results = []
    for index, (r, mode, precision, variant) in enumerate(schedule, 1):
        name = '{}-{}-{}-r{}'.format(mode, precision, variant, r)
        csv_path = out / (name + '.csv')
        if variant in ('C', 'D'):
            config = out / (name + '.yaml')
            config.write_text('''%YAML:1.0
---
schema_version: 1
source: "{source}"
display: {display}
max_frames: {frames}
capture_width: {width}
capture_height: {height}
capture_fps: 30
capture_fourcc: "MJPG"
capture_buffers: 2
timing_csv: "{csv}"
models:
  - {{ id: person, type: yolo, engine: "{engine}", input_stage: camera, graph: 1 }}
'''.format(source=args.source, display=int(mode == 'display'), frames=args.frames,
           width=args.width, height=args.height, csv=csv_path, engine=engines[precision]))
            command = [str(bins[variant]), str(config)]
        else:
            command = [str(bins[variant]), str(engines[precision]), args.source, 'camera',
                       '--capture-width', str(args.width), '--capture-height', str(args.height),
                       '--capture-fps', '30', '--capture-fourcc', 'MJPG', '--capture-buffers', '2',
                       '--max-frames', str(args.frames), '--timing-csv', str(csv_path)]
            if mode == 'headless': command.append('--no-display')
        print('START {}/{} {}'.format(index, len(schedule), name), flush=True)
        record = {'name': name, 'round': r, 'mode': mode, 'precision': precision,
                  'variant': variant, 'argv': command, 'utc': datetime.now(timezone.utc).isoformat()}
        telemetry = None
        start = time.monotonic()
        try:
            with (out / (name + '.tegrastats.log')).open('x') as stats, (out / (name + '.log')).open('x') as log:
                telemetry = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=stats, stderr=subprocess.STDOUT)
                p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=max(120, args.frames / 5))
                record['returncode'] = p.returncode
            record['process_wall_s'] = time.monotonic() - start
            if p.returncode: raise RuntimeError('Inference failed: ' + name)
            log_text = (out / (name + '.log')).read_text()
            for required in ['Capture buffers: requested=2, actual=2',
                             'Actual capture: {}x{}'.format(args.width, args.height),
                             'Source FPS: 30;', 'Frames: {}'.format(args.frames)]:
                if required not in log_text: raise RuntimeError('Missing capture verification: ' + required)
            if query(['v4l2-ctl', '-d', args.source, '--get-fmt-video'])['output'].find("'MJPG'") < 0:
                raise RuntimeError('Actual camera FOURCC is not MJPG')
            with csv_path.open() as f: rows = list(csv.DictReader(f))
            if len(rows) != args.frames: raise RuntimeError('Unexpected CSV frame count')
            loop = re.search(r'Measured loop wall: ([\d.]+) ms', log_text)
            record['reported_loop_ms'] = float(loop.group(1)) if loop else None
            record['detected_counts_at_logged_frames'] = [int(x) for x in re.findall(r'Frame \d+: detections=(\d+)', log_text)]
            record['csv_sha256'] = sha(csv_path)
        except Exception as e:
            record['error'] = str(e)
            raise
        finally:
            if telemetry is not None:
                telemetry.terminate()
                try: telemetry.wait(timeout=5)
                except subprocess.TimeoutExpired: telemetry.kill(); telemetry.wait()
            results.append(record)
            dump(out / 'runs.json', results)
        print('DONE {} wall={:.2f}s'.format(name, record['process_wall_s']), flush=True)
    metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['final_engine_hashes'] = {k: sha(p) for k, p in engines.items()}
    if any(metadata['final_engine_hashes'][k] != metadata['engines'][k]['sha256'] for k in engines):
        raise RuntimeError('Engine changed during test')
    dump(out / 'metadata.json', metadata)
    print('ALL {} REAL-CAMERA RUNS PASSED'.format(len(results)), flush=True)

if __name__ == '__main__':
    main()
