#!/usr/bin/env python3
"""Recorded commands for this experiment; run only after engine build succeeds."""
from pathlib import Path
import subprocess, json, hashlib, datetime, time, shlex
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
engines = {
    'fp16': 'engines/yolov8m_fp16_maxn.plan',
    'int8_old': 'engines/yolov8m_int8.plan',
    'int8_new': 'engines/int8-baseline-v1/yolov8m_int8.plan',
}
records = []
def run(name, cmd, required=True):
    start = datetime.datetime.now().astimezone().isoformat()
    t = time.monotonic()
    with (OUT / (name + '.log')).open('x') as log:
        log.write('Command: ' + shlex.join(cmd) + '\nStarted: ' + start + '\n')
        log.flush()
        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write('\nExit code: ' + str(result.returncode) + '\n')
    records.append(dict(name=name, command=cmd, started=start, seconds=time.monotonic()-t, exit_code=result.returncode))
    (OUT/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
    print(name, 'exit=', result.returncode, flush=True)
    if required and result.returncode: raise RuntimeError(name + ' failed')

artifacts = list(engines.values()) + ['models/yolov8m.onnx','data/reference/input.bin','data/images/test.jpg','data/splits/person_temporal_v1/manifest.csv','cache/int8-baseline-v1/yolov8m_int8.cache','cache/int8-baseline-v1/yolov8m_int8.cache.meta','build/benchmark_yolo_cuda','build/yolo_infer_raw','build/libyolo_calibrator.a','build/libcuda_yolo.a','build/libtrt_runner.a']
(OUT/'artifacts.json').write_text(json.dumps({p: {'bytes': (ROOT/p).stat().st_size, 'sha256': hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in artifacts},indent=2)+'\n')
run('power_mode', ['/usr/sbin/nvpmodel','-q'], False)
run('clocks', ['/usr/bin/jetson_clocks','--show'], False)
run('system', ['uname','-a'])
run('processes', ['ps','-eo','pid,comm,pcpu,pmem'], False)
with (OUT/'tegrastats.log').open('x') as telemetry:
    monitor = subprocess.Popen(['/usr/bin/tegrastats','--interval','1000'], stdout=telemetry, stderr=subprocess.STDOUT)
    try:
        for round_id, order in enumerate([['fp16','int8_old','int8_new'], ['int8_new','int8_old','fp16'], ['int8_old','fp16','int8_new']],1):
            for label in order:
                run(f'benchmark_r{round_id}_{label}', ['build/benchmark_yolo_cuda',engines[label],'data/images/test.jpg','graph'])
        for label, engine in engines.items():
            output = str(OUT / (label + '_output.bin'))
            run('infer_'+label, ['build/yolo_infer_raw',engine,'data/reference/input.bin',output])
            run('postprocess_'+label, ['build/yolo_postprocess_raw',output])
    finally:
        monitor.terminate()
        try: monitor.wait(timeout=5)
        except subprocess.TimeoutExpired: monitor.kill();monitor.wait()
print('All benchmark/inference commands finished', flush=True)
