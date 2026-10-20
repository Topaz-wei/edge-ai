"""Capture/compare immutable YOLO artifacts using existing engines; never builds engines."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO = ROOT / 'tensorrt-yolo'
ENGINES = {'fp16': YOLO/'engines/yolov8m_fp16.plan',
           'mixed_int8': YOLO/'debug/trial007_sub_model22/yolov8m_int8.plan'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def require_equal(a, b):
    if digest(a) != digest(b): raise RuntimeError('Artifact differs: {} != {}'.format(a, b))

def new_directory(path):
    path.mkdir(parents=True, exist_ok=False)

def capture(output, build):
    new_directory(output)
    manifest = {'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'platform': platform.platform(), 'engines': {k:digest(v) for k,v in ENGINES.items()},
                'commands': [], 'artifacts': {}}
    def run(args, log):
        with (output/log).open('w') as f:
            p = subprocess.run([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT)
        manifest['commands'].append({'argv': [str(x) for x in args], 'returncode':p.returncode, 'log':log})
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        if p.returncode: raise RuntimeError('Command failed; see '+str(output/log))
    image = YOLO/'data/images/test.jpg'
    manifest['image_sha256'] = digest(image)
    run([build/'yolo_preprocess', image, output/'input.bin'], 'preprocess.log')
    for name, engine in ENGINES.items():
        run([build/'yolo_infer_raw', engine, output/'input.bin', output/(name+'.bin')], name+'.log')
        for graph in (True, False):
            tag = name + ('-graph' if graph else '-cuda')
            args = [build/'yolo_infer_cuda', engine, image, output/(tag+'.png')]
            if not graph: args.append('--no-graph')
            run(args, tag+'.log')
            # Ignore variable destination path, retain all numeric detection text.
            lines = (output/(tag+'.log')).read_text().splitlines()
            (output/(tag+'.txt')).write_text('\n'.join(x for x in lines if not x.startswith(('Visualization saved to:', '[TensorRT]'))))
        run([build/'benchmark_yolo_cuda', engine, image, 'graph'], name+'-benchmark.log')
        inputs = sorted((YOLO/'validation/compare-v2/inputs').rglob('*.bin'))
        if inputs:
            batch = output/(name+'-batch'); batch.mkdir()
            pairs = []
            for i, path in enumerate(inputs):
                pairs.extend([str(path), str(batch/('{:04d}.bin'.format(i)))])
            (output/(name+'-batch.txt')).write_text('\n'.join(pairs)+'\n')
            run([build/'yolo_infer_batch', engine, output/(name+'-batch.txt')], name+'-batch.log')
            manifest['validation_inputs'] = {str(p.relative_to(YOLO)):digest(p) for p in inputs}
    for p in sorted(output.rglob('*')):
        if p.suffix in ('.bin','.png') or p.name.endswith(('-graph.txt','-cuda.txt')):
            manifest['artifacts'][str(p.relative_to(output))] = digest(p)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

def compare(baseline, output):
    a, b = [json.loads((p/'manifest.json').read_text()) for p in (baseline,output)]
    for key in ('engines','image_sha256','validation_inputs','artifacts'):
        if a.get(key) != b.get(key): raise RuntimeError('Regression mismatch: '+key)
    for rel, expected in a['artifacts'].items():
        if digest(baseline/rel) != expected: raise RuntimeError('Baseline modified: '+rel)
        require_equal(baseline/rel, output/rel)
    print('PASS: {} artifacts byte-identical; engine/input hashes unchanged'.format(len(a['artifacts'])))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('capture','compare'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--build', type=Path, default=YOLO/'build')
    args = ap.parse_args()
    if args.mode == 'capture': capture(args.output.resolve(), args.build.resolve())
    else:
        if not args.baseline: ap.error('compare requires --baseline')
        compare(args.baseline.resolve(), args.output.resolve())
