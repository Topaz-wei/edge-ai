"""Recompute original reviewed-set metrics from frozen before/after raw outputs (CPU)."""
import sys, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'tensorrt-yolo'
sys.path.insert(0,str(root/'scripts'))
from validate_reviewed_engines import evaluate, postprocess, load_output
samples=json.loads((root/'validation/compare-v2/manifest.json').read_text())['samples']
by_stem={s['sample_id'].replace('/','_'):s for s in samples}
inputs=sorted((root/'validation/compare-v2/inputs').rglob('*.bin'))
report={}
for version in ['baseline','after']:
    report[version]={}
    for engine in ['fp16','mixed_int8']:
        records=[]
        for i,p in enumerate(inputs):
            s=by_stem[p.stem]
            raw=root/('cache/shared-runtime-'+version)/(engine+'-batch')/('{:04d}.bin'.format(i))
            det=postprocess(load_output(raw),.001,.7)[:300]
            records.append(dict(sample_id=s['sample_id'],gt=s['gt'],predictions=[dict(confidence=d['confidence'],xyxy=d['box_business'].tolist()) for d in det]))
        report[version][engine]=evaluate(records)[0]
assert report['baseline']==report['after'], 'Metrics changed'
(root/'cache/shared-runtime-after/metrics-comparison.json').write_text(json.dumps(report,indent=2))
for k,v in report['after'].items():print(k,{x:v[x] for x in ['images','precision_07','recall_07','map50','map50_95']})
