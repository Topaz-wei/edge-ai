#!/usr/bin/env python3
"""Evaluate reviewed person labels using existing business preprocessing and raw TRT inference."""
import argparse,csv,json,os,subprocess,shutil
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/tensorrt-yolo-ultralytics')
import cv2,numpy as np,torch,ultralytics
from ultralytics.utils.metrics import box_iou,ap_per_class
from ultralytics.engine.validator import BaseValidator
from prepare_yolo_reference import ROOT,prepare_reference,sha256
from postprocess_yolo_output import load_output,postprocess


def match(gt,boxes):
    gt=torch.tensor(np.asarray(gt).reshape(-1,4),dtype=torch.float32)
    boxes=torch.tensor(np.asarray(boxes).reshape(-1,4),dtype=torch.float32)
    return BaseValidator.match_predictions(SimpleNamespace(iouv=torch.linspace(.5,.95,10)),
        torch.zeros(len(boxes)),torch.zeros(len(gt)),box_iou(gt,boxes)).numpy()


def evaluate(records,plot_dir=None):
    tp=[];scores=[];targets=[];TP=FP=FN=0;per_image=[]
    for r in records:
        boxes=np.array([d['xyxy'] for d in r['predictions']]).reshape(-1,4)
        conf=np.array([d['confidence'] for d in r['predictions']])
        gt=r['gt'];correct=match(gt,boxes)
        tp.append(correct);scores.append(conf);targets.extend([0]*len(gt))
        # Rematch after confidence filtering: low-score boxes cannot consume a GT.
        selected=conf>=.7;fixed=match(gt,boxes[selected]);n=int(fixed[:,0].sum())
        fp=int(selected.sum())-n;fn=len(gt)-n;TP+=n;FP+=fp;FN+=fn
        per_image.append(dict(sample_id=r['sample_id'],gt=len(gt),predictions_at_07=int(selected.sum()),tp=n,fp=fp,fn=fn))
    tp=np.concatenate(tp);scores=np.concatenate(scores)
    result=ap_per_class(tp,scores,np.zeros(len(scores)),np.array(targets),plot=plot_dir is not None,
                        save_dir=plot_dir or Path(),names={0:'person'})
    return dict(images=len(records),ground_truth_boxes=len(targets),precision_07=TP/max(TP+FP,1),recall_07=TP/max(TP+FN,1),
        tp=TP,fp=FP,fn=FN,map50=float(result[5][:,0].mean()),map50_95=float(result[5].mean()),
        ap_by_iou=result[5].tolist(),precision_at_best_f1=float(result[2].mean()),recall_at_best_f1=float(result[3].mean())),per_image


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--engine',action='append',required=True,help='name=path')
    ap.add_argument('--batch-runner',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Output exists; choose a new experiment directory')
    assert ultralytics.__version__=='8.3.173'
    pack=ROOT/'data/annotations/validation-v1'
    with (pack/'review.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==115 and all(r['status']=='reviewed' for r in rows)
    initial={e['sample_id']:e for e in json.loads((pack/'proposals.json').read_text())['entries']}
    engines=[]
    for spec in args.engine:
        name,path=spec.split('=',1);path=Path(path).resolve()
        assert name and Path(name).name==name and name not in ('.','..') and path.is_file()
        engines.append((name,path))
    assert len(set(n for n,p in engines))==len(engines)
    samples=[]
    for r in rows:
        image=pack/r['image'];label=pack/r['label'];sample=r['sample_id']
        assert sha256(image)==initial[sample]['source_sha256'],sample
        gt=[]
        for line in label.read_text().splitlines():
            if not line.strip():continue
            v=list(map(float,line.split()));assert len(v)==5 and v[0]==0 and np.isfinite(v).all()
            _,cx,cy,w,h=v;assert 0<w<=1 and 0<h<=1
            b=[(cx-w/2)*360,(cy-h/2)*640,(cx+w/2)*360,(cy+h/2)*640]
            assert b[0]>=-1e-4 and b[1]>=-1e-4 and b[2]<=360.0001 and b[3]<=640.0001
            gt.append(b)
        samples.append(dict(sample_id=sample,image=str(image),gt=gt,image_sha256=sha256(image),label_sha256=sha256(label)))
    args.output.mkdir(parents=True)
    shutil.copytree(pack/'labels',args.output/'labels_snapshot')
    shutil.copyfile(pack/'review.csv',args.output/'review_snapshot.csv')
    metadata=dict(ultralytics=ultralytics.__version__,script_sha256=sha256(__file__),
        preprocessing_sha256=sha256(ROOT/'scripts/prepare_yolo_reference.py'),postprocessing_sha256=sha256(ROOT/'scripts/postprocess_yolo_output.py'),
        infer_binary_sha256=sha256(args.batch_runner),batch_runner=str(args.batch_runner.resolve()),
        settings=dict(ap_confidence=.001,nms_iou=.7,max_detections=300,precision_recall_confidence=.7,precision_recall_iou=.5,
                      ap_iou='0.50:0.05:0.95',classes='person only; existing argmax-over-80 then person filter',preprocessing='existing business masks and 640 letterbox'),
        samples=samples,engines={n:dict(path=str(p),sha256=sha256(p)) for n,p in engines})
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    inputs=args.output/'inputs';inputs.mkdir()
    for s in samples:
        im=cv2.imread(s['image'],cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION);assert im.shape==(640,360,3)
        tensor=prepare_reference(im,'business')[3]
        tensor.astype('<f4').tofile(str(inputs/(s['sample_id'].replace('/','_')+'.bin')))
    summary={}
    for name,engine in engines:
        out=args.output/name;out.mkdir();rawdir=out/'raw';rawdir.mkdir();logs=out/'infer_logs';logs.mkdir()
        manifest=out/'batch_inputs.txt'
        manifest.write_text(''.join(str((inputs/(s['sample_id'].replace('/','_')+'.bin')).resolve())+'\n'+str((rawdir/(s['sample_id'].replace('/','_')+'.bin')).resolve())+'\n' for s in samples))
        with (out/'inference.log').open('w') as log:
            subprocess.run([str(args.batch_runner.resolve()),str(engine),str(manifest.resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
        records=[]
        for i,s in enumerate(samples,1):
            stem=s['sample_id'].replace('/','_');raw=rawdir/(stem+'.bin')
            det=postprocess(load_output(raw),.001,.7)[:300]
            records.append(dict(sample_id=s['sample_id'],gt=s['gt'],predictions=[dict(confidence=d['confidence'],xyxy=d['box_business'].tolist()) for d in det]))
            print('%s %d/%d %s detections=%d'%(name,i,len(samples),s['sample_id'],len(det)),flush=True)
        (out/'predictions.json').write_text(json.dumps(records,indent=2)+'\n')
        metrics,per_image=evaluate(records,out)
        metrics['by_group']={g:evaluate([r for r in records if r['sample_id'].startswith(g+'/')])[0] for g in sorted({r['sample_id'].split('/')[0] for r in records})}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        with (out/'per_image.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(per_image[0]));w.writeheader();w.writerows(per_image)
        summary[name]=metrics
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(name,json.dumps(metrics),flush=True)
    print('DONE',args.output,flush=True)

if __name__=='__main__':main()
