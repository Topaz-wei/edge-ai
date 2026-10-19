#!/usr/bin/env python3
"""Generate unreviewed person proposals and an offline review pack; never ground truth."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import numpy as np
import cv2
from prepare_yolo_reference import ROOT,prepare_reference,sha256
from postprocess_yolo_output import postprocess


def yolo_line(box,width,height):
    x1,y1,x2,y2=map(float,box)
    if not all(np.isfinite([x1,y1,x2,y2])) or not (0<=x1<x2<=width and 0<=y1<y2<=height):
        raise ValueError('Invalid original-image box: '+str(box))
    return '0 %.8f %.8f %.8f %.8f'%((x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height)


def save_image(path,image):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image):raise IOError('Cannot save '+str(path))


def preview(original,masked,proposals,sample):
    panes=[]
    for title,source in [('ORIGINAL',original),('MODEL INPUT (MASKED)',masked)]:
        canvas=np.full((690,360,3),245,np.uint8);canvas[50:]=source
        cv2.putText(canvas,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(canvas,sample+' PENDING',(8,40),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,160),1,cv2.LINE_AA)
        for i,p in enumerate(proposals):
            x1,y1,x2,y2=map(lambda v:int(round(v)),p['xyxy'])
            x1=min(359,x1);x2=min(359,x2);y1=min(639,y1);y2=min(639,y2)
            color=(0,140,0) if p['confidence']>=.7 else (0,120,230)
            cv2.rectangle(canvas,(x1,y1+50),(x2,y2+50),color,2)
            cv2.putText(canvas,'#%d person %.3f'%(i+1,p['confidence']),(max(0,x1),max(63,y1+46)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
        if not proposals:
            cv2.putText(canvas,'NO PROPOSALS - CHECK MISSES',(8,75),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,220),1,cv2.LINE_AA)
        panes.append(canvas)
    return np.concatenate(panes,axis=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,default=ROOT/'data/splits/person_temporal_v1/validation')
    parser.add_argument('--weights',type=Path,default=ROOT/'models/yolov8m.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confidence',type=float,default=.25)
    parser.add_argument('--iou',type=float,default=.7)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists; refusing to overwrite labels or review decisions')
    if not 0<args.confidence<1 or not 0<args.iou<1:parser.error('Thresholds must be in (0,1)')
    if not args.weights.is_file():parser.error('Local model required; downloads disabled')
    files=sorted(args.images.rglob('*.jpg'),key=lambda p:(p.parent.name,int(p.stem)))
    if not files:parser.error('No validation JPG images')
    # Verify this is exactly the existing fixed split, never auto-repartition.
    with (args.images.parent/'manifest.csv').open() as f:
        expected={row['sha256'] for row in csv.DictReader(f) if row['split']=='validation'}
    if len(files)!=len(expected) or {sha256(p) for p in files}!=expected:
        raise ValueError('Validation source does not match split manifest')
    os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/tensorrt-yolo-ultralytics')
    import torch
    import ultralytics
    from ultralytics import YOLO
    from run_pytorch_reference import select_raw_output
    if ultralytics.__version__!='8.3.173':raise RuntimeError('Use ultralytics8.3.173 reference environment')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; GPU access required')
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=YOLO(str(args.weights)).model.to(args.device).float().eval()
    args.output.mkdir(parents=True)
    (args.output/'classes.txt').write_text('person\n')
    entries=[];review=[];groups={}
    def save_progress():
        with (args.output/'review.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['sample_id','status','image','label','preview','proposal_count','min_confidence','notes'])
            w.writeheader();w.writerows(review)
    for i,path in enumerate(files):
        rel=path.relative_to(args.images);sample=rel.with_suffix('').as_posix()
        original=cv2.imread(str(path),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if original is None:raise ValueError('Cannot decode '+str(path))
        _,masked,_,tensor,_,_=prepare_reference(original,'business')
        with torch.inference_mode():
            raw=select_raw_output(model(torch.from_numpy(tensor).to(args.device)))
        detections=postprocess(raw,args.confidence,args.iou)
        proposed=[];discarded=0
        for d in detections:
            box=d['box_business'].tolist()
            if box[2]<=box[0] or box[3]<=box[1]:discarded+=1;continue
            proposed.append({'class_id':0,'confidence':d['confidence'],'xyxy':box,'candidate':d['candidate']})
        image_rel=Path('images')/rel;label_rel=Path('labels')/rel.with_suffix('.txt');preview_rel=Path('previews')/rel.with_suffix('.jpg')
        (args.output/image_rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,args.output/image_rel)
        (args.output/label_rel).parent.mkdir(parents=True,exist_ok=True)
        (args.output/label_rel).write_text(''.join(yolo_line(p['xyxy'],360,640)+'\n' for p in proposed))
        save_image(args.output/preview_rel,preview(original,masked,proposed,sample))
        entry={'sample_id':sample,'source':str(path.resolve()),'source_sha256':sha256(path),'image':str(image_rel),'label':str(label_rel),
               'initial_label_sha256':sha256(args.output/label_rel),'width':360,'height':640,'status':'pending','proposals':proposed,'discarded_degenerate':discarded}
        entries.append(entry)
        review.append(dict(sample_id=sample,status='pending',image=str(image_rel),label=str(label_rel),preview=str(preview_rel),
                           proposal_count=len(proposed),min_confidence=min((p['confidence'] for p in proposed),default=''),notes=''))
        groups.setdefault(rel.parent.name,[]).append(preview_rel)
        save_progress()
        print('%d/%d %s proposals=%d'%(i+1,len(files),sample,len(proposed)),flush=True)
    for group,previews in groups.items():
        for start in range(0,len(previews),16):
            items=previews[start:start+16];sheet=np.full((345*((len(items)+3)//4),1440,3),255,np.uint8)
            for k,p in enumerate(items):
                small=cv2.resize(cv2.imread(str(args.output/p)),(360,345))
                y,x=(k//4)*345,(k%4)*360;sheet[y:y+345,x:x+360]=small
            save_image(args.output/'contact_sheets'/('%s_%02d.jpg'%(group,start//16+1)),sheet)
    metadata={'schema_version':1,'status':'UNREVIEWED_PROPOSALS_NOT_GROUND_TRUTH','created':datetime.datetime.now().astimezone().isoformat(),
              'weights':str(args.weights.resolve()),'weights_sha256':sha256(args.weights),'device':args.device,
              'versions':{'torch':torch.__version__,'ultralytics':ultralytics.__version__,'numpy':np.__version__,'opencv':cv2.__version__},
              'preprocessing':'business image; existing headrest/window masks; fixed640 centered letterbox; RGB FP32 /255; TF32 off',
              'label_coordinates':'original business image, normalized xywh; class0 person; no confidence column',
              'confidence':args.confidence,'nms_iou':args.iou,'script_sha256':sha256(__file__),
              'preprocess_script_sha256':sha256(ROOT/'scripts/prepare_yolo_reference.py'),
              'postprocess_script_sha256':sha256(ROOT/'scripts/postprocess_yolo_output.py'),
              'images':len(entries),'proposals':sum(len(e['proposals']) for e in entries),
              'empty_images':sum(not e['proposals'] for e in entries),'groups':{k:len(v) for k,v in groups.items()},'entries':entries}
    (args.output/'proposals.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.output/'README.md').write_text('''# Validation person 预标注审核包

所有标签均待人工审核，不是ground truth。原始validation图片、划分和校准缓存未修改。

## 审核入口

1. 先打开 contact_sheets/ 的分人物缩略图，逐张检查，不要只检查低分图片。
2. previews/ 下是原分辨率带框对照：左侧原图，右侧模型看到的遮罩图。绿色分数>=0.7，橙色为0.25–0.7；颜色不代表标注正确。
3. 在标注工具中打开 images/ 原图，使用 classes.txt 的person类别及 labels/ 下YOLO预标注；修改并保存 labels/ 对应文件。
4. 审核完一张，在 review.csv 将 status 从 pending 改为 reviewed；notes 可记录补框、删框、改框或待讨论问题。不确定时填 needs_review，不能当成完成。

## 标注规则（全包统一）

- 只标person，class_id=0。标注原始360×640业务图，不标在带框预览或640×640 letterbox图上。
- 删除误检，补上所有原图中能确认的人；不能因为模型漏检、分数低或处于遮罩区域就忽略。右侧遮罩图用于解释模型盲区，不作为删去真实目标的依据。
- 框紧贴原图中可见的人体范围；画面外部分截到边界，不凭空补全不可见身体。遮挡严重、目标身份不清时标 needs_review，先统一规则再完成。
- 每人一个框；包含评估范围内的所有人员，不只保留最高分的人。空标签只代表模型没有提议，必须人工确认确实无人，才能当作负样本。
- YOLO文本每行：0 center_x center_y width height，后四项除以原图宽高，范围[0,1]。标签中没有置信度。
- 标注时不要参照INT8/mixed输出来修改真实框，也不要为了提高指标删掉模型难以检测的目标。

## 文件说明

- images/：原图副本；labels/：供你修改的YOLO标签。
- previews/、contact_sheets/：生成时的静态预览；修改标签后不会自动刷新，以 labels/ 为准。
- review.csv：人工审核状态，初始全部pending。
- proposals.json：不可变的初始提议、置信度、图像/标签哈希和模型来源；不要用它覆盖你的人工标签。
- classes.txt：仅person。尚未生成mAP/Precision/Recall评估结果。

这是辅助标注使用的0.25阈值，和之后业务指标的0.7阈值无关。所有115张都审核确认后，才能将labels作为真实标注进入评估。
''')
    print(json.dumps({k:metadata[k] for k in ('images','proposals','empty_images','groups','status')},indent=2))
if __name__=='__main__':main()
