"""GPU integration test: existing engines, generated lossless videos, no camera access."""
import argparse
import cv2, subprocess, json, csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]; y=root/'tensorrt-yolo'
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--build',type=Path,default=root/'build-shared/tensorrt-yolo')
ap.add_argument('--baseline-build',type=Path,default=y/'build-baseline')
args=ap.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
build=args.build.resolve();baseline=args.baseline_build.resolve()
im=cv2.imread(str(y/'data/images/test.jpg'))
for name,frame in [('business',im),('camera',cv2.transpose(im))]:
    w=cv2.VideoWriter(str(out/(name+'.avi')),cv2.VideoWriter_fourcc(*'FFV1'),30,(frame.shape[1],frame.shape[0]))
    assert w.isOpened()
    for _ in range(6): w.write(frame)
    w.release()
engine=y/'debug/trial007_sub_model22/yolov8m_int8.plan'
commands=[]
def run(args,tag,cwd=root):
    with (out/(tag+'.log')).open('w') as f:
        p=subprocess.run([str(x) for x in args],cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
    commands.append({'argv':[str(x) for x in args],'returncode':p.returncode})
    assert p.returncode==0,tag
for stage in ['business','camera']:
    for graph in [True,False]:
        for version,binary in [('old',baseline/'yolo_video_cuda'),('new',build/'yolo_video_cuda')]:
            tag='{}-{}-{}'.format(version,stage,graph)
            args=[binary,engine,out/(stage+'.avi'),stage,'--no-display','--max-frames','6','--timing-csv',out/(tag+'.csv'),'--output',out/(tag+'.mp4')]
            if not graph:args+=['--no-graph']
            run(args,tag)
            rows=list(csv.DictReader((out/(tag+'.csv')).open()));assert len(rows)==6
    config=out/(stage+'.yaml')
    config.write_text('''%YAML:1.0
---
schema_version: 1
source: "STAGE.avi"
display: 0
max_frames: 6
output: "STAGE-config.mp4"
timing_csv: "STAGE-config.csv"
models:
  - { id: first, type: yolo, engine: "ENGINE", input_stage: STAGE, graph: 1 }
  - { id: second, type: yolo, engine: "SECOND_ENGINE", input_stage: STAGE, graph: 0 }
'''.replace('STAGE',stage).replace('SECOND_ENGINE',str(y/'engines/yolov8m_fp16.plan' if stage=='business' else engine)).replace('ENGINE',str(engine)))
    run([build/'edge_video',config],stage+'-config',Path('/tmp'))
    rows=list(csv.DictReader((out/(stage+'-config.csv')).open()));assert len(rows)==6
    assert all(float(r['first_wall_ms'])>0 and float(r['second_wall_ms'])>0 for r in rows)
    video=cv2.VideoCapture(str(out/(stage+'-config.mp4')));ok,frame=video.read();assert ok
    assert frame.shape[:2]==(im.shape[:2] if stage=='business' else im.shape[:2][::-1]);video.release()
(out/'commands.json').write_text(json.dumps(commands,indent=2))
print('PASS: old/new camera/business Graph on/off, 6-frame recording/CSV; config two instances from different cwd')
