from common import *
import cv2, numpy as np, shutil, subprocess
OUT.mkdir(exist_ok=True);(OUT/'frames').mkdir(exist_ok=True)
for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/obsidian_experiment_topics.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:
    (OUT/'before').mkdir(exist_ok=True); shutil.copyfile(ROOT/n,OUT/'before'/Path(n).name)
video=ROOT/'data/bdd100k/videos/val/b1ca2e5d-84cf9134.mov';cap=cv2.VideoCapture(str(video)); assert cap.isOpened();wanted=[1153,1158,1159,1160,1164,1165];receipts=[]
rt=read(PREV/'runtime.json')
for frame in range(max(wanted)+1):
    ok,img=cap.read();assert ok
    if frame not in wanted:continue
    cv2.imwrite(str(OUT/'frames'/f'{frame}.png'),img)
    # Numerical inspection figure only; full source pixels are preserved above.
    crop=img[220:430,650:1280].copy();cv2.imwrite(str(OUT/'frames'/f'{frame}_detail.png'),cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
    receipts.append(dict(frame=frame,pixel_sha256=hashlib.sha256(img.tobytes()).hexdigest(),file_sha256=sha(OUT/'frames'/f'{frame}.png')))
cap.release();save('inspection_inputs.json',dict(video_sha256=sha(video),frames=receipts,head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
print('FRAMES_READY')
