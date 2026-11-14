from common import *
import cv2,numpy as np
p=read(OUT/'design.json');video=ROOT/p['video'];h=sha(video);cap=cv2.VideoCapture(str(video));assert cap.isOpened();folder=OUT/'frames';folder.mkdir(exist_ok=False);prior=read(ROOT/'research_mask_propagation_20261114/frames.json')['records'];records=[]
for fr in range(1208):
 ok,im=cap.read();assert ok,fr
 if fr<1141:continue
 dest=folder/f'{fr-1141:05d}.jpg';assert cv2.imwrite(str(dest),im,[cv2.IMWRITE_JPEG_QUALITY,100]);digest=sha(dest)
 if fr>=1159:assert digest==prior[fr-1159]['sha256'],fr
 if fr==1141:
  cv2.imwrite(str(OUT/'reference_1141.png'),im);cv2.imwrite(str(OUT/'reference_1141_detail.png'),cv2.resize(im[200:450,900:1280],(1140,750)))
 records.append(dict(frame=fr,path=rel(dest),sha256=digest))
cap.release();assert sha(video)==h;save('frames.json',dict(records=records,video_sha256=h,overlap_with_primary_identical_jpegs=49,detail_coordinates='fullframe=(900+x/3,200+y/3)'));print('EARLY_FRAMES_VERIFIED')
