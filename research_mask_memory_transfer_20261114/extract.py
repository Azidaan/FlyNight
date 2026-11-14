from common import *
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'design.json');manifest=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for r in p['records']:
 video=ROOT/f'data/bdd100k/videos/val/{r["video"]}.mov';assert video.is_file();before=sha(video);cap=cv2.VideoCapture(str(video));assert cap.isOpened();folder=OUT/'frames'/r['id'];folder.mkdir(parents=True,exist_ok=False);start=r['frame'];queries=[start+x for x in p['query_offsets']];records=[];sheet=Image.new('RGB',(1600,440),'#20232a');draw=ImageDraw.Draw(sheet)
 crop=r['crop'];cx=(crop[0]+crop[2])//2;cy=(crop[1]+crop[3])//2;side=max(180,2*(crop[2]-crop[0]));x1=max(0,min(1280-side,cx-side//2));y1=max(0,min(720-side,cy-side//2));detail=[x1,y1,x1+side,y1+side]
 for frame in range(start+25):
  ok,im=cap.read();assert ok,(r['id'],frame)
  if frame<start:continue
  if frame==start:assert np.array_equal(im,cv2.imread(str(ROOT/r['image']))),r['id']
  jpg=folder/f'{frame-start:05d}.jpg';assert cv2.imwrite(str(jpg),im,[cv2.IMWRITE_JPEG_QUALITY,100]);records.append(dict(frame=frame,path=rel(jpg),sha256=sha(jpg)))
  if frame in [start]+queries:
   png=OUT/f'{r["video"]}_{frame}.png';assert cv2.imwrite(str(png),im);j=([start]+queries).index(frame);patch=im[y1:y1+side,x1:x1+side];sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(patch,(400,400)),cv2.COLOR_BGR2RGB)),(j*400,30));draw.text((j*400+5,5),str(frame)+(' начало' if frame==start else ''),font=font,fill='white')
 cap.release();assert sha(video)==before;sheet.save(OUT/(r['id']+'_before_predictions.png'));manifest.append(dict(id=r['id'],video=rel(video),video_sha256=before,detail=detail,frames=records,reference_png_exact=True))
save('frames.json',dict(records=manifest));print('THREE_SEQUENCES_VERIFIED',flush=True)
