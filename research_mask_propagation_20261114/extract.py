from common import *
import cv2,numpy as np,time
from PIL import Image,ImageDraw,ImageFont
selection=dict(registered=now(),video='data/bdd100k/videos/val/b1ca2e5d-84cf9134.mov',start=1159,end=1207,query_frames=[1165,1171,1183,1195,1207],reason='First familiar query plus four fixed12-frame steps until the end of the same1209-frame video. Selection precedes propagation predictions. Keep absence/occlusion cases; do not replace poor results.',decode='Sequential from frame0, never seek. Compare decoded1159 and1165 to archived PNG pixels before using.')
save('selection.json',selection);video=ROOT/selection['video'];h=sha(video);cap=cv2.VideoCapture(str(video));assert cap.isOpened();frames=OUT/'frames';frames.mkdir(exist_ok=False);records=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);sheet=Image.new('RGB',(1280,3*390),'#20232a');draw=ImageDraw.Draw(sheet);i=0
for index in range(selection['end']+1):
 ok,im=cap.read();assert ok,index
 if index<selection['start']:continue
 if index in [1159,1165]:
  old=cv2.imread(str(ROOT/f'research_visible_part_2179_20261114/frames/{index}.png'));assert np.array_equal(old,im),(index,'Sequential frame differs from archived pixel input')
 path=frames/f'{index-selection["start"]:05d}.jpg';assert cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,100]);decoded=cv2.imread(str(path));records.append(dict(frame=index,path=rel(path),sha256=sha(path),jpeg_mean_abs_error=float(np.abs(im.astype(float)-decoded).mean())))
 if index in [1159]+selection['query_frames']:
  png=OUT/f'frame_{index}.png';assert cv2.imwrite(str(png),im);sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im,(640,360)),cv2.COLOR_BGR2RGB)),((i%2)*640,(i//2)*390+30));draw.text(((i%2)*640+8,(i//2)*390+4),str(index),fill='white',font=font);i+=1
cap.release();assert sha(video)==h;sheet.save(OUT/'contact_before_predictions.png')
detail=Image.new('RGB',(1590,1120),'#20232a');draw=ImageDraw.Draw(detail)
for j,index in enumerate([1159]+selection['query_frames']):
 im=cv2.imread(str(OUT/f'frame_{index}.png'));detail.paste(Image.fromarray(cv2.cvtColor(im[100:630,750:1280],cv2.COLOR_BGR2RGB)),((j%3)*530,(j//3)*560+30));draw.text(((j%3)*530+8,(j//3)*560+4),str(index)+'  x750,y100',fill='white',font=font)
detail.save(OUT/'detail_before_predictions.png');save('frames.json',dict(video_sha256=h,records=records,decode_verified_against=[1159,1165]));print('SEQUENTIAL_FRAMES_VERIFIED',len(records),flush=True)
