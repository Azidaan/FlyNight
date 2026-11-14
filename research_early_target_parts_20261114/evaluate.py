from common import *
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');old=ROOT/'research_mask_propagation_20261114';truth=np.load(old/'evaluation_truth.npz');masks=np.load(OUT/'predictions.npz');scores=np.load(OUT/'scores.npz');rows=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);neg=cv2.fillPoly(np.zeros((720,1280),'uint8'),[np.rint(np.array([[867,185],[1135,181],[1135,266],[864,267]])/3+[900,200]).astype('int32')],1).astype(bool)
for key in masks.files:
 src,mode,f=key.split('__');fr=int(f);m=masks[key];assert np.array_equal(m,scores[key]>0);t=truth[f];pos=t==1;n=(t>=0)&~pos;tp=int((m&pos).sum());fp=int((m&n).sum());fn=int((~m&pos).sum());rows.append(dict(source=src,mode=mode,frame=fr,tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn),recall=tp/int(pos.sum()),fpr=fp/int(n.sum()),predicted_pixels=int(m.sum()),extra_neighbor_pixels=int((m&neg).sum()) if fr==1171 else None,extra_neighbor_fpr=float((m&neg).sum()/neg.sum()) if fr==1171 else None))
means={s+'__'+mode:float(np.mean([v['iou'] for v in rows if v['source']==s and v['mode']==mode])) for s,mode in sorted({(r['source'],r['mode']) for r in rows})};hard=next(r for r in rows if r['source']=='both' and r['mode']=='first_only' and r['frame']==1171);recover=hard['recall']>=.5 and hard['fpr']<=.1;gain=means['both__first_only']>max(means['late__first_only'],means['early__first_only']);save('result.json',dict(records=rows,means=means,recovery_passed=bool(recover),increment_passed=bool(gain),scientific_passed=bool(recover and gain),scope='Manual pastmask and GT-assisted earlier reference selection. Same historical clip only. Additional neighborcore was frozen before this test but originated as a post-result diagnostic of previous run.'))
for fr in p['queries']:
 im=cv2.cvtColor(cv2.imread(str(old/f'frame_{fr}.png')),cv2.COLOR_BGR2RGB);sheet=Image.new('RGB',(4*450,393),'#20232a');draw=ImageDraw.Draw(sheet)
 for j,(title,source) in enumerate([('Кадр, видимые участки цели',None),('Только поздний образец','late'),('Только ранний образец','early'),('Оба образца','both')]):
  image=im.copy();mask=masks[source+'__first_only__'+str(fr)] if source else None
  if mask is not None:image[mask]=(image[mask]*.4+np.array([30,235,50])*.6).astype('uint8')
  panel=cv2.resize(image[200:450,900:1280],(450,296));known=cv2.resize((truth[str(fr)][200:450,900:1280]==1).astype('uint8'),(450,296),interpolation=cv2.INTER_NEAREST);contours,_=cv2.findContours(known,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(255,205,10),1);sheet.paste(Image.fromarray(panel),(j*450,39));draw.text((j*450+6,9),title,font=font,fill='white')
 draw.text((6,350),f'Кадр {fr}. Зелёное — прогноз; жёлтый контур — известная видимая цель. Память хранит только заданные прошлые образцы.',font=font,fill='white');sheet.save(OUT/f'comparison_{fr}.png')
print(json.dumps(dict(means=means,hard=[r for r in rows if r['frame']==1171],passed=recover and gain)),flush=True)
