from common import *
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');output=np.load(OUT/'predictions.npz');prompts=np.load(OUT/'prompts.npz');rows=[];means={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for c in p['cases']:
 cid=c['id'];truth=np.load(ROOT/c['truth']);x,y,x2,y2=c['crop'];sheet=Image.new('RGB',(400*4,285*len(c['queries'])),'#21252a');d=ImageDraw.Draw(sheet);qs=[]
 for j,fr in enumerate(c['queries']):
  im=cv2.imread(str(ROOT/next(i['path'] for i in c['frames'] if i['frame']==fr)));t=truth[c['truth_prefix']+str(fr)];k=t>=0;pos=t==1;neg=k&~pos
  for i,(src,title) in enumerate([('', 'Исходный кадр'),('manual','Ручная прошлая маска'),('yolo','Прошлая маска YOLO'),('box','Прошлая рамка в SAM 2')]):
   img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
   if src:
    m=output[f'{cid}__{src}__{fr}'];tp=int((m&pos).sum());fp=int((m&neg).sum());fn=int((~m&pos).sum());rec=dict(case=cid,frame=fr,source=src,tp=tp,fp=fp,fn=fn,target_pixels=int(pos.sum()),negative_pixels=int(neg.sum()),predicted_pixels=int(m.sum()),iou=tp/(tp+fp+fn) if pos.any() else None,recall=tp/int(pos.sum()) if pos.any() else None,fpr=fp/int(neg.sum()) if neg.any() else None);rows.append(rec);qs.append(rec);img[m]=(img[m]*.4+np.array([30,235,50])*.6).astype('uint8')
   patch=cv2.resize(img[y:y2,x:x2],(400,240));core=cv2.resize(pos[y:y2,x:x2].astype('uint8'),(400,240),interpolation=cv2.INTER_NEAREST);contours,_=cv2.findContours(core,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(patch,contours,-1,(255,205,10),1);sheet.paste(Image.fromarray(patch),(i*400,j*285+29));d.text((i*400+4,j*285+5),f'{fr} · {title}',font=font,fill='white')
 sheet.save(OUT/(cid+'_comparison.jpg'),quality=90);means[cid]={s:float(np.mean(v)) if (v:=[r['iou'] for r in qs if r['source']==s and r['iou'] is not None]) else None for s in ['manual','yolo','box']}
hard=next(r for r in rows if r['case']=='old2179' and r['frame']==1171 and r['source']=='yolo');eligible=[r for r in rows if r['case']!='old2179' and r['source']=='yolo' and r['target_pixels']];assert len(eligible)==6
passed=lambda r:r['recall']>=.5 and r['fpr']<=.1
save('result.json',dict(records=rows,means=means,hard_passed=passed(hard),external_passed=all(passed(r) for r in eligible),scientific_passed=passed(hard) and all(passed(r) for r in eligible),scope='Automatic initial mask *given* retrospective target boxes; not automatic identity discovery. Only known visible cores evaluated; three queries without confirmed positive left undefined.'))
