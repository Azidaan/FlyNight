from common import *
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');manifest=read(OUT/'frames.json');truth=np.load(OUT/'query_cores.npz');pred=np.load(OUT/'predictions.npz');scores=np.load(OUT/'scores.npz');views=read(OUT/'annotation_views.json')['records'];rows=[];means={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
for c in manifest['records']:
 cid=c['id'];vid=Path(c['video']).stem;ref=c['frames'][0]['frame'];sheet=Image.new('RGB',(1500,1020),'#20232a');draw=ImageDraw.Draw(sheet)
 for i,off in enumerate([6,12,24]):
  frame=ref+off;t=truth[vid+'__'+str(frame)];pos=t==1;neg=(t>=0)&~pos;v=next(w for w in views if w['video']==vid and w['frame']==frame);x,y,rr,bb=v['annotation_only_crop'];im=cv2.cvtColor(cv2.imread(str(OUT/f'{vid}_{frame}.png')),cv2.COLOR_BGR2RGB)
  for j,(variant,name) in enumerate([(None,'Кадр и ручная область'),('ordinary','Обычная память'),('first_only','Только первый образец'),('flow','Перенос движением'),('static','Неподвижная маска')]):
   picture=im.copy()
   if variant:
    key=cid+'__'+variant+'__'+str(frame);mask=pred[key];tp=int((mask&pos).sum());fp=int((mask&neg).sum());fn=int((~mask&pos).sum());n=int(neg.sum());positive=int(pos.sum());row=dict(case=cid,video=vid,frame=frame,variant=variant,target_pixels=positive,negative_pixels=n,tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn) if positive and tp+fp+fn else None,recall=tp/positive if positive else None,fpr=fp/n if n else None,neighbor_fpr=float((mask&(t==2)).sum()/(t==2).sum()) if (t==2).any() else None,predicted_pixels=int(mask.sum()));rows.append(row)
    if variant in ['ordinary','first_only']:assert np.array_equal(mask,scores[key]>0)
    picture[mask]=(picture[mask]*.4+np.array([30,235,50])*.6).astype('uint8')
   panel=cv2.resize(picture[y:bb,x:rr],(300,300));source=cv2.resize(pos[y:bb,x:rr].astype('uint8'),(300,300),interpolation=cv2.INTER_NEAREST);contours,_=cv2.findContours(source,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(255,205,10),1)
   sheet.paste(Image.fromarray(panel),(j*300,i*335+30));draw.text((j*300+3,i*335+6),str(frame)+' · '+name,font=font,fill='white')
 sheet.save(OUT/(cid+'_comparison.png'));local=[z for z in rows if z['case']==cid];means[cid]={v:float(np.mean([z['iou'] for z in local if z['variant']==v and z['iou'] is not None])) if any(z['variant']==v and z['iou'] is not None for z in local) else None for v in ['ordinary','first_only','flow','static']}
valid=[z for z in rows if z['variant']=='first_only' and z['target_pixels']];quality=bool(valid and all(z['recall']>=.5 and z['fpr']<=.1 for z in valid));atleast2=all(sum(z['case']==cid for z in valid)>=2 for cid in means);available=[v for v in means.values() if v['first_only'] is not None];wins=sum(v['first_only']>v['ordinary'] for v in available);increment=bool(wins>=2 and np.mean([v['first_only'] for v in available])>np.mean([v['ordinary'] for v in available])) if atleast2 else None
save('result.json',dict(records=rows,means_by_video=means,quality_passed=quality,added_information_passed=increment,all_video_readiness=atleast2,observed_wins_on_evaluable_videos=wins,n_evaluable_videos=len(available),scientific_passed=bool(quality and increment is True),scope='Six visible queries and three edge queries without confident target visibility. No positivity or zero IoU assigned to missing targets. All nine outputs retained; unknown pixels are not negatives.'))
print(json.dumps(dict(means=means,quality=quality,increment=increment,wins=wins,edge=[z for z in rows if z['variant'] in ['ordinary','first_only'] and not z['target_pixels']])),flush=True)
