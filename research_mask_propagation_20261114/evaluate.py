from common import *
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');assert read(OUT/'execution_result.json')['passed'];pred=np.load(OUT/'predictions.npz');scores=np.load(OUT/'scores.npz');truths=dict(np.load(OUT/'manual_query_cores_v2.npz'));z=np.load(ROOT/p['familiar_query_labels']);t=np.full((720,1280),-1,np.int8)
for name,cls in [('background',0),('target',1),('neighbor',2),('occluder',3)]:
 m=z['1165_'+name].astype(bool);assert (t[m]<0).all(),name;t[m]=cls
region=np.zeros(t.shape,bool);region[200:450,900:1280]=True;t[~region]=-1;truths['1165']=t
records=[];initial=np.load(ROOT/p['initial_mask'])['1159_target'].astype(bool)
for frame in p['query_frames']:
 truth=truths[str(frame)];valid=truth>=0;target=truth==1;negative=(truth>=0)&~target
 for variant in ['sam','flow','static']:
  key=variant+'__'+str(frame);mask=pred[key];assert mask.shape==truth.shape and mask.dtype==np.bool_
  if variant=='sam':assert np.array_equal(mask,scores[key]>0)
  if variant=='static':assert np.array_equal(mask,initial)
  tp=int((mask&target).sum());fp=int((mask&negative).sum());fn=int((~mask&target).sum());row=dict(frame=frame,variant=variant,tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn) if tp+fp+fn else None,recall=tp/int(target.sum()) if target.any() else None,fpr=fp/int(negative.sum()) if negative.any() else None,target_pixels=int(target.sum()),negative_pixels=int(negative.sum()),predicted_pixels_whole_frame=int(mask.sum()),inside_initial_target_coverage=float((target&initial).sum()/target.sum()) if target.any() else None)
  row['fpr_by_class']={str(cls):float((mask&(truth==cls)).sum()/(truth==cls).sum()) if (truth==cls).any() else None for cls in [0,2,3]};records.append(row)
with (OUT/'evaluation_truth.npz').open('xb') as f:np.savez_compressed(f,**truths)
quality=all(r['recall'] is not None and r['recall']>=.5 and r['fpr'] is not None and r['fpr']<=.1 for r in records if r['variant']=='sam');wins=[]
for frame in p['query_frames'][1:]:
 rows={r['variant']:r for r in records if r['frame']==frame};wins.append(bool(rows['sam']['target_pixels']>0 and rows['sam']['iou']>max(rows['static']['iou'],rows['flow']['iou'])))
save('result.json',dict(records=records,quality_passed=quality,added_information_passed=sum(wins)>=3,later_frames_wins_over_both=sum(wins),scientific_passed=bool(quality and sum(wins)>=3),scope='Knownpixel maskIoU and false positives on partial visiblecores. One knownvideo, laterqueries; noIDmetrics, nofullmaskgroundtruth.',source_unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items())))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
for frame in p['query_frames']:
 im=cv2.imread(str(OUT/f'frame_{frame}.png'));truth=truths[str(frame)];sheet=Image.new('RGB',(2120,595),'#20232a');draw=ImageDraw.Draw(sheet)
 for j,(name,variant) in enumerate([('Текущий кадр',None),('SAM 2 по прошлой маске','sam'),('Перенос движением','flow'),('Неподвижная маска','static')]):
  picture=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  if variant:
   mask=pred[variant+'__'+str(frame)];picture[mask]=(picture[mask]*.4+[20* .6,235*.6,50*.6]).astype('uint8')
  crop=picture[100:630,750:1280];sheet.paste(Image.fromarray(crop),(530*j,35));draw.text((530*j+6,7),name,font=font,fill='white')
 draw.text((8,572),f'Кадр {frame}. Зелёное — прогноз цели. Только первая маска дана вручную; текущих подсказок нет.',font=font,fill='white');sheet.save(OUT/f'comparison_{frame}.png')
print(json.dumps(records,ensure_ascii=False),flush=True)
