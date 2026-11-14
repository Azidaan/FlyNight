from common import *
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
protocol=read(OUT/'protocol.json');field=read(OUT/'field_execution.json');truth=np.load(OUT/'query_cores.npz');raw=np.load(OUT/'predictions.npz');scores=np.load(OUT/'scores.npz');fieldmask=np.load(OUT/'field_predictions.npz');fieldscore=np.load(OUT/'field_scores.npz');initial=np.load(OUT/'initial_masks.npz');primary=read(OUT/'result.json');rows=[];checks=[];means={};initrows=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
def measure(mask,t):
 v=t>=0;encoded=(t[v]==1).astype(int)*2+mask[v].astype(int);tn,fp,fn,tp=np.bincount(encoded,minlength=4);return dict(tp=int(tp),fp=int(fp),fn=int(fn),recall=float(tp/(tp+fn)) if tp+fn else None,fpr=float(fp/(fp+tn)) if fp+tn else None,iou=float(tp/(tp+fp+fn)) if tp+fn else None,predicted_pixels=int(mask.sum()))
for row in primary['records']:
 key=row['case']+'__'+row['variant']+'__'+str(row['frame']);vals=measure(raw[key],truth[row['video']+'__'+str(row['frame'])]);assert all(row[k]==v for k,v in vals.items()),key
 if row['variant'] in ['ordinary','first_only']:assert np.array_equal(raw[key],scores[key]>0)
 checks.append(key)
for rec in protocol['design']['records']:
 cid=rec['id'];vid=rec['video'];ref=rec['frame'];l,t,rr,b=field['windows'][cid];sheet=Image.new('RGB',(1500,1010),'#20232a');draw=ImageDraw.Draw(sheet)
 for view in ['context','zoom']:
  for mode in ['ordinary','first_only']:
   for fr in [ref,ref+6,ref+12,ref+24]:
    key=cid+'__'+view+'__'+mode+'__'+str(fr);s=fieldscore[key]>0;mask=fieldmask[key]
    if view=='zoom':restored=np.zeros(mask.shape,bool);restored[t:b,l:rr]=s;assert np.array_equal(restored,mask)
    else:assert np.array_equal(s,mask)
    if fr==ref:
     a=initial[cid];initrows.append(dict(case=cid,view=view,mode=mode,input_pixels=int(a.sum()),output_pixels=int(mask.sum()),iou=float((a&mask).sum()/(a|mask).sum())));continue
    tr=truth[vid+'__'+str(fr)];m=measure(mask,tr);rows.append(dict(case=cid,frame=fr,view=view,mode=mode,target_pixels=int((tr==1).sum()),target_pixels_inside_window=int((tr[t:b,l:rr]==1).sum()),**m))
 for j,off in enumerate([6,12,24]):
  fr=ref+off;tr=truth[vid+'__'+str(fr)];a=next(x['annotation_only_crop'] for x in read(OUT/'annotation_views.json')['records'] if x['video']==vid and x['frame']==fr);x,y,x2,y2=a;im=cv2.cvtColor(cv2.imread(str(OUT/f'{vid}_{fr}.png')),cv2.COLOR_BGR2RGB)
  variants=[('Исходный кадр',None),('Полный, первый образец',raw[cid+'__first_only__'+str(fr)]),('Без внешнего фона',fieldmask[cid+'__context__first_only__'+str(fr)]),('Увеличенное окно',fieldmask[cid+'__zoom__first_only__'+str(fr)]),('Перенос движением',raw[cid+'__flow__'+str(fr)])]
  for i,(title,mask) in enumerate(variants):
   im2=im.copy()
   if mask is not None:im2[mask]=(im2[mask]*.4+np.array([30,235,50])*.6).astype('uint8')
   panel=cv2.resize(im2[y:y2,x:x2],(300,300));m=cv2.resize((tr[y:y2,x:x2]==1).astype('uint8'),(300,300),interpolation=cv2.INTER_NEAREST);contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(255,205,10),1);sheet.paste(Image.fromarray(panel),(300*i,j*335+30));draw.text((300*i+3,j*335+5),str(fr)+' · '+title,font=font,fill='white')
 sheet.save(OUT/(cid+'_field_comparison.png'));means[cid]={}
 for view in ['context','zoom']:
  for mode in ['ordinary','first_only']:
   data=[z['iou'] for z in rows if z['case']==cid and z['view']==view and z['mode']==mode and z['iou'] is not None];means[cid][view+'__'+mode]=float(np.mean(data)) if data else None
assert all(sha(ROOT/n)==h for n,h in read(OUT/'field_protocol.json')['inputs'].items())
save('field_result.json',dict(records=rows,means=means,initialization=initrows,all_known_targets_inside_fixed_windows=all(z['target_pixels']==z['target_pixels_inside_window'] for z in rows),scope='Post-primary input rendering diagnostic. MeanIoU on labeledcores, not new-video validation of chosen window; missing target has undefinedIoU.'))
save('audit.json',dict(passed=True,primary_count_reconstructions=len(checks),field_mask_score_reconstructions=len(fieldmask.files),primary_exact_count_keys=checks,source_unchanged=True,artifacts={n:sha(OUT/n) for n in ['result.json','field_result.json','predictions.npz','scores.npz','field_predictions.npz','field_scores.npz','initial_masks.npz','query_cores.npz']},scope='Reconstructed binary forecasts from saved scores/field transforms and recomputed primary confusion counts. Does not promote sparse labels to full groundtruth.'));print(json.dumps(means),flush=True)
