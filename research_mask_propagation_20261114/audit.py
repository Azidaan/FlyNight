from common import *
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');r=read(OUT/'result.json');truth=np.load(OUT/'evaluation_truth.npz');pred=np.load(OUT/'predictions.npz');memory=np.load(OUT/'memory_predictions.npz');scores=np.load(OUT/'memory_scores.npz');mr=read(OUT/'memory_result.json');rows=[];checks=[]
def recompute(mask,t):
 valid=t>=0;target=t==1;code=(target[valid].astype(int)*2+mask[valid].astype(int));tn,fp,fn,tp=np.bincount(code,minlength=4);return dict(tp=int(tp),fp=int(fp),fn=int(fn),iou=float(tp/(tp+fp+fn)) if tp+fp+fn else None,recall=float(tp/(tp+fn)) if tp+fn else None,fpr=float(fp/(fp+tn)) if fp+tn else None)
for row in r['records']:
 vals=recompute(pred[row['variant']+'__'+str(row['frame'])],truth[str(row['frame'])]);assert all(row[k]==v for k,v in vals.items());checks.append(dict(frame=row['frame'],variant=row['variant']))
for frame in range(1159,1208):
 assert np.array_equal(memory['replay__'+str(frame)],pred['sam__'+str(frame)])
for frame in p['query_frames']:
 for variant in ['replay','initial_memory_only']:
  mask=memory[variant+'__'+str(frame)];assert np.array_equal(mask,scores[variant+'__'+str(frame)]>0);rows.append(dict(frame=frame,variant=variant,predicted_pixels=int(mask.sum()),**recompute(mask,truth[str(frame)])))
assert all(not t['nonconditioning_after_clear'] for t in mr['traces'] if t['variant']=='initial_memory_only')
for n,h in read(OUT/'memory_protocol_v2.json')['inputs'].items():assert sha(ROOT/n)==h,n
trajectory=[]
for frame in range(1159,1208):
 for variant,mask in [('primary',pred['sam__'+str(frame)]),('initial_memory_only',memory['initial_memory_only__'+str(frame)])]:
  y,x=np.where(mask);trajectory.append(dict(frame=frame,variant=variant,area=len(x),centroid=[float(x.mean()),float(y.mean())] if len(x) else None))
save('memory_evaluation.json',dict(records=rows,trajectory=trajectory,scope='Post-primary controlledmemoryintervention; keeps same input frames/weights/initialmask/threshold and changes only retained selfpredicted memory. Does not establish all causes or new-video generalization.'))
save('audit.json',dict(passed=True,primary_metric_checks=checks,memory_metric_records=len(rows),baseline_replay_exact_frames=49,source_unchanged=True,initial_memory_only_has_no_retained_selfpredictions=True,primary_negative_preserved=not r['scientific_passed'],artifacts={n:sha(OUT/n) for n in ['protocol.json','result.json','predictions.npz','scores.npz','manual_query_cores_v2.npz','memory_protocol_v2.json','memory_result.json','memory_predictions.npz','memory_scores.npz','memory_evaluation.json']},scope='Independent confusion-count recomputation and array/replay/hash checks; conditioning-memory invariance is asserted inside the recorded worker, not independently reconstructed from saved tensors.'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
for frame in p['query_frames']:
 im=cv2.imread(str(OUT/f'frame_{frame}.png'));sheet=Image.new('RGB',(1590,595),'#20232a');draw=ImageDraw.Draw(sheet)
 for j,(name,mask) in enumerate([('Текущий кадр',None),('Обычная память SAM 2',pred['sam__'+str(frame)]),('Только первый образец в памяти',memory['initial_memory_only__'+str(frame)])]):
  picture=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  if mask is not None:picture[mask]=(picture[mask]*.4+np.array([20,235,50])*.6).astype('uint8')
  sheet.paste(Image.fromarray(picture[100:630,750:1280]),(j*530,35));draw.text((j*530+6,7),name,font=font,fill='white')
 draw.text((8,572),f'Кадр {frame}. Изменено только сохранение собственных прогнозов; новые подсказки и пороги не добавлены.',font=font,fill='white');sheet.save(OUT/f'memory_comparison_{frame}.png')
print(json.dumps(rows),flush=True)
