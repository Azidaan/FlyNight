from common import *
import cv2,numpy as np
from scipy.stats import rankdata
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');assert read(OUT/'inference.json')['maps_sha256']==sha(OUT/'maps.npz');maps=np.load(OUT/'maps.npz');A=ROOT/'research_mask_propagation_20261114';B=ROOT/'research_mask_memory_transfer_20261114';oldpred=np.load(A/'predictions.npz');oldmem=np.load(A/'memory_predictions.npz');newpred=np.load(B/'predictions.npz');initials=np.load(B/'initial_masks.npz');q=[.1,.25,.5,.75,.9];variants=p['candidates'];selections=[];descriptors={};decisions={};masks={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
def candidates(c,fr):
 if c['id']=='old2179':return [oldpred['sam__'+str(fr)],oldmem['initial_memory_only__'+str(fr)],oldpred['flow__'+str(fr)],oldpred['static__'+str(fr)]]
 return [newpred[c['id']+'__'+v+'__'+str(fr)] for v in variants]
def fields(c,fr):
 im=cv2.cvtColor(cv2.imread(str(ROOT/c['image_dir']/c['image_pattern'].format(frame=fr))),cv2.COLOR_BGR2RGB);dep=maps[c['id']+'__'+str(fr)];rank=(rankdata(dep.ravel(),method='average').reshape(dep.shape)-.5)/dep.size;yy=np.broadcast_to(np.arange(dep.shape[0],dtype=float)[:,None]/(dep.shape[0]-1),dep.shape);return im,rank,yy
def desc(mask,fields):
 im,rank,yy=fields;assert mask.any();return dict(RGB=np.quantile(im[mask]/255.,q,axis=0).flatten(),Depth=np.quantile(rank[mask],q),Y=np.quantile(yy[mask],q))
for c in p['cases']:
 cid=c['id'];first=np.load(ROOT/read(A/'protocol.json')['initial_mask'])['1159_target'].astype(bool) if cid=='old2179' else initials[cid];reference=desc(first,fields(c,c['reference']));descriptors.update({cid+'__reference__'+n:x for n,x in reference.items()})
 for fr in c['queries']:
  fs=fields(c,fr);cs=candidates(c,fr);cost={k:np.full(4,np.inf) for k in reference}
  for j,m in enumerate(cs):
   if not m.any():continue
   d=desc(m,fs);descriptors.update({cid+'__'+str(fr)+'__'+variants[j]+'__'+n:x for n,x in d.items()})
   for k in reference:cost[k][j]=np.abs(reference[k]-d[k]).mean()
  finite=np.isfinite(cost['RGB']);rank={k:rankdata(v,method='average') for k,v in cost.items()};comb={'RGB':cost['RGB'],'Depth':cost['Depth'],'Y':cost['Y'],'RGB+Depth':rank['RGB']+rank['Depth'],'RGB+Y':rank['RGB']+rank['Y']};picked={}
  for name,v in comb.items():
   idx=int(np.argmin(np.where(finite,v,np.inf))) if finite.any() else -1;picked[name]=variants[idx] if idx>=0 else 'empty';masks[cid+'__'+str(fr)+'__'+name]=cs[idx] if idx>=0 else np.zeros(first.shape,bool)
  selections.append(dict(case=cid,frame=fr,cost={k:[float(x) if np.isfinite(x) else None for x in v] for k,v in cost.items()},selected=picked,areas={n:int(m.sum()) for n,m in zip(variants,cs)}));decisions[cid+'__'+str(fr)]=picked
with (OUT/'descriptors.npz').open('xb') as f:np.savez_compressed(f,**descriptors)
with (OUT/'selected_masks.npz').open('xb') as f:np.savez_compressed(f,**masks)
# Freeze all decisions before opening either evaluation truth archive.
save('choices.json',dict(records=selections,descriptors_sha256=sha(OUT/'descriptors.npz'),selected_masks_sha256=sha(OUT/'selected_masks.npz'),query_truth_read=False))
oldtruth=np.load(A/'evaluation_truth.npz');newtruth=np.load(B/'query_cores.npz');rows=[];means={};allnames=variants+['RGB','Depth','Y','RGB+Depth','RGB+Y','oracle']
def evaluate(mask,t):
 y=t==1;n=(t>=0)&~y;tp=int((mask&y).sum());fp=int((mask&n).sum());fn=int((~mask&y).sum());return dict(tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn) if y.any() else None,recall=tp/int(y.sum()) if y.any() else None,fpr=fp/int(n.sum()) if n.any() else None,predicted_pixels=int(mask.sum()))
for c in p['cases']:
 cid=c['id'];means[cid]={}
 for fr in c['queries']:
  truth=oldtruth[str(fr)] if cid=='old2179' else newtruth[c['video']+'__'+str(fr)];cs=candidates(c,fr);available={v:evaluate(m,truth) for v,m in zip(variants,cs)}
  for name,met in available.items():rows.append(dict(case=cid,frame=fr,method=name,selected=name,**met))
  for name in ['RGB','Depth','Y','RGB+Depth','RGB+Y']:rows.append(dict(case=cid,frame=fr,method=name,selected=decisions[cid+'__'+str(fr)][name],**evaluate(masks[cid+'__'+str(fr)+'__'+name],truth)))
  if (truth==1).any():best=max(variants,key=lambda n:available[n]['iou']);rows.append(dict(case=cid,frame=fr,method='oracle',selected=best,**available[best]))
 for name in allnames:
  data=[r['iou'] for r in rows if r['case']==cid and r['method']==name and r['iou'] is not None];means[cid][name]=float(np.mean(data)) if data else None
valid=[v for v in means.values() if v['ordinary'] is not None];overall={n:float(np.mean([v[n] for v in valid])) for n in allnames};wins=sum(v['RGB+Depth']>v['RGB'] for v in valid);inc=wins>=2 and overall['RGB+Depth']>overall['RGB'];specific=overall['RGB+Depth']>overall['RGB+Y'] and overall['RGB+Depth']>=max(overall[n] for n in variants)
save('result.json',dict(records=rows,means_by_video=means,video_equal_means=overall,depth_improves_rgb_videos=wins,increment_passed=bool(inc),specificity_and_baseline_passed=bool(specific),scientific_passed=bool(inc and specific),scope='Conditional on four precomputed masks and partial manually identified targetregions. Oracle uses truth after selection only. Forced choice can give false output when target notvisible. No trackerID or metricdepth evaluated.'))
print(json.dumps(dict(means=means,overall=overall,wins=wins)),flush=True)
