from official import *
from collections import Counter
p=read(OUT/'protocol.json');s=read(OUT/'actions.json');flow=np.load(OUT/'motion.npz');arr=np.load(ROOT/'research_return_cycle_20261114/outputs.npz');rr=s['records'];verified=0
for q in s['steps']:
 ids=[rid for i,rid in q['proposal']];score=np.zeros((2,len(ids)))
 for i in (0,1):
  a=flow[f"{q['case']}__{q['frame']}__{i}"]
  for j,rid in enumerate(ids):
   b=arr[f"{q['case']}__{q['frame']}__obs__{rid}"];tn,fp,fn,tp=np.bincount(2*a.ravel().astype('uint8')+b.ravel().astype('uint8'),minlength=4);score[i,j]=tp/max(tp+fp+fn,1);verified+=1
 assert np.allclose(score,q['scores'],rtol=0,atol=1e-12)
 paired=sorted([[i,rid] for i,rid in q['proposal'] if score[i,ids.index(rid)]>.5 and np.count_nonzero(score[i]>=score[i,ids.index(rid)])==1 and score[i,ids.index(rid)]==max(score[i]) and np.argmax(score[:,ids.index(rid)])==i and score[i,ids.index(rid)]>score[1-i,ids.index(rid)]])
 assert paired==sorted(q['accepted'])
stats={}
for sp in ['diagnostic','fresh']:
 gr=[q for q in rr if q['split']==sp];vs=sorted({q['video'] for q in gr});snap={k:{n:int(sum(q[k]==n for q in gr)) for n in ['correct','wrong','deferred']} for k in ['original_status','cycle_status','motion_status']};pairs={}
 for metric in ['correct','wrong']:
  dif=np.array([np.mean([int(q['motion_status']==metric)-int(q['original_status']==metric) for q in gr if q['video']==v]) for v in vs]);ci=np.quantile(dif[np.random.default_rng(42).integers(len(vs),size=(10000,len(vs)))].mean(1),[.025,.975]);pairs[metric]=dict(mean=float(dif.mean()),interval=ci.tolist())
 stats[sp]=dict(stats=snap,change=pairs,videos=len(vs))
old=next(q for q in s['old'] if q['frame']==1171);old_ok=old['accepted']==[[0,111062]] and next(q for q in s['old'] if q['frame']==1165)['accepted']==[];f=stats['fresh'];gain=f['change']['wrong']['interval'][1]<0;keep=f['stats']['motion_status']['correct']>=f['stats']['original_status']['correct'];save('result.json',dict(scientific_passed=bool(old_ok and gain and keep),old_ok=old_ok,fresh_wrong_reduction=bool(gain),fresh_correct_kept=bool(keep),stats=stats,old=s['old'],seconds_median=float(np.median(s['seconds_per_flow'])),scope='Independent propagation conditional on same initialization, same current detector and segmentation proposals. Independent algorithms, not independent data. No proposals introduced by flow; policy can only abstain.'))
save('audit.json',dict(passed=True,recomputed_pixels_to_iou=verified,checks=len(s['steps']),result_sha256=sha(OUT/'result.json')))
