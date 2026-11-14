from official import *
import time
cv2.setNumThreads(2);p=read(OUT/'protocol.json');parent=ROOT/'research_return_cycle_20261114';cs=read(parent/'cases.json')['cases'];decs=read(parent/'decisions.json')['records'];rr=read(parent/'result.json')['records'];arr=np.load(parent/'outputs.npz');yy,xx=np.indices((720,1280),'float32');out={};steps=[];timing=[];rows=[];witness=[]
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def jacc(a,b):
 inter=np.count_nonzero(a&b);u=np.count_nonzero(a)+np.count_nonzero(b)-inter;return inter/u if u else 0.
for c in cs+[dict(id='old2179',video='b1ca2e5d-84cf9134',split='old',frames=[dict(image=rel(ROOT/'research_mask_propagation_20261114/frames'/f'{j:05d}.jpg')) for j in range(13)])]:
 cid=c['id'];frames=c['frames'];old=cid=='old2179';m=[arr[f'{cid}__past__{i}'].astype('uint8') for i in (0,1)];gray=cv2.imread(str(ROOT/frames[0]['image']),0)
 for j in range(1,len(frames)):
  nowim=cv2.imread(str(ROOT/frames[j]['image']),0);start=time.perf_counter();# backward map current -> previous for remap
  flow=cv2.calcOpticalFlowFarneback(nowim,gray,None,.5,3,15,3,5,1.2,0);m=[cv2.remap(a,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT) for a in m];gray=nowim;timing.append(time.perf_counter()-start)
  if j not in ([6,12] if old else [1,2,4]):continue
  fr=1159+j if old else j;e=next(r for r in decs if r['case']==cid and r['frame']==fr);actions=dict(e['forward']);chosen=[(i,rid,arr[f'{cid}__{fr}__obs__{rid}']) for i,rid in e['forward']];scores=[[jacc(a.astype(bool),b) for _,_,b in chosen] for a in m];good=[]
  for pos,(i,rid,mask) in enumerate(chosen):
   v=scores[i][pos];a=np.asarray(scores);agree=v>.5 and np.all(v>np.delete(a[i],pos)) and np.all(v>np.delete(a[:,pos],i));assert {0,1}>=set(actions)
   if agree:good.append([i,rid])
  if old:witness.append(dict(frame=fr,original=e['forward'],accepted=good,overlaps=scores))
  for i in [0,1]:
   rid=actions.get(i);approved=dict(good).get(i);out[f'{cid}__{fr}__{i}']=m[i].astype(bool)
   if not old:
    r=next(q for q in rr if q['case']==cid and q['frame']==j and q['slot']==i and q['mode']=='forward');back=next(q for q in rr if q['case']==cid and q['frame']==j and q['slot']==i and q['mode']=='confirmed');rows.append(dict(case=cid,video=c['video'],split=c['split'],frame=j,slot=i,proposed=rid,approved=approved,reverse=back['raw_id'],original_status=r['status'],cycle_status=back['status'],motion_status=r['status'] if approved is not None else 'deferred',visible=r['visible']))
  steps.append(dict(case=cid,frame=fr,proposal=e['forward'],accepted=good,scores=scores))
with (OUT/'motion.npz').open('xb') as f:np.savez_compressed(f,**out)
save('actions.json',dict(records=rows,steps=steps,old=witness,seconds_per_flow=timing));print('FLOW_DECISIONS',len(rows),len(timing),flush=True)
