from common import *
import gzip
import numpy as np
p=verify();native=ROOT/'runs/domain_matched_native_20261109';v='b23b2649-8349d2a1';f=539
with gzip.open(native/'rollouts/C34'/(v+'.events.json.gz'),'rt',encoding='utf8') as h:d=json.load(h)
e=next(e for e in d['events'] if e['frame']==f);i,r=next((i,r) for i,r in enumerate(e['rows']) if r['tid']==232);m=np.array(r['mean'][:4]);box=np.r_[m[:2]-m[2:]/2,m[:2]+m[2:]/2]
with np.load(ROOT/read(native/'protocol.json')['videos'][v]['detections'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
out=[]
for rid in np.flatnonzero(raw['frame_idx']==f):
    b=raw['xyxy'][rid].astype(float);inter=np.maximum(0,np.minimum(box[2:],b[2:])-np.maximum(box[:2],b[:2])).prod();u=np.prod(box[2:]-box[:2])+np.prod(b[2:]-b[:2])-inter;ov=float(inter/u);confidence=float(raw['conf'][rid]);cls=int(raw['cls'][rid]);stage='high' if confidence>=.25 else 'low' if confidence>.1 else 'excluded'
    # Independent scalar intersection and area check, not another tracker call.
    ii=max(0,min(box[2],b[2])-max(box[0],b[0]))*max(0,min(box[3],b[3])-max(box[1],b[1]));uu=(box[2]-box[0])*(box[3]-box[1])+(b[2]-b[0])*(b[3]-b[1])-ii;assert abs(ov-ii/uu)<1e-15
    out.append(dict(raw_id=int(rid),confidence=confidence,cls=cls,box=b.tolist(),iou=ov,score_fused_cost=1-ov*confidence,plain_iou_cost=1-ov,stage=stage))
high=[x for x in out if x['stage']=='high'];low=[x for x in out if x['stage']=='low'];assert min(e['corrected'][i])>.8 and not any(a==i for a,b in e['actual_first']);assert all(x['score_fused_cost']>.5 and x['plain_iou_cost']>.5 for x in low)
assert not any(a['tid']==232 and a['frame']==539 for a in d['assignments']);next_row=next(rr for ee in d['events'] if ee['frame']==544 for rr in ee['rows'] if rr['tid']==232);assert next_row['state']==2 and next_row['history'][-1]['frame']==538
save('drop232.json',dict(passed=True,video=v,frame=f,tid=232,predicted_box=box.tolist(),initial_state=r['state'],saved_high_min_cost=min(e['corrected'][i]),max_iou_all=max(x['iou'] for x in out),max_iou_low=max(x['iou'] for x in low),best_detections=sorted(out,key=lambda x:x['iou'],reverse=True)[:8],low_count=len(low),high_count=len(high),total_detections=len(out),no_first_match=True,no_second_match_possible=True,plain_iou_second_also_cannot_match=True,first_unassigned_after=538,later_state=next_row['state'],source_readout_proposals=[q for q in e['proposals'] if q['old_id']==232],scope='Exact algorithmic failure at the first loss after538. No annotation at539, so no ground-truth identity or physical cause is invented. The previously known score-fused second-stage impossibility is not sufficient to explain this case: removing fusion still cannot match any low-score box at the unchanged0.5 threshold.',script_sha256=sha(__file__)))
print('DROP232_VERIFIED',max(x['iou'] for x in out),min(e['corrected'][i]))
