"""Evaluation-only provenance trace; no descriptor or runtime selector imports."""
from common import *
import gzip, numpy as np
ep=read(PREV/'evaluation_plan.json'); clock=read(ROOT/ep['clock'])['anchors']; labels=read(ROOT/ep['labels']); by={int(x.get('frameIndex',i)):x for i,x in enumerate(labels)}
with np.load(ROOT/ep['detector_file'],allow_pickle=False) as z: raw={k:z[k].copy() for k in ['xyxy','conf','cls','frame_idx']}
with gzip.open(ROOT/ep['source_archive'],'rt',encoding='utf8') as f: archive=json.load(f)
def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1])); union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0
def ownership(frame,box):
    if str(frame) not in clock: return None
    a=clock[str(frame)]; row=by[a['frameIndex']]; assert row['name']==a['name']; found=[]
    for obj in row.get('labels') or []:
        if not obj.get('box2d') or (obj.get('attributes') or {}).get('crowd',False): continue
        b=[obj['box2d'][k] for k in ['x1','y1','x2','y2']]; score=overlap(box,b)
        if score>0: found.append(dict(identity=str(obj['id']),iou=score,covered_object_fraction=max(0,min(box[2],b[2])-max(box[0],b[0]))*max(0,min(box[3],b[3])-max(box[1],b[1]))/((b[2]-b[0])*(b[3]-b[1]))))
    return sorted(found,key=lambda x:-x['iou'])
rows=[]
for e in archive['events']:
    if not 1147<=e['frame']<=1165: continue
    for r in e['rows']:
        if r['tid'] not in [2179,2445]: continue
        rid=e['assigned'].get(str(r['tid'])); ref=r['history'][-1] if r['history'] else None
        item=dict(frame=e['frame'],tid=r['tid'],state=r['state'],assigned_raw_id=rid,exact_anchor=str(e['frame']) in clock,mean=r['mean'])
        if rid is not None:
            assert int(raw['frame_idx'][rid])==e['frame']
            item.update(detection=raw['xyxy'][rid].tolist(),confidence=float(raw['conf'][rid]),detection_ownership=ownership(e['frame'],raw['xyxy'][rid]))
        if ref:
            box=raw['xyxy'][ref['raw_id']].tolist(); item['last_history']=dict(frame=ref['frame'],raw_id=ref['raw_id'],filtered_box=ref['box'],raw_box=box,filtered_raw_iou=overlap(ref['box'],box),filtered_ownership=ownership(ref['frame'],ref['box']),raw_ownership=ownership(ref['frame'],box))
        rows.append(item)
observations=[]
for o in read(OUT/'descriptor_plan.json')['observations']:
    observations.append(dict(**o,filtered_raw_iou=overlap(o['filtered'],o['raw']),filtered_ownership=ownership(o['frame'],o['filtered']),raw_ownership=ownership(o['frame'],o['raw'])))
save('trace.json',dict(passed=True,rows=rows,observations=observations,exact_anchors_in_window=sorted(int(x) for x in clock if 1147<=int(x)<=1165),scope='Exact anchors only, overlap diagnostics do not label unannotated frames or prove first physical switch. Filtered box and raw detection are measured separately.'))
print('TRACE_READY',len(rows),len(observations),flush=True)
