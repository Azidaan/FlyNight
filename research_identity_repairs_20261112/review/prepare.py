"""Snapshot all five complete assignment matrices and exact observed references.

Metadata only: no solver, descriptors, policies, tracker or neural execution.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,gzip,hashlib,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
stage=read(ROOT/'research_stage_localization_20261112/protocol.json');native=read(ROOT/'runs/domain_matched_native_20261109/protocol.json');old_reid=read(ROOT/'research_remaining_causes_20261112/independent_review/reid/result.json');low=read(ROOT/'research_remaining_causes_20261112/independent_review/reid/low_protocol_v3.json')
sources=['research_stage_localization_20261112/protocol.json','runs/domain_matched_native_20261109/protocol.json','research_failure_localization_20261112/stage1.json','research_failure_localization_20261112/stage2.json','research_remaining_causes_20261112/independent_review/reid/result.json','research_remaining_causes_20261112/independent_review/reid/descriptors.npz','research_remaining_causes_20261112/independent_review/reid/low_protocol_v3.json','research_remaining_causes_20261112/independent_review/reid/low_result.json','research_remaining_causes_20261112/independent_review/reid/low_descriptors.npz','weights/masa/masa_r50.pth','tracking/masa_r50.py','tracking/gpu_lock.py']
frames={};snapshots=[]
def slot(v,fr,box,usage):
 key=v+'|'+str(fr);d=frames.setdefault(key,dict(video=v,frame=fr,video_path=native['videos'][v]['video_path'],boxes=[],usages=[]))
 box=list(map(float,box))
 if box not in d['boxes']:d['boxes'].append(box);d['usages'].append([])
 idx=d['boxes'].index(box);d['usages'][idx].append(usage);return [key,idx]
for cc in stage['cases']:
 c=cc['case'];v=c['video'];eventpath=ROOT/'runs/domain_matched_native_20261109/rollouts/C34'/(v+'.events.json.gz');sources.append(str(eventpath.relative_to(ROOT)).replace('\\','/'))
 with gzip.open(eventpath,'rt',encoding='utf8') as f:d=json.load(f)
 e=next(x for x in d['events'] if x['frame']==c['frame']);assert len(e['C'])==len(e['rows']) and len(e['high_raw_ids'])==len(e['C'][0]);target=next(q for q in e['proposals'] if q['old_id']==c['old_id'] and q['raw_id']==c['raw_id']);rows=[]
 for i,row in enumerate(e['rows']):
  h=row['history'];last=h[-1] if h else None;ref=None
  if last:
   assert 0<=last['frame']<e['frame'];ref=dict(frame=last['frame'],box=last['box'],raw_id=last.get('raw_id'),raw_origin_method=last.get('raw_origin_method'),public=last.get('public'),crop=last.get('crop'),slot=slot(v,last['frame'],last['box'],dict(old_case=c['old_id'],row=i,tid=row['tid'],role='last_observed_reference')))
  mean=np.asarray(row['mean'],float);pred=np.r_[mean[:2]-mean[2:4]/2,mean[:2]+mean[2:4]/2].tolist()
  rows.append(dict(row=i,tid=row['tid'],state=row['state'],cls=row['cls'],start_frame=row['start_frame'],mean=row['mean'],covariance=row['covariance'],predicted_xyxy=pred,history=h,reference=ref,current_crop=row['current_crop'],search=row['search']))
 candidates=next(x['candidates'] for x in low['cases'] if x['old_id']==c['old_id'])
 snapshots.append(dict(video=v,frame=c['frame'],old_case=c['old_id'],label=c['label'],target_tid=c['old_id'],target_raw_id=c['raw_id'],target_row=target['row'],target_column=target['column'],rows=rows,high_raw_ids=e['high_raw_ids'],base_cost=e['C'],corrected_cost=e['corrected'],votes=e['votes'],native_first=e['native_first'],actual_first=e['actual_first'],emitted_ids=e['emitted_ids'],assigned=e['assigned'],duplicates=e['duplicates'],proposals=e['proposals'],raw_current_car_candidates=candidates,reference_slot=slot(v,cc['sequence'][cc['endpoints'][0]['length']-1][0],c['reference_box'],dict(old_case=c['old_id'],role='existing_target_reference_witness')),evaluation=dict(label=c['label'],target_tid=c['old_id'],target_raw_id=c['raw_id'],annotation_clock=f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json',annotation_boxes=f'data/bdd100k/labels/box_track_20/val/{v}.json')))
 sources.extend([native['videos'][v]['video_path'],native['videos'][v]['detections'],f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json',f'data/bdd100k/labels/box_track_20/val/{v}.json'])
save('assignment_snapshots.json',dict(prepared=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),sources={n:sha(ROOT/n) for n in sorted(set(sources))},snapshots=snapshots,scope='Read-only data preparation; actual assignments copied, not recomputed. Evaluation fields are forbidden inputs to repair policies. Current snapshot hashes bind files as read today, never replace historical run checksums retroactively.'))
save('reference_extraction_plan.json',dict(prepared=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),snapshot_sha256=sha(OUT/'assignment_snapshots.json'),frames=frames,mode='One shared immutable frame backbone; exact last observed history box, not predicted state box, for appearance reference. No annotation input. Decode sequentially from0. Reuse/check target existing5reference descriptors from old frozen ledger; allnew refs use own exact frame/box/source ledger. Batch order fixed before forward to avoid known batch composition numerical mismatch. NoGPUworkperformed.',deadline='2026-11-13T07:35:00+03:00'))
print('METADATA_PREPARED',len(snapshots),sum(len(s['rows']) for s in snapshots),len(frames),sum(len(f['boxes']) for f in frames.values()))
