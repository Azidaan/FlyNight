"""Prepare complete A event inputs and shared frame/ROI ledger; no GT reads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,gzip,hashlib
from pathlib import Path
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
selection=read(OUT/'selected_events_A_v2.json');inv=read(OUT/'inventory_v2.json');proposal=read(OUT/'proposed_selection_protocol_v2.json');native=read(ROOT/'runs/domain_matched_native_20261109/protocol.json')
excluded=sorted(v for v,roles in inv['video_exposure'].items() if any(role!='historical_native_validation_aggregate_and_baseline' for role,source in roles))
assert not(set(proposal['rule']['A_videos'])&set(excluded));frames={};events=[];archives={};rawcache={};sources={}
def slot(v,frame,box,use):
 key=v+'|'+str(frame);x=frames.setdefault(key,dict(video=v,frame=frame,video_path=native['videos'][v]['video_path'],boxes=[],usages=[]));box=list(map(float,box))
 if box not in x['boxes']:x['boxes'].append(box);x['usages'].append([])
 i=x['boxes'].index(box);x['usages'][i].append(use);return [key,i]
for chosen in selection['events']:
 v=chosen['video'];fr=chosen['frame'];path=chosen['event_archive'];assert v not in excluded
 if path not in archives:
  with gzip.open(ROOT/path,'rt',encoding='utf8') as f:archives[path]={e['frame']:e for e in json.load(f)['events']}
  sources[path]=sha(ROOT/path)
 if v not in rawcache:
  with np.load(ROOT/native['videos'][v]['detections']) as z:rawcache[v]={k:z[k] for k in ['xyxy','frame_idx','conf','cls']}
  for n in [native['videos'][v]['detections'],native['videos'][v]['video_path'],f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json',f'data/bdd100k/labels/box_track_20/val/{v}.json']:sources[n]=sha(ROOT/n)
 e=archives[path][fr];raw=rawcache[v];eligible=[p for p in e['proposals'] if p.get('geometry') is not None];assert [(p['old_id'],p['raw_id']) for p in eligible]==[(p['old_id'],p['raw_id']) for p in chosen['proposal_keys']]
 rows=[]
 for i,row in enumerate(e['rows']):
  mean=np.asarray(row['mean'],float);pred=np.r_[mean[:2]-mean[2:4]/2,mean[:2]+mean[2:4]/2].tolist();last=row['history'][-1] if row['history'] else None;ref=None
  if last:
   assert last['frame']<fr;ref=dict(frame=last['frame'],box=last['box'],raw_id=last.get('raw_id'),crop=last.get('crop'),slot=slot(v,last['frame'],last['box'],dict(event_key=chosen['event_key'],row=i,tid=row['tid'],role='past_observation')))
  rows.append(dict(row=i,tid=row['tid'],state=row['state'],cls=row['cls'],start_frame=row['start_frame'],predicted_xyxy=pred,predicted_box=pred,mean=row['mean'],covariance=row['covariance'],reference=ref,reference_slot=ref['slot'] if ref else None,history=row['history'],current_crop=row['current_crop'],search=row['search']))
 ids=np.flatnonzero((raw['frame_idx']==fr)&(raw['cls']==2));candidates=[]
 for i in ids:
  box=raw['xyxy'][i].astype(float).tolist();s=slot(v,fr,box,dict(event_key=chosen['event_key'],raw_id=int(i),role='current_raw_car'))
  candidates.append(dict(raw_id=int(i),box=box,confidence=float(raw['conf'][i]),cls=int(raw['cls'][i]),current_slot=s,slot=s,native_first=bool(raw['conf'][i]>=.25),native_second=bool(.1<raw['conf'][i]<.25)))
 events.append(dict(event_key=chosen['event_key'],video=v,frame=fr,event_archive=path,rows=rows,native_proposals=e['proposals'],proposals=e['proposals'],eligible_proposal_keys=chosen['proposal_keys'],assigned=e['assigned'],actual_first=e['actual_first'],native_first=e['native_first'],high_raw_ids=e['high_raw_ids'],base_cost=e['C'],corrected_cost=e['corrected'],votes=e['votes'],emitted_ids=e['emitted_ids'],duplicates=e['duplicates'],raw_candidates=candidates,raw_current_car_candidates=candidates,evaluation_sources=dict(clock=f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json',labels=f'data/bdd100k/labels/box_track_20/val/{v}.json')))
for n in ['selected_events_A_v2.json','inventory_v2.json','proposed_selection_protocol_v2.json']:sources[str((OUT/n).relative_to(ROOT)).replace('\\','/')]=sha(OUT/n)
save('A_runtime_manifest_v2.json',dict(cohort='A',events=events,snapshots=events,excluded_videos=excluded,source_sha256=sources,selection_sha256=sha(OUT/'selected_events_A_v2.json'),scope=proposal['rule']['scope'],GT_read_during_preparation=False,repair_outcomes_computed=False))
save('A_descriptor_plan_v2.json',dict(cohort='A',frames=frames,runtime_manifest_sha256=sha(OUT/'A_runtime_manifest_v2.json'),source_sha256=sources,excluded_videos=excluded,policy='ALLstrictAevents,allcompetitorpastrows+allrawcurrentcars; exactboxes, no silentclipping. Sequentialdecodeonce/video; oneGPUowner. Bnotincluded. NoGT/ownavailabilityselection.',deadline=proposal['deadline']))
print(json.dumps(dict(events=len(events),eligible_proposals=sum(len(e['eligible_proposal_keys']) for e in events),rows=sum(len(e['rows']) for e in events),current_candidates=sum(len(e['raw_candidates']) for e in events),unique_frames=len(frames),unique_rois=sum(len(f['boxes']) for f in frames.values()),rows_missing_reference=sum(r['reference'] is None for e in events for r in e['rows']),excluded_videos=len(excluded)),ensure_ascii=False))
