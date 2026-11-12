"""Evaluation-only metadata audit: never passes annotations to repair policy."""
import json,hashlib,math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));den=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
 return inter/den if den>0 else 0.
runtime=read(OUT/'A_runtime_manifest_v2.json');caches={};sources={};rows=[];targets=[]
def labels(video,frame):
 if video not in caches:
  cp=f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json';lp=f'data/bdd100k/labels/box_track_20/val/{video}.json';ap=f'runs/z13_val20_matched_20261024/cache/{video}/alignment.json';clock=read(ROOT/cp);gt=read(ROOT/lp);align=read(ROOT/ap);by={int(x.get('frameIndex',i)):x for i,x in enumerate(gt)}
  assert {int(k):(x['name'],x['frameIndex']) for k,x in clock['anchors'].items()}=={x['video_frame_index']:(x['reference_name'],x['reference_index']) for x in align['matches']}
  caches[video]=(clock['anchors'],by)
  for p in [cp,lp,ap]:sources[p]=sha(ROOT/p)
 anchors,by=caches[video]
 if str(frame) not in anchors:return None
 anchor=anchors[str(frame)];g=by[anchor['frameIndex']];assert g['name']==anchor['name']
 return [dict(identity=str(o['id']),box=[o['box2d'][k] for k in ['x1','y1','x2','y2']]) for o in g['labels'] if o['category']=='car' and not o.get('attributes',{}).get('crowd',False)]
def hits(box,objects):return [dict(identity=o['identity'],iou=iou(box,o['box'])) for o in objects if iou(box,o['box'])>=.5]
for event in runtime['events']:
 v=event['video'];current=labels(v,event['frame']);row_results={}
 for row in event['rows']:
  bindings=[]
  for observed in row['history']:
   annotations=labels(v,observed['frame'])
   if annotations is not None:bindings.append(dict(frame=observed['frame'],hits=hits(observed['box'],annotations)))
  unique=[b['hits'][0]['identity'] for b in bindings if len(b['hits'])==1];expected=None
  if not bindings:reason='no_exact_annotated_observation_in_saved_recent_history'
  elif any(len(b['hits'])!=1 for b in bindings):reason='annotated_history_observation_unmatched_or_ambiguous'
  elif len(set(unique))!=1:reason='inconsistent_annotated_history_identities'
  else:reason='consistent_exact_anchored_history';expected=unique[0]
  last=row['reference'];lastlabels=None if last is None else labels(v,last['frame']);lasthits=[] if lastlabels is None else hits(last['box'],lastlabels)
  primary=bool(current is not None and lastlabels is not None and len(lasthits)==1)
  anchored=bool(current is not None and expected is not None)
  r=dict(event_key=event['event_key'],video=v,frame=event['frame'],tid=row['tid'],state=row['state'],cls=row['cls'],current_exact_anchor=current is not None,last_reference_exact_anchor=lastlabels is not None,last_reference_unique_identity=lasthits[0]['identity'] if len(lasthits)==1 else None,strict_last_reference_evaluable=primary,annotated_history_observations=bindings,annotated_history_count=len(bindings),history_identity_status=reason,expected_anchored_persistent_identity=expected,anchored_history_evaluable=anchored,expected_identity_present_in_current_annotation=bool(anchored and any(o['identity']==expected for o in current)),scope='Expectedtrackidentityatpastlabelledobservation; NOTgroundtruthidentityoflastunannotatedappearancecrop. No propagation/interpolation ofpixelGT. Recentannotatedhistoryinconsistency remainsunknown/historyproblem.')
  rows.append(r);row_results[row['tid']]=r
 for p in event['eligible_proposal_keys']:targets.append(dict(proposal_key=event['event_key']+'|'+str(p['old_id'])+'|'+str(p['raw_id']),raw_id=p['raw_id'],**row_results[p['old_id']]))
summary=dict(events=len(runtime['events']),proposals=len(targets),current_exact=sum(r['current_exact_anchor'] for r in targets),last_reference_exact=sum(r['last_reference_exact_anchor'] for r in targets),both_current_last_reference_exact=sum(r['current_exact_anchor'] and r['last_reference_exact_anchor'] for r in targets),strict_unique_last_reference_evaluable=sum(r['strict_last_reference_evaluable'] for r in targets),any_exact_history=sum(r['annotated_history_count']>0 for r in targets),current_exact_and_any_history=sum(r['current_exact_anchor'] and r['annotated_history_count']>0 for r in targets),anchored_history_evaluable=sum(r['anchored_history_evaluable'] for r in targets),anchored_identity_present_current=sum(r['expected_identity_present_in_current_annotation'] for r in targets),target_history_status=dict(Counter(r['history_identity_status'] for r in targets)),all_rows=len(rows),all_rows_anchored_evaluable=sum(r['anchored_history_evaluable'] for r in rows))
result=dict(passed=True,summary=summary,targets=targets,rows=rows,source_sha256=sources,runtime_manifest_sha256=sha(OUT/'A_runtime_manifest_v2.json'),evaluation_units_frozen_before_repair_outcomes=True,algorithm_gets_annotations=False,nearest_or_interpolated_GT=False,new_repair_outcomes_computed=False)
with (OUT/'anchored_readiness.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(summary,ensure_ascii=False))
