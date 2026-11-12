"""Independent metadata-only leakage/whole-population audit, no repair scores."""
import json,gzip,csv,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=read(OUT/'proposed_selection_protocol_v2.json');A=read(OUT/'selected_events_A_v2.json');B=read(OUT/'selected_events_B_RESERVED_v2.json');inventory=read(OUT/'inventory_v2.json');runtime=read(OUT/'A_runtime_manifest_v2.json')
native=read(ROOT/'runs/domain_matched_native_20261109/protocol.json');fit=read(ROOT/'research_domain_matched_reader_20261109/protocol_v2.json');old=read(ROOT/'research_failure_localization_20261112/protocol.json');training=read(ROOT/'runs/train_cache/selection.json')
fitvideos={r['video'] for r in fit['edges']};oldvideos={r['video'] for r in old['selected']};trainvideos={v for vv in training['videos'].values() for v in vv};av=set(A['videos']);bv=set(B['videos']);assert len(fit['edges'])==26 and len(fitvideos)==20 and len(oldvideos)==11 and len(trainvideos)==200
assert not(av&bv or av&fitvideos or bv&fitvideos or av&oldvideos or bv&oldvideos or av&trainvideos or bv&trainvideos);assert av|bv==set(native['videos'])-oldvideos
salt=protocol['rule']['salt'];ordered=sorted(av|bv,key=lambda v:hashlib.sha256((salt+v).encode()).hexdigest());assert A['videos']==ordered[:5] and B['videos']==ordered[5:]
expected=[]
for v in A['videos']:
 with gzip.open(ROOT/f'runs/domain_matched_native_20261109/rollouts/C34/{v}.events.json.gz','rt',encoding='utf8') as f:events=json.load(f)['events']
 for e in events:
  pp=[p for p in e['proposals'] if p.get('geometry') is not None]
  for p in pp:assert p['reason'] in ['head_accept','head_refuse']
  if pp:expected.append((v+'|'+str(e['frame']),[(p['old_id'],p['raw_id']) for p in pp]))
actual=[(e['event_key'],[(p['old_id'],p['raw_id']) for p in e['proposal_keys']]) for e in A['events']];assert expected==actual;assert {e['event_key'] for e in runtime['events']}=={k for k,p in expected};assert not({k for k,p in expected}&set(inventory['explicit_prior_event_exposure']))
cohorts=defaultdict(lambda:dict(videos=set(),uids=set(),rows=0))
with gzip.open(ROOT/'runs/flyvis_exclusive_t4t5_20261107/choices.csv.gz','rt',encoding='utf8') as f:
 for r in csv.DictReader(f):
  c=cohorts[r['split']];c['videos'].add(r['video']);c['uids'].add((r['video'],r['uid']));c['rows']+=1
assert all(not((av|bv)&c['videos']) for c in cohorts.values())
summary={split:dict(contributing_videos=len(c['videos']),unique_video_uid_queries=len(c['uids']),raw_candidate_pair_rows=c['rows']) for split,c in cohorts.items()}
result=dict(passed=True,A_videos=len(av),A_events=len(A['events']),A_proposals=sum(len(pp) for key,pp in expected),B_reserved_videos=len(bv),B_metadata_events=len(B['events']),fit26_videos=len(fitvideos),earlier_cohorts=summary,detector_training_videos=len(trainvideos),prior_detailed_failure_videos=len(oldvideos),all_eligible_A_events_retained=True,no_known_prior_event_overlap=True,rule_hash_order_exact=True,annotation_or_repair_outcome_selection=False,B_repair_outcomes_read=False,scope='Independentwholepopulationmetadataaudit; native20aggregatehistoricallyseen. Correctedstrict9videoA5/B4 before anynewrepairoutcomes.',source_sha256={str((OUT/n).relative_to(ROOT)).replace('\\','/'):sha(OUT/n) for n in ['inventory_v2.json','selected_events_A_v2.json','selected_events_B_RESERVED_v2.json','proposed_selection_protocol_v2.json','A_runtime_manifest_v2.json']})
with (OUT/'leakage_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
