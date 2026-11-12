"""Complete fit84 provenance and sparse-label clock audit; metadata only."""
import json,gzip,csv,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
inv=read(OUT/'inventory.json');protocol=read(OUT/'proposed_selection_protocol.json');cohorts={};oldqueryuids=[]
source='runs/flyvis_exclusive_t4t5_20261107/choices.csv.gz'
with gzip.open(ROOT/source,'rt',encoding='utf8') as f:
 for r in csv.DictReader(f):
  if r['split'] not in cohorts:cohorts[r['split']]=dict(records=0,videos=set())
  cohorts[r['split']]['records']+=1;cohorts[r['split']]['videos'].add(r['video']);oldqueryuids.append({k:r[k] for k in ['split','video','uid','j','k']})
  role=['earlier_identity_'+r['split'],source]
  inv['video_exposure'].setdefault(r['video'],[])
  if role not in inv['video_exposure'][r['video']]:inv['video_exposure'][r['video']].append(role)
inv['source_sha256'][source]=sha(ROOT/source);inv['source_sha256']['research_identity_holdout_20261112/review/inventory.json']=sha(OUT/'inventory.json')
previous=read(ROOT/'research_failure_localization_20261112/protocol.json');old11=sorted({r['video'] for r in previous['selected']});strict=sorted(set(inv['canonical_native_videos'])-set(old11)-set(inv['current_head_fit']['videos']))
assert len(old11)==11 and len(strict)==9
inv['prior_failure_29_videos']=old11;inv['strict_repair_choice_video_unseen_pool']=strict;inv['earlier_all_fit_query_cohorts']={k:dict(records=v['records'],videos=len(v['videos']),video_ids=sorted(v['videos'])) for k,v in cohorts.items()};inv['prior_query_uid_bindings']=oldqueryuids
inv['correction']=dict(reason='Originalqueries.json deliberatelyincludesfit20/evaluate26only; nowdirectsourcechoices.csv.gz inventoriesfit84too. Prior29failureanalysis spans11videos, whilefivecausalrepaircases span5. Initial15pool is newevent-disjoint, notdisjointfromall11oldanalysisvideos.',original_inventory_preserved=True,all_correction_before_new_repair_outcomes=True)
save('inventory_v2.json',inv)
clockrows=[];pervideo=[]
for v in protocol['rule']['A_videos']:
 cp=f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json';ap=f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json';lp=f'data/bdd100k/labels/box_track_20/val/{v}.json';clock=read(ROOT/cp);alignment=read(ROOT/ap);labels=read(ROOT/lp)
 routed={int(k):(a['name'],int(a['frameIndex'])) for k,a in clock['anchors'].items()};full={m['video_frame_index']:(m['reference_name'],m['reference_index']) for m in alignment['matches']};assert routed==full;assert len(labels)==len(full)
 pervideo.append(dict(video=v,raw_frames=clock['frames'],clock_anchors=len(routed),full_alignment_matches=len(full),annotation_rows=len(labels),clock_is_full_alignment=True,sources={n:sha(ROOT/n) for n in [cp,ap,lp]}))
 for event in read(OUT/'selected_events_A.json')['events']:
  if event['video']!=v:continue
  for p in event['proposal_keys']:
   clockrows.append(dict(event_key=event['event_key'],video=v,frame=event['frame'],old_id=p['old_id'],raw_id=p['raw_id'],reference_frame=p['reference_frame'],current_exact_anchor=event['frame'] in routed,reference_exact_anchor=p['reference_frame'] in routed,strict_video_unseen=v in strict))
save('clock_readiness.json',dict(passed=True,proposals=clockrows,per_video=pervideo,total=len(clockrows),both_exact=sum(x['current_exact_anchor'] and x['reference_exact_anchor'] for x in clockrows),current_exact=sum(x['current_exact_anchor'] for x in clockrows),reference_exact=sum(x['reference_exact_anchor'] for x in clockrows),strict_video_unseen_proposals=sum(x['strict_video_unseen'] for x in clockrows),strict_video_unseen_both_exact=sum(x['strict_video_unseen'] and x['current_exact_anchor'] and x['reference_exact_anchor'] for x in clockrows),conclusion='ClockcoversallavailableGTframes, notsparsesubset. BDDMOTannotations are5Hz vsraw30Hz. Timestamp/namealignment cannotmanufacturenewGTbetweenanchors. CarryingtrackerIDfromlastanchor iscircularforIDqualityunlessindependentphysicalcorrespondenceverified. No nearest/interpolatedGT.',official_source='https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst',GT_identity_values_used_for_selection=False,repair_outcomes_computed=False))
salt=protocol['rule']['salt'];ordered=sorted(strict,key=lambda v:hashlib.sha256((salt+v).encode()).hexdigest());proposalA=ordered[:5];proposalB=ordered[5:]
save('strict_video_selection_proposal.json',dict(created=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),state='metadata_proposal_only_waiting_parent_scope_decision',pool=strict,prior_failure_videos=old11,rule='SAMEoriginalhashsalt/order, first5strictvideosA,next4B; no outcomesorGTidentityselection',A_videos=proposalA,B_reserved_videos=proposalB,alternatively_keep_original_A83_new_events=dict(A_videos=protocol['rule']['A_videos'],strict_video_unseen_A=[v for v in protocol['rule']['A_videos'] if v in strict],prior_video_seen_A=[v for v in protocol['rule']['A_videos'] if v in old11]),initial_8_7_manifests_preserved=True,new_repair_outcomes_read=False))
print(json.dumps(dict(fit_cohorts=inv['earlier_all_fit_query_cohorts'],prior_failure_videos=len(old11),strict_pool=len(strict),A83_both_exact=sum(x['current_exact_anchor'] and x['reference_exact_anchor'] for x in clockrows),strict_A83_both_exact=sum(x['strict_video_unseen'] and x['current_exact_anchor'] and x['reference_exact_anchor'] for x in clockrows)),ensure_ascii=False))
