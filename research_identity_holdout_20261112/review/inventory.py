"""Leakage and selection metadata only; never evaluate a new repair outcome."""
import json,re,hashlib,gzip,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PAT=re.compile(r'^[0-9a-f]{8}-[0-9a-f]{8}$')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def video_records(x):
 if isinstance(x,dict):
  if isinstance(x.get('video'),str) and PAT.fullmatch(x['video']):yield x
  for k,v in x.items():
   if k not in ['sources','inputs','snapshots','source_sha256','old_variants','channels','scores']:yield from video_records(v)
 elif isinstance(x,list):
  for v in x:yield from video_records(v)
sources={};exposures={};events={}
def expose(video,role,source,event=None):
 exposures.setdefault(video,[]).append(dict(role=role,source=source))
 if event:events.setdefault(video+'|'+str(event['frame']),[]).append(dict(role=role,source=source,**event))
def take(name,role,fields=None):
 p=ROOT/name
 if not p.exists():return None
 d=read(p);sources[name]=sha(p);nodes=[d] if fields is None else [d[k] for k in fields if k in d]
 for node in nodes:
  for r in video_records(node):
   event={k:r[k] for k in ['frame','old_id','raw_id','tid','reference_frame'] if k in r}
   expose(r['video'],role,name,event if 'frame' in event else None)
 return d
train=take('research_domain_matched_reader_20261109/protocol_v2.json','current_identity_head_fit',['edges']);fitvideos={r['video'] for r in train['edges']};assert len(train['edges'])==26 and len(fitvideos)==20
agreement=take('research_agreement_readout_20261112/protocol.json','agreement_train_fit',['train']);take('research_agreement_readout_20261112/protocol.json','agreement_screen_choice',['screen'])
queries=take('research_identity_assignment_protocol_20261108/queries.json','earlier_identity_fit_or_evaluation',['rows']);querycohorts={}
for r in queries['rows']:querycohorts.setdefault(r['split'],set()).add(r['video'])
for name in ['runs/identity_channel_head_20261109/protocol.json','runs/identity_channel_training_readiness_20261109/protocol.json']:
 d=take(name,'earlier_identity_training_or_readiness',['cases'])
 for v in d.get('videos',[]):expose(v,'earlier_identity_training_or_readiness',name)
det=take('runs/train_cache/selection.json','detector_training_cache',[])
for split,vids in det['videos'].items():
 for v in vids:expose(v,'detector_training_cache_'+split,'runs/train_cache/selection.json')
oldcases=take('research_stage_localization_20261112/protocol.json','causal_screen_five',['cases']);screenvideos={c['case']['video'] for c in oldcases['cases']}
take('research_failure_localization_20261112/protocol.json','failure_analysis_selected_29',['selected'])
probe_sources=[]
for folder in ROOT.glob('research_*20261112'):
 if any(tag in folder.name for tag in ['probe','localization','agreement','weekly_causal','id_birth','id_state','detector_output']):
  p=folder/'protocol.json'
  if p.exists():
   name=p.relative_to(ROOT).as_posix();take(name,'prior_causal_or_identity_screen',['cases','screen','selected','edges','targets']);probe_sources.append(name)
native=take('runs/domain_matched_native_20261109/protocol.json','historical_native_validation',[]);nativevideos=set(native['videos'])
for v in nativevideos:expose(v,'historical_native_validation_aggregate_and_baseline','runs/domain_matched_native_20261109/protocol.json')
assignment=take('research_identity_assignment_protocol_20261108/protocol_v2.json','cohort_provenance',[])
assert assignment['cohorts']['future_val20']['researcher_unseen'] is False
assert not fitvideos&nativevideos
# Filename-only archive enumeration, including ignored immutable run artifacts.
listed=subprocess.check_output(['rg','--files','--no-ignore','runs','-g','*.events.json.gz'],cwd=ROOT,text=True).splitlines();archivebyvideo={}
for name in listed:
 v=Path(name).name.removesuffix('.events.json.gz')
 if PAT.fullmatch(v):archivebyvideo.setdefault(v,[]).append(name.replace('\\','/'))
localval={p.stem for p in (ROOT/'data/bdd100k/videos/val').glob('*.mov')};archivevideos=set(archivebyvideo);unseenarchive=archivevideos-nativevideos-set(exposures)
pool=sorted(nativevideos-screenvideos-fitvideos);assert len(pool)==15
salt='identity-repair-independent-video-split-v1|20261112|'
ordered=sorted(pool,key=lambda v:hashlib.sha256((salt+v).encode()).hexdigest());A=ordered[:8];B=ordered[8:]
selected={};counts={};eventmetadata={}
for split,vids in [('A',A),('B_reserved_untouched',B)]:
 rows=[];pervideo={}
 for v in vids:
  source=f'runs/domain_matched_native_20261109/rollouts/C34/{v}.events.json.gz';sources[source]=sha(ROOT/source)
  with gzip.open(ROOT/source,'rt',encoding='utf8') as f:archive=json.load(f)
  nprops=0;nevents=0
  for event in archive['events']:
   eligible=[p for p in event['proposals'] if p.get('geometry') is not None]
   for p in eligible:assert p['reason'] in ['head_accept','head_refuse']
   if not eligible:continue
   key=v+'|'+str(event['frame']);assert key not in events,'Prior explicitly examined event in chosennewvideo'
   nevents+=1;nprops+=len(eligible)
   rows.append(dict(event_key=key,video=v,frame=event['frame'],event_archive=source,proposal_keys=[dict(old_id=p['old_id'],raw_id=p['raw_id'],row=p['row'],column=p['column'],reference_frame=p['reference_frame']) for p in eligible],eligible_proposals=len(eligible),saved_rows=len(event['rows']),saved_high_columns=len(event['high_raw_ids'])))
  pervideo[v]=dict(eligible_events=nevents,eligible_proposals=nprops)
 selected[split]=rows;counts[split]=dict(videos=len(vids),events=len(rows),proposals=sum(r['eligible_proposals'] for r in rows),per_video=pervideo)
 role='new_for_repair_choice_within_historically_seen_validation_video'
 save('selected_events_'+('A' if split=='A' else 'B_RESERVED')+'.json',dict(split=split,events=rows,eligibility='ALLoriginalnativeproposals with geometrypresent; bothheadsigns included, noGTselection.',scope=role,repair_outcomes_read=False))
now=datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
save('inventory.json',dict(prepared=now,source_sha256=sources,video_exposure={v:sorted({(r['role'],r['source']) for r in refs}) for v,refs in sorted(exposures.items())},explicit_prior_event_exposure=events,current_head_fit=dict(edges=26,videos=sorted(fitvideos)),earlier_query_cohorts={k:dict(videos=len(v),ids=sorted(v)) for k,v in querycohorts.items()},prior_five_screen_videos=sorted(screenvideos),probe_protocols=probe_sources,canonical_native_videos=sorted(nativevideos),local_val_count=len(localval),all_existing_event_archive_videos=sorted(archivevideos),archive_paths_by_video=archivebyvideo,archive_videos_outside_canonical_native=sorted(archivevideos-nativevideos),unexposed_existing_archive_videos=sorted(unseenarchive),local_val_outside_native20=sorted(localval-nativevideos),split_counts=counts,scope='Currentidentitydevelopment leakageaudit, notproof ofallpossibleexternalpretrainingimages or completehistoricprojectexposure. Existingnative20explicitlyresearcher_unseen:false; splitvideos are disjointfrom5repairchoicevideos and26edge20trainfit, buthistoricalnativeaggregatewasviewed.',new_repair_outcomes_read=False))
production={p.relative_to(ROOT).as_posix():sha(p) for pattern in ['identity_repair*.py','appearance_recovery_policy.py'] for p in (ROOT/'tracking').glob(pattern)}
save('proposed_selection_protocol.json',dict(prepared=now,inventory_sha256=sha(OUT/'inventory.json'),source_sha256=sources,frozen_component_sha256=production,rule=dict(unit='wholevideo',salt=salt,order='SHA256(UTF8(salt+video)) lexicographicascending',exclude='ALL5repairchoicevideos,ALL26edge20trainingvideos,ALLexplicitpreviouslyexaminedevents; native20historicaggregateexposuredisclosed',A_videos=A,B_reserved_videos=B,eligible='Alloriginalnativeproposals with geometrypresent, bothhead_accept/head_refuse signs; allcompetitorrows retained; noannotation/availability/prediction-scorefilter',scope='Newrepair-choice video/event test on previouslyusedvalidationcohort, NOTfullyunseenprojectvideo finaltest'),population_counts=counts,budget='EvaluateALLeligibleAevents/proposals; nooutcome-dependentcap ordiscard. Ifdeadline preventscompletion, reportincompleteandallunprocessedkeys; do notpromotepartialsample. Bmetadataonly, noCensus/MASA/repairoutcomes until separatelyfrozenpostfailurevalidation.',evaluation='Exactexistingrawframe annotationclock only; unavailablepast/currentannotation remainsunknown. Neverremoveunknownfromattemptdenominator; explicitcorrect/wrong/unknown actioncounts, protectedownership, dupclaims, completeLAPmatrixchecks; eventwiseANDper-videopairedresults. NoGTselection. Baselineandmatchedappearancewithoutnewcorecontrol mandatory.',failure_policy='PreserveAfrozenholdoutresult. AnyrepairafterAerror becomesnewdevelopmentusingA; registerchangesseparately beforeB, keepBuntouched. No repeatedB-tuning; afterBfailure requiresanotheruntouchedpopulation or honestunresolvedstatus.',deadline='2026-11-13T07:35:00+03:00',new_model_calls=0,tracker_calls=0))
print(json.dumps(dict(head_fit_videos=len(fitvideos),prior_screen_videos=len(screenvideos),explicit_prior_events=len(events),all_archive_videos=len(archivevideos),unexposed_archives=len(unseenarchive),pool=len(pool),A=counts['A'],B=counts['B_reserved_untouched']),ensure_ascii=False))
