"""Freeze corrected video-disjoint split before new repair outcomes."""
import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
inv=read(OUT/'inventory_v2.json');strict=read(OUT/'strict_video_selection_proposal.json');old=read(OUT/'proposed_selection_protocol.json');all_events=read(OUT/'selected_events_A.json')['events']+read(OUT/'selected_events_B_RESERVED.json')['events'];counts={}
for name,vids in [('A',strict['A_videos']),('B_RESERVED',strict['B_reserved_videos'])]:
 rows=sorted([e for e in all_events if e['video'] in vids],key=lambda e:(vids.index(e['video']),e['frame']));assert all(e['event_key'] not in inv['explicit_prior_event_exposure'] for e in rows);assert not(set(vids)&set(inv['prior_failure_29_videos']))
 counts[name]=dict(videos=len(vids),events=len(rows),proposals=sum(e['eligible_proposals'] for e in rows))
 save('selected_events_'+name+'_v2.json',dict(split=name,events=rows,videos=vids,eligibility='ALLnativegeometrypresentproposals/bothheadsigns; noGTselection oroutcomecap.',scope='VideodisjointALL11priorfailureanalysisvideos,ALL5repairchoicevideos,ALLtraining/fittingcohorts; historicalnativeaggregate20seen, NOTgloballyresearcherunseenprojectvideos.',prior_8_7_draft_preserved_unexecuted=True,new_repair_outcomes_read=False))
p=dict(old);p['prepared']=datetime.now(ZoneInfo('Europe/Moscow')).isoformat();p['inventory_sha256']=sha(OUT/'inventory_v2.json');p['rule']=dict(old['rule'],A_videos=strict['A_videos'],B_reserved_videos=strict['B_reserved_videos'],exclude='ALL11priorfailureanalysisvideos(and5repairchoicecontained),ALLtraining/fittingvideos,ALLexplicitpreviouslyexaminedevents',scope='Video-disjointpriorrepairandfailurechoice, historicalnativecohortaggregateviewed; notunseenprojectfinalvideoevaluation');p['population_counts']=counts;p['correction']='Beforeanynewrepairoutcomes: initial8/7draftexcludedonly5causalchoicevideos; exposureaudithas11failureanalysisvideos. Parentchoosesstrict9poolA5/B4 withSAMEhashsalt. Originaldraftspreserved.';p['annotation_evaluation_units']=dict(primary='CurrentexactGTframe+exactlastreferenceGTframe, ifavailable; otherwiseunknown.',separate_anchored_history='ExpectedpersistentID from>=1exactannotatedhistoricalrowobservation, allrecentannotatedhistoryobservations uniqueIoU>=.5 sameID. Currentproposal evaluatedonlyoncurrentexactGTframe. DoesNOTassignlabeltolastunannotatedcrop. Unknown/ambiguous/inconsistenthistoryseparate; contaminationafteranchorcountsashistory/systemerror.',algorithm_annotation_inputs=False,nearest_or_interpolated_GT=False);p['source_sha256'].update({str((OUT/n).relative_to(ROOT)).replace('\\','/'):sha(OUT/n) for n in ['inventory_v2.json','strict_video_selection_proposal.json','selected_events_A_v2.json','selected_events_B_RESERVED_v2.json','proposed_selection_protocol.json']})
save('proposed_selection_protocol_v2.json',p);print(json.dumps(dict(counts=counts,A=p['rule']['A_videos'],B=p['rule']['B_reserved_videos'],sha256=sha(OUT/'proposed_selection_protocol_v2.json'))))
