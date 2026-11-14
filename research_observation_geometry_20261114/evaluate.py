from official import *
from itertools import product
p=read(OUT/'protocol.json');s=read(OUT/'scored_candidates.json');segments=np.load(OUT/'best_masks.npz');ref=ROOT/'research_missing_candidate_repair_20261114';cc={c['id']:c for c in read(ref/'cases.json')['cases']};z=zipfile.ZipFile(ROOT/p['labelzip']);db={};n=0;checked=0
for q in s['pairs']:
 for name in ['mask','box']:
  a=np.array(q[name+'_scores']);sel=q[name+'_assign'];v=sum(1+a[i,j] for i,j in enumerate(sel) if j is not None)
  eligible=[[None]+[j for j,x in enumerate(row) if x>=.5] for row in a];mx=max(sum(1+a[i,j] for i,j in enumerate(js) if j is not None) for js in product(*eligible) if js[0] is None or js[0]!=js[1]);assert abs(mx-v)<1e-10;checked+=1
for r in s['records']:
 if not r['initial_missed'] or r['best_mask_idx'] is None:continue
 c=cc[r['case']];v=c['video'];idx=c['index']
 if v not in db:db[v]=labels(z,v)
 t,k,_=masks(z,v,db[v][idx],r['identity']);mm=segments[f"{r['case']}__{r['slot']}__{r['arm']}"]&k;inter=int((t&mm).sum());union=int((t|mm).sum());assert abs(r['best_mask_iou']-inter/union)<1e-9;n+=1
full=[q for q in s['records'] if q['arm']=='full'];stats={}
for split in ['familiar','fresh']:
 rr=[q for q in full if q['split']==split];vs=sorted({q['video'] for q in rr});missing=[q for q in rr if q['initial_missed']];extra=sum(x['conditional_extra_owner'] is not None for x in missing);d=np.array([np.mean([float(x['conditional_extra_owner'] is not None) for x in rr if x['video']==v]) for v in vs]);lo,hi=np.quantile(d[np.random.default_rng(42).integers(len(vs),size=(10000,len(vs)))].mean(1),[.025,.975]);stats[split]=dict(total=len(rr),missing=len(missing),maximum_single_mask_available=sum(r['best_mask_iou']>=.5 for r in missing),one_to_one_extra=extra,video_equal_delta=d.mean(),interval=[float(lo),float(hi)],box_claims_failed_mask_quality=sum(q['assigned_by_box'] is not None and q['mask_quality_of_box_choice']<.5 for q in rr),mask_and_box_disagree=sum(q['assigned_by_box']!=q['assigned_by_mask'] for q in rr),sufficient_reassigned=sum(q['assigned_by_mask'] is not None for q in rr),videos=len(vs))
diagnoses=[]
for r in [q for q in full if q['initial_missed']]:
 rr=[q for q in s['records'] if q['case']==r['case'] and q['slot']==r['slot']];d={q['arm']:q for q in rr}
 reason='ready_original_car_mask' if r['conditional_extra_owner'] is not None else 'original_car_mask_blocked_by_owner' if r['best_mask_iou']>=.5 else 'only_tiled_car_mask' if d['tile']['best_mask_iou']>=.5 else 'only_other_class_mask' if max(d['fullall']['best_mask_iou'],d['tileall']['best_mask_iou'])>=.5 else 'no_sufficient_mask_in_tested_outputs'
 diagnoses.append(dict(case=r['case'],split=r['split'],slot=r['slot'],official_id=r['identity'],reason=reason,box_annotated_tight=r['visible_tight_box_iou'],by_arm={q['arm']:dict(max_box=q['best_box_iou'],max_mask=q['best_mask_iou'],chosen=q['best_mask_source_box']) for q in rr}))
f=stats['fresh'];save('result.json',dict(scientific_passed=bool(f['one_to_one_extra']>0 and f['interval'][0]>0),stats=stats,diagnoses=diagnoses,scope='Independent proposals but privileged target matching via query annotations; measured observation availability, not a working association policy. No oracle matching deployed.'))
save('audit.json',dict(passed=True,independent_all_global_optima=checked,independent_best_mask_pixel_counts=n,result_sha256=sha(OUT/'result.json'),scores_sha256=sha(OUT/'scored_candidates.json')))
