from official import *
p=read(OUT/'protocol.json');pairs=read(OUT/'pairs.json')['cases'];out=read(OUT/'choices.json')['records'];auto=np.load(OUT/'auto_candidates.npz');oracle=np.load(OUT/'oracle_candidates.npz');lab=zipfile.ZipFile(ROOT/p['labelzip']);rows=[];fields=['RGB','Depth','Y','Geom','RGB+Depth','RGB+Y','rank_RGB+Depth','rank_RGB+Y'];cache={}
def mask_iou(a,b,k):
 u=((a|b)&k).sum();return float((a&b&k).sum()/u) if u else 0.
for d in out:
 c=next(c for c in pairs if c['id']==d['case']);cid=c['id'];v=c['video'];off=d['offset'];j=d['shuffled'];branch=d['branch'];z=auto if branch=='auto' else oracle
 if v not in cache:cache[v]=labels(lab,v)
 target=[oracle[f'{cid}__{i}__{off}'] for i in [0,1]];m=[z[f'{cid}__{i}__{off}'] for i in [j,1-j]];k=(raster(lab,v,cache[v][c['index']+off])[:,:,1]&3)==0;visible=all(t.any() for t in target);J=np.array([[mask_iou(a,b,k) for b in m] for a in target]);valid=[int(min(J[0,swap],J[1,1-swap])>=.5) for swap in [0,1]];answer=valid.index(1) if sum(valid)==1 else None;flags={}
 for source in fields:
  choice=d['decisions'].get(source);flags[source]=bool(answer is not None and choice is not None and choice==answer) if visible else None
 rows.append(dict(case=cid,video=v,offset=off,branch=branch,both_visible=bool(visible),mask_match=J.tolist(),matched_permutation=answer,correct=flags,reason='missing_truth' if not visible else 'missing_reference_or_candidate' if d['missing_candidate_or_reference'] else 'no_unambiguous_pair_above_half_mask_IoU' if answer is None else 'pairs_evaluable'))
def stats(branch):
 subset=[r for r in rows if r['branch']==branch and r['both_visible']];videos=sorted({r['video'] for r in subset});mean={v:{s:float(np.mean([r['correct'][s] for r in subset if r['video']==v])) for s in fields} for v in videos};tv={s:float(np.mean([x[s] for x in mean.values()])) for s in fields} if videos else {};status={reason:sum(r['reason']==reason for r in rows if r['branch']==branch) for reason in sorted({r['reason'] for r in rows})}
 if not videos:return dict(usable=0,eligible=0,means={},video_equal={},reasons=status)
 arr=np.array([mean[v]['RGB+Depth']-mean[v]['RGB'] for v in videos]);idx=np.random.default_rng(42).integers(len(videos),size=(10000,len(videos)));lo,hi=np.quantile(arr[idx].mean(1),[.025,.975]);return dict(usable=len(videos),eligible=len(subset),means=mean,video_equal=tv,reasons=status,delta_over_RGB=float(arr.mean()),paired_interval=[float(lo),float(hi)],correct_counts={s:int(sum(r['correct'][s] for r in subset)) for s in fields})
aa=stats('auto');oo=stats('oracle');ok=aa['usable']>=6 and aa['paired_interval'][0]>0 and aa['video_equal']['RGB+Depth']>max(aa['video_equal'][x] for x in ['RGB','RGB+Y','Geom'])
save('result.json',dict(records=rows,auto=aa,oracle=oo,scientific_passed=bool(ok),scope='Two-object conditional global assignment. Only both-visible intervals enter main denominator. Missing/unusable automatic pair = no correct association. Oracle masks separate. Comparing inferred depth stability does not validate the sign of physical depth or evaluate tracking metrics.'))
print('DEPTH_GLOBAL',aa,oo,flush=True)
