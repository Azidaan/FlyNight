from official import *
from collections import Counter
s=read(OUT/'actions.json');vec=np.load(OUT/'vectors.npz');out={};episodes=[];n=0
for q in s['traces']:
 past=vec[q['case']+'__past'].astype('float64');cur=vec[f"{q['case']}__{q['frame']}"].astype('float64');v=np.einsum('ij,kj->ik',past,cur);assert np.allclose(v,np.asarray(q['cos']),rtol=0,atol=1e-12);n+=1
for m in ['mask','appearance','fill_missing']:
 r=[x for x in s['records'] if x['mode']==m and x['slot']==0];seq=[]
 for c in sorted({x['case'] for x in r}):
  sub=[x for x in r if x['case']==c];a=next(x for x in sub if x['return_frame']);gap=[x['status'] for x in sub if not x['return_frame']];success=all(x=='wait' for x in gap) and a['status']=='correct';q=dict(case=c,video=a['video'],mode=m,success=success,wrong_during=gap.count('wrong'),return_status=a['status']);seq.append(q)
 episodes+=seq;out[m]=dict(complete=sum(q['success'] for q in seq),episodes=len(seq),wrong_during=sum(q['wrong_during'] for q in seq),correct_return=sum(q['return_status']=='correct' for q in seq),wrong_return=sum(q['return_status']=='wrong' for q in seq),waiting_return=sum(q['return_status']=='wait' for q in seq))
vs=sorted({a['case'] for a in episodes});delta=np.array([int(next(q['success'] for q in episodes if q['case']==c and q['mode']=='fill_missing'))-int(next(q['success'] for q in episodes if q['case']==c and q['mode']=='mask')) for c in vs]);ci=np.quantile(delta[np.random.default_rng(42).integers(len(delta),size=(10000,len(delta)))].mean(1),[.025,.975]);safe=out['fill_missing']['wrong_during']<=out['mask']['wrong_during'] and out['fill_missing']['wrong_return']<=out['mask']['wrong_return'];gain=ci[0]>0;rank=[q for q in s['ranks'] if q['slot']==0 and q['source_available'] and q['admissible_correct_ids']];save('result.json',dict(scientific_passed=bool(safe and gain),gain=bool(gain),safe=bool(safe),stats=out,episodes=episodes,paired_complete_delta=float(delta.mean()),interval=ci.tolist(),rank_one=sum(q['first_correct_rank']==1 for q in rank),rankable=len(rank),scope='Fixed appearance control on reused natural-gap episodes; identity conditioned at source. No tuning, no counterfactual tracker outputs.'))
save('audit.json',dict(passed=True,independent_cosine_matrix_count=n,result_sha256=sha(OUT/'result.json')))
