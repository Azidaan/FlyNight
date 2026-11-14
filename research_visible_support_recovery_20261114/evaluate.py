from official import *
from collections import Counter
a=read(OUT/'actions.json');r=a['rows'];modes=['jaccard','prior_coverage','both_coverages'];stats={};reproduced=0
for q in a['matrices']:
 ids=q['ids'];names=q['owners'];S=np.array(q['scores']);b=dict(q['reserved']);ps=[]
 for i,t in enumerate(names):
  for j,o in enumerate(ids):
   if S[i,j]>.5 and (S[i,j]>np.delete(S[i],j)).all() and S[i,j]>S[1-i,j] and b.get(o,t)==t:ps.append([t,o])
 assert ps==q['proposed'] and set(tuple(x) for x in q['baseline'])<=set(tuple(x) for x in q['combined']);reproduced+=1
for name in modes:
 rs=[q for q in r if q['name']==name];stats[name]={k:sum(q['status']==k for q in rs) for k in ['correct','wrong','deferred']}
q=a['case1176']['prior_coverage'];old=a['old'];known=q['far']==111813 and next(x for x in old if x['method']=='prior_coverage' and x['frame']==1165)['far'] is None and next(x for x in old if x['method']=='prior_coverage' and x['frame']==1171)['far']==111062;vs=sorted({x['video'] for x in r});delta=np.array([np.mean([x['status']=='correct' for x in r if x['video']==v and x['name']=='prior_coverage'])-np.mean([x['status']=='correct' for x in r if x['video']==v and x['name']=='jaccard']) for v in vs]);ci=np.quantile(delta[np.random.default_rng(42).integers(len(vs),size=(10000,len(vs)))].mean(1),[.025,.975]);safe=stats['prior_coverage']['wrong']<=stats['jaccard']['wrong'];save('result.json',dict(scientific_passed=bool(known and safe and ci[0]>0),old1176_passed=bool(known),no_extra_errors=safe,gain=bool(ci[0]>0),stats=stats,case1176=a['case1176'],old=old,mean=float(delta.mean()),interval=ci.tolist(),scope='No hidden ignore information at prediction. Use asymmetric spatial overlap as secondary observation, keep existing assignments. Known diagnostic cases and previously reused 11 videos; no complete-tracker claim.'))
save('audit.json',dict(passed=True,selection_matrices_checked=reproduced,result_sha256=sha(OUT/'result.json')))
