from official import *
from collections import Counter
s=read(OUT/'actions.json');rr=s['records'];by={};arms=['fixed_separate','fixed_exclusive','ordinary_separate','ordinary_exclusive'];src=read(ROOT/'research_return_cycle_20261114/result.json')['records'];checked=0
for q in s['matrices']:
 m=np.asarray(q['S']);chk=[[i,rid] for i in range(2) for j,rid in enumerate(q['ids']) if m[i,j]>.5 and (m[i,j]>np.delete(m[i],j)).all() and m[i,j]>m[1-i,j]];assert chk==q['actions'];checked+=1
for q in rr:
 if q['arm']=='fixed_separate':
  x=next(x for x in src if x['case']==q['case'] and x['frame']==q['offset'] and x['slot']==q['slot'] and x['mode']=='forward');assert q['action']==x['raw_id'] and q['status']==x['status']
for sp in ['fresh','diagnostic']:
 r=[x for x in rr if x['split']==sp];stats={}
 for arm in arms:
  sub=[x for x in r if x['arm']==arm];stats[arm]={n:sum(x['status']==n for x in sub) for n in ['correct','wrong','deferred']}
 changes={}
 for arm in arms[1:]:
  vs=sorted({x['video'] for x in r});ds=np.array([np.mean([x['status']=='correct' for x in r if x['video']==v and x['arm']==arm])-np.mean([x['status']=='correct' for x in r if x['video']==v and x['arm']==arms[0]]) for v in vs]);ci=np.quantile(ds[np.random.default_rng(42).integers(len(vs),size=(10000,len(vs)))].mean(1),[.025,.975]);changes[arm]=dict(mean=float(ds.mean()),interval=ci.tolist())
 by[sp]=dict(stats=stats,changes=changes)
fixed=by['fresh']['stats'][arms[0]];new=by['fresh']['stats'][arms[-1]];inc=by['fresh']['changes'][arms[-1]]['interval'][0]>0;safe=new['wrong']<fixed['wrong'];sample=[q for q in rr if q['case']=='b1d7b3ac-0bdb47dc_149' and q['offset']==4 and q['slot']==0];oldcorrect=any(q['arm']==arms[-1] and q['status']=='correct' for q in sample);save('result.json',dict(scientific_passed=bool(inc and safe and oldcorrect),improved=bool(inc),safe=safe,hardcase=oldcorrect,stats=by,hard_example=sample,scope='2 by 2 factorial. Same SAM2 and detection heads; only temporal memory and pixel ownership readout changed. Tested reused scenes; no representation or threshold change.'))
save('audit.json',dict(passed=True,matrices_recomputed=checked,unchanged_baseline_decisions=sum(x['arm']==arms[0] for x in rr),result_sha256=sha(OUT/'result.json')))
