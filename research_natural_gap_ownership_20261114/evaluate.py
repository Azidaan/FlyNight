from official import *
from collections import Counter
p=read(OUT/'protocol.json');cs=read(OUT/'cases.json');actions=read(OUT/'actions.json');rs=actions['records'];arr=np.load(OUT/'predictions.npz');lab=zipfile.ZipFile(ROOT/p['labelzip']);stats={};checked=0
for tr in actions['traces']:
 for n,ms in tr['scores'].items():
  a=np.asarray(ms);pairs=[]
  for i,row in enumerate(a):
   for j,x in enumerate(row):
    if x>.5 and (x>np.delete(row,j)).all() and (x>np.delete(a[:,j],i)).all():pairs.append([i,tr['rids'][j]])
  assert tr['actions'][n]==pairs;checked+=1
for c in cs['cases']:
 r0=[r for r in rs if r['case']==c['id'] and r['method']=='mask' and r['slot']==0];db=labels(lab,c['video'])
 for r in r0:
  if r['action'] is None and not r['first_return']:continue
  m=arr[f"{c['id']}__{r['frame']}__chosen"];t,k,o=masks(lab,c['video'],db[r['frame']+c['prior']],c['target']);me=measure(m,t,k,o)
  if r['action'] is not None and t.any():assert abs(r['target_iou']-me['iou'])<1e-9
per=[]
for n in ['past_box','predicted_box','mask']:
 a=[r for r in rs if r['method']==n and r['slot']==0];episodes=[]
 for c in cs['cases']:
  gr=[x for x in a if x['case']==c['id']];ret=next(x for x in gr if x['first_return']);gap=[x for x in gr if not x['first_return']];during=Counter(x['status'] for x in gap);verified=all(x['status']=='wait' for x in gap) and not any(x['ignored'] for x in gap);guard=verified and ret['status']=='correct';episodes.append(dict(video=c['video'],case=c['id'],method=n,gap= dict(during),return_status=ret['status'],return_available=ret['available_current'],source_available=ret['prior_available'],no_output_all_scorable_gap=verified,full_sequence_pass=guard))
 per.extend(episodes);stats[n]=dict(episodes=len(episodes),complete_pass=sum(x['full_sequence_pass'] for x in episodes),return_correct=sum(x['return_status']=='correct' for x in episodes),return_wrong=sum(x['return_status']=='wrong' for x in episodes),return_wait=sum(x['return_status']=='wait' for x in episodes),during_wrong=sum(x['gap'].get('wrong',0) for x in episodes),during_wait=sum(x['gap'].get('wait',0) for x in episodes),during_unknown=sum(x['gap'].get('unknown_ignore',0) for x in episodes),no_source=sum(not x['source_available'] for x in episodes),no_return_observation=sum(not x['return_available'] for x in episodes))
ds={}
for n in ['past_box','predicted_box']:
 vs=sorted({x['video'] for x in per});vol=[sum(x['full_sequence_pass'] for x in per if x['method']=='mask' and x['video']==v)-sum(x['full_sequence_pass'] for x in per if x['method']==n and x['video']==v) for v in vs]
 if vs:
  delta=np.array(vol);ci=np.quantile(delta[np.random.default_rng(42).integers(len(vs),size=(10000,len(vs)))].mean(1),[.025,.975]);ds[n]=dict(mean=float(delta.mean()),interval=ci.tolist())
 else:ds[n]=dict(mean=None,interval=[None,None])
sz=stats['mask'];gate1=sz['episodes']>=8;gate2=gate1 and sz['complete_pass']>=sz['episodes']/2 and sz['during_wrong']==0;gate3=gate1 and all(q['interval'][0]>0 for q in ds.values());save('result.json',dict(scientific_passed=bool(gate1 and gate2 and gate3),availability_passed=gate1,safety_passed=gate2,increment_passed=gate3,stats=stats,episodes=per,differences=ds,candidate_episodes=len(cs['candidate_events']),videos_scanned=len(cs['availability']),scope='Natural gaps in official segmentation ID, not proof of physical disappearance. Initial association conditioned on past official box. No full tracker or use of future masks for inference.'))
save('audit.json',dict(passed=True,decisions_recomputed=checked,source_cases=len(cs['cases']),result_sha256=sha(OUT/'result.json')))
