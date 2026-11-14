from official import *
from collections import Counter
p=read(OUT/'protocol.json');s=read(OUT/'measurements.json');r=s['records'];mat=s['assignment_matrices'];M0=np.load(OUT/'interventions.npz');parent=ROOT/'research_return_cycle_20261114';original=read(parent/'result.json');cases={q['id']:q for q in read(parent/'cases.json')['cases']};lab=zipfile.ZipFile(ROOT/p['labelzip']);cache={};count=0
for q in r:
 c=cases[q['case']];idx=c['index'];v=c['video'];frame=q['frame'];i=q['slot']
 if v not in cache:cache[v]=labels(lab,v)
 t,k,other=masks(lab,v,cache[v][idx+frame],q['id'])
 for name,met in [('original','prior_overlap'),('truepast','truepast_prior_overlap')]:
  m=M0[f"{q['case']}__{frame}__{i}__{name}"];tn,fp,fn,tp=np.bincount(t[k].astype(int)*2+m[k],minlength=4);j=tp/(tp+fp+fn) if (tp+fn)>0 else None;assert (j is None and q[met] is None) or (j is not None and abs(j-q[met])<1e-12);count+=1
for a in mat:
 A=np.asarray(a['scores']);ps=[]
 for i,line in enumerate(A):
  for j,v in enumerate(line):
   if v>.5 and (v>np.delete(line,j)).all() and (v>np.delete(A[:,j],i)).all():ps.append([i,a['rids'][j]])
 assert ps==a['actions']
bad=[q for q in r if q['status']=='wrong'];assert len(bad)==len(original['error_attribution'])==13;states=Counter(q['stage'] for q in bad);available=[q for q in bad if q['visible']];stats={}
for n in ['runtime','prior_labels_only','query_labels_only']:
 stats[n]={sp:dict(Counter(q['scores'][n]['status'] for q in r if q['split']==sp)) for sp in ['fresh','diagnostic']}
save('result.json',dict(scientific_passed=len(available)>0 and all(q['stage']!='competition_or_observation_not_resolved' for q in available),breakdown=states,errors=bad,stats=stats,checked_all_correct_and_deferred_on_same_clips=len(r)-len(bad),scope='Observed stage split and isolated interventions; replacing labels is a diagnostic ceiling, never a deployable repair. Stage cannot distinguish every upstream cause of a missing current observation.'))
save('audit.json',dict(passed=True,prior_pixel_counts_recomputed=count,independent_global_pairings=len(mat),reproduced_source_error_keys=all(any(q['case']==x['case'] and q['slot']==x['slot'] and q['frame']==x['frame'] for q in bad) for x in original['error_attribution']),result_sha256=sha(OUT/'result.json')))
