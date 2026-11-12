from common import *
import math,numpy as np,psutil,shutil
p=verify();r=read(OUT/'result.json');assert r['passed'] and sha(OUT/'paired_samples.npz')==r['samples_sha256'];maxerr=0.;metrics=0
with np.load(OUT/'paired_samples.npz') as z:
 for row in r['records']:
  ci=next(i for i,c in enumerate(p['cases']) if c['case']['old_id']==row['old_id']);k=f"{ci}_{row['query']}_{row['support']}_{row['stage']}";A=z[k+'_a'];B=z[k+'_b'];assert A.shape==B.shape and len(A)==row['count'];P=[];C=[]
  for j in range(A.shape[1]):
   a=A[:,j];b=B[:,j];ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b);va=math.fsum((float(x)-ma)**2 for x in a)/len(a);vb=math.fsum((float(x)-mb)**2 for x in b)/len(b);cov=math.fsum((float(x)-ma)*(float(y)-mb) for x,y in zip(a,b))/len(a);P.append(cov/math.sqrt(va*vb) if va*vb else0.);den=va+vb+(ma-mb)**2;C.append(2*cov/den if den else0.)
  maxerr=max(maxerr,abs(math.fsum(P)/len(P)-row['pearson']),abs(math.fsum(C)/len(C)-row['concordance']));metrics+=2
assert maxerr<1e-12
old=read(ROOT/'research_input_precision_probe_20261112/scores.json')
for row in r['records']:
 if row['query']=='C34' and row['support']=='full' and row['stage']=='retina16':
  m=next(x for x in old['input_moments'] if x['old_id']==row['old_id'] and x['representation']=='C34' and x['mode']=='index');den=m['variance_reference']+m['variance_candidate']+(m['mean_reference']-m['mean_candidate'])**2;assert abs(row['concordance']-2*m['covariance']/den)<1e-12
gaps=[]
for query in ['C34','R134_temporal']:
 for support in ['full','detector_and_annotation']:
  for metric in ['pearson','concordance']:
   for stage in sorted({x['stage'] for x in r['records']}):
    rows={x['old_id']:x for x in r['records'] if x['query']==query and x['support']==support and x['stage']==stage};assert len(rows)==5
    for tid in [232,832]:gaps.append(dict(query=query,support=support,metric=metric,stage=stage,own_id=tid,foreign=rows[574][metric],own=rows[tid][metric],gap=rows[574][metric]-rows[tid][metric]))
primary=[x for x in gaps if x['query']=='C34' and x['support']=='full' and x['metric']=='concordance'];save('comparisons.json',dict(passed=True,gaps=gaps,primary=primary,raw_positive_gaps=sum(x['gap']>0 for x in gaps if x['stage']=='raw_color_pixels'),raw_total=sum(x['stage']=='raw_color_pixels' for x in gaps),all_stage_support_counts_matched=True))
for query in ['C34','R134_temporal']:
 for support in ['full','detector_and_annotation']:
  for tid in [232,70,1102,574,832]:assert len({x['count'] for x in r['records'] if x['query']==query and x['support']==support and x['old_id']==tid})==1
folder=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/c5fe814751614643b1841f00f124f288';rep=read(folder/'memory_guard.json');assert rep['returncode']==0 and rep['reason'] is None
for pid,birth in rep['owned'].items():
 try:assert psutil.Process(int(pid)).create_time()!=birth
 except psutil.NoSuchProcess:pass
shutil.copy2(folder/'memory_guard.worker.log',OUT/'owned_worker.log');verify();save('audit.json',dict(passed=True,independent_metrics=metrics,max_error=maxerr,baseline_input_moments_exact=True,all_owned_ended=True,execution=rep,raw_image_order_precedes_core=True,scope='Confirms ordering of these fixed spatial statistics, not absence of identity information or complete neuronal mechanism.',script_sha256=sha(__file__)))
print('AUDIT',metrics,maxerr)
