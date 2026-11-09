"""Five preregistered paired contrasts, with exact official metric pooling."""
from evaluation_common import *
start(1800)
import gzip
import numpy as np
p=read(OUT/'protocol.json');phase=read(OUT/'evaluation_protocol.json');verify(phase)
assert read(OUT/'metrics_complete.json')['passed']
videos=list(p['videos']);n=len(videos)
folders={name:ROOT/path for name,path in phase['reused_metrics'].items()}
folders.update({a:OUT/'metrics'/a for a in p['arms']});arms=list(folders)
def pooled(c,w):
    tp=w@c['TP'];fp=w@c['FP'];fn=w@c['FN'];aa=w@c['AA']
    assa=aa/np.maximum(1,tp);deta=tp/np.maximum(1,tp+fp+fn)
    return dict(HOTA=100*np.sqrt(deta*assa).mean(-1),AssA=100*assa.mean(-1),DetA=100*deta.mean(-1))
rng=np.random.default_rng(42);indices=rng.integers(0,n,(20000,n));counts=np.stack([np.bincount(row,minlength=n) for row in indices])
point={};samples={};all_metrics={}
for arm,folder in folders.items():
    data=[read(folder/(v+'.json')) for v in videos];assert all(r['passed'] for r in data)
    for v,r in zip(videos,data):assert sha(folder/(v+'.details.json.gz'))==r['details_sha256']
    c={k:np.array([r['components'][k] for r in data],float) for k in ['TP','FP','FN','AA']}
    point[arm]={k:float(v[0]) for k,v in pooled(c,np.ones((1,n))).items()}
    ids={k:sum(r['components'][k] for r in data) for k in ['IDTP','IDFP','IDFN']}
    point[arm]['IDF1']=100*ids['IDTP']/max(1.,ids['IDTP']+.5*ids['IDFP']+.5*ids['IDFN'])
    for k in ['AssA','IDF1']:point[arm]['mean_video_'+k]=float(np.mean([r['metrics'][k] for r in data]))
    for k in ['IDSw','CLR_FP','CLR_FN']:point[arm][k]=int(sum(r['metrics'][k] for r in data))
    all_metrics[arm]={v:r['metrics'] for v,r in zip(videos,data)};samples[arm]=pooled(c,counts)['HOTA']
def compare(a,b):
    delta=samples[a]-samples[b]
    harm=dict(AssA=point[a]['AssA']-point[b]['AssA'],IDF1=point[a]['IDF1']-point[b]['IDF1'],
        AssA_mean=point[a]['mean_video_AssA']-point[b]['mean_video_AssA'],IDF1_mean=point[a]['mean_video_IDF1']-point[b]['mean_video_IDF1'],
        IDSw=point[a]['IDSw']-point[b]['IDSw'],FP=point[a]['CLR_FP']-point[b]['CLR_FP'],FN=point[a]['CLR_FN']-point[b]['CLR_FN'])
    return dict(delta_pooled_HOTA=point[a]['HOTA']-point[b]['HOTA'],ci95=np.percentile(delta,[2.5,97.5]),ci99_primary_family=np.percentile(delta,[.5,99.5]),harm=harm,
        harm_passed=all(harm[k]>=0 for k in ['AssA','IDF1','AssA_mean','IDF1_mean']) and all(harm[k]<=0 for k in ['IDSw','FP','FN']))
comparisons={a+'-'+b:compare(a,b) for a,b in phase['primary_comparisons']}
secondary={a+'-'+b:compare(a,b) for a,b in phase['secondary_comparisons']}
accepted=all(c['ci99_primary_family'][0]>0 and c['harm_passed'] for c in comparisons.values())
from trackeval.metrics.hota import HOTA
from evaluation.hota_eval import VALID
official=HOTA();max_error=0.;checks=0
for arm,folder in folders.items():
    detail=[]
    for v in videos:
        with gzip.open(folder/(v+'.details.json.gz'),'rt',encoding='utf8') as f:d=json.load(f)[v]
        detail.append({c:{k:np.asarray(x) if isinstance(x,list) else x for k,x in d[c]['HOTA'].items()} for c in VALID if c in d})
    for index,selected in enumerate([np.arange(n),*indices[:20]]):
        cls={}
        for c in VALID:
            seq={str(i):detail[int(j)][c] for i,j in enumerate(selected) if c in detail[int(j)]}
            if seq:cls[c]=official.combine_sequences(seq)
        value=float(official.combine_classes_det_averaged(cls)['HOTA'].mean()*100)
        expected=point[arm]['HOTA'] if index==0 else float(samples[arm][index-1]);error=abs(value-expected)
        assert error<1e-9;max_error=max(max_error,error);checks+=1
verify(phase)
write('comparison_report.json',dict(passed=True,pooled=point,per_video=all_metrics,comparisons=comparisons,secondary=secondary,
    accepted_flyvis_system_and_increment=accepted,development_validation=True,independent_final_test=False,
    bootstrap=dict(samples=20000,seed=42,unit='paired_video',family=5,primary_ci=99,official_pooling_checks=checks,max_error=max_error),
    evaluation_sha256=sha(OUT/'evaluation_protocol.json'),limits='Twenty historically studied development-val videos. A local permission is not an emitted old ID. No post-outcome changes of heads,thresholds,alpha,dtype or selected videos.'))
status('statistics_complete',accepted=accepted);print('POOLED',json.dumps(cast(point)),flush=True);print('ACCEPTED',accepted,flush=True)
