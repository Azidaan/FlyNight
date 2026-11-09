"""Unchanged official TrackEval on new outputs; reuse both ready controls."""
from common import *
start('cpu',timeout=7200)
import gzip,time
import numpy as np
from evaluation.hota_eval import evaluate_hota,VALID
p=read(OUT/'protocol.json');phase=read(OUT/'evaluation_protocol.json');verify(phase)
assert all(read(OUT/(a+'_complete.json'))['passed'] for a in p['arms'])
for arm in p['arms']:
    for i,video in enumerate(p['videos']):
        folder=OUT/'metrics'/arm;folder.mkdir(parents=True,exist_ok=True)
        marker=folder/(video+'.json');details=folder/(video+'.details.json.gz')
        receipt=read(OUT/'rollouts'/arm/(video+'.receipt.json'))
        for n,h in receipt['artifacts'].items():assert sha(ROOT/n)==h
        output=OUT/'outputs'/arm/(video+'.json')
        if marker.exists():
            saved=read(marker);assert saved['output_sha256']==sha(output) and saved['details_sha256']==sha(details)
            continue
        assert not details.exists(),'Partial details require explicit recovery'
        status('official_metrics',arm=arm,video=video,done=i,total=20)
        metrics=evaluate_hota(OUT/'outputs'/arm,[video],split='val',details_path=details)
        with gzip.open(details,'rt',encoding='utf8') as f:data=json.load(f)[video]
        H=[data[c]['HOTA'] for c in VALID if c in data]
        tp=sum(np.array(v['HOTA_TP'],float) for v in H);fp=sum(np.array(v['HOTA_FP'],float) for v in H)
        fn=sum(np.array(v['HOTA_FN'],float) for v in H);aa=sum(np.array(v['AssA'],float)*np.array(v['HOTA_TP'],float) for v in H)
        hota=float(100*np.sqrt((tp/np.maximum(1,tp+fp+fn))*(aa/np.maximum(1,tp))).mean())
        assa=float(100*(aa/np.maximum(1,tp)).mean())
        assert abs(hota-metrics['HOTA'])<1e-9 and abs(assa-metrics['AssA'])<1e-9
        identities={k:sum(int(data[c]['Identity'][k]) for c in VALID if c in data) for k in ['IDTP','IDFP','IDFN']}
        idf1=100*identities['IDTP']/max(1.,identities['IDTP']+.5*identities['IDFP']+.5*identities['IDFN'])
        assert abs(idf1-metrics['IDF1'])<1e-9
        write(marker.relative_to(OUT),dict(passed=True,video=video,arm=arm,metrics=metrics,
            components=dict(TP=tp,FP=fp,FN=fn,AA=aa,**identities),details_sha256=sha(details),
            output_sha256=sha(output),evaluation_sha256=sha(OUT/'evaluation_protocol.json')))
        print('OFFICIAL',arm,video,metrics['HOTA'],flush=True)
verify(phase);write('metrics_complete.json',dict(passed=True,new_video_evaluations=80,
    reused_controls=['native_B0','legacy_unfiltered_G'],evaluation_sha256=sha(OUT/'evaluation_protocol.json')))
status('official_metrics_complete',new_evaluations=80)
