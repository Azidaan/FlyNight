"""Independent age selection, residual assignments, known labels and intervals."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_singleton_age_20261108.common import *


def main(obs):
    import numpy as np
    from ultralytics.trackers.utils import matching
    from research_singleton_free_20261108.audit import independent_labels
    phase('comparison_phase.json');report=read(RUN/'comparison_report.json');selection=read(RUN/'selection.json');protocol=read(RUN/'protocol.json');checks=Counter()
    assert read(OWN/'independent_audit.json')['passed'] and read(OWN_FULL/'independent_audit.json')['passed']
    for name,h in report['outputs'].items():assert sha(ROOT/name)==h
    for manifest in report['source_reuse_manifests'].values():
        for name,h in manifest.items():assert sha(ROOT/name)==h
    exports=read(BASE/'export_complete.json')['videos'];cal=[r for r in read(OWN/'calibration_rows.json')['rows'] if r['primary_known']];known=[]
    for video in sorted({r['video'] for r in cal}):
        snapshots=read(ROOT/exports[video]['path'])['snapshots']
        for r in cal:
            if r['video']==video:
                t=snapshots[str(r['frame'])]['rows'][r['row']];known.append(dict(uid=r['uid'],positive=r['positive'],gap=r['frame']-t['history'][-1]['frame'],support=r['support']))
    assert {r['uid']:r for r in known}=={r['uid']:r for r in selection['calibration_rows']};P=sum(r['positive'] for r in known);N=len(known)-P
    wanted=[]
    for cap in protocol['calibration']['caps']:
        accepted=[r for r in known if r['support']>=3 and r['gap']<=cap];tp=sum(r['positive'] for r in accepted);fp=len(accepted)-tp
        wanted.append(dict(cap=cap,true_positive=tp,false_positive=fp,youden=tp/P-fp/N))
    assert wanted==selection['candidates'];best=max(wanted,key=lambda r:(r['youden'],-r['false_positive'],r['true_positive'],-r['cap']));assert best==selection['selected'] and best['cap']==selection['cap'];cap=best['cap']
    ledgers={s:{a:read(RUN/'outcomes'/(s+'_'+a+'.json')) for a in ['B0','GEO','SUPPORT','IMAGE','FLYVIS','AGE']} for s in ['fit20','evaluate26']}
    for video in sorted({r['video'] for s in ledgers.values() for r in s['AGE']['rows']}):
        entry=exports[video];path=ROOT/entry['path'];assert sha(path)==entry['sha256'];data=read(path);scores=read(OWN/'scores'/(video+'.json'))['frames'] if (OWN/'scores'/(video+'.json')).exists() else {}
        with np.load(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz',allow_pickle=False) as z:raw={n:z[n] for n in z.files}
        label=independent_labels(video,raw);frames=read(RUN/'counterfactual'/video/'AGE.json')['frames']
        rows={int(r['uid']):r for split in ledgers.values() for r in split['AGE']['rows'] if r['video']==video}
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as matrices:
            for frame,result in frames.items():
                snap=data['snapshots'][frame];C=matrices['cost_'+frame];native,rr,cc=matching.linear_assignment(C,thresh=.8);native=np.asarray(native,int).reshape(-1,2)
                D=np.full((len(rr),len(cc)),np.inf,dtype=C.dtype);ri={int(i):k for k,i in enumerate(rr)};ci={int(j):k for k,j in enumerate(cc)}
                for e in scores.get(frame,{}).values():
                    i=e['row'];j=e['column'];track=snap['rows'][i];gap=int(frame)-track['history'][-1]['frame']
                    if e['free'] and e['support']>=3 and gap<=cap:
                        assert i in ri and j in ci and track['state']==2 and C[i,j]<1 and track['cls']==snap['classes'][j]
                        D[ri[i],ci[j]]=C[i,j]-.2
                added=np.asarray(matching.linear_assignment(D,thresh=.8)[0],int).reshape(-1,2)
                expected=sorted([*map(tuple,native),*((int(rr[i]),int(cc[j])) for i,j in added)])
                assert sorted(map(tuple,result['first']))==expected;checks['residual_matrices']+=1
                for r in rows.values():
                    if r['frame']!=int(frame):continue
                    out=r['outcome'];raw_id=result['final'].get(str(r['tid']));identity=label(int(frame)).get(raw_id)
                    kind='none' if raw_id is None else ('unknown' if identity is None or out['target'] is None else ('correct' if identity==out['target'] else 'wrong'))
                    assert out['raw_id']==raw_id and out['identity']==identity and out['kind']==kind and out['J']=={'correct':1,'wrong':-1,'none':0,'unknown':0}[kind];checks['outcome_rows']+=1
        status('independent_recency_conditional_audit',video=video,checks=dict(checks),guard=obs.read())
    counts={};per_video={}
    for split,arms in ledgers.items():
        counts[split]={};per_video[split]={}
        for arm,ledger in arms.items():
            totals=Counter();pv=defaultdict(Counter)
            for r in ledger['rows']:
                if r['primary_known']:
                    totals['N']+=1;totals['J']+=r['outcome']['J'];totals[r['outcome']['kind']]+=1;pv[r['video']]['N']+=1;pv[r['video']]['J']+=r['outcome']['J']
            for key in ['N','J','correct','wrong','none','unknown']:assert totals[key]==ledger['totals'].get(key,0)
            counts[split][arm]=totals;per_video[split][arm]=pv
    for key,value in report['comparisons'].items():
        a,b=key.split('-');videos=sorted(ledgers['evaluate26'][a]['per_video']);pv=per_video['evaluate26'];d=np.array([pv[a][v]['J']-pv[b][v]['J'] for v in videos]);n=np.array([pv[a][v]['N'] for v in videos])
        ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)));sample=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1))
        assert int(d.sum())==value['delta_J'] and np.allclose(np.percentile(sample,[2.5,97.5]),value['ci95'],rtol=0,atol=1e-12)
        assert np.allclose(np.percentile(sample,[2.5/14,100-2.5/14]),value['ci_joint14'],rtol=0,atol=1e-12);checks['paired_intervals']+=1
    gate=report['comparisons']['AGE-B0']['ci95'][0]>0 and counts['evaluate26']['AGE']['wrong']<=counts['evaluate26']['B0']['wrong']
    gate=gate or read(OWN_FULL/'comparison_report.json')['accepted_selected_system'];assert bool(gate)==report['full_gate']['proceed'];phase('comparison_phase.json')
    write(RUN/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(RUN/'comparison_report.json'),selection_sha256=sha(RUN/'selection.json'),full_gate=bool(gate)),new=True)


if __name__=='__main__':main(guard())
