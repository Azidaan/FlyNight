"""Independent verification of selected costs, final assignments, labels and statistics."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
import numpy as np
from ultralytics.trackers.utils import matching
from research_identity_assignment_run_20261108.common import *


def main(obs):
    verify_phase('audit_phase.json');report=read(RUN/'comparison_report.json');selection=read(RUN/'selection.json')
    assert report['selection_sha256']==sha(RUN/'selection.json')
    ex=read(RUN/'export_complete.json')['videos'];scores=read(RUN/'scores_complete.json')['videos'];count=Counter()
    arms=list(report['parameters']);known={}
    for name,h in report['outputs'].items():assert sha(ROOT/name)==h
    # Independently classify every primary ledger row, retaining sparse unknowns.
    for split in ['fit20','evaluate26']:
        known[split]={}
        for arm in arms:
            data=read(RUN/'outcomes'/f'{split}_{arm}.json');c=Counter();by=defaultdict(Counter)
            for r in data['rows']:
                o=r['outcome'];identity=o['identity'];target=o['target']
                kind='none' if o['raw_id'] is None else ('unknown' if identity is None or target is None else ('correct' if identity==target else 'wrong'))
                j={'correct':1,'wrong':-1,'none':0,'unknown':0}[kind]
                assert o['kind']==kind and o['J']==j
                if r['primary_known']:c[kind]+=1;c['J']+=j;c['N']+=1;by[r['video']]['J']+=j;by[r['video']]['N']+=1
                count['classified_ledger_rows']+=1
            for k in ['correct','wrong','none','unknown','J','N']:assert c[k]==data['totals'].get(k,0),(split,arm,k)
            for v,vc in by.items():
                for k in ['J','N']:assert vc[k]==data['per_video'][v].get(k,0)
            known[split][arm]=data
    for number,(video,entry) in enumerate(ex.items()):
        path=ROOT/entry['path'];assert sha(path)==entry['sha256'];data=read(path)
        assert sha(path.with_name(video+'.states.zip'))==data['states_sha256']
        scenario_cache={}
        plan=read(RUN/'scores'/(video+'.json'));assert sha(RUN/'scores'/(video+'.json'))==scores[video]
        assert sha(path.with_suffix('.npz'))==data['matrices_sha256']
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as matrices:
            for frame,snapshot in data['snapshots'].items():
                groups={q['split'] for q in snapshot['queries'] if q['split'] in known}
                if not groups:continue
                assert len(groups)==1;split=groups.pop();p=plan['frames'][frame]
                for arm,settings in report['parameters'].items():
                    C=matrices['cost_'+frame];cost=C.copy()
                    for i,j in p['votes']:cost[i,j]-=settings['alpha']
                    for j,col in p['columns'].items():
                        if not col['geometry_valid'] or len(col['rows'])<2:continue
                        rr=np.array(col['rows']);beta=settings['beta']
                        if arm=='G_SHIFT':cost[rr,int(j)]-=beta;continue
                        values=col['scores'].get(arm)
                        if values is None or any(v is None for v in values):continue
                        a=np.clip(np.array(values,float),-1,1)
                        cost[rr,int(j)]-=beta*(a-a.mean())
                    expected=np.asarray(matching.linear_assignment(cost,thresh=.8)[0]).reshape(-1,2).tolist()
                    if arm=='B0':actual=snapshot['native_matches'];final=snapshot['native_final_assignments'];emitted={int(o[4]) for o in snapshot['native_output']};final={k:v for k,v in final.items() if int(k) in emitted}
                    else:
                        if arm not in scenario_cache:scenario_cache[arm]=read(RUN/'counterfactual'/video/f"{arm}_a{settings['alpha']:g}_b{settings['beta']:g}.json")
                        result=scenario_cache[arm]['frames'][frame]
                        actual=result['first'];final=result['final']
                    assert sorted(expected)==sorted(actual),(video,frame,arm,'selected costs')
                    rows=[r for r in known[split][arm]['rows'] if r['video']==video and r['frame']==int(frame)]
                    for row in rows:assert row['outcome']['raw_id']==final.get(str(row['tid']))
                    count['selected_matrices']+=1
        status('independent_result_audit',complete=number+1,total=len(ex),checks=dict(count),guard=obs.read())
    # Recompute paired-video intervals independently from the classified ledger.
    for key,ci in report['evaluate26_comparisons'].items():
        arm,base=key.split('-');A=known['evaluate26'][arm];B=known['evaluate26'][base]
        vids=sorted(set(A['per_video'])|set(B['per_video']));d=np.array([A['per_video'][v].get('J',0)-B['per_video'][v].get('J',0) for v in vids]);n=np.array([A['per_video'][v].get('N',0) for v in vids])
        rng=np.random.default_rng(42);sample=[]
        for start in range(0,20000,1000):
            indices=rng.integers(0,len(vids),(1000,len(vids)));sample.extend((100*d[indices].sum(1)/np.maximum(1,n[indices].sum(1))).tolist())
        assert int(d.sum())==ci['delta_J'] and np.allclose(np.percentile(sample,[2.5,97.5]),ci['ci95'],rtol=0,atol=1e-12)
        assert np.allclose(np.percentile(sample,[.3125,99.6875]),ci['ci99_375'],rtol=0,atol=1e-12)
        count['bootstrap_comparisons']+=1
    # Selection criterion and ties checked from the frozen fit20 candidate tables.
    def winner(rows,param):return max(rows,key=lambda x:(x['totals'].get('J',0),-x['totals']['changed_native_bindings'],-x[param]))[param]
    assert selection['alpha']==winner(selection['alpha_candidates'],'alpha')
    for arm,rows in selection['beta_candidates'].items():assert selection['parameters'][arm]['beta']==winner(rows,'beta')
    proceed=any(report['evaluate26_comparisons'][a+'-B0']['ci95'][0]>0 for a in ['G','G_FLYVIS'])
    assert proceed==report['full_tracker_gate']['proceed'];assert read(RUN/'restored_zero_parity.json')['passed']
    verify_phase('audit_phase.json')
    write(RUN/'independent_audit.json',dict(passed=True,checks=dict(count),comparison_sha256=sha(RUN/'comparison_report.json'),
        selection_sha256=sha(RUN/'selection.json'),full_tracker_gate=proceed,not_checked='Conditional replay is not a causal full-video HOTA result.'),new=True)


if __name__=='__main__':main(guard())
