"""Evaluate stored blinded margins and estimate uncertainty by resampling whole videos."""
import csv
import gzip
from pathlib import Path
import sys
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import flyvis_pair_support_overlap_20261107 as io
RUN=Path('runs/flyvis_exclusive_t4t5_20261107')
def evaluate():
    import numpy as np
    p=io.read(RUN/'protocol.json');m=io.read(RUN/'measurement.json')
    assert io.sha(RUN/'choices.csv.gz')==m['rows_sha256'] and io.sha(RUN/'protocol.json')==m['protocol_sha256'] and io.sha(RUN/'inputs_used.json')==m['inputs_sha256']
    cats=defaultdict(Counter); flags=defaultdict(Counter); scope=defaultdict(lambda:defaultdict(Counter));seen=set()
    def vote(s):return 0 if abs(s)<=1e-10 else (1 if s>0 else -1)
    with gzip.open(RUN/'choices.csv.gz','rt',encoding='ascii') as f:
        for z in csv.DictReader(f):
            sp=z['split']; v=z['video'];u,j,k=(int(z[n]) for n in ['uid','j','k'])
            assert (u,j,k) not in seen;seen.add((u,j,k))
            a,b,inter=(int(z[n]) for n in ('n_a','n_b','shared'))
            ex=min(a,b)>inter; assert ex==bool(z['exclusive_margin'])
            w=vote(float(z['whole_margin']));assert w==int(z['whole_choice'])
            e=vote(float(z['exclusive_margin'])) if ex else w
            if ex:assert e==int(z['exclusive_choice'])
            if not inter:assert e==w and float(z['exclusive_margin'])==float(z['whole_margin'])
            valid=z['label_reason']=='ok';two=z['exactly2']=='1'
            if two:
                flags[sp][z['label_reason']]+=1;cats[sp]['exactly_two']+=1
                cats[sp]['no_exclusive']+=int(not ex);cats[sp]['labeled_no_exclusive']+=int(not ex and valid)
            if not valid:assert int(z['target_side'])==0;continue
            truth=int(z['target_side']);assert abs(truth)==1
            h=1+w*truth;g=1+e*truth # twice correctness: 0 miss, 1 tie, 2 correct
            groups=['all_admitted_pairs_fallback']
            if two:groups+=['exactly_two_fallback']
            if two and ex:groups+=['primary']
            if two and ex and inter:groups+=['overlap_both_unique']
            if two and ex and not inter:groups+=['disjoint']
            for group in groups:
                c=scope[sp,group][v]
                for field,value in [('n',1),('whole_credit_x2',h),('unique_credit_x2',g),('whole_correct',h==2),('unique_correct',g==2),
                     ('whole_ties',h==1),('unique_ties',g==1),('wrong_to_right',h==0 and g==2),('right_to_wrong',h==2 and g==0),('changed_choices',w!=e)]:c[field]+=value
    assert len(seen)==m['pairs']
    out={}; per={}
    for split,vids in p['videos'].items():
        out[split]={};per[split]={}
        for group in ('primary','overlap_both_unique','disjoint','exactly_two_fallback','all_admitted_pairs_fallback'):
            by=scope[split,group];c=Counter()
            for v in vids:c.update(by[v])
            n=c['n'];assert n
            cnt=np.array([by[v]['n'] for v in vids]); dif=np.array([(by[v]['unique_credit_x2']-by[v]['whole_credit_x2'])*50.0 for v in vids])
            draws=np.random.default_rng(42).integers(len(vids),size=(20000,len(vids)));den=cnt[draws].sum(1);ok=den>0
            interval=np.percentile(dif[draws].sum(1)[ok]/den[ok],[2.5,97.5]).tolist()
            out[split][group]=dict(**dict(c),videos_with_evaluable_pairs=int((cnt>0).sum()),whole_pct=50*c['whole_credit_x2']/n,
                exclusive_pct=50*c['unique_credit_x2']/n,difference_pp=float(dif.sum()/n),ci95_pp=interval,valid_bootstrap=int(ok.sum()))
            per[split][group]={v:dict(x) for v,x in by.items()}
        assert sum(flags[split].values())==cats[split]['exactly_two']
        assert out[split]['primary']['n']+cats[split]['labeled_no_exclusive']==flags[split]['ok']==out[split]['exactly_two_fallback']['n']
        assert out[split]['primary']['n']==out[split]['overlap_both_unique']['n']+out[split]['disjoint']['n']
    s=out['evaluate26']['primary']
    assert cats['evaluate26']['exactly_two']==1368 and sum(x['exactly_two'] for x in cats.values())==6045
    result=dict(status='improved' if s['ci95_pp'][0]>0 else 'worse' if s['ci95_pp'][1]<0 else 'inconclusive',
        cohorts=cats,labels=flags,comparison=out,measurement_sha256=io.sha(RUN/'measurement.json'),report_units='paired percentage points; 0.5 credit for ties')
    io.dump(RUN/'per_video.json',per,True);io.dump(RUN/'report.json',result,True)
    print('Primary',s,'status',result['status'])
if __name__=='__main__':evaluate()
