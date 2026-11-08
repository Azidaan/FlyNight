"""Independent own-score arithmetic, fit-only choice, costs, labels and intervals."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_singleton_verify_20261108.common import *


def check_scores(obs):
    import numpy as np
    from scipy.spatial import cKDTree
    inventory=read(RUN/'inventory_recovery1.json');manifest=read(RUN/'features_complete.json')['files'];exports=read(BASE/'export_complete.json')['videos']
    geometry=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json')
    counts=Counter();groups=defaultdict(list)
    for e in inventory['events']:groups[e['video']].append(e)
    for number,(video,events) in enumerate(groups.items()):
        snapshots=read(ROOT/exports[video]['path'])['snapshots'];scorefile=RUN/'scores'/(video+'.json')
        assert sha(scorefile)==read(RUN/'scores_complete.json')['videos'][video];saved=read(scorefile)['frames'];features={}
        def get(key):
            if key not in features:
                path=RUN/'features'/(key+'.npz');assert sha(path)==manifest[key]
                with np.load(path,allow_pickle=False) as z:features[key]={k:z[k].astype(float) for k in z.files}
                features[key]['IMAGE']=(features[key]['IMAGE']-norm['mean'])/np.where(np.array(norm['sd'])>=1e-12,norm['sd'],1)
            return features[key]
        for e in events:
            snap=snapshots[str(e['frame'])];row=snap['rows'][e['row']];j=e['column'];g=row['geometry'][row['choices'].index(j)]
            current=get(e['current_key']);reference=get(e['reference_key']);old=row['history'][-1]
            box=np.array(snap['boxes'][j],float);B=np.array(old['box'],float);c0=np.array(old['crop'],float);ct=np.array(row['current_crop'],float)
            points=np.array(geometry['points'],float);xy0=c0[:2]+points*(c0[2:]-c0[:2])/geometry['eye_size'];xyt=ct[:2]+points*(ct[2:]-ct[:2])/geometry['eye_size']
            inside=((xy0>=B[:2])&(xy0<=B[2:])).all(1);p=np.flatnonzero(((xyt>=box[:2])&(xyt<=box[2:])).all(1))
            projected=B[:2]+g['offset']+(xyt[p]-box[:2])/g['scale'];distance,q=cKDTree(xy0).query(projected)
            ok=((projected>=B[:2])&(projected<=B[2:])).all(1)&inside[q]&(distance<=.6*geometry['spacing']*(c0[2]-c0[0])/geometry['eye_size'])
            ok &=current['DIS_VALID'][p].astype(bool)&reference['DIS_VALID'][q].astype(bool);p=p[ok];q=q[ok];unique=np.unique(q)
            values={};ids=np.flatnonzero(inside&reference['DIS_VALID'].astype(bool))
            def cosine(x,y,center):
                if center:x=x-x.mean(0);y=y-y.mean(0)
                xx=np.sum(x*x);yy=np.sum(y*y)
                return None if xx<1e-12 or yy<1e-12 else float(np.clip(np.sum(x*y)/np.sqrt(xx*yy),-1,1))
            for kind in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:
                if len(unique)<3:value=None
                elif kind=='SUPPORT':value=1.
                elif kind=='RANDOM':value=2*(int.from_bytes(hashlib.sha256(f"{video}|{e['frame']}|{row['tid']}|{j}|42".encode()).digest()[:8],'little')/2**64)-1
                else:
                    name='C' if kind in ['FLYVIS','PERM'] else kind;x=np.stack([current[name][p[q==cell]].mean(0) for cell in unique]);y=reference[name]
                    if kind=='DIS':value=cosine(x/np.array(g['scale']),y[unique],False)
                    elif kind!='PERM':value=cosine(x,y[unique],True)
                    else:
                        v=[]
                        for rep in range(32):
                            seed=int.from_bytes(hashlib.sha256(f"{video}|{row['tid']}|{old['frame']}|{rep}|42".encode()).digest()[:8],'little')
                            copied=y.copy();copied[ids]=y[np.random.default_rng(seed).permutation(ids)];v.append(cosine(x,copied[unique],True))
                        value=None if any(t is None for t in v) else float(np.mean(v))
                values[kind]=value
            record=saved[str(e['frame'])][f"{e['row']},{j}"];assert record['support']==(len(unique) if len(unique)>=3 else 0)
            for kind,value in values.items():
                measured=record['scores'][kind];assert (value is None)==(measured is None)
                if value is not None:assert abs(value-measured)<1e-12,(video,e['frame'],kind,value,measured)
                counts['independent_scores']+=1
        status('independent_absolute_scores',complete=number+1,total=len(groups),checks=dict(counts),guard=obs.read())
    return counts


def main(obs):
    import numpy as np
    from ultralytics.trackers.utils import matching
    from research_singleton_free_20261108.audit import independent_labels
    phase('audit_phase.json');report=read(RUN/'comparison_report.json');selection=read(RUN/'selection.json');checks=check_scores(obs)
    for name,h in report['outputs'].items():assert sha(ROOT/name)==h
    for sources in report['reused_source_manifests'].values():
        for name,h in sources.items():assert sha(ROOT/name)==h
    ledgers={split:{a:read(ROOT/name) for name in report['outputs'] if Path(name).name.startswith(split+'_') for a in [read(ROOT/name)['arm']]} for split in ['fit20','evaluate26']}
    calibration=read(RUN/'calibration_rows.json');by_cal=defaultdict(list)
    for row in calibration['rows']:by_cal[row['video']].append(row)
    exports=read(BASE/'export_complete.json')['videos'];videos={r['video'] for d in ledgers.values() for ledger in d.values() for r in ledger['rows']}
    for number,video in enumerate(sorted(videos)):
        entry=exports[video];path=ROOT/entry['path'];assert sha(path)==entry['sha256'];data=read(path)
        rawpath=ROOT/f'runs/train_cache/{video}/detections/raw_video.npz'
        with np.load(rawpath,allow_pickle=False) as z:raw={n:z[n] for n in z.files}
        label=independent_labels(video,raw);scores=read(RUN/'scores'/(video+'.json'))['frames'] if (RUN/'scores'/(video+'.json')).exists() else {}
        for row in by_cal[video]:
            snap=data['snapshots'][str(row['frame'])];identity=label(row['frame']).get(snap['raw_ids'][row['column']])
            assert identity==row['identity']
            if row['primary_known']:assert row['positive']==(identity==row['target'])
        new={arm:read(RUN/'counterfactual'/video/(arm+'.json'))['frames'] for arm in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']}
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as matrices:
            for frame,snap in data['snapshots'].items():
                split=next((q['split'] for q in snap['queries'] if q['split'] in ledgers),None)
                if split is None:continue
                C=matrices['cost_'+frame];native,rr,cc=matching.linear_assignment(C,thresh=.8);native=np.asarray(native,int).reshape(-1,2)
                for arm,frames in new.items():
                    D=np.full((len(rr),len(cc)),np.inf,dtype=C.dtype);ri={int(i):k for k,i in enumerate(rr)};ci={int(j):k for k,j in enumerate(cc)}
                    for e in scores.get(frame,{}).values():
                        threshold=selection['thresholds'][arm];value=e['scores'][arm]
                        accept=e['free'] and threshold!='off' and value is not None and value>=threshold
                        if accept:
                            i=e['row'];j=e['column'];assert i in ri and j in ci and snap['rows'][i]['state']==2 and C[i,j]<1
                            D[ri[i],ci[j]]=C[i,j]-.2
                    added=np.asarray(matching.linear_assignment(D,thresh=.8)[0],int).reshape(-1,2)
                    wanted=sorted([*map(tuple,native),*((int(rr[i]),int(cc[j])) for i,j in added)])
                    measured=frames[frame];assert sorted(map(tuple,measured['first']))==wanted
                    assert set(map(tuple,native)).issubset(set(map(tuple,measured['first'])))
                    for i,j in measured['first']:
                        tid=str(snap['rows'][i]['tid'])
                        if tid in measured['final']:assert measured['final'][tid]==snap['raw_ids'][j]
                    for row in ledgers[split][arm]['rows']:
                        if row['video']!=video or row['frame']!=int(frame):continue
                        result=row['outcome'];raw_id=measured['final'].get(str(row['tid']));identity=label(int(frame)).get(raw_id)
                        assert result['raw_id']==raw_id and result['identity']==identity
                        kind='none' if raw_id is None else ('unknown' if identity is None or result['target'] is None else ('correct' if identity==result['target'] else 'wrong'))
                        assert result['kind']==kind and result['J']=={'none':0,'unknown':0,'correct':1,'wrong':-1}[kind];checks['independent_outcome_rows']+=1
                    checks['independent_matrices']+=1
        status('independent_own_template_outcomes',complete=number+1,total=len(videos),checks=dict(checks),guard=obs.read())
    known=[r for r in calibration['rows'] if r['primary_known']];positive=sum(r['positive'] for r in known);negative=len(known)-positive
    selected={}
    for arm,candidates in selection['candidates'].items():
        for row in candidates:
            t=row['threshold'];tp=fp=0
            for r in known:
                v=r['scores'][arm];accepted=t!='off' and v is not None and v>=t
                tp+=int(accepted and r['positive']);fp+=int(accepted and not r['positive'])
            assert (tp,fp)==(row['true_positive'],row['false_positive']) and abs(tp/positive-fp/negative-row['youden'])<1e-12
        best=max(candidates,key=lambda r:(r['youden'],-r['false_positive'],r['true_positive'],float('inf') if r['threshold']=='off' else r['threshold']))
        assert selection['thresholds'][arm]==best['threshold'];selected[arm]=best
    order=['IMAGE','R1','DIS','FLYVIS'];chosen=max(order,key=lambda a:(selected[a]['youden'],-selected[a]['false_positive'],selected[a]['true_positive'],-order.index(a)))
    assert chosen==selection['selected_system']==report['selected_system']
    totals={};per_video={}
    for arm,ledger in ledgers['evaluate26'].items():
        c=Counter();v=defaultdict(Counter)
        for row in ledger['rows']:
            if row['primary_known']:
                c['N']+=1;c['J']+=row['outcome']['J'];c[row['outcome']['kind']]+=1;v[row['video']]['J']+=row['outcome']['J'];v[row['video']]['N']+=1
        for k in ['N','J','correct','wrong','none','unknown']:assert c[k]==ledger['totals'].get(k,0)
        totals[arm]=c;per_video[arm]=v
    for key,value in report['comparisons'].items():
        a,b=key.split('-');videos=sorted(ledgers['evaluate26'][a]['per_video']);d=np.array([per_video[a][v]['J']-per_video[b][v]['J'] for v in videos]);n=np.array([per_video[a][v]['N'] for v in videos])
        ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)));sample=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1))
        assert int(d.sum())==value['delta_J'] and np.allclose(np.percentile(sample,[2.5,97.5]),value['ci95'],rtol=0,atol=1e-12)
        assert np.allclose(np.percentile(sample,[2.5/11,100-2.5/11]),value['ci_corrected'],rtol=0,atol=1e-12);checks['independent_intervals']+=1
    proceed=any(report['comparisons'][a+'-B0']['ci95'][0]>0 and totals[a]['wrong']<=totals['B0']['wrong'] for a in [chosen,'SUPPORT'])
    assert proceed==report['full_tracker_gate']['proceed'];phase('audit_phase.json')
    write(RUN/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(RUN/'comparison_report.json'),
        selection_sha256=sha(RUN/'selection.json'),full_gate=bool(proceed),scope='Read-only arithmetic/identity audit; not full-tracker acceptance.'),new=True)


if __name__=='__main__':main(guard())
