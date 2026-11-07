"""Describe fixed 68-term linear readout on the same five preselected failures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyvis_five_flip_review_20261107 import BASE, CH, OUT, MAP, SEED, LABEL, sha, load, np, json


def calculate():
    dfile = OUT/'decomposition.json'
    d = load(dfile)
    expected = d['inputs_sha256']
    groups = {
        'T4_T5': list(range(4, 12)),
        'other_direct': [*range(4), *range(12,34)],
        'neighbor_T4_T5': list(range(38,46)),
        'neighbor_other': [*range(34,38), *range(46,68)]}
    assert sorted(sum(groups.values(),[])) == list(range(68))
    def source(base, uid):
        matches = []
        for k in expected:
            p = Path(k)
            if p.parent == base and p.suffix == '.npz':
                with np.load(p,allow_pickle=False) as data:
                    for i in np.flatnonzero(data['uid'] == uid): matches.append((p,int(i)))
        assert len(matches) == 1
        p,i = matches[0]; assert sha(p) == expected[p.as_posix()]
        return p,i
    rows=[]
    for c in d['cases']:
        uid,v = c['uid'],c['video']
        path = LABEL/'supervision'/(v+'.json')
        assert sha(path)==expected[path.as_posix()]
        lab=next(r for r in load(path)['rows'] if r['uid']==uid)
        t=lab['current_identity'].index(lab['target_identity'])
        pm,i=source(MAP,uid)
        with np.load(pm,allow_pickle=False) as z:
            X=z['full'][i].astype('float64');g,o=z['retinal_mask'][i,1+t],z['retinal_mask'][i,2-t]
        ps,i=source(SEED,uid)
        with np.load(ps,allow_pickle=False) as z: w=z['seed'][i].astype('float64')
        assert X.shape==(721,68) and w.shape==(68,) and np.isfinite(w).all() and np.isfinite(X).all()
        score={}
        for policy,am,bm in [('all',g,o),('only',g&~o,o&~g)]:
            dif=(X[am].mean(0)-X[bm].mean(0))*w
            parts={k:float(dif[indices].sum()) for k,indices in groups.items()}
            parts['direct34']=parts['T4_T5']+parts['other_direct']
            together=float(dif.sum())
            cell_scores=X@w
            reference=float(cell_scores[am].mean()-cell_scores[bm].mean())
            assert abs(together-reference)<1e-10 and abs(together-sum(parts[k] for k in groups))<1e-10
            assert abs(parts['T4_T5']-c['target_minus_alternative'][policy]['score'])<1e-10
            parts['full68']=together;score[policy]=parts
        rows.append(dict(uid=uid,video=v,frame=c['cached_frame'],target_minus_alternative=score))
        print(uid,'all/only',str({p:{k:round(x,5) for k,x in y.items()} for p,y in score.items()}))
    meta=dict(analysis='Descriptive on the five cases selected by a T4/T5 flip only. No generalization or causal cell intervention.',
        source_sha256=sha(Path(__file__)),parent_sha256=sha(dfile), indices_zero_based=groups, rows=rows)
    with (OUT/'other_channels.json').open('x',encoding='utf-8') as f:
        json.dump(meta,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


if __name__=='__main__':
    calculate()
