"""Prospective, CPU-only screen of the saved linear readout, not the recurrent state."""
from pathlib import Path
import csv, gzip, json, sys, subprocess
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import flyvis_pair_support_overlap_20261107 as io
from scripts.flyvis_five_flip_review_20261107 import MAP, SEED, BASE, LABEL
R=Path('runs/flyvis_atlas_readout_20261107')
ID='flyvis_atlas_readout_v1'
WEB=Path('ui/flyvis_atlas/dist/data')
SCI=Path('scripts/flyvis_atlas_readout_20261107.py')
AUD=Path('scripts/flyvis_atlas_readout_audit_20261107.py')
CON=Path('tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json')
def prepare():
    typ=io.read(CON)['output_units']; assert len(set(typ))==34
    c=io.read(io.CAT);assert not any(x['id']==ID for x in c['experiments'])
    q='Меняется ли выбор верного продолжения при использовании всех 34 прямых типов вместо T4/T5 и при ослаблении по одному типу или пространственной колонке?'
    p=dict(id=ID,time=datetime.now(timezone.utc).isoformat(),git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      question=q,types=typ,split='Same 172 previously evaluated historical exactly-two labeled pairs, 26 videos including those with zero eligible pairs; no new selection.',
      hypothesis='Additional direct output channels can change identification under the fixed historic target filter. Primary gain requires paired video-bootstrap 95% lower bound above zero; harm if upper below zero. Exploration across channels cannot rescue this comparison.',
      readout='Whole original candidate masks. Compare direct 34 vs T4/T5 projected eight with same historical weights. Rank by unrounded dot product, tie <=1e-10 receives half credit.',
      probe='Only standardized final maps, after recurrence and before pooling. Clone maps; multiply one direct type across the eye OR the 34 types of one position by 0.9. Matched sign control uses 1.1, identity 1.0. Positions act through whole-mask pooling with unchanged denominator. Exclude neighbor-derived channels. Not silencing individual neurons inside the core.',
      audit='Check unchanged unselected values, no original mutation, no training; compare independently calculated effects from per-cell projected scores; assert all margins agree <1e-9 and choices exact. Reproduce previous eight-channel decisions on every row. Evaluate raw output independently, not with predictor labels.',
      statistics='Pooled paired differences in percentage points. Video bootstrap, 20000 resamples, seed 42. Screen: exact two-sided sign test on video totals of paired credit differences excluding zero-video differences; Holm family 755 for weakening only, p<.05. Report all, including neutral. 95% intervals descriptive for individual probes. Strongest alternative: small perturbation plus large rank margins hides useful types.',
      budget='CPU only, bounded chunks, existing system RAM guard; no inference, training or tracker re-run. Save raw before audit.',
      sources={k.as_posix():io.sha(k) for k in [BASE/'choices.csv.gz',BASE/'measurement.json',BASE/'inputs_used.json',BASE/'protocol.json',SCI,AUD,CON]})
    R.mkdir(exist_ok=False);io.dump(R/'protocol.json',p,True)
    c['experiments'].append(dict(id=ID,title='Какие выходные типы и колонки FlyVis меняют выбор из двух рамок',area='flyvis',state='ready',
         question=q,hypothesis=p['hypothesis'],expected='Отдельные компоненты меняют выбор; польза 34 каналов против восьми определяется заранее установленным интервалом.',
         started=p['time'][:10],acceptance=[dict(criterion=p['audit'],met=None),dict(criterion=p['hypothesis'],met=None)],
         evidence=[dict(kind='file',ref=(R/'protocol.json').as_posix(),claim='Фиксация до измерения')]))
    io.dump(io.CAT,c);io.journal('проверка выходных типов и колонок FlyVis: все 172 пары, целые рамки, прежние коэффициенты, 34 типа и 721 позиция, ±10% только на считывании. Критерии: '+(R/'protocol.json').as_posix()+'.')

def measure():
    sys.path.insert(0,str(io.GUARD.parent));from system_memory_guard import guard_entrypoint
    guard=guard_entrypoint(timeout=1200)
    import numpy as np
    p=io.read(R/'protocol.json')
    for f,h in p['sources'].items():assert io.sha(Path(f))==h
    manifest=io.read(BASE/'inputs_used.json'); sources={}
    def verify(f):
        k=f.as_posix()
        if k not in sources:assert manifest[k]==io.sha(f);sources[k]=manifest[k]
    with gzip.open(BASE/'choices.csv.gz','rt',encoding='ascii') as f:
        cases={int(r['uid']):r for r in csv.DictReader(f) if r['split']=='evaluate26' and r['exactly2']=='1' and r['label_reason']=='ok' and r['exclusive_choice']}
    assert len(cases)==172
    def subset(path):
        for f in sorted(path.glob('chunk_*.npz')):
            with np.load(f,allow_pickle=False) as z:uids=z['uid']; ii=[(i,int(u)) for i,u in enumerate(uids) if int(u) in cases]
            if ii:verify(f);yield f,ii
    weights={}
    for f,ii in subset(SEED):
        with np.load(f,allow_pickle=False) as z:s=z['seed'][:,:34].astype(float)
        for i,u in ii:assert u not in weights;weights[u]=s[i]
    assert weights.keys()==cases.keys()
    records={};errors=0.
    for f,ii in subset(MAP):
        with np.load(f,allow_pickle=False) as z:X=z['full'][:,:,:34].copy();mask=z['retinal_mask']
        for i,u in ii:
            r=cases[u];w=weights[u];x=X[i].astype(float);before=x.copy()
            assert x.shape==(721,34) and np.isfinite(x).all() and np.isfinite(w).all()
            a,b=mask[i,1:3]; assert a.sum()==int(r['n_a']) and b.sum()==int(r['n_b']) and (a&b).sum()==int(r['shared'])
            coeff=a/a.sum()-b/b.sum(); dx=x[a].mean(0)-x[b].mean(0)
            orig=dx@w; ref=dx[4:12]@w[4:12]
            assert abs(ref-float(r['whole_margin']))<1e-9
            identity=x.copy();identity*=1.;assert np.array_equal(identity,x)
            typ=w*(coeff@x); pos=coeff*(x@w)
            out=[]; alt=[]
            for scale,dest in [(0.9,out),(1.1,alt)]:
                # Explicit cloned type intervention on each of 34 channels.
                for j in range(34):
                    cp=x.copy();cp[:,j]*=scale
                    assert np.array_equal(cp[:,:j],x[:,:j]) and np.array_equal(cp[:,j+1:],x[:,j+1:])
                    dest.append((cp[a].mean(0)-cp[b].mean(0))@w)
                # For each single-column edit, untouched rows cancel vs original.
                # Clone all 721 independent target ROWS; each row is one independent
                # counterfactual, never use the simultaneously edited map as a candidate.
                edits=x.copy(); edits*=scale
                dest.extend(orig+((edits-x)@w)*coeff)
                expected=orig+(scale-1)*np.r_[typ,pos];err=np.max(np.abs(expected-dest))
                errors=max(errors,err)
                assert err<1e-9 and np.array_equal(np.where(np.abs(expected)<=1e-10,0,np.sign(expected)),np.where(np.abs(dest)<=1e-10,0,np.sign(dest)))
            assert np.array_equal(before,x) and u not in records
            records[u]=(orig,ref,out,alt)
    assert records.keys()==cases.keys()
    us=sorted(records);margins=np.array([[records[u][0],records[u][1],*records[u][2],*records[u][3]] for u in us])
    # Supervision used only by the later evaluator, never to compute margin.
    assert margins.shape==(172,1512) and np.isfinite(margins).all()
    with (R/'raw.npz').open('xb') as f:np.savez_compressed(f,uid=us,margins=margins)
    for f,h in sources.items():assert io.sha(Path(f))==h
    io.dump(R/'input_checks.json',dict(sources=sources,raw_sha256=io.sha(R/'raw.npz'),max_error=errors,
        system_guard=guard.read(),protocol_sha256=io.sha(R/'protocol.json')),True)
    print('172 rows, 1512 conditions including controls; max_abs_error',errors)

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:measure()
