"""Paired historical diagnostic; the only change is the retinal support of a candidate."""
import csv
import gzip
import itertools
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import flyvis_pair_support_overlap_20261107 as common
os.chdir(ROOT)
RUN=Path('runs/flyvis_exclusive_t4t5_20261107')
PAIRS=common.RUN/'recovery1'
MAP=Path('runs/z13_spatial_set_readout_20261025/recovery2/features')
SEED=Path('runs/z13_region_assignment_20261025/recovery2/features')
IDENTS=Path('runs/z13_temporal_pair_readiness_20261025')
ID='flyvis_exclusive_t4t5_v1'
SHAS=[PAIRS/'protocol.json',PAIRS/'report.json',PAIRS/'audit.json',PAIRS/'pairs.csv.gz',MAP/'report.json',SEED/'report.json',IDENTS/'report.json',
      IDENTS/'protocol.json',common.OLD/'recovery1/protocol.json',Path('tracking/flyvis_activity.py'),common.GUARD,
      Path('scripts/z13_spatial_set_readout_20261025.py'),Path('scripts/z13_filter_descriptor_fixed_20261024.py'),
      Path('scripts/z13_identity_gate_readout_20261026.py'),Path('scripts/z13_temporal_pair_readiness_20261025.py'),
      Path('scripts/flyvis_exclusive_t4t5_20261107.py'),Path('scripts/audit_flyvis_exclusive_t4t5_20261107.py')]
load=common.read
dump=common.dump
sha=common.sha

def prepare():
    c=load(common.CAT);assert not RUN.exists() and not any(e['id']==ID for e in c['experiments'])
    old=load(common.OLD/'recovery1/protocol.json')
    q='Улучшает ли удаление общих колонок из двух рамок выбор правильного продолжения по сохранённым каналам T4/T5?'
    p=dict(id=ID,time=datetime.now().astimezone().isoformat(),git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),question=q,
      videos={k:old[k] for k in ['evaluate26','fit84','fit20']},model='C: original frozen FlyVis; no D, training, recalibration or tracker replay.',
      score='Same previously stored observed-exemplar target filter coefficients 4:12 in original 68-channel feature space, only eight central standardized T4a-d/T5a-d channels, no neighbouring-channel means. Dot product with arithmetic mean of the eight channels over a mask. Highest score wins. This fixed partial readout is not an optimal new T4/T5 classifier.',
      inputs='Use only already frozen full per-cell arrays and masks, match each candidate ID/UID/mask with old retinal list and pair cardinalities. Offline known-target and detection identity only for evaluating scores, never scoring. Old highest-confidence same-class IoU matching to GT as originally saved; label ambiguity is not a negative.',
      comparison='A: all centres in each box. B: only centres in that box and not the OTHER candidate. Exactly same seed, direction and scoring. If either exclusive set empty, no exclusive choice; count separately, operational fallback uses whole-box for this pair. No nearest fill. Disjoint sets reproduce scores exactly. Absolute score tie tolerance 1e-10: tied ranking receives 1/2, not an arbitrary ID choice.',
      primary='Frozen historical evaluate26, exactly two admitted real candidates, known query and target, both candidate identities known, exactly one is target, BOTH exclusive sets nonempty. Pooled paired percentage-point difference exclusive minus whole. Cluster bootstrap all 26 source videos with replacement, 20000 samples seed42, percentile 95% CI, dropping only resamples with zero denominator. Support for improvement only if lower bound >0, deterioration if upper <0, otherwise inconclusive.',
      secondary='Prespecified descriptive: primary restricted to intersecting supports; all label-evaluable exactly-two with fallback; all unordered admitted pairs from 2+ competitions; separate fit20 and fit84 historical context, not independent replications. Report wrong->right/right->wrong, ties and losses of coverage, not HOTA. Diagnostic only, no selection of a subgroup rescues primary.',
      limitations='Known track target from historical observation, both candidate IDs from existing annotated frames. Some unrecognizable/missing/duplicate objects unscorable. Reused training-source videos; no end-to-end tracking or novel data. Frozen target coefficients originally estimated using observable past examples, not labels of candidate pairs.',
      acceptance='Verify manifests, every admitted pair and old mask; whole-box mean exact against the stored same-array FP32 mean; independent score per pair via per-cell score sums minus shared cells. Primary defined prospectively above.',
      sha256={str(f):sha(f) for f in SHAS})
    RUN.mkdir();dump(RUN/'protocol.json',p,True)
    c['experiments'].append(dict(id=ID,title='Выбор цели по собственным ячейкам T4/T5 двух рамок',area='flyvis',state='ready',started=p['time'][:10],
      question=q,hypothesis='Сигнал общих ячеек может смешивать движения; исключение может также выбросить полезную часть цели.',
      expected='Больше верных выборов при одной и той же оценке сходства с прошлой целью.',
      acceptance=[dict(criterion=p['acceptance'],met=None),dict(criterion=p['primary'],met=None)],
      evidence=[dict(kind='file',ref=(RUN/'protocol.json').as_posix(),claim='Постановка до подсчёта результатов')]))
    dump(common.CAT,c);common.journal('проверка сигнала T4/T5 собственных ячеек (`'+ID+'`). *Постановка.* '+q+' Исторические 26 видео, ровно два кандидата, один известный верный ответ. Тот же прошлый фильтр и каналы в обоих вариантах; без обучения. Пустые множества отдельно. Протокол: '+str(RUN/'protocol.json')+'.')

def compute(guard):
    import numpy as np
    import psutil
    from tracking.flyvis_activity import OUTPUT_CELL_TYPES
    assert tuple(OUTPUT_CELL_TYPES[4:12])==tuple(f'T{t}{c}' for t in (4,5) for c in 'abcd')
    p=load(RUN/'protocol.json'); started=time.perf_counter();checked={}
    def verify(f,h):
        key=f.as_posix()
        if key not in checked: assert sha(f)==h,f;checked[key]=h
        assert checked[key]==h
    for f,h in p['sha256'].items():verify(Path(f),h)
    mr=load(MAP/'report.json');sr=load(SEED/'report.json'); ir=load(IDENTS/'report.json'); base=load(PAIRS/'report.json')
    assert mr['complete'] and mr['all_same_full_mean_exact'] and sr['all_masks_and_choices_equal'] and ir['complete'] and load(PAIRS/'audit.json')['passed']
    assert sha(PAIRS/'pairs.csv.gz')==base['sha256'][str(PAIRS/'pairs.csv.gz')]
    oldpairs=defaultdict(dict)
    with gzip.open(PAIRS/'pairs.csv.gz','rt',encoding='ascii') as f:
        for row in csv.DictReader(f):
            if row['admitted']=='1':
                u,j,k=(int(row[t]) for t in ('uid','j','k'))
                oldpairs[u][j,k]=(row['split'],row['video'],int(row['na']),int(row['nb']),int(row['shared']),int(row['exactly_two_admitted']))
    def shards(base,report,split):
        prefix=(base/'C'/split).as_posix()+'/'
        ks=sorted(k for k in report['manifest'] if k.startswith(prefix))
        assert ks
        for k in ks:
            m=Path(k);verify(m,report['manifest'][k]);f=m.with_suffix('.npz');verify(f,load(m)['sha256']);yield f
    eps=1e-10
    score_tol=0.;count=0;ncases=0; all_uids=set()
    def vote(g):return 1 if g>eps else -1 if g< -eps else 0
    output=RUN/'choices.csv.gz'
    with gzip.open(output,'xt',encoding='ascii',newline='') as fp:
      wr=csv.writer(fp);wr.writerow('split video uid j k exactly2 category n_a n_b shared label_reason target_side whole_margin exclusive_margin whole_choice exclusive_choice'.split())
      for split,folder in [('evaluate26','train26'),('fit84','fit84'),('fit20','fit20')]:
        seed={}; records={}
        for f in shards(SEED,sr,folder):
            with np.load(f,allow_pickle=False) as z:
                uu,ss=z['uid'],z['seed'][:,4:12].astype(float)
                for u,w in zip(uu,ss,strict=True):
                    assert int(u) not in seed and np.isfinite(w).all();seed[int(u)]=w.copy()
        for video in p['videos'][split]:
            mark=IDENTS/'videos'/(video+'.json');verify(mark,ir['manifest'][video]);h=load(mark)
            idx=IDENTS/'inputs'/(video+'.json'); lab=IDENTS/'supervision'/(video+'.json')
            verify(idx,h['input_sha256']);verify(lab,h['supervision_sha256'])
            m={x['uid']:x for x in load(lab)['rows']}
            for x in load(idx)['rows']:
                u=x['uid']; y=m.pop(u)
                assert x['video']==video and u not in records
                records[u]=(video,x['current_choices'],x['current_masks'],y['flags']['known'],y['target_identity'],y['current_identity'])
            assert not m
        assert set(seed)==set(records)
        seen=set()
        for f in shards(MAP,mr,folder):
            with np.load(f,allow_pickle=False) as z:
                uu=z['uid']; masks=z['retinal_mask']; whole=z['pooled'][:,:,4:12];full=z['full'][:,: ,4:12].copy()
            assert full.shape==(len(uu),721,8)
            for ix,uu0 in enumerate(uu):
                u=int(uu0);assert u not in seen and u not in all_uids;seen.add(u);all_uids.add(u)
                vid,ids,mm,known,tar,identity=records[u];assert len(ids)==len(mm)==len(identity)
                assert masks[ix,0].all() and not masks[ix,len(ids)+1:].any()
                if len(ids)<2:assert u not in oldpairs;continue
                w=seed[u];X=full[ix].astype(np.float64); assert np.isfinite(X).all()
                Z=X@w # independent algebraic check: score each cell first
                scores=[]; sums=[]
                for j,inds in enumerate(mm,1):
                    mask=masks[ix,j];assert np.array_equal(np.flatnonzero(mask),inds) and mask.any()
                    avg=X[mask].mean(0);assert np.array_equal(avg.astype(np.float32),whole[ix,j])
                    scores.append(float(avg@w));sums.append(float(Z[mask].sum()))
                pairs=oldpairs[u]; assert len(pairs)==len(ids)*(len(ids)-1)//2
                for j,k in itertools.combinations(range(len(ids)),2):
                    ja,kb=ids[j],ids[k]; a=masks[ix,j+1];b=masks[ix,k+1]; A=a&~b;B=b&~a;C=a&b
                    na,nb,ns=(int(x.sum()) for x in (a,b,C));only_a=na-ns;only_b=nb-ns
                    assert pairs[ja,kb]==(split,vid,na,nb,ns,int(len(ids)==2))
                    whole_diff=scores[j]-scores[k]; independent_whole=sums[j]/na-sums[k]/nb
                    delta=abs(whole_diff-independent_whole);assert delta<1e-9 and vote(whole_diff)==vote(independent_whole)
                    exclusive=None
                    if only_a and only_b:
                        exclusive=float((X[A].mean(0)@w)-(X[B].mean(0)@w))
                        cs=Z[C].sum();repeat=(sums[j]-cs)/only_a-(sums[k]-cs)/only_b
                        delta=max(delta,abs(exclusive-repeat));assert delta<1e-9 and vote(exclusive)==vote(repeat)
                        if not ns:assert exclusive==whole_diff
                    score_tol=max(score_tol,delta)
                    # Only the evaluator uses identity, after both signal scores have been computed.
                    reason='ok';answer=0
                    if not known or not tar:reason='unknown_target_or_query'
                    elif not identity[j] or not identity[k]:reason='unknown_candidate'
                    elif (identity[j]==tar)==(identity[k]==tar):reason='both_target' if identity[j]==tar else 'neither_target'
                    else:answer=1 if identity[j]==tar else -1
                    cat=common.category(na,nb,ns)
                    wr.writerow([split,vid,u,ja,kb,int(len(ids)==2),cat,na,nb,ns,reason,answer,whole_diff,exclusive,vote(whole_diff),None if exclusive is None else vote(exclusive)])
                    count+=1
        assert seen==set(records);ncases+=len(seen)
        print(split,len(seen),'queries complete',flush=True)
    assert count==sum(len(x) for x in oldpairs.values()) and ncases==37296
    dump(RUN/'inputs_used.json',checked,True)
    dump(RUN/'measurement.json',dict(full_arm='C',channels=list(OUTPUT_CELL_TYPES[4:12]),queries=ncases,pairs=count,score_max_abs_disagreement=score_tol,
        all_historical_pair_bindings_exact=True,whole_mean_fp32_exact=True,independent_choices_exact=True,no_training=True,no_gpu=True,
        seconds=time.perf_counter()-started,worker_peak_bytes=psutil.Process().memory_info().peak_wset,guard=guard.read(),
        rows_sha256=sha(output),protocol_sha256=sha(RUN/'protocol.json'),inputs_sha256=sha(RUN/'inputs_used.json')),True)

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--compute' in sys.argv:
        sys.path.insert(0,str(common.GUARD.parent));from system_memory_guard import guard_entrypoint
        compute(guard_entrypoint(timeout=1200))
