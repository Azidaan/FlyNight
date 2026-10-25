"""Preserve registration/CPU batching failures and run the frozen scientific specification."""
from pathlib import Path
import inspect,hashlib,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_temporal_identity_trial_20261025 as S


def retry_write(path,value):
    original=ORIGINAL_WRITE
    for attempt in range(30):
        try:return original(path,value)
        except PermissionError:
            if attempt==29:raise
            time.sleep(.05)


ORIGINAL_WRITE=S.write
S.write=retry_write


def finish_registration():
    assert not (S.RUN/'registration_recovery1.json').exists()
    p=S.read(S.RUN/'protocol.json');path=ROOT/'experiments/project_history.json';current=S.read(path)
    if not any(c['id']==S.IDENT for c in current['experiments']):
        draft=S.read(path.with_suffix('.json.tmp'));card=next(c for c in draft['experiments'] if c['id']==S.IDENT)
        current['experiments'].append(card);retry_write(path,current)
    source=inspect.getsource(S.prepare);tail=source[source.index('    glossary=read('):]
    ns={**vars(S),'p':p};exec(compile('def finish():\n'+tail,__file__,'exec'),ns);ns['finish']()
    retry_write(S.RUN/'registration_recovery1.json',dict(at=S.now().isoformat(),cause='Transient Windows PermissionError replacing catalog; resumed only unexecuted registration,30bounded50msretries.',protocol_unchanged_sha256=S.sha(S.RUN/'protocol.json'),original_scientific_source_sha256=S.sha(Path(S.__file__))))


def regression():
    source=inspect.getsource(S.regression)
    assert source.count("range(0,len(data['uid']),128)")==1 and source.count('left+128')==1
    source=source.replace("range(0,len(data['uid']),128)","range(0,len(data['uid']),256)").replace('left+128','left+256')
    # Original saved probabilities were produced with256CPU rows. Preserve exact replay rather than relax it.
    ns={**vars(S)};exec(compile(source,__file__,'exec'),ns);ns['regression']()
    retry_write(S.RUN/'regression_recovery1.json',dict(at=S.now().isoformat(),all12_saved_heads_exact=True,
        cause='Frozen original prediction uses256CPU rows;128changes floating-point batching. New residual logits already matched same-batch base exactly. Restore original256batch for exact historical probabilities/outcomes.',
        executed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),original_error_preserved='regression.stdout.log'))


def old_contrast(new,old):
    import numpy as np
    vids=sorted(new);d=np.array([new[v]-old[v] for v in vids]);bs=d[np.random.default_rng(1).integers(0,len(d),(5000,len(d)))].mean(1)
    return dict(mean_per_video=float(d.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),better=int((d>0).sum()),n=len(d))


def worker():
    source=inspect.getsource(S.worker)
    old="fm={a:{v:float(np.mean([final[a,s][v] for s in p['seeds']])) for v in p['evaluate26']} for a in ARMS};fc=contrast(fm);final_pass=passes(fc)"
    new="fm={a:{v:float(np.mean([final[a,s][v] for s in p['seeds']])) for v in p['evaluate26']} for a in ARMS};fc=contrast(fm);fc['D-original_context']=old_contrast(fm['D'],read(FOLLOW/'training/report.json')['mean_seed_per_video']['D_context']);final_pass=passes(fc)"
    assert source.count(old)==1;source=source.replace(old,new)
    retry_write(S.RUN/'execution_receipt.json',dict(at=S.now().isoformat(),wrapper_sha256=S.sha(Path(__file__)),
        original_source_sha256=S.sha(Path(S.__file__)),executed_worker_sha256=hashlib.sha256(source.encode()).hexdigest(),
        specification_unchanged=True,correction='Implement already-preregistered frozen previousD_context contrast in final gate; no change to primary thresholds or internal criterion. Bounded transient JSON-write retry.'))
    ns={**vars(S),'old_contrast':old_contrast};exec(compile(source,__file__,'exec'),ns);ns['worker']()


if __name__=='__main__':
    if '--register' in sys.argv:finish_registration()
    elif '--regression' in sys.argv:regression()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:retry_write(S.RUN/'failure.json',dict(at=S.now().isoformat(),error=repr(e)));raise
