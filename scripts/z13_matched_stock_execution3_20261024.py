"""Stock execution2 with verified FPS metadata path, unchanged scientific worker."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_matched_stock_execution2_20261024 as E
from scripts.z13_matched_fps_recovery_20261024 import install,OVERLAY
S=E.S


def verify():
    E.verify();p=S.read(S.STAGE/'execution_v3.json')
    for n,h in p['sha256'].items():assert S.sha(ROOT/n)==h,n


if __name__=='__main__':
    if '--prepare' in sys.argv:
        assert not (S.STAGE/'controller.json').exists() and not (S.STAGE/'execution_v3.json').exists()
        names=['scripts/z13_matched_stock_execution3_20261024.py','scripts/z13_matched_fps_recovery_20261024.py',(OVERLAY/'manifest.json').relative_to(ROOT).as_posix(),'scripts/z13_record_matched_trackers_20261024.py']
        S.write(S.STAGE/'execution_v3.json',dict(at=S.now().isoformat(),sha256={n:S.sha(ROOT/n) for n in names},change='Same execution2 worker, read FPS from validated metadata overlay. Preserve science protocol and source; preserve previous errors; parent completion follows verified recovery chain.'))
    elif '--worker' in sys.argv:
        verify();install();S.__file__=str(Path(__file__));ns=vars(S).copy();exec(compile(E.transformed(),str(Path(__file__)),'exec'),ns)
        try:ns['worker']()
        except Exception as e:S.write(S.STAGE/'status.json',dict(state='failed',stage=str(e),updated=S.now().isoformat()));raise
    elif '--job' in sys.argv:
        verify();install();mode,arm,video=sys.argv[2:5];assert mode in ('signature','masa') and video in S.VIDEOS and arm in S.SYSTEMS+('all',)
        result=S.build_signature(arm,video) if mode=='signature' else S.build_masa(video)
        paths=[S.RUN/arm/'signatures'/video/'signatures.npz'] if mode=='signature' else [S.RUN/a/'masa'/(video+'.npz') for a in S.SYSTEMS]+[S.STAGE/'baseline_masa_replay'/(video+'.npz')]
        S.write(S.STAGE/'jobs'/f'{mode}_{arm}_{video}.json',dict(result=result,outputs={q.relative_to(ROOT).as_posix():S.sha(q) for q in paths},at=S.now().isoformat()))
    else:raise RuntimeError('Explicit service mode required')
