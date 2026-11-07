"""Service recovery: join three ordered logs by their stable query identity."""
from pathlib import Path
import inspect
import sys
import shutil
import hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import flyvis_pair_support_overlap_20261107 as old
RUN=old.RUN/'recovery1'
source=inspect.getsource(old.compute)
needle='for r,m,b in zip(rr,tr,br):'
replacement='''m_by_uid={m['uid']:m for m in tr}; b_by_uid={b['uid']:b for b in br}
                assert {r['uid'] for r in rr}==set(m_by_uid)==set(b_by_uid)
                for r in rr:
                    m=m_by_uid[r['uid']]; b=b_by_uid[r['uid']]'''
assert source.count(needle)==1
source=source.replace(needle,replacement)
if __name__=='__main__':
    if '--prepare' in sys.argv:
        RUN.mkdir(exist_ok=False)
        p=old.read(old.RUN/'protocol.json')
        p['recovery']='First video had nonidentical row order but equal sets of unique IDs. Old stage stopped before any completed video; keep its five provisional pair rows unused. Join by UID; retain all frame, track, kind, candidate and quadrant checks. No change to populations or outcomes.'
        p['sha256'][(old.RUN/'protocol.json').as_posix()]=old.sha(old.RUN/'protocol.json')
        for path in [Path(__file__).resolve().relative_to(ROOT),Path('scripts/audit_flyvis_pair_support_overlap_20261107.py')]:p['sha256'][path.as_posix()]=old.sha(path)
        p['executed_sha256']=hashlib.sha256(source.encode()).hexdigest()
        old.dump(RUN/'protocol.json',p,True)
    elif '--compute' in sys.argv:
        assert hashlib.sha256(source.encode()).hexdigest()==old.read(RUN/'protocol.json')['executed_sha256']
        scope=dict(vars(old),RUN=RUN);exec(compile(source,__file__,'exec'),scope);scope['compute']()
    elif '--audit' in sys.argv:
        from scripts import audit_flyvis_pair_support_overlap_20261107 as audit
        audit.RUN=ROOT/RUN;audit.main()
