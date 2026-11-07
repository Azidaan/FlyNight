"""Use the accepted dev replay engine for two fresh-admission controls."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import assignment_write_dev_20261029 as engine

def tracker_factory(args,server=None,weight=0.,shadow=False,policy='bounded'):
    from tracking.query_bounded_write import BoundedWriteTracker,PeriodicWriteTracker
    assert policy in ('bounded','periodic')
    return (BoundedWriteTracker if policy=='bounded' else PeriodicWriteTracker)(args,server,weight,shadow)

def main():
    source=inspect.getsource(engine.main).replace('assignment_write_dev','bounded_write_dev')
    source=source.replace('from tracking.query_assignment_write import AssignmentWriteTracker','from scripts.bounded_write_dev_20261029 import tracker_factory as AssignmentWriteTracker')
    old="trackers.update({n+'_guarded':AssignmentWriteTracker(args,s,p['appearance_weight']) for n,s in servers.items()})"
    assert source.count(old)==1
    source=source.replace(old,"\n    for n,s in servers.items():\n        for policy in ['bounded','periodic']:trackers[n+'_'+policy]=AssignmentWriteTracker(args,s,p['appearance_weight'],policy=policy)")
    assert source.count("for n in ['r50_guarded','detic_guarded']:")==1
    source=source.replace("for n in ['r50_guarded','detic_guarded']:","for n in p['new_variants']:")
    source=source.replace('New assignment-based write admission on exact dev vectors','New bounded and periodic write controls on exact dev vectors')
    ns=dict(vars(engine));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
