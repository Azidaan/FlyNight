"""New cost intervention on the exact accepted development vectors."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import assignment_write_dev_20261029 as engine

def tracker_factory(args,server=None,weight=0.,shadow=False,policy='centered'):
    from tracking.query_cost_use import CostUseTracker
    return CostUseTracker(args,server,weight,shadow,policy)

def main():
    source=inspect.getsource(engine.main).replace('assignment_write_dev','cost_use_dev')
    source=source.replace('from tracking.query_assignment_write import AssignmentWriteTracker','from scripts.cost_use_dev_20261029 import tracker_factory as AssignmentWriteTracker')
    old="trackers.update({n+'_guarded':AssignmentWriteTracker(args,s,p['appearance_weight']) for n,s in servers.items()})"
    assert source.count(old)==1
    source=source.replace(old,"\n    for name in p['new_variants']:\n        encoder,policy=name.split('_');trackers[name]=AssignmentWriteTracker(args,servers[encoder],p['appearance_weight'],policy=policy)")
    assert source.count("for n in ['r50_guarded','detic_guarded']:")==1
    source=source.replace("for n in ['r50_guarded','detic_guarded']:","for n in p['new_variants']:")
    source=source.replace('New assignment-based write admission on exact dev vectors','New centered and signal-free shifted costs on exact dev vectors')
    source=source.replace('not full KeepTrack','not full Deep SORT or BoT-SORT')
    source=source.replace('not scientific memory benefit','not scientific appearance-cost benefit').replace('May retain stale or initially wrong memory.','Cost changes can reward wrong identity; common shift may explain improvements.')
    source=source.replace('            assert time.time()<deadline-30',"            assert __import__('shutil').disk_usage(ROOT).free>5_000_000_000,'Binding disk reserve5GB'\n            assert time.time()<deadline-30")
    ns=dict(vars(engine));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
