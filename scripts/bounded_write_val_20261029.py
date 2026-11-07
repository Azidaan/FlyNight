"""Same verified stream, new bounded-versus-periodic write controls."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import memory_write_val_20261029 as engine

def tracker_factory(args,server=None,weight=0.,shadow=False,policy='bounded',beta=.9):
    from tracking.query_bounded_write import BoundedWriteTracker,PeriodicWriteTracker
    from tracking.query_memory_write import MemoryWriteTracker
    if not weight:return MemoryWriteTracker(args)
    assert policy in ('bounded','periodic','cautious')
    return (PeriodicWriteTracker if policy=='periodic' else BoundedWriteTracker)(args,server,weight,shadow)

def main():
    source=inspect.getsource(engine.main).replace('memory_write_val','bounded_write_val').replace('Deep OC-SORT','KeepTrack')
    source=source.replace('from tracking.query_memory_write import MemoryWriteTracker','from scripts.bounded_write_val_20261029 import tracker_factory as MemoryWriteTracker')
    old='per={n:{} for n in [*reference_names,*names]}';assert source.count(old)==1
    source=source.replace(old,"per={n:{} for n in [*reference_names,'r50_guarded','detic_guarded',*names]}")
    old="for n,k in reference_names.items():per[n][video]=original['metrics'][k]";assert source.count(old)==1
    source=source.replace(old,old+"\n        prior_guard=service.read(RUN/'assignment_write_val/videos'/video/'report.json')\n        for e in ['r50','detic']:per[e+'_guarded'][video]=prior_guard['metrics'][e+'_guarded']")
    old="for n,k in reference_names.items():pooled[n]=references['pooled'][k]";assert source.count(old)==1
    source=source.replace(old,old+"\n    prior_guard_audit=service.read(RUN/'assignment_write_val_audit/report.json')\n    for e in ['r50','detic']:pooled[e+'_guarded']=prior_guard_audit['pooled'][e+'_guarded']")
    assert source.count("per[e+'_cautious']")==1 and source.count("for b in ['last','mean']")==1
    source=source.replace("per[e+'_cautious']","per[e+'_bounded']").replace("for b in ['last','mean']","for b in ['last','periodic']")
    old="native_comparisons={n:compare(per[n],per['native']) for n in names}";assert source.count(old)==1
    source=source.replace(old,old+"\n    mechanism_comparisons={e:compare(per[e+'_bounded'],per[e+'_guarded']) for e in ['r50','detic']}\n    if audit:assert mechanism_comparisons==service.read(gpu/'report.json')['mechanism_comparisons']")
    old='primary_comparisons=comparisons,native_comparisons=native_comparisons,pooled=pooled,';assert source.count(old)==1
    source=source.replace(old,old+'mechanism_comparisons=mechanism_comparisons,')
    ns=dict(vars(engine));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
