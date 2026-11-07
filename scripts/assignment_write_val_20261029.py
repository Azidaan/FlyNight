"""Reuse the verified stream/audit engine; change only the admitted-write tracker."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import memory_write_val_20261029 as engine

def tracker_factory(args,server=None,weight=0.,shadow=False,policy='last',beta=.9):
    from tracking.query_memory_write import MemoryWriteTracker
    from tracking.query_assignment_write import AssignmentWriteTracker
    return AssignmentWriteTracker(args,server,weight,shadow) if weight else MemoryWriteTracker(args)

def main():
    source=inspect.getsource(engine.main)
    source=source.replace('memory_write_val','assignment_write_val')
    source=source.replace('Deep OC-SORT','KeepTrack')
    source=source.replace('from tracking.query_memory_write import MemoryWriteTracker','from scripts.assignment_write_val_20261029 import tracker_factory as MemoryWriteTracker')
    old='per={n:{} for n in [*reference_names,*names]}'
    assert source.count(old)==1
    source=source.replace(old,"per={n:{} for n in [*reference_names,'r50_mean','detic_mean',*names]}")
    old="for n,k in reference_names.items():per[n][video]=original['metrics'][k]"
    assert source.count(old)==1
    source=source.replace(old,old+"\n        previous_mean=service.read(RUN/'memory_write_val/videos'/video/'report.json')\n        for e in ['r50','detic']:per[e+'_mean'][video]=previous_mean['metrics'][e+'_mean']")
    old="for n,k in reference_names.items():pooled[n]=references['pooled'][k]"
    assert source.count(old)==1
    source=source.replace(old,old+"\n    previous_audit=service.read(RUN/'memory_write_val_audit/report.json')\n    for e in ['r50','detic']:pooled[e+'_mean']=previous_audit['pooled'][e+'_mean']")
    assert source.count("per[e+'_cautious']")==1
    source=source.replace("per[e+'_cautious']","per[e+'_guarded']")
    ns=dict(vars(engine));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
