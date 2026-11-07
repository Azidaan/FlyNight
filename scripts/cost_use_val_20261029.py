"""Use the verified full stream/audit engine with three new cost trackers."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import memory_write_val_20261029 as engine

def compare_extended(a,b):
    import numpy as np
    assert set(a)==set(b) and len(a)==20
    videos=sorted(a);d=np.array([a[v]['HOTA']-b[v]['HOTA'] for v in videos])
    means=d[np.random.default_rng(1).integers(0,20,(20000,20))].mean(1)
    return dict(mean_per_video=float(d.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
        ci_bonferroni_9875=np.quantile(means,[.00625,.99375]).tolist(),
        ci_global7=np.quantile(means,[.05/14,1-.05/14]).tolist(),better=int((d>0).sum()),n=20)

def descriptive_compare(a,b):
    r=engine.compare(a,b)
    return {k:v for k,v in r.items() if k!='ci_bonferroni_9875'}

def tracker_factory(args,server=None,weight=0.,shadow=False,policy='centered',beta=None):
    from tracking.query_cost_use import CostUseTracker
    mode='centered' if policy=='cautious' else policy # original engine's zero-shadow label
    return CostUseTracker(args,server,weight,shadow,mode)

def main():
    source=inspect.getsource(engine.main).replace('memory_write_val','cost_use_val')
    source=source.replace('from tracking.query_memory_write import MemoryWriteTracker','from scripts.cost_use_val_20261029 import tracker_factory as MemoryWriteTracker')
    old="comparisons={e+'_'+b:compare(per[e+'_cautious'],per[e+'_'+b]) for e in ['r50','detic'] for b in ['last','mean']}"
    assert source.count(old)==1
    source=source.replace(old,"comparisons={e+'_'+b:compare(per[e+'_centered'],per[e+'_last'] if b=='old' else per['r50_shift']) for e in ['r50','detic'] for b in ['old','shift']}")
    old="native_comparisons={n:compare(per[n],per['native']) for n in names}"
    assert source.count(old)==1
    source=source.replace(old,old+"\n    extractor_descriptive=descriptive_compare(per['detic_centered'],per['r50_centered'])\n    if audit:\n        assert native_comparisons==service.read(gpu/'report.json')['native_comparisons']\n        assert extractor_descriptive==service.read(gpu/'report.json')['extractor_descriptive']")
    old='primary_comparisons=comparisons,native_comparisons=native_comparisons,pooled=pooled,'
    assert source.count(old)==1
    source=source.replace(old,old+'extractor_descriptive=extractor_descriptive,')
    source=source.replace('New memory policies on exact cached GPU vectors','New cost rules on exact cached GPU vectors')
    source=source.replace('Memory write policy in our fixed query tracker, not complete Deep OC-SORT or FlyVis core; old exact GPU FP32 vectors reused, predictor arithmetic unchanged.',
        'Appearance-cost use in our query tracker; last-observation memory fixed. Exact old GPU FP32 vectors reused; only float32 cost rule changes. Not full Deep SORT/BoT-SORT or FlyVis core.')
    ns={**vars(engine),'compare':compare_extended,'descriptive_compare':descriptive_compare}
    exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
