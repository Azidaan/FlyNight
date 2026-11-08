"""Same predefined return metrics; reuse identical saved trajectories and labels."""
import os,sys,shutil
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_singleton_free_20261108.common import *


def main(obs):
    import numpy as np
    from research_singleton_free_20261108.full_paths import FULL,raw_path
    import research_identity_assignment_run_20261108.full_precision_manager as bridge
    from research_identity_assignment_run_20261108.full_additional import measure
    bridge.FULL=FULL;bridge.raw_path=raw_path
    phase('full_phase.json');p=read(FULL/'protocol.json');assert read(FULL/'rollouts_complete.json')['passed'];summary={};cache={}
    parent=BASE/'full_precision'
    for arm in p['parameters']:
        totals=Counter();sources={}
        for video,entry in p['videos'].items():
            path=FULL/'additional'/arm/(video+'.json');raw=raw_path(p,arm,video);signature=(video,sha(raw))
            if path.exists():result=read(path)
            elif signature in cache:result=dict(cache[signature],arm=arm,reused_identical_raw=True);write(path,result,new=True)
            else:
                if arm=='B0':
                    source=parent/'additional/B0'/(video+'.json');result=read(source)
                    assert result['raw_sha256']==sha(raw) and result['clock_sha256']==sha(FULL/'runtime_inputs'/(video+'.json'))
                    result=dict(result,reused_from=source.relative_to(ROOT).as_posix());write(path,result,new=True)
                else:
                    metric=read(FULL/'metrics'/arm/(video+'.json'));source=metric.get('reused_metric_source');reusable=None
                    if source:
                        source_path=(ROOT/source).resolve()
                        try:relative=source_path.relative_to((parent/'metrics').resolve())
                        except ValueError:relative=None
                        if relative is not None:
                            oldpath=parent/'additional'/relative
                            if oldpath.exists():
                                reusable=read(oldpath)
                                assert metric['identical_output_arrays_verified'] and reusable['clock_sha256']==sha(FULL/'runtime_inputs'/(video+'.json'))
                                result=dict(reusable,arm=arm,raw_sha256=sha(raw),reused_from=oldpath.relative_to(ROOT).as_posix(),same_output_array_verified=True)
                    if reusable is None:result=measure(video,arm,p)
                    write(path,result,new=True)
                cache[signature]=result
            cache.setdefault(signature,result)
            totals.update(result['totals']);sources[path.relative_to(ROOT).as_posix()]=sha(path)
        summary[arm]=dict(totals=dict(totals),sources=sources)
        status('full_first_problem_return_metrics',arm=arm,guard=obs.read())
    phase('full_phase.json');write(FULL/'additional_report.json',dict(passed=True,arms=summary,
        definitions_source_sha256=sha(BASE/'full_execution_phase.json'),scope='Same sparse-label return definitions, unknown/censored explicit. No GT in tracker.'),new=True)


if __name__=='__main__':main(guard())
