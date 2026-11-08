"""Same sparse-anchor return and sustained-identity definitions as earlier pilots."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_bbox_normalized_full_20261108.common import *


def main(obs):
    import numpy as np
    import research_identity_assignment_run_20261108.full_precision_manager as bridge
    from research_identity_assignment_run_20261108.full_additional import measure
    bridge.FULL=FULL;bridge.raw_path=lambda protocol,arm,video:FULL/'rollouts'/arm/(video+'.npz')
    full_phase();p=read(FULL/'protocol.json');assert read(FULL/'rollouts_complete.json')['passed'];cache={};summary={}
    for arm in p['parameters']:
        totals=Counter();files={}
        for video in p['videos']:
            path=FULL/'additional'/arm/(video+'.json');raw=bridge.raw_path(p,arm,video)
            with np.load(raw,allow_pickle=False) as z:signature=(video,hashlib.sha256(z['outputs'].tobytes()).hexdigest())
            if path.exists():r=read(path)
            elif arm=='B0':
                source=PARENT/'additional/B0'/(video+'.json');old=read(source);assert old['raw_sha256']==sha(raw) and old['clock_sha256']==sha(FULL/'runtime_inputs'/(video+'.json'));r=dict(old,reused_from=source.relative_to(ROOT).as_posix());write(path,r,new=True)
            elif signature in cache:
                old,source=cache[signature];r=dict(old,arm=arm,raw_sha256=sha(raw),reused_from=source.relative_to(ROOT).as_posix(),identical_output_array_verified=True);write(path,r,new=True)
            else:r=measure(video,arm,p);write(path,r,new=True)
            assert r['raw_sha256']==sha(raw);cache.setdefault(signature,(r,path));totals.update(r['totals']);files[path.relative_to(ROOT).as_posix()]=sha(path)
        summary[arm]=dict(totals=dict(totals),sources=files);status('native_confirm_full_identity_returns',arm=arm,guard=obs.read())
    full_phase();write(FULL/'additional_report.json',dict(passed=True,arms=summary,scope='Identical sparse actual-anchor definitions, unknown and censored separate. Public canonical IDs compared; native states are unchanged.'),new=True)


if __name__=='__main__':main(guard())
