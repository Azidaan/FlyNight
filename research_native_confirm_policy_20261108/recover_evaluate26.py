import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def main(obs):
    p=phase('evaluate26_metadata_phase.json')
    import research_singleton_native_confirm_20261108.native_trace as native
    native.status=lambda stage,**v:status('native_evaluate26_metadata_recovery',**v)
    results={}
    for video in p['videos']:results[video]=native.trace(video,obs,p)
    phase('evaluate26_metadata_phase.json');write(RUN/'evaluate26_metadata_complete.json',dict(passed=True,videos=results,labels_opened=False,new_core_calls=0,new_detector_calls=0),new=True)


if __name__=='__main__':main(guard())
