import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_spatial_signals_20261108.common import *


def main(obs):
    import research_singleton_verify_20261108.prepare_inputs as source
    source.RUN=RUN;source.BASE=RUN/'reuse_pool';source.phase=phase;source.status=status;source.main(obs)
    sources=read(RUN/'input_phase.json')['sources'].copy();sources['runs/bbox_spatial_signals_20261108/input_complete.json']=sha(RUN/'input_complete.json')
    for path in (RUN/'retinal').glob('*'):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'feature_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources),new=True)


if __name__=='__main__':main(guard())
