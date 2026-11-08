import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_20261108.common import *


def main(obs):
    configure_paths()
    import research_singleton_free_20261108.full_additional as module
    module.RUN=RUN;module.phase=phase;module.status=status;module.main(obs)


if __name__=='__main__':main(guard())
