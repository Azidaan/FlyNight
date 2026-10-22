"""Service-only recovery: preserve original mathematics, enable crash traceback."""
import faulthandler
import sys
from pathlib import Path
faulthandler.enable(all_threads=True)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.z13_localization_labels_20261023 as original
old_train=original.train_steps

def traced_train(*args, **kwargs):
    print('Entering frozen train_steps; maxsteps=',kwargs.get('maxsteps'),flush=True)
    result=old_train(*args,**kwargs)
    print('Leaving frozen train_steps',flush=True)
    return result

original.train_steps=traced_train
original.main()
