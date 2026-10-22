"""Service-only CPU mask fix, identical frozen worker and scientific settings."""
import faulthandler
import os
from pathlib import Path
import sys
import time
faulthandler.enable(all_threads=True)
# On this Windows cu130 runtime, -1 crashes CPU autograd; empty mask passes.
# The original successful candidate_data104 CPU run also used an empty mask.
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.z13_localization_labels_20261023 as original
try:
    original.worker()
    original.write(original.RUN/'status.json',dict(state='completed',stage='Результаты18моделей сохранены',updated=original.now().isoformat(),elapsed_seconds=round(time.monotonic()-original.T0,2),models_done=18,models_total=18))
except Exception as exc:
    original.write(original.RUN/'status.json',dict(state='failed',stage=str(exc),updated=original.now().isoformat(),elapsed_seconds=round(time.monotonic()-original.T0,2)))
    raise
