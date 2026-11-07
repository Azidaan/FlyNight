"""Только до импорта NumPy задать тот же BLAS-режим,что был у производителя."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_masa_feature_contribution_20261027 import main
if __name__=='__main__':main()
