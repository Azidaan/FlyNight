import os,sys,runpy
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
mode=sys.argv[1]
if mode=='validate':
 sys.argv=[str(ROOT/'scripts/validate_project_history.py')];runpy.run_path(sys.argv[0],run_name='__main__')
else:
 assert mode=='tests'
 import pytest
 raise SystemExit(pytest.main(['tests/test_identity_repair_components.py','tests/test_project_history.py','-q']))
