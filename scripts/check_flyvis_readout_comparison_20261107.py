"""One guarded sequential CPU check, imported before Torch."""
import os,sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
if __name__=='__main__':
    job=guard_entrypoint(timeout=2400);mode=sys.argv[1]
    if mode=='validator':
        sys.argv=['scripts/validate_project_history.py'];runpy.run_path(sys.argv[0],run_name='__main__')
    elif mode=='audit':
        from scripts.audit_flyvis_readout_comparison_20261107 import main
        main()
    else:
        import pytest
        files={'unit':['tests/test_flyvis_cached_readouts.py'],'catalog':['tests/test_project_history.py'],'graph':['tests/test_obsidian_fundamental_overview.py']}[mode]
        raise SystemExit(pytest.main([*files,'-q','--basetemp',str(job.run_dir/'pytest_temp')]))
