"""Run one mandatory check below the existing independent memory supervisor."""
import os,sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
if __name__=='__main__':
    job=guard_entrypoint(timeout=1200)
    mode=sys.argv[1]
    if mode=='validator':
        sys.argv=['scripts/validate_project_history.py']
        runpy.run_path(sys.argv[0],run_name='__main__')
    elif mode in ('catalog','graph'):
        import pytest
        test='tests/test_project_history.py' if mode=='catalog' else 'tests/test_obsidian_fundamental_overview.py'
        raise SystemExit(pytest.main([test,'-q','--basetemp',str(job.run_dir/'pytest_temp')]))
    else:
        raise ValueError(mode)
