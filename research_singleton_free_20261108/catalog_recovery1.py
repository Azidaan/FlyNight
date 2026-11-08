"""Repeat the required suite with an explicit unchanged-file witness."""
import os,sys,runpy
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main(obs):
    phase();paths=['experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'scripts/validate_project_history.py','tests/test_project_history.py'];before={n:sha(ROOT/n) for n in paths}
    original=sys.argv
    try:
        sys.argv=['scripts/validate_project_history.py']
        try:runpy.run_path(str(ROOT/sys.argv[0]),run_name='__main__')
        except SystemExit as exc:assert exc.code in [0,None],exc.code
    finally:sys.argv=original
    import pytest
    assert pytest.main(['-q','tests/test_project_history.py','--basetemp',str(RUN/'catalog_pytest_temp_recovery1')])==0
    assert all(sha(ROOT/n)==h for n,h in before.items()),'A catalog/test input changed during the stable check.'
    write(RUN/'catalog_check_recovery1.json',dict(passed=True,tests=58,inputs_unchanged=before,
        original_failure_preserved='runs/singleton_free_20261108/children/catalog.ended.json'),new=True)


if __name__=='__main__':main(guard())
