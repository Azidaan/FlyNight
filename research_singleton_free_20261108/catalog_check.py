"""Required catalog validator and tests, one externally guarded CPU process."""
import os,sys,runpy
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main(obs):
    phase();original=sys.argv
    try:
        sys.argv=['scripts/validate_project_history.py']
        try:runpy.run_path(str(ROOT/sys.argv[0]),run_name='__main__')
        except SystemExit as exc:assert exc.code in [0,None],exc.code
    finally:sys.argv=original
    import pytest
    assert pytest.main(['-q','tests/test_project_history.py','--basetemp',str(RUN/'catalog_pytest_temp')])==0
    write(RUN/'catalog_check.json',dict(passed=True,catalog_sha256=sha(ROOT/'experiments/project_history.json'),
        glossary_sha256=sha(ROOT/'experiments/metrics_glossary.json')),new=True)


if __name__=='__main__':main(guard())
