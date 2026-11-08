import os,sys,runpy
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_full_20261108.common import *


def main(obs):
    before={n:sha(ROOT/n) for n in ['experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md']};old=sys.argv
    try:
        sys.argv=['scripts/validate_project_history.py']
        try:runpy.run_path(str(ROOT/sys.argv[0]),run_name='__main__')
        except SystemExit as e:assert e.code in [None,0]
    finally:sys.argv=old
    import pytest
    assert pytest.main(['-q','tests/test_project_history.py','--basetemp',str(FULL/'catalog_temp')])==0
    assert all(sha(ROOT/n)==h for n,h in before.items());write(FULL/'catalog_check.json',dict(passed=True,tests=58,inputs_unchanged=before),new=True)


if __name__=='__main__':main(guard())
