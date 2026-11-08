import os,sys,runpy
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import read,write,sha
from research_singleton_verify_20261108.common import guard
RUN=ROOT/'runs/singleton_native_confirm_20261108'


def main(obs):
    names=['experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md'];before={n:sha(ROOT/n) for n in names};old=sys.argv
    try:
        sys.argv=['scripts/validate_project_history.py']
        try:runpy.run_path(str(ROOT/sys.argv[0]),run_name='__main__')
        except SystemExit as e:assert e.code in [0,None]
    finally:sys.argv=old
    import pytest
    assert pytest.main(['-q','tests/test_project_history.py','--basetemp',str(RUN/'catalog_pytest_temp')])==0
    assert all(sha(ROOT/n)==h for n,h in before.items());write(RUN/'catalog_check.json',dict(passed=True,tests=58,inputs_unchanged=before),new=True)


if __name__=='__main__':main(guard())
