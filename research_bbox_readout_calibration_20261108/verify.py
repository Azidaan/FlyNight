import sys,runpy,os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_readout_calibration_20261108.common import *


def main(obs):
    phase();before={n:sha(ROOT/n) for n in ['experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md']};previous=sys.argv
    try:
        sys.argv=['scripts/validate_project_history.py']
        try:runpy.run_path(str(ROOT/sys.argv[0]),run_name='__main__')
        except SystemExit as e:assert e.code in [None,0]
    finally:sys.argv=previous
    import pytest
    assert pytest.main(['-q','tests/test_project_history.py','--basetemp',str(RUN/'test_temp')])==0
    assert all(sha(ROOT/n)==h for n,h in before.items());write(RUN/'verification.json',dict(passed=True,inputs_unchanged=before),new=True)


if __name__=='__main__':main(guard())
