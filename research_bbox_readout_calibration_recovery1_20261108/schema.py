import os,sys,runpy
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_readout_calibration_20261108.common import *


def main(obs):
    phase();before=sha(ROOT/'experiments/project_history.json');old=sys.argv
    try:
        sys.argv=['scripts/validate_project_history.py']
        try:runpy.run_path(str(ROOT/sys.argv[0]),run_name='__main__')
        except SystemExit as e:assert e.code in [None,0]
    finally:sys.argv=old
    assert before==sha(ROOT/'experiments/project_history.json');write(RUN/'recovery1/schema_passed.json',dict(passed=True,catalog_sha256=before,full_test_suite_required_before_publication=True),new=True)


if __name__=='__main__':main(guard())
