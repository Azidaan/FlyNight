import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import write
from research_singleton_verify_20261108.common import guard


def main(obs):
    import pytest
    assert pytest.main(['-q','tests/test_native_confirm_identity.py','--basetemp',str(ROOT/'runs/singleton_native_confirm_20261108/policy_unit_temp')])==0
    write(ROOT/'runs/singleton_native_confirm_20261108/policy_units.json',dict(passed=True,tests=6,scope='ID-only linking, working-ID protection, first-public requirement, single ownership, collision revocation, shared solver, zero mode. No quality claim.'),new=True)


if __name__=='__main__':main(guard())
