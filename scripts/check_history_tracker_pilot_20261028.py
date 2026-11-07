import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    f=RUN/'history_tracker_checks'
    proof=service.read(RUN/'history_tracker_real100_interface.json')
    frozen=service.read(RUN/'history_tracker_pilot/frozen.json')
    assert proof['passed'] and proof['raw_native_exact'] and proof['zero_full_output_state_exact']
    assert proof['checkpoint_sha256']==frozen['models']['v0_seed42']['sha256']
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_history_pair_model.py','tests/test_history_pair_runtime.py','tests/test_history_tracker_summary.py','-q']]):
        with (f/f'check_{i}.log').open('w',encoding='utf-8') as out:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=300)
        assert r.returncode==0
    service.write(f/'report.json',dict(passed=True,updated=service.now()))


if __name__=='__main__':main()
