import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_full_recovery1_20261108.common import *


def main():
    began=time.monotonic()
    while not (RECOVERY/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RECOVERY/'deadline_registered.json').exists();recovery_phase();full_phase()
    from research_native_confirm_risk_20261108.controller import run
    run('full_recovery1_views','research_native_confirm_full_recovery1_20261108/prepare_views.py');run('full_recovery1_features','research_native_confirm_full_recovery1_20261108/features_wrapper.py')
    for name in ['rollouts','evaluate','additional','statistics','audit']:run('full_recovery1_'+name,f'research_native_confirm_full_20261108/{name}.py')
    assert read(FULL/'independent_audit.json')['passed'];write(FULL/'queue_result.json',dict(passed=True,comparison_sha256=sha(FULL/'comparison_report.json'),independent_audit_sha256=sha(FULL/'independent_audit.json'),technical_recovery=1),new=True);status('native_confirm_full_pilot_complete',technical_recovery=1)


if __name__=='__main__':main()
