import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_readout_calibration_20261108.common import *


def main():
    t=time.monotonic()
    while not (RUN/'deadline_registered.json').exists() and time.monotonic()-t<10:time.sleep(.1)
    assert (RUN/'deadline_registered.json').exists();phase();import research_native_confirm_risk_20261108.controller as launcher;launcher.RUN=RUN;launcher.status=status;launcher.run('verify','research_bbox_readout_calibration_20261108/verify.py')
    while not (DATA/'queue_result.json').exists():
        assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
        for path in (DATA/'children').glob('*.ended.json'):assert read(path)['returncode']==0,str(path)
        time.sleep(3)
    assert read(DATA/'queue_result.json')['passed'] and not read(DATA/'queue_result.json').get('stopped_at_budget_gate')
    for name in ['compare','audit']:launcher.run(name,f'research_bbox_readout_calibration_20261108/{name}.py')
    write(RUN/'queue_result.json',dict(passed=True,comparison_sha256=sha(RUN/'comparison_report.json'),audit_sha256=sha(RUN/'independent_audit.json'),fit_only=True,no_new26_or20val=True),new=True);status('fit_readout_calibration_finished')


if __name__=='__main__':main()
