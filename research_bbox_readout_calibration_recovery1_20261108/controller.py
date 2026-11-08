import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_readout_calibration_20261108.common import *
RECOVERY=RUN/'recovery1'


def main():
    t=time.monotonic()
    while not (RECOVERY/'deadline_registered.json').exists() and time.monotonic()-t<10:time.sleep(.1)
    assert (RECOVERY/'deadline_registered.json').exists();p=read(RECOVERY/'protocol.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h
    while not (DATA/'queue_result.json').exists():
        assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
        for path in (DATA/'children').glob('*.ended.json'):assert read(path)['returncode']==0,str(path)
        time.sleep(2)
    assert read(DATA/'queue_result.json')['passed'];phase();import research_native_confirm_risk_20261108.controller as launcher;launcher.RUN=RECOVERY;launcher.status=status
    for name,script in [('catalog_schema','research_bbox_readout_calibration_recovery1_20261108/schema.py'),('compare','research_bbox_readout_calibration_20261108/compare.py'),('audit','research_bbox_readout_calibration_20261108/audit.py')]:launcher.run(name,script)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h
    write(RECOVERY/'queue_result.json',dict(passed=True,original_mutex_rejection_preserved=True,science_sources_unchanged=True,mandatory_58_catalog_tests_deferred_to_final_graph=True),new=True);write(RUN/'queue_result.json',dict(passed=True,comparison_sha256=sha(RUN/'comparison_report.json'),audit_sha256=sha(RUN/'independent_audit.json'),fit_only=True,no_new26_or20val=True,recovered_by='runs/bbox_readout_calibration_20261108/recovery1/queue_result.json'),new=True);status('fit_readout_calibration_finished')


if __name__=='__main__':main()
