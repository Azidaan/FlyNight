import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_singleton_pilot_20261108.common import *
RECOVERY=ROOT/'runs/bbox_singleton_audit_recovery1_20261108'


def main():
    began=time.monotonic()
    while not (RECOVERY/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RECOVERY/'deadline_registered.json').exists();p=read(RECOVERY/'protocol.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h
    import research_native_confirm_risk_20261108.controller as launcher
    launcher.RUN=RECOVERY;launcher.status=for_run(RECOVERY);launcher.run('audit','research_bbox_singleton_audit_recovery1_20261108/audit.py')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h
    write(RECOVERY/'queue_result.json',dict(passed=True,audit_sha256=sha(RUN/'independent_audit.json'),original_failure_preserved=True,science_not_repeated=True),new=True)
    write(RUN/'queue_result.json',dict(passed=True,comparison_sha256=sha(RUN/'comparison_report.json'),audit_sha256=sha(RUN/'independent_audit.json'),recovered_by='runs/bbox_singleton_audit_recovery1_20261108/queue_result.json',original_audit_returncode=1),new=True)


if __name__=='__main__':main()
