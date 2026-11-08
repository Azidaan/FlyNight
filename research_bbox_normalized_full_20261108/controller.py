import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_normalized_full_20261108.common import *


def main():
    began=time.monotonic()
    while not (FULL/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (FULL/'deadline_registered.json').exists();full_phase()
    import research_native_confirm_risk_20261108.controller as launcher
    launcher.RUN=FULL;launcher.status=status
    for name in ['catalog_check','native_trace','prepare_views','features','rollouts','evaluate','additional','statistics','audit']:launcher.run(name,f'research_bbox_normalized_full_20261108/{name}.py')
    assert read(FULL/'independent_audit.json')['passed'];write(FULL/'queue_result.json',dict(passed=True,comparison_sha256=sha(FULL/'comparison_report.json'),audit_sha256=sha(FULL/'independent_audit.json')),new=True);status('normalized_full_pilot_finished')


if __name__=='__main__':main()
