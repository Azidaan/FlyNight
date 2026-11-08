import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_normalized_signals_20261108.common import *


def main():
    phase('comparison_phase.json');waiting=for_run(RUN/'comparison_planner')
    while not (RUN/'data_stage_complete.json').exists():
        assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
        for path in (RUN/'children').glob('*.ended.json'):assert read(path)['returncode']==0,('Data stage failed; preserve original logs and recover explicitly.',str(path))
        waiting('waiting_for_registered_map_data');time.sleep(5)
    complete=read(RUN/'data_stage_complete.json')
    if complete.get('restoration_stopped_at_logical_readiness'):
        write(RUN/'queue_result.json',dict(passed=True,stopped_before_neural=True,reason='Logical readiness failed; no signal comparison or full20.'),new=True);return
    assert complete['passed'];import research_native_confirm_risk_20261108.controller as launcher
    launcher.RUN=RUN;launcher.status=status
    for name in ['compare','audit']:launcher.run(name,f'research_bbox_normalized_signals_20261108/{name}.py')
    write(RUN/'queue_result.json',dict(passed=True,comparison_sha256=sha(RUN/'comparison_report.json'),audit_sha256=sha(RUN/'independent_audit.json')),new=True);status('normalized_spatial_comparison_finished')


if __name__=='__main__':main()
