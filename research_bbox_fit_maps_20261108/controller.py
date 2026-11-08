import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_fit_maps_20261108.common import *


def main():
    began=time.monotonic()
    while not (RUN/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'deadline_registered.json').exists();phase();import research_native_confirm_risk_20261108.controller as launcher;launcher.RUN=RUN;launcher.status=status
    for name in ['verify','inventory']:launcher.run(name,f'research_bbox_fit_maps_20261108/{name}.py')
    if not read(RUN/'inventory.json')['input_restore_gate']['proceed']:write(RUN/'queue_result.json',dict(passed=True,stopped_at_budget_gate=True,new_core_calls=0),new=True);return
    for name in ['setup_data','inputs','features','score']:launcher.run(name,f'research_bbox_fit_maps_20261108/{name}.py')
    write(RUN/'queue_result.json',dict(passed=True,data_only=True,features_sha256=sha(RUN/'features_complete.json'),scores_sha256=sha(RUN/'scores_complete.json'),no_classifier_or_full_tracker_evaluation=True),new=True);status('fit_data_restoration_finished')


if __name__=='__main__':main()
