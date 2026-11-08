"""One bounded local transition from completed CPU inputs to missing GPU maps."""
import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main():
    receipt=read(RUN/'feature_queue_protocol.json');parent=ROOT/receipt['input_guard_report']
    while True:
        state=read(parent)
        if state['returncode'] is not None:
            if state['returncode']!=0 or not (RUN/'input_complete.json').exists():
                write(RUN/'feature_queue_result.json',dict(passed=False,stage='inputs',parent=state),new=True);return
            break
        if datetime.now(MSK)>=datetime(2026,11,9,7,35,tzinfo=MSK):raise TimeoutError('MSK deadline')
        time.sleep(2)
    time.sleep(2)
    assert read(RUN/'input_complete.json')['passed']
    names=['research_identity_assignment_run_20261108/extract_features.py','research_identity_assignment_run_20261108/common.py',
       'research_identity_assignment_run_20261108/queue_features.py','scripts/z13_target_model.py','tracking/z13_readout.py',
       'tracking/flyvis_checkpoint.py','tracking/flyvis_motion.py','tracking/flyvis_adapter.py',
       'research_spatial_identity_20261107/extract_reference.py','runs/identity_assignment_20261108/input_complete.json',
       'runs/z13_local_peak_20261021/geometry.json','runs/z13_target_center_dev_20261018/C/normalization.json',
       'runs/z13_target_center_dev_20261018/R1/normalization.json','runs/flyvis/results/flow/0000/000/best_chkpt',
       'runs/spatial_identity_maps_20261107/features_completed.json','runs/z13_spatial_set_readout_20261025/recovery2/features/report.json']
    write(RUN/'feature_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in names},
        action='Reuse matching current UID maps and old reference maps. Infer only previously absent causal sequences. Same frozen checkpoint and normalizations.'),new=True)
    status('feature_recovery_starting')
    with (RUN/'feature_launcher.log').open('x',encoding='utf-8') as out,(RUN/'feature_launcher.err').open('x',encoding='utf-8') as err:
        r=subprocess.run([sys.executable,'-u','research_identity_assignment_run_20261108/extract_features.py'],cwd=ROOT,stdout=out,stderr=err)
    write(RUN/'feature_queue_result.json',dict(passed=r.returncode==0,returncode=r.returncode,features_complete=(RUN/'features_complete.json').exists()),new=True)


if __name__=='__main__':main()
