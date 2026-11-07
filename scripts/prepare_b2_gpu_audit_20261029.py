"""Register unchanged 1200-loss replay on original device, new provenance."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    folder=RUN/'b2_training_gpu_predictor_audit';assert not folder.exists();folder.mkdir()
    script='scripts/audit_b2_training_gpu_predictor_20261029.py'
    files=[script,'scripts/audit_b2_sparse_training_20261029.py','tracking/b2_pilot_runtime.py',
           'runs/continuation_20261028/b2_sparse_training_audit_recovery3/controller_result.json',
           'runs/continuation_20261028/b2_predictor_device_profile_recovery1/report.json']
    service.write(folder/'protocol.json',dict(budget_seconds=86400,loss_absolute_tolerance=3e-5,
        scope='New independently executed all1200loss audit on original GPU with original predictor setup, frozen checkpoint/GT/splits/criteria; original CPU failure remains, CPU-GPU loss equivalence not claimed',
        reason='First-video directGPU60losses pass original tolerance, manual detector setup does not; CPU original-predictor loss equality remains unconfirmed',
        sha256={f:service.sha(ROOT/f) for f in files},bounded_by_global_save=True))
    service.write(RUN/'ready_jobs/143_b2_training_gpu_predictor_audit.json',dict(name='143_b2_training_gpu_predictor_audit',folder=folder.relative_to(ROOT).as_posix(),
        script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))
    print(folder)

if __name__=='__main__':main()
