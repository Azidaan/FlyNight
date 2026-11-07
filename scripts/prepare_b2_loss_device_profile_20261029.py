"""Register one diagnostic with existing queue/guard, preserving failed audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    folder=RUN/'b2_loss_device_profile';assert not folder.exists();folder.mkdir()
    source=ROOT/'scripts/audit_b2_sparse_training_20261029.py'
    script='scripts/profile_b2_loss_device_20261029.py'
    service.write(folder/'protocol.json',dict(budget_seconds=86400,bounded_by_global_save=True,
        scope='First complete dev video, all six arms, GPU replay only to discriminate device numerics from checkpoint/runtime mismatch; no scientific promotion and no tolerance change',
        loss_absolute_tolerance=3e-5,source_sha256=service.sha(source),
        failed_cpu_audit_sha256=service.sha(RUN/'b2_sparse_training_audit_recovery3/controller_result.json'),
        script_sha256=service.sha(ROOT/script),scientific_verdict=None))
    service.write(RUN/'ready_jobs/139_b2_loss_device_profile.json',dict(name='139_b2_loss_device_profile',folder=folder.relative_to(ROOT).as_posix(),script=script,
        script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))
    print(folder)

if __name__=='__main__':main()
