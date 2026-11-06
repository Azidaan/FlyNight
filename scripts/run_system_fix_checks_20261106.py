"""A bounded combined regression suite for the changes made in this audit."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
TESTS=[
    'test_flyvis_adapter.py','test_flyvis_checkpoint_strict.py','test_flyvis_device_contract.py',
    'test_flyvis_connectome_transfer.py','test_flyvis_motion_contracts.py','test_flyvis_tracker.py',
    'test_b1_timestamped.py','test_flyvis_temporal_residual.py','test_b1_tracker_runtime.py',
    'test_b1_real_inputs.py','test_tracker_lifecycle_reset.py','test_b2_lifecycle_isolation.py',
    'test_lazy_appearance_association.py','test_query_appearance_comparison.py',
    'test_query_assignment_write.py','test_query_bounded_write.py','test_query_memory_write.py',
    'test_frame_masa_appearance.py','test_history_pair_runtime.py','test_history_training_data.py',
    'test_history_pair_model.py','test_detection_cache_integrity.py','test_q_cache_readonly_cli.py',
    'test_flyvis_assignment_contracts.py','test_retina_state_contracts.py',
    'test_z13_checkpoint_transaction.py','test_z13_stop_guard_fallback.py','test_z13_resume.py',
    'test_motion_birth_reset.py','test_motion_augmented_bytetrack.py','test_motion_source_bytetrack.py',
    'test_birth_policy.py','test_xywh_bytetrack.py',
    'test_classical_motion_geometry.py','test_classical_motion_state.py',
    'test_flyvis_tracker_input_state.py','test_flyvis_secondary_geometry.py',
    'test_motion_calibration_binding.py','test_gpu_lock_lifecycle.py','test_gpu_lock_aliases.py',
    'test_learned_head_contracts.py','test_reliability_head_contracts.py',
    'test_flyvis_hdf5_transaction.py','test_flyvis_observation_transaction.py',
    'test_flyvis_activity_readout.py','test_b1_deterministic_runtime.py',
    'test_b1_deterministic_flag_restore.py','test_b1_residual_transaction.py',
]


def main():
    out=ROOT/'runs/flyvis_repairs_20261106'/('integrated_'+str(time.time_ns()))
    out.mkdir()
    child = '''import json,sys
from pathlib import Path
import torch
import pytest
torch.set_default_device('cpu')
before=torch.cuda.is_initialized()
code=int(pytest.main(sys.argv[2:]))
report=dict(pytest_returncode=code,cuda_initialized_before=before,
            cuda_initialized_after=torch.cuda.is_initialized(),
            torch_default_device=str(torch.get_default_device()))
Path(sys.argv[1]).write_text(json.dumps(report,indent=2),encoding='utf-8')
raise SystemExit(code or (2 if report['cuda_initialized_after'] else 0))
'''
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-c',child,str(out/'runtime.json'),
             *['tests/'+name for name in TESTS],'-q','-p','no:cacheprovider',
             '--basetemp='+str(out/'temp')]
    source_names=sorted(set(['tests/'+t for t in TESTS]+[
        p.relative_to(ROOT).as_posix() for folder in ('tracking','scripts')
        for p in (ROOT/folder).glob('*.py')]))
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
    (out/'command.json').write_text(json.dumps(dict(command=command,source_sha256=sources),indent=2),encoding='utf-8')
    with (out/'tests.log').open('wb') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    changed=[name for name,previous in sources.items()
             if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=previous]
    (out/'result.json').write_text(json.dumps(dict(returncode=result.returncode,
        passed=result.returncode==0 and not changed,concurrently_changed_sources=changed,
        test_files=len(TESTS),runtime=json.loads((out/'runtime.json').read_text(encoding='utf-8'))
            if (out/'runtime.json').exists() else None)),encoding='utf-8')
    print(str(out))
    print((out/'tests.log').read_text(encoding='utf-8',errors='replace')[-5000:])
    raise SystemExit(result.returncode or (2 if changed else 0))


if __name__=='__main__':
    main()
