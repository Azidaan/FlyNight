"""Save the tested preregistration checkpoint without active caches or unrelated changes."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_temporal_pair_readiness_20261025 import read,write,sha,now
BASE=ROOT/'runs/z13_temporal_identity_20261025'


def main():
    assert 'OK: каталог согласован с источниками' in (BASE/'catalog_registration.log').read_text(encoding='utf8')
    assert '63 passed' in (BASE/'tests_registration.log').read_text(encoding='utf8')
    assert read(BASE/'training/original_regression.json')['all_exact']
    assert not (BASE/'setup_publication.json').exists()
    names=['scripts/z13_temporal_identity_cuda_probe_20261025.py','scripts/z13_temporal_identity_features_20261025.py',
        'scripts/z13_temporal_identity_trial_20261025.py','scripts/z13_temporal_identity_trial_recovery1_20261025.py',
        'scripts/z13_temporal_identity_queue_20261025.py',Path(__file__).relative_to(ROOT).as_posix(),
        'tracking/z13_temporal_identity_readout.py','tests/test_z13_temporal_identity_readout.py',
        'experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'runs/z13_temporal_region_features_20261025/stalled_transfer_audit_20261025.json',
        'runs/z13_temporal_region_features_20261025/controller_result.json']
    for rel in ['legacy_preservation.json','catalog_registration.log','tests_registration.log',
        'cuda_probe/protocol.json','cuda_probe/report.json','cuda_probe/controller_result.json',
        'features/protocol.json','features/price.json','features/cached_index_parity.json',
        'training/protocol.json','training/original_regression.json','training/registration_recovery1.json',
        'training/regression_recovery1.json','training/regression.stdout.log','training/regression_recovery1.stdout.log']:
        names.append((BASE/rel).relative_to(ROOT).as_posix())
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT).decode('utf8').strip()
    assert git('branch','--show-current')=='main' and not git('diff','--cached','--name-only')
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,check=True,capture_output=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged==set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Register matched temporal identity readout and preserve CUDA stall recovery',
        '-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True,capture_output=True)
    write(BASE/'setup_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names,
        trained_result_not_yet_claimed=True,feature_cache_not_uploaded=True))
    print(commit)


if __name__=='__main__':main()
