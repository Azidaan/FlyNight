"""Publish only this bounded readiness audit and its own derived graph notes."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import z13_temporal_pair_readiness_20261025 as S
from scripts.z13_build_vault_preserving_edits import verify_local_edits
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    run=S.RUN
    assert not (run/'publication.json').exists()
    assert S.read(run/'recorded.json')['complete']
    assert 'OK: каталог согласован с источниками' in (run/'catalog_validation.log').read_text(encoding='utf8')
    assert '62 passed' in (run/'tests.log').read_text(encoding='utf8')
    assert S.read(run/'label_provenance_and_cost_audit.json')['all_candidate_identity_labels_rebound_to_primary_annotations']
    control=S.read(run/'controller_result.json');ps=WindowsProcesses()
    assert control['all_owned_ended'] and all(ps.creation(int(pid))!=birth for pid,birth in control['owned'].items())
    protected=S.read(run/'vault_preservation.json')['protected_existing_edits']
    verify_local_edits({(ROOT/name).resolve():digest for name,digest in protected.items()})
    initial=S.read(run/'foreign_snapshot.json');assert S.sha(ROOT/'.gitignore')==initial['.gitignore']
    # The input and identity-label dataset stays local; publish reproducible indexes and hashes.
    files={}
    for folder in ('inputs','supervision'):
        for path in (run/folder).glob('*.json'):
            files[path.relative_to(ROOT).as_posix()]=dict(sha256=S.sha(path),bytes=path.stat().st_size)
    S.write(run/'local_dataset_manifest.json',dict(at=S.now().isoformat(),files=files,
        all_260_files_local=len(files)==260,source='Original inputs and labels are reconstructed by the frozen preparation script; no neural activity computed.'))
    S.write(run/'final_checks.json',dict(at=S.now().isoformat(),catalog_ok=True,tests_passed=62,
        all_owned_ended=True,protected_graph_and_gitignore_unchanged=True,graph_rebuilt=True,
        report_sha256=S.sha(run/'report.json'),local_dataset_files=len(files),
        note='Two pre-existing validator warnings concern placeholder fly_input_norm_hota_v1 paths; no new warning.'))
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()
    assert git('branch','--show-current')=='main'
    assert not git('diff','--cached','--name-only')
    names=['scripts/z13_temporal_pair_readiness_20261025.py','scripts/z13_record_temporal_pair_readiness_20261025.py',
        'scripts/z13_temporal_pair_prepare_recovery1_20261025.py','scripts/z13_audit_temporal_pair_labels_20261025.py',
        Path(__file__).relative_to(ROOT).as_posix(),'tests/test_z13_temporal_pair_readiness.py',
        'experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md']
    names += [x.relative_to(ROOT).as_posix() for x in run.glob('*.json') if x.name!='publication.json']
    names += [x.relative_to(ROOT).as_posix() for x in (run/'videos').glob('*.json')]
    names += [(run/n).relative_to(ROOT).as_posix() for n in ['catalog_validation.log','tests.log','vault_build.log']]
    own_graph=['ObsidianFlyVis/04 Термины и метрики/Термины и метрики.md',
        'ObsidianFlyVis/04 Термины и метрики/Идеальный запас действий при наличии временной пары.md',
        'ObsidianFlyVis/04 Термины и метрики/Охват причинно доступных пар областей.md',
        'ObsidianFlyVis/06 Журнал/Дни/2026-10-25.md']
    own_graph += [x.relative_to(ROOT).as_posix() for x in (ROOT/'ObsidianFlyVis/06 Журнал').glob('*итог подготовки реальных временных пар*')]
    assert all(n not in {x.replace(chr(92),'/') for x in protected} for n in own_graph)
    names=sorted(set(names+own_graph))
    for name in names:
        (ROOT/name).resolve().relative_to(ROOT.resolve());assert (ROOT/name).is_file()
    for i in range(0,len(names),20):
        subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,check=True,capture_output=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''}
    assert staged==set(names),(staged-set(names),set(names)-staged)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Prepare causal region pairs and audit identity supervision on130videos',
        '-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True,capture_output=True)
    verify_local_edits({(ROOT/name).resolve():digest for name,digest in protected.items()})
    assert S.sha(ROOT/'.gitignore')==initial['.gitignore']
    S.write(run/'publication.json',dict(at=S.now().isoformat(),commit=commit,push_completed=True,paths=names,
        local_dataset_retained=True,large_dataset_uploaded=False))
    print(commit)


if __name__=='__main__':
    main()
