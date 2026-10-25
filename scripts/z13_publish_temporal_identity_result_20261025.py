"""Publish only the completed trial, primary receipts and owned generated documents."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_temporal_pair_readiness_20261025 import read,write,sha,now
BASE=ROOT/'runs/z13_temporal_identity_20261025'


def main():
    assert read(BASE/'records_finalized.json')['complete']
    assert read(BASE/'training/report.json')['models']==90
    assert not read(BASE/'training/report.json')['internal_pass']
    assert read(BASE/'training/recovery2/controller_result.json')['all_owned_ended']
    assert 'OK: каталог согласован с источниками' in (BASE/'catalog_final.log').read_text(encoding='utf8')
    assert '65 passed' in (BASE/'tests_final.log').read_text(encoding='utf8')
    assert not (BASE/'publication.json').exists()
    git=lambda *a:subprocess.check_output(['git','--no-optional-locks',*a],cwd=ROOT).decode('utf8').strip()
    assert git('branch','--show-current')=='main' and not git('diff','--cached','--name-only')
    preservation=read(BASE/'vault_publication.json');assert preservation['official_generator_completed']
    protected={p.replace('\\','/'):h for p,h in preservation['protected_existing_edits'].items()}
    for rel,digest in protected.items():
        path=ROOT/rel
        assert (not path.exists()) if digest is None else path.is_file() and sha(path)==digest
    names=[
        'scripts/z13_record_temporal_identity_result_20261025.py',
        'scripts/z13_temporal_identity_continuation2_20261025.py',
        'scripts/z13_temporal_identity_final_20261025.py',
        'scripts/z13_temporal_identity_vectorized_recovery2_20261025.py',
        'scripts/z13_temporal_identity_vectorized_recovery3_20261025.py',
        'scripts/z13_finalize_temporal_identity_records_20261025.py',
        Path(__file__).relative_to(ROOT).as_posix(),
        'tests/test_z13_temporal_identity_vectorized.py',
        'scripts/build_flyvis_architecture.py','scripts/flyvis_bottlenecks_layout.py','scripts/obsidian_fly_problems.py',
        'experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md']
    for path in BASE.rglob('*'):
        # Exact source arrays/checkpoints are local, with immutable manifests.
        if path.is_file() and path.suffix in ('.json','.log') and path.name!='rows.json':
            names.append(path.relative_to(ROOT).as_posix())
    out=ROOT/'dissertation/architecture'
    for stem in ['FlyVis_architecture','FlyVis_bottlenecks']:
        for path in out.glob(stem+'*'):
            if path.is_file() and path.suffix in ('.drawio','.svg','.png','.json'):
                names.append(path.relative_to(ROOT).as_posix())
    names += ['dissertation/architecture/validation.json','dissertation/architecture/FlyVis_temporal_identity_detail.png']
    changed=set()
    for args in [('diff','--name-only','-z','HEAD','--','ObsidianFlyVis'),
                 ('ls-files','--others','--exclude-standard','-z','--','ObsidianFlyVis')]:
        changed.update(p for p in subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').split('\0') if p)
    graph=sorted(changed-set(protected))
    assert graph and all((ROOT/p).is_file() and Path(p).suffix=='.md' for p in graph)
    names=sorted(set(names+graph));assert '.gitignore' not in names
    for rel in names:
        path=(ROOT/rel).resolve();path.relative_to(ROOT)
        assert path.is_file() and rel not in protected
    write(BASE/'publication_manifest.json',dict(created=now().isoformat(),paths=names,
        protected_existing_graph_paths=len(protected),large_data_local_manifest_sha256=sha(BASE/'local_data_manifest.json'),
        primary_source_sha256=sha(BASE/'training/internal_report.json'),all_trials_finished=True))
    names.append((BASE/'publication_manifest.json').relative_to(ROOT).as_posix())
    for left in range(0,len(names),20):
        subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[left:left+20]],cwd=ROOT,check=True,capture_output=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''}
    # A few setup artifacts are unchanged and are deliberately already in history.
    assert staged<=set(names) and len(staged)>150
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Record negative temporal identity readout with matched controls and full audit',
        '-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True,capture_output=True)
    for rel,digest in protected.items():
        path=ROOT/rel
        assert (not path.exists()) if digest is None else path.is_file() and sha(path)==digest
    write(BASE/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,
        committed_files=len(staged),paths=sorted(staged),protected_graph_files_unchanged=True,
        no_weights_or_large_arrays_uploaded=True,primary_pass=False,models=90))
    print(commit)


if __name__=='__main__':main()
