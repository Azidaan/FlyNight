"""Build a pinned package allowlist and test it without changing the Git index."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
CONFIG = json.loads((OUT/'package_inputs.json').read_text(encoding='utf-8'))


def git(*args, data=None):
    result = subprocess.run(['git', *args], cwd=ROOT, input=data, capture_output=True)
    return result.stdout.decode('utf-8').strip() if result.returncode == 0 else None


def descriptor(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def check_sources():
    rows = []
    for source in CONFIG['sources']:
        name = source['path']
        row = descriptor(ROOT/name)
        assert row['sha256'] == source['expected_sha256'], 'Source drift: '+name
        frozen = OUT/'frozen_sources'/name
        assert frozen.read_bytes() == (ROOT/name).read_bytes(), 'Immutable source snapshot mismatch: '+name
        row['head_blob'] = git('rev-parse', '--verify', '-q', 'HEAD:'+name)
        row['index_blob'] = git('rev-parse', '--verify', '-q', ':'+name)
        assert row['index_blob'] == row['head_blob'], 'Unexpected staged source: '+name
        if row['head_blob']:
            assert source['snapshots'], 'Missing original snapshot: '+name
            snapshot = ROOT/source['snapshots'][0]
            row['baseline'] = descriptor(snapshot)
            row['baseline_equals_HEAD'] = git('hash-object', '--path='+name, '--stdin',
                data=snapshot.read_bytes()) == row['head_blob']
            assert row['baseline_equals_HEAD'], 'Preexisting source changes: '+name
            row['classification'] = 'tracked_clean_at_start'
        else:
            assert not source['snapshots'], 'Prior-untracked source requires a separate patch: '+name
            row['classification'] = 'new_by_current_task'
        row['successive_snapshots'] = [descriptor(ROOT/p) for p in source['snapshots']]
        rows.append(row)
    return rows


def run_tests(cwd, label):
    command = [str(ROOT/'.venv/Scripts/python.exe'), '-m', 'pytest', *CONFIG['tests'],
               '-q', '-p', 'no:cacheprovider', '--basetemp='+str(OUT/(label+'_temp_'+str(time.time_ns())))]
    started = time.monotonic()
    with (OUT/(label+'_tests.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return dict(passed=result.returncode == 0, exit_code=result.returncode,
                command=command, cwd=str(cwd), seconds=time.monotonic()-started,
                log=str((OUT/(label+'_tests.log')).relative_to(ROOT)).replace('\\','/'))


def validate(rows):
    shared = run_tests(ROOT, 'shared')
    isolated = OUT/('isolated_'+str(time.time_ns()))
    isolated.mkdir()
    raw = subprocess.run(['git', 'archive', 'HEAD', 'tracking', 'scripts', 'tests',
        'models', 'datasets', 'evaluation'], cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for item in archive.getmembers():
            (isolated/item.name).resolve().relative_to(isolated.resolve())
            if not (item.isfile() or item.isdir()):
                raise ValueError('Unexpected archive entry '+item.name)
        archive.extractall(isolated)
    for row in rows:
        destination = isolated/row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT/'frozen_sources'/row['path'], destination)
    isolated_result = run_tests(isolated, 'isolated')
    result = dict(head=git('rev-parse', 'HEAD'), passed=shared['passed'] and isolated_result['passed'],
        shared=shared, isolated=isolated_result, overlay=rows,
        scope=f'HEAD plus exactly the {len(rows)} specified frozen source/test files; no vendor runtime or unrelated dirty sources')
    (OUT/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main(perform_validation):
    rows = check_sources()
    result = validate(rows) if perform_validation else json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    assert result['passed'], 'Package tests failed; inspect validation.json and logs'
    assert result['head'] == git('rev-parse', 'HEAD'), 'HEAD changed after validation'
    assert result['overlay'] == rows, 'Source overlay changed after validation'
    check_sources()
    sources = [row['path'] for row in rows]
    (OUT/'source_paths.txt').write_text('\n'.join(sources)+'\n', encoding='utf-8')
    commands = dict(shared_tests=result['shared']['command'], isolated_tests=result['isolated']['command'],
        rebuild=[str(ROOT/'.venv/Scripts/python.exe'), str(Path(__file__).relative_to(ROOT)), '--validate'],
        staging_for_parent_only=['git add --pathspec-from-file='+str((OUT/'source_paths.txt').relative_to(ROOT)),
                                'git add -f --pathspec-from-file='+str((OUT/'artifact_paths.txt').relative_to(ROOT))],
        native_probe_policy='Probe reports record their actual source versions. Repeated native probes must use separate output names/directories to preserve earlier evidence. No training, HOTA or GPU run is authorized by this script.')
    (OUT/'commands.json').write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding='utf-8')
    controls = [str((OUT/p).relative_to(ROOT)).replace('\\','/') for p in
        ('build_package.py','package_inputs.json','frozen_source_manifest.json','source_paths.txt','commands.json','validation.json',
         'shared_tests.log','isolated_tests.log')]
    artifacts = list(dict.fromkeys(CONFIG['artifact_paths']+controls))
    manifest_name = str((OUT/'manifest.json').relative_to(ROOT)).replace('\\','/')
    pathspec_name = str((OUT/'artifact_paths.txt').relative_to(ROOT)).replace('\\','/')
    (OUT/'artifact_paths.txt').write_text('\n'.join(artifacts+[manifest_name,pathspec_name])+'\n', encoding='utf-8')
    manifest = dict(head=git('rev-parse','HEAD'), index_changes=git('diff','--cached','--name-status'),
        sources=rows, artifacts=[descriptor(ROOT/p) for p in artifacts+[pathspec_name]],
        manifest_self_hash='The manifest is not recursively self-hashed; all other allowlisted files have exact SHA256 and byte size.',
        input_manifests=CONFIG['manifest_inputs'], scope=CONFIG['scope'],
        excluded='All preexisting untracked source, B1/B2/history source patches, learned-head followups, observation-transaction and HDF5 followups, plan/catalogue/Obsidian, weights, datasets and experiment caches. Frozen source copies are local test input only, not additional publication payload.',
        included_npz='Only small synthetic/native diagnostic outputs and independently recomputed endpoint errors; no dataset or model cache.',
        verification='Shared tree and HEAD archive with only this overlay both passed. Full source hashes and commands are in validation.json.',
        limits='Legacy geometry remains default after calibration/ROI counterevidence; new geometric modes are explicit candidates with no HOTA claim. Final native parity uses CPU. The historical original_native_report accidentally initialized CUDA through the old timer; original_native_v2_report explicitly corrects this. GpuLock canonical aliases coordinate only new-code processes in one Windows session; old processes require normal completion or restart.')
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(sources=len(rows), tracked_sources=sum(r['head_blob'] is not None for r in rows),
        artifacts=len(artifacts)+2, bytes=sum(r['size'] for r in rows)+sum(r['size'] for r in manifest['artifacts']),
        passed=True, manifest=manifest_name), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    main(parser.parse_args().validate)
