"""Prepare an exact allowlist; never stage, commit, clean or reset Git files."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
FIX = 'runs/flyvis_system_fixes_20261106'
REPAIR = 'runs/flyvis_repairs_20261106'
SOURCES = [
    'tracking/flyvis_adapter.py', 'tracking/flyvis_checkpoint.py',
    'tracking/flyvis_connectome_swap.py', 'tracking/detection_cache.py',
    'tracking/inference_with_q.py', 'tracking/cache_integrity.py',
    'tracking/cache_readonly.py', 'tracking/z13_checkpoint.py',
    'scripts/z13_window_runner.py', 'scripts/run_tracking_three_modes.py',
    'scripts/sweep_lqa_k.py', 'scripts/q_aggregation_ablation.py',
    'scripts/correlation_experiment.py', 'scripts/check_detection_cache_parity_20261106.py',
    'tests/test_flyvis_checkpoint_strict.py', 'tests/test_flyvis_device_contract.py',
    'tests/test_flyvis_connectome_transfer.py', 'tests/test_detection_cache_integrity.py',
    'tests/test_q_cache_readonly_cli.py', 'tests/test_z13_checkpoint_transaction.py',
    'tests/test_z13_stop_guard_fallback.py',
]
TESTS = ['tests/test_flyvis_adapter.py', *[p for p in SOURCES if p.startswith('tests/')],
         'tests/test_z13_resume.py']
ARTIFACTS = [
    FIX+'/loading_native_report.json', FIX+'/loading_native_probe.py',
    FIX+'/connectome_transfer_native_report.json', FIX+'/connectome_transfer_native_probe.py',
    FIX+'/loading_before/flyvis_connectome_swap.py',
    FIX+'/q_cache_legacy_read_report.json', FIX+'/q_cache_legacy_read_probe.py',
    FIX+'/stop_guard_probe_before_fix.json', FIX+'/stop_guard_probe.json', FIX+'/stop_guard_probe.py',
    FIX+'/z13_transaction_before.log', FIX+'/z13_transaction_and_stop_final.log',
    REPAIR+'/cache_before/manifest.json', REPAIR+'/cache_before/detection_cache.py',
    REPAIR+'/cache_before/inference_with_q.py', REPAIR+'/cache_native_v2/report.json',
    REPAIR+'/stable_loading_cache_tests.log', REPAIR+'/cache_after.log',
]

def git(*args, data=None):
    result = subprocess.run(['git', *args], cwd=ROOT, input=data, capture_output=True)
    return result.stdout.decode('utf-8').strip() if result.returncode == 0 else None

def descriptor(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), size=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def baseline_for(name):
    directory = REPAIR+'/cache_before' if name in ('tracking/detection_cache.py', 'tracking/inference_with_q.py') else FIX+'/loading_before'
    return ROOT/directory/Path(name).name

def validate_isolated():
    isolated = OUT/('isolated_'+str(time.time_ns()))
    isolated.mkdir()
    archived = subprocess.run(['git', 'archive', 'HEAD', 'tracking', 'scripts', 'tests',
        'models', 'datasets', 'evaluation'], cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
        for member in archive.getmembers():
            (isolated/member.name).resolve().relative_to(isolated.resolve())
            if not (member.isfile() or member.isdir()):
                raise ValueError('Unexpected archive entry '+member.name)
        archive.extractall(isolated)
    for name in SOURCES:
        destination = isolated/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    command = [str(ROOT/'.venv/Scripts/python.exe'), '-m', 'pytest', *TESTS,
               '-q', '-p', 'no:cacheprovider', '--basetemp='+str(isolated/'pytest_tmp')]
    log = OUT/'isolated_tests.log'
    started = time.monotonic()
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.run(command, cwd=isolated, stdout=stream, stderr=subprocess.STDOUT)
    result = dict(passed=process.returncode == 0, exit_code=process.returncode,
        seconds=time.monotonic()-started, command=command, cwd=str(isolated),
        head=git('rev-parse', 'HEAD'), overlay=[descriptor(ROOT/p) for p in SOURCES],
        scope='Tracked HEAD archive plus only explicit source allowlist; no B2, motion or other dirty worktree source overlay')
    (OUT/'isolated_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result

def main(validate=False):
    if validate:
        result = validate_isolated()
        if not result['passed']:
            print(json.dumps(result, ensure_ascii=False))
            raise SystemExit(result['exit_code'])
    validation = json.loads((OUT/'isolated_validation.json').read_text(encoding='utf-8'))
    assert validation['passed']
    assert validation['overlay'] == [descriptor(ROOT/p) for p in SOURCES], 'Source changed after isolated validation'
    rows = []
    for name in SOURCES:
        row = descriptor(ROOT/name)
        row.update(head_blob=git('rev-parse', 'HEAD:'+name), index_blob=git('rev-parse', ':'+name))
        row['classification'] = 'tracked_clean_at_start' if row['head_blob'] else 'new_by_current_task'
        if row['head_blob']:
            baseline = baseline_for(name)
            row['baseline'] = descriptor(baseline)
            canonical = git('hash-object', '--path='+name, '--stdin', data=baseline.read_bytes())
            row['baseline_equals_HEAD'] = canonical == row['head_blob']
            assert row['baseline_equals_HEAD'], name
            assert row['index_blob'] == row['head_blob'], 'Unexpected staged edits: '+name
        rows.append(row)
    command_list = dict(
        stable_tests=[str(ROOT/'.venv/Scripts/python.exe'), '-m', 'pytest', *TESTS,
                      '-q', '-p', 'no:cacheprovider', '--basetemp='+REPAIR+'/stable_loading_cache_temp'],
        isolated_validation=[str(ROOT/'.venv/Scripts/python.exe'), str(Path(__file__).relative_to(ROOT)), '--validate'],
        native_probes=[['.venv/Scripts/python.exe', 'scripts/check_detection_cache_parity_20261106.py'],
                       ['.venv/Scripts/python.exe', FIX+'/loading_native_probe.py'],
                       ['.venv/Scripts/python.exe', FIX+'/connectome_transfer_native_probe.py'],
                       ['.venv/Scripts/python.exe', FIX+'/q_cache_legacy_read_probe.py'],
                       ['.venv/Scripts/python.exe', FIX+'/stop_guard_probe.py']],
        prerequisites='Native FlyVis probes require the existing vendor runtime/checkpoints/connectome JSON; detector probe requires the existing baseline weights and three train JPGs; legacy read probe requires the existing saved Q cache. These inputs are not part of this code package.',
        output_policy='CPU detector probe defaults to a unique output path. Other historical probes use their recorded report paths; preserve prior reports or copy the probe to a separately named artifact before rerunning.',
        staging_for_parent_only=['git add --pathspec-from-file='+str((OUT/'source_paths.txt').relative_to(ROOT)),
                                'git add -f --pathspec-from-file='+str((OUT/'artifact_paths.txt').relative_to(ROOT))])
    (OUT/'commands.json').write_text(json.dumps(command_list, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'source_paths.txt').write_text('\n'.join(SOURCES)+'\n', encoding='utf-8')
    control = [str((OUT/name).relative_to(ROOT)).replace('\\','/') for name in
               ('build_package.py', 'commands.json', 'source_paths.txt', 'isolated_validation.json', 'isolated_tests.log')]
    artifacts = ARTIFACTS + control
    for name in artifacts:
        assert (ROOT/name).is_file(), name
    manifest_path = str((OUT/'manifest.json').relative_to(ROOT)).replace('\\','/')
    pathspec_path = str((OUT/'artifact_paths.txt').relative_to(ROOT)).replace('\\','/')
    (OUT/'artifact_paths.txt').write_text('\n'.join([*artifacts, manifest_path, pathspec_path])+'\n', encoding='utf-8')
    manifest = dict(head=git('rev-parse', 'HEAD'), index_changes=git('diff', '--cached', '--name-status'),
        sources=rows, artifacts=[descriptor(ROOT/p) for p in [*artifacts, pathspec_path]],
        manifest_self_hash='Not recursively self-hashed; Git commit records this manifest. Every other package file is listed with exact SHA256 and size.',
        excluded='All B2 originals/patches/tests; all motion/K1/classical/retina edits; all temporal/history lifecycle changes; plan/catalogue/Obsidian; weights, videos, NPZ caches, temp directories and unrelated untracked files.',
        verification='isolated_validation.json pins the exact tested overlay; stable_loading_cache_tests.log records the corresponding shared-worktree run.',
        review_limits='CUDA execution was not exercised. Stop fallback confirms the direct owned child only. Historical native reports retain their actual source hashes and scopes; no HOTA or detector quality claim.')
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(source_files=len(rows), artifact_files=len(artifacts)+2,
                         payload_bytes=sum(row['size'] for row in rows)+sum(row['size'] for row in manifest['artifacts']),
                         manifest=manifest_path, isolated_passed=True), ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    main(parser.parse_args().validate)
