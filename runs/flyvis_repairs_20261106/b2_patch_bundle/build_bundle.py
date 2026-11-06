"""Preserve only the authored B2 diff/test, then verify exact-byte application."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
SOURCE = 'tracking/b2_pilot_runtime.py'
TEST = 'tests/test_b2_lifecycle_isolation.py'
BASELINE = 'runs/flyvis_system_fixes_20261106/loading_before/b2_pilot_runtime.py'
OLD_MANIFEST = 'runs/flyvis_system_fixes_20261106/loading_prior_untracked_b2_manifest.json'


def descriptor(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def git(*args, cwd=ROOT, env=None):
    # Repository discovery alone does not disable the user's global autocrlf.
    # Pin output bytes for this subprocess without changing any Git config.
    command = ['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf', *args]
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True)
    return dict(command=command, exit_code=result.returncode,
                stdout=result.stdout.decode('utf-8', errors='replace'),
                stderr=result.stderr.decode('utf-8', errors='replace'))


def main():
    previous = json.loads((ROOT/OLD_MANIFEST).read_text(encoding='utf-8'))
    before, after, test = descriptor(ROOT/BASELINE), descriptor(ROOT/SOURCE), descriptor(ROOT/TEST)
    assert before['sha256'] == previous['baseline_sha256']
    assert after['sha256'] == previous['expected_after_sha256'], 'B2 source changed after original repair'
    assert test['sha256'] == previous['new_test_sha256'], 'B2 test changed after original repair'
    assert git('ls-files','--error-unmatch',SOURCE)['exit_code'] != 0, 'Source classification changed'
    old_patch = ROOT/previous['patch']
    assert descriptor(old_patch)['sha256'] == previous['patch_sha256']
    shutil.copyfile(old_patch, OUT/'b2_lifecycle.patch')
    test_patch = ''.join(difflib.unified_diff([], (ROOT/TEST).read_text(encoding='utf-8').splitlines(keepends=True),
                                            fromfile='/dev/null', tofile='b/'+TEST))
    (OUT/'b2_lifecycle_test.patch').write_text(test_patch, encoding='utf-8', newline='\n')
    scratch = OUT/('apply_check_'+str(time.time_ns()))
    (scratch/'tracking').mkdir(parents=True)
    shutil.copyfile(ROOT/BASELINE, scratch/SOURCE)
    env = os.environ.copy()
    for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_COMMON_DIR'):
        env.pop(key, None)
    env['GIT_CEILING_DIRECTORIES'] = str(scratch.parent)
    discovery = git('rev-parse','--show-toplevel',cwd=scratch,env=env)
    assert discovery['exit_code'] != 0, 'Temporary apply check discovered a Git repository'
    calls = [discovery]
    patches = [OUT/'b2_lifecycle.patch', OUT/'b2_lifecycle_test.patch']
    for patch in patches:
        for flags in (['--check'], []):
            result = git('apply', *flags, '--whitespace=nowarn', str(patch), cwd=scratch, env=env)
            calls.append(result)
            assert result['exit_code'] == 0, result
    source_exact = (scratch/SOURCE).read_bytes() == (ROOT/SOURCE).read_bytes()
    test_exact = (scratch/TEST).read_bytes() == (ROOT/TEST).read_bytes()
    assert source_exact and test_exact, 'Patch application did not preserve exact file bytes'
    for patch in reversed(patches):
        result = git('apply','--reverse','--whitespace=nowarn',str(patch),cwd=scratch,env=env)
        calls.append(result)
        assert result['exit_code'] == 0, result
    reverse_exact = (scratch/SOURCE).read_bytes() == (ROOT/BASELINE).read_bytes() and not (scratch/TEST).exists()
    assert reverse_exact
    report = dict(passed=True, current_source_exact_bytes=source_exact, new_test_exact_bytes=test_exact,
        reverse_restores_exact_baseline=reverse_exact, scratch=str(scratch.relative_to(ROOT)).replace('\\','/'),
        repository_discovery_disabled=True, line_endings='Explicit process-local core.autocrlf=false/core.eol=lf; no source normalization or Git config write',
        commands=calls)
    (OUT/'apply_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    dependencies=[]
    for name in previous['dependencies']:
        row=descriptor(ROOT/name)
        tracked=git('ls-files','--error-unmatch',name)['exit_code']==0
        row['classification']='tracked_dependency' if tracked else 'preexisting_untracked_source_not_in_bundle'
        dependencies.append(row)
    manifest=dict(source=after,baseline=before,new_test=test,
        patches=[descriptor(p) for p in patches],dependencies=dependencies,
        original_manifest=descriptor(ROOT/OLD_MANIFEST),application_report=descriptor(OUT/'apply_report.json'),
        publication_boundary='Only patch/provenance files, not the complete preexisting untracked source or its dependency tree. The test is stored as a new-file patch, not installed as an executable test in a partial checkout.',
        application_requirement='The recipient must provide the exact original source whose SHA256 is recorded. Reject a different baseline; do not apply with offset/fuzz as proof of correctness.',
        limitations='The dependency list was collected by conservative AST imports; it is provenance, not a self-contained runnable environment. No original source, original test or repository index was modified by this check.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    names=['build_bundle.py','manifest.json','apply_report.json','b2_lifecycle.patch','b2_lifecycle_test.patch','first_apply_byte_mismatch.json']
    (OUT/'artifact_paths.txt').write_text('\n'.join(str((OUT/n).relative_to(ROOT)).replace('\\','/') for n in [*names,'artifact_paths.txt'])+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,files=len(names)+1,source_sha256=after['sha256'],baseline_sha256=before['sha256']),ensure_ascii=False))


if __name__ == '__main__':
    main()
