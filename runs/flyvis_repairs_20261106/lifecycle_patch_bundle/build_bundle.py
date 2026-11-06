"""Bundle only this audit's edits to preexisting untracked lifecycle modules."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
SOURCES=[f'tracking/{name}.py' for name in (
    'history_pair_runtime','lazy_appearance_association','query_appearance_comparison',
    'frame_masa_appearance','query_assignment_write','query_memory_write','query_bounded_write')]
NEW=['tests/test_tracker_lifecycle_reset.py','scripts/check_lifecycle_parity_20261106.py']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    original=json.loads((OUT.parent/'parent_baseline.json').read_text(encoding='utf-8'))
    scratch=OUT/('apply_check_'+str(time.time_ns())); scratch.mkdir()
    env=os.environ.copy()
    for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_COMMON_DIR'):
        env.pop(key,None)
    env['GIT_CEILING_DIRECTORIES']=str(scratch.parent)
    calls=[]
    def git(*args):
        command=['git','-c','core.autocrlf=false','-c','core.eol=lf',*args]
        result=subprocess.run(command,cwd=scratch,env=env,capture_output=True)
        calls.append(dict(command=command,returncode=result.returncode,
            stdout=result.stdout.decode('utf-8',errors='replace'),stderr=result.stderr.decode('utf-8',errors='replace')))
        return result.returncode
    assert git('rev-parse','--show-toplevel')!=0
    descriptors=[]; patches=[]; before_bytes={}
    with zipfile.ZipFile(OUT.parent/'parent_baseline.zip') as archive:
        for name in SOURCES+NEW:
            old=archive.read(name) if name in SOURCES else b''
            new=(ROOT/name).read_bytes()
            if name in SOURCES:
                assert digest(old)==original['files'][name]['sha256']
                target=scratch/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(old)
                before_bytes[name]=old
            patch=''.join(difflib.unified_diff(old.decode('utf-8').splitlines(keepends=True),
                new.decode('utf-8').splitlines(keepends=True),
                fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
            assert patch, name+' has no changes'
            patch_path=OUT/(Path(name).stem+'.patch'); patch_path.write_bytes(patch.encode('utf-8'))
            patches.append(patch_path)
            descriptors.append(dict(path=name,baseline_sha256=digest(old) if old else None,
                current_sha256=digest(new),patch=patch_path.relative_to(ROOT).as_posix(),
                patch_sha256=digest(patch_path.read_bytes()),
                classification='prior_untracked_modified' if old else 'new_requires_prior_untracked_dependencies'))
    for patch in patches:
        assert git('apply','--check','--whitespace=nowarn',str(patch))==0
        assert git('apply','--whitespace=nowarn',str(patch))==0
    assert all((scratch/name).read_bytes()==(ROOT/name).read_bytes() for name in SOURCES+NEW)
    for patch in reversed(patches):
        assert git('apply','--reverse','--whitespace=nowarn',str(patch))==0
    assert all((scratch/name).read_bytes()==old for name,old in before_bytes.items())
    assert all(not (scratch/name).exists() for name in NEW)
    report=dict(passed=True,exact_forward_bytes=True,exact_reverse_bytes=True,commands=calls,
        scratch=scratch.relative_to(ROOT).as_posix(),original_workspace_sources_unchanged=True)
    (OUT/'apply_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest=dict(files=descriptors,baseline_archive_sha256=digest((OUT.parent/'parent_baseline.zip').read_bytes()),
        baseline_archive='runs/flyvis_repairs_20261106/parent_baseline.zip (local only; contains preexisting user documents)',
        boundaries='Patches only. Exact original source hashes are required. Does not publish the untracked dependency tree or claim a self-contained checkout.',
        validation=['runs/flyvis_repairs_20261106/lifecycle_parity.json','runs/flyvis_repairs_20261106/lifecycle_parity_outputs.npz'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    names=[Path(__file__),*patches,OUT/'apply_report.json',OUT/'manifest.json',OUT/'artifact_paths.txt']
    (OUT/'artifact_paths.txt').write_text('\n'.join(p.relative_to(ROOT).as_posix() for p in names)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,modified_sources=len(SOURCES),new_files_as_patches=len(NEW))))


if __name__=='__main__':
    main()
