"""Package existing/new agent changes without touching production files or Git index."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/flyvis_repairs_20261106/temporal_heads_bundle_v2'
def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def git(*args):return subprocess.run(['git',*args],cwd=ROOT,capture_output=True)
def write_bytes_exact(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes()==data, f'Refusing changed existing bundle artifact: {path}'
    else:path.write_bytes(data)

changes=[
    ('tracking/learned_recovery_admission.py','prior_untracked_user_source',ROOT/'runs/flyvis_repairs_20261106/learned_head_review/original/learned_recovery_admission.py'),
    ('tracking/b1_timestamped.py','new_agent_file',None),
    ('tests/test_b1_timestamped.py','new_agent_file',None),
    ('scripts/b1_timestamped_smoke_20261106.py','new_agent_file',None),
    ('tests/test_learned_head_contracts.py','new_agent_file',None),
]
assert not OUT.exists(), 'Use a new bundle directory after any subsequent change'
OUT.mkdir(parents=True)
entries=[];patches=[]
for name,role,baseline in changes:
    current=(ROOT/name).read_bytes();before=b'' if baseline is None else baseline.read_bytes()
    original_file='/dev/null' if baseline is None else 'a/'+name
    patch=''.join(difflib.unified_diff(before.decode('utf-8').splitlines(keepends=True),
        current.decode('utf-8').splitlines(keepends=True),fromfile=original_file,tofile='b/'+name)).encode('utf-8')
    assert patch
    patch_name='patches/'+name.replace('/','__')+'.patch'
    write_bytes_exact(OUT/patch_name,patch);patches.append(patch)
    write_bytes_exact(OUT/'after'/name,current)
    if baseline is not None:write_bytes_exact(OUT/'baseline'/name,before)
    tracked=git('ls-files','--error-unmatch','--',name).returncode==0
    head=git('show','HEAD:'+name) if tracked else None
    if role=='tracked_clean_before_agent':assert head.returncode==0 and head.stdout==before
    entries.append(dict(path=name,role=role,baseline_path=None if baseline is None else baseline.relative_to(ROOT).as_posix(),
                        baseline_sha256=None if baseline is None else sha_bytes(before),after_sha256=sha_bytes(current),
                        patch=patch_name,patch_sha256=sha_bytes(patch),git_tracked=tracked,
                        baseline_equals_HEAD=None if head is None else head.stdout==before))
write_bytes_exact(OUT/'combined.patch',b''.join(patches))

# Static local-import closure is conservative: includes imports in optional
# methods, does not claim to discover dynamic imports or external weight files.
pending=[name for name,_,_ in changes];seen=set();dependencies=[];external=set()
while pending:
    name=pending.pop()
    if name in seen:continue
    seen.add(name);path=ROOT/name
    if not path.is_file():continue
    source=path.read_bytes();tree=ast.parse(source.decode('utf-8-sig'))
    imported=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imported.extend(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module:
            imported.append(node.module)
            imported.extend(node.module+'.'+a.name for a in node.names if a.name!='*')
    for module in imported:
        candidate=Path(*module.split('.')).with_suffix('.py')
        package=Path(*module.split('.'))/'__init__.py'
        found=next((p for p in (candidate,package) if (ROOT/p).is_file()),None)
        if found is not None:pending.append(found.as_posix())
        else:external.add(module.split('.')[0])
    if name in [n for n,_,_ in changes]:continue
    tracked=git('ls-files','--error-unmatch','--',name).returncode==0
    head=git('show','HEAD:'+name) if tracked else None
    write_bytes_exact(OUT/'dependencies'/name,source)
    dependencies.append(dict(path=name,sha256=sha_bytes(source),git_tracked=tracked,
        equals_HEAD=None if head is None else head.returncode==0 and head.stdout==source,
        classification='external dependency snapshot; no ownership or clean status inferred from untracked state'))

verification=OUT/'isolated_apply_check'
verification.mkdir()
env=os.environ.copy();env['GIT_CEILING_DIRECTORIES']=str(verification.parent)
for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'):env.pop(key,None)
discovery=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=verification,env=env,capture_output=True)
assert discovery.returncode!=0, 'Verification directory unexpectedly discovered a Git repository'
for row in entries:
    if row['baseline_sha256'] is not None:
        write_bytes_exact(verification/row['path'],(OUT/'baseline'/row['path']).read_bytes())
commands=[]
for args in [('apply','--check',str(OUT/'combined.patch')),('apply',str(OUT/'combined.patch'))]:
    result=subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf',*args],cwd=verification,env=env,capture_output=True)
    commands.append(dict(command=list(args),returncode=result.returncode,stderr=result.stderr.decode('utf-8',errors='replace')))
    assert result.returncode==0, commands[-1]
for row in entries:assert sha(verification/row['path'])==row['after_sha256']
result=subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf','apply','--reverse',str(OUT/'combined.patch')],cwd=verification,env=env,capture_output=True)
commands.append(dict(command=['apply','--reverse','combined.patch'],returncode=result.returncode,stderr=result.stderr.decode('utf-8',errors='replace')))
assert result.returncode==0
for row in entries:
    path=verification/row['path']
    if row['baseline_sha256'] is None:assert not path.exists()
    else:assert sha(path)==row['baseline_sha256']
for row in entries:assert sha(ROOT/row['path'])==row['after_sha256']
for row in dependencies:assert sha(ROOT/row['path'])==row['sha256']
evidence=[
    'runs/flyvis_timestamp_repair_20261106/real_D_v3/protocol.json',
    'runs/flyvis_timestamp_repair_20261106/real_D_v3/report.json',
    'runs/flyvis_timestamp_repair_20261106/independent_validation_v3.json',
    'runs/flyvis_timestamp_repair_20261106/tests_v3.log',
    'runs/flyvis_timestamp_repair_20261106/README.txt',
    'runs/flyvis_timestamp_repair_20261106/real_D_v4/protocol.json',
    'runs/flyvis_timestamp_repair_20261106/real_D_v4/report.json',
    'runs/flyvis_repairs_20261106/b1_finite_transition/tests_joint_final.log',
    'runs/flyvis_repairs_20261106/b1_finite_transition/before_tests.log',
    'runs/flyvis_repairs_20261106/b1_finite_transition/head_boundary_before.log',
    'runs/flyvis_repairs_20261106/b1_finite_transition/REPORT.txt',
    'runs/flyvis_timestamp_repair_20261106/real_D_v5/protocol.json',
    'runs/flyvis_timestamp_repair_20261106/real_D_v5/report.json',
    'runs/flyvis_timestamp_repair_20261106/real_D_v5/runtime_binding.json',
    'runs/flyvis_timestamp_repair_20261106/independent_validation_v5.json',
    'runs/flyvis_repairs_20261106/learned_head_review/report.json',
    'runs/flyvis_repairs_20261106/learned_head_review/after_report.json',
    'runs/flyvis_repairs_20261106/learned_head_review/after_tests_final_v2.log',
    'runs/flyvis_repairs_20261106/learned_head_review/REPORT.txt',
    'runs/flyvis_repairs_20261106/head_test_split/report.json',
    'runs/flyvis_repairs_20261106/head_test_split/tests_joint_after_split.log',
    'runs/flyvis_repairs_20261106/head_test_split/reliability_standalone.log',
    'runs/flyvis_repairs_20261106/head_test_split/tests_final_runtime_dependency.log',
    'runs/flyvis_repairs_20261106/historical_residual_independent/report.json',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/timestamped_after_report.json',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/TIMESTAMPED_INDEPENDENT_REVIEW.txt',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/timestamped_review_manifest.json',
]
manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),changes=entries,
    combined_patch_sha256=sha(OUT/'combined.patch'),dependencies=sorted(dependencies,key=lambda x:x['path']),
    dependency_limit='Static local import closure only; optional branches may be included, dynamic imports, installed packages and external model/data files are not copied or certified.',
    external_top_level_imports=sorted(external),evidence={name:sha(ROOT/name) for name in evidence},
    verification=dict(no_git_repository_discovered=True,commands=commands,forward_equals_after_exact=True,
                      reverse_equals_baseline_exact=True,production_files_unchanged=True,index_and_git_history_unchanged=True),
    boundary='Patch delivery only. Prior-untracked user sources remain prior-untracked unless parent separately authorizes staging. No trained historical weights or scientific artifacts changed. B1 checkpoint semantics require separate new training.',
    script_sha256=sha(__file__))
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(changes=len(entries),dependencies=len(dependencies),combined_patch_sha256=manifest['combined_patch_sha256'],verification=manifest['verification']),ensure_ascii=False,indent=2))
