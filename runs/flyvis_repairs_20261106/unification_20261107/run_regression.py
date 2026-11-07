"""Sequential CPU regression groups with a system-memory supervisor."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,hashlib,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from system_memory_guard import guard_entrypoint
job=guard_entrypoint()
import psutil
# The external supervisor checks total system memory.
OUT=Path(__file__).resolve().parent
group=sys.argv[1]
limit=None
old=json.loads((ROOT/'runs/flyvis_repairs_20261106/integrated_1794000756588983700/command.json').read_bytes())
selected={x for x in old['command'] if x.startswith('tests/') and x.endswith('.py')}
untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z','tests'],cwd=ROOT).decode().split('\0')
selected.update(x for x in untracked if x.endswith('.py'))
selected.update(['tests/test_flyvis_state_transplant.py','tests/test_obsidian_checked_hypotheses.py','tests/test_obsidian_fundamental_overview.py'])
all_tests={p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').glob('test_*.py')}
files=sorted(selected if group in ('integration','integration_light') else all_tests-selected-{'tests/test_project_history.py'})
if group in ('native','native4'):files=['tests/test_flyvis_activity_readout.py::test_estimator_exposes_internal_activity_of_current_frame_only']
assert group in ('integration','integration_light','remaining','native','native4') and files
out=job.run_dir/group
out.mkdir(exist_ok=False)
def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ['tracking','scripts','tests','evaluation'] for p in (ROOT/directory).rglob('*.py')}
before=hashes()
(out/'command.json').write_text(json.dumps(dict(files=files,source_sha256=before,limit_bytes=limit),indent=2),encoding='utf-8')
import torch,pytest
torch.set_default_device('cpu');torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
started=time.monotonic();code=1
try:
    extra=['-k','not test_estimator_exposes_internal_activity_of_current_frame_only'] if group=='integration_light' else []
    code=int(pytest.main(files+extra+['-q','-p','no:cacheprovider','--basetemp='+str((out/'temp').as_posix())]))
finally:
    after=hashes()
    report=dict(returncode=code,seconds=time.monotonic()-started,files=len(files),job=job.read(),
        cuda_initialized=torch.cuda.is_initialized(),source_unchanged=before==after,
        changed_sources=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)))
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
raise SystemExit(code or (2 if report['cuda_initialized'] or not report['source_unchanged'] else 0))
