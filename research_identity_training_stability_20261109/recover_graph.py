"""Finish the failed graph phase, reusing unchanged already-green catalog tests."""
import os,sys,json,hashlib,contextlib,tempfile,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
stop=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(600,(stop-datetime.now(stop.tzinfo)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(OUT/'graph_recovery_protocol.json');tests=read(OUT/'final_catalog_tests_receipt.json')
for n,h in p['scientific_sources'].items():assert sha(OUT/n)==h,n
stable={}
for n,h in tests['stable_sources'].items():
    expected=p['changed'][n]['after'] if n in p['changed'] else h
    assert sha(ROOT/n)==expected,n;stable[n]=expected
from scripts.validate_project_history import main as validate
sys.argv=['scripts/validate_project_history.py']
with (OUT/'recovery_catalog_validation.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f):assert validate()==0
from scripts.build_obsidian_vault import main as build
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis'
def protected(path):return {f.relative_to(path).as_posix():sha(f) for f in path.rglob('*.md') if any('Гипотез' in part for part in f.parts)}
before=protected(vault)
def generate(path):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(path)];assert build()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
    preview=Path(tmp)/'ObsidianFlyVis';shutil.copytree(vault,preview);generate(preview);graph=audit(preview);assert protected(preview)==before;generate(vault);assert audit(vault)==graph
assert protected(vault)==before and all(sha(ROOT/n)==h for n,h in stable.items())
out=dict(passed=True,tests=58,tests_reused=True,tests_receipt_sha256=sha(OUT/'final_catalog_tests_receipt.json'),graph=graph,stable_sources=stable,hypothesis_files=len(before),hypotheses_unchanged=True,
    recovery_protocol_sha256=sha(OUT/'graph_recovery_protocol.json'),failure_guard=p['failure_guard'],scientific_sources_unchanged=True,goal_achieved=False)
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('GRAPH_RECOVERY_PASSED',graph['note_count'],len(before),flush=True)
