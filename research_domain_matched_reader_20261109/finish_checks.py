"""Finish after passed catalogue tests and a corrected citation metadata key."""
import os,sys,json,hashlib,sysconfig,contextlib,tempfile,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=OUT]
import queue
assert Path(queue.__file__).resolve()==Path(sysconfig.get_path('stdlib'),'queue.py').resolve()
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
guard_entrypoint(timeout=min(900,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tests=read(OUT/'catalog_tests_receipt.json');fix=read(OUT/'literature_schema_recovery.json')
assert tests['passed'] and tests['tests']==58 and tests['exit_code']==0
assert sha(OUT/'catalog_tests.log')==tests['log_sha256']
stable=dict(tests['stable_sources']);assert sha(ROOT/fix['snapshot'])==fix['old_sha256']==stable[fix['source']]
stable[fix['source']]=fix['new_sha256']
assert all(sha(ROOT/name)==digest for name,digest in stable.items())
from scripts.validate_project_history import main as validate
sys.argv=['scripts/validate_project_history.py']
with (OUT/'catalog_validation_after_metadata_fix.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f):assert validate()==0
from scripts.build_obsidian_vault import main as build
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis'
def hypotheses(path):return {f.relative_to(path).as_posix():sha(f) for f in path.rglob('*.md') if any('Гипотез' in p for p in f.parts)}
protected=hypotheses(vault)
def generate(path):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(path)];assert build()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
    preview=Path(tmp)/'ObsidianFlyVis';shutil.copytree(vault,preview)
    generate(preview);graph=audit(preview);assert hypotheses(preview)==protected
    generate(vault);assert audit(vault)==graph
assert hypotheses(vault)==protected and all(sha(ROOT/name)==digest for name,digest in stable.items())
for phase in ['protocol_v2.json','execution_protocol.json','input_recovery_protocol.json']:
    for name,digest in read(OUT/phase)['sources'].items():assert sha(ROOT/name)==digest,(phase,name)
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,catalog_tests=58,catalog_tests_reused=True,
    tests_evidence='catalog_tests_receipt.json; catalog_tests.log',graph=graph,hypothesis_files=len(protected),hypotheses_unchanged=True,stable_sources=stable,
    citation_metadata_fix=fix,scientific_audit_sha256=sha(OUT/'audit.json'),goal_achieved=False,
    scope='Completed coordinate/domain diagnostics, matched20video-fold comparison, final training heads and hypothesis-preserving graph. No native-ID benefit claimed.'),f,ensure_ascii=False,indent=2)
print('FINISHED_CHECKS',58,graph['note_count'],len(protected),flush=True)
