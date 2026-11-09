"""Validate the new records and graph; preserve all shared hypothesis bytes."""
import os,sys,json,hashlib,tempfile,shutil,contextlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(1200,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
assert read(OUT/'recording.json')['goal_achieved'] is False
stable={name:sha(ROOT/name) for name in ['experiments/project_history.json','experiments/metrics_glossary.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}
with (OUT/'checks_protocol.json').open('x',encoding='utf8') as f:json.dump(dict(registered=datetime.now(STOP.tzinfo).isoformat(),stable_sources=stable,source_sha256=sha(Path(__file__)),scope='Validate completed coordinate/domain records, literature and hypothesis-preserving graph; no scientific reruns.'),f,ensure_ascii=False,indent=2)
from scripts.validate_project_history import main as validate
sys.argv=['scripts/validate_project_history.py']
with (OUT/'catalog_validation.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f):assert validate()==0
import pytest
class Counts:
    passed=0;collected=0
    def pytest_runtest_logreport(self,report):
        if report.when=='call' and report.passed:self.passed+=1
    def pytest_sessionfinish(self,session,exitstatus):self.collected=session.testscollected
counts=Counts()
with (OUT/'catalog_tests.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f):code=pytest.main(['tests/test_project_history.py','-q'],plugins=[counts])
assert code==0 and counts.passed==counts.collected and counts.passed>=58,(code,counts.passed,counts.collected)
with (OUT/'catalog_tests_receipt.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,tests=counts.passed,exit_code=code,stable_sources=stable,log_sha256=sha(OUT/'catalog_tests.log')),f,indent=2)
assert all(sha(ROOT/name)==digest for name,digest in stable.items())
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
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,catalog_tests=counts.passed,graph=graph,hypothesis_files=len(protected),hypotheses_unchanged=True,stable_sources=stable,
    scientific_audit_sha256=sha(OUT/'audit.json'),goal_achieved=False,scope='Completed diagnostic stages and fixed domain-head comparison. No native ID benefit claimed.'),f,ensure_ascii=False,indent=2)
print('RECORDS_AND_GRAPH_CHECKED',counts.passed,graph['note_count'],len(protected),flush=True)
