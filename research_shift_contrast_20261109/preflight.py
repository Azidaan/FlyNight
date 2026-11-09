"""Required catalogue checks and meaningful invariants of the new reader."""
import os,sys,json,hashlib,contextlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(900,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
import psutil,numpy as np
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stable={n:sha(ROOT/n) for n in ['experiments/project_history.json','experiments/metrics_glossary.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}
from tracking.shift_profile_contrast import AXES,nonneighbors,subtract_sidelobes
rng=np.random.default_rng(20261109);r=rng.uniform(-.4,.4,(19,34));winner=4;value=subtract_sidelobes(r,winner)
shift=rng.uniform(-.1,.1,34);np.testing.assert_allclose(subtract_sidelobes(r+shift,winner),value,atol=1e-15,rtol=0)
permutation=rng.permutation(19);new_winner=int(np.flatnonzero(permutation==winner)[0]);np.testing.assert_allclose(subtract_sidelobes(r[permutation],new_winner,AXES[permutation]),value,atol=1e-15,rtol=0)
np.testing.assert_allclose(subtract_sidelobes(np.tile(shift,(19,1)),winner),np.zeros(34),atol=1e-15,rtol=0)
assert min(map(len,nonneighbors()))==12 and max(map(len,nonneighbors()))==15
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
assert all(sha(ROOT/n)==v for n,v in stable.items())
with (OUT/'preflight.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,tests=counts.passed,reader_invariants=['per-channelconstantoffset','permutationofshiftorder','constantprofilezero'],reader_sha256=sha(ROOT/'tracking/shift_profile_contrast.py'),stable_sources=stable,goal_achieved=False),f,indent=2)
print('CONTRAST_PREFLIGHT_PASSED',counts.passed,flush=True)
