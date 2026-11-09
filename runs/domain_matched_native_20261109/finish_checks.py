"""Check final scientific records and preserve every shared hypothesis file."""
from evaluation_common import *
start(1800)
import tempfile,shutil,contextlib
assert read(OUT/'recording.json')['passed'] and read(OUT/'runtime_audit.json')['passed']
stable={name:sha(ROOT/name) for name in ['experiments/project_history.json','experiments/metrics_glossary.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}
write('final_checks_protocol.json',dict(registered=datetime.now(MSK).isoformat(),stable_sources=stable,source_sha256=sha(Path(__file__)),scope='Validate the completed fixed native comparison, direct-ID fate, current synthesis and protected graph; no scientific rerun.'))
from scripts.validate_project_history import main as validate
sys.argv=['scripts/validate_project_history.py']
with (OUT/'final_catalog_validation.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f):assert validate()==0
import pytest
class Counts:
    passed=0;collected=0
    def pytest_runtest_logreport(self,report):
        if report.when=='call' and report.passed:self.passed+=1
    def pytest_sessionfinish(self,session,exitstatus):self.collected=session.testscollected
counts=Counts()
with (OUT/'final_catalog_tests.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f):code=pytest.main(['tests/test_project_history.py','-q'],plugins=[counts])
assert code==0 and counts.passed==counts.collected and counts.passed>=58,(code,counts.passed,counts.collected)
write('final_catalog_tests_receipt.json',dict(passed=True,tests=counts.passed,exit_code=code,stable_sources=stable,log_sha256=sha(OUT/'final_catalog_tests.log')))
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
for phase in ['execution_protocol.json','evaluation_protocol.json','audit_protocol.json']:verify(read(OUT/phase))
write('final_checks.json',dict(passed=True,catalog_tests=counts.passed,graph=graph,hypothesis_files=len(protected),hypotheses_unchanged=True,stable_sources=stable,
    scientific_comparison_sha256=sha(OUT/'comparison_report.json'),runtime_audit_sha256=sha(OUT/'runtime_audit.json'),fate_sha256=sha(OUT/'id_fate.json'),goal_achieved=False,
    scope='Completed development-native pilot; independent final validation remains unrun.'))
print('FINAL_NATIVE_RECORDS_CHECKED',counts.passed,graph['note_count'],len(protected),flush=True)
