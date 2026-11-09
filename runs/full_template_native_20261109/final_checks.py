"""Validate fixed scientific receipts, records and a preview before the live graph."""
from common import *
start('cpu',timeout=1200)
import contextlib,runpy,tempfile,shutil
assert not (OUT/'final_checks.json').exists()
phases=['protocol.json','input_recovery_protocol.json','execution_protocol.json','status_recovery_protocol.json',
    'cache_registry_protocol.json','evaluation_protocol.json','fate_recovery_protocol.json',
    'boundary_protocol.json','integration_audit_protocol.json','centering_protocol.json']
metadata_alias={'experiments/useful_literature.json':OUT/'useful_literature_at_integration.json'}
def verify_all():
    snapshots=[]
    for phase in phases:
        for name,digest in read(OUT/phase)['sources'].items():
            path=ROOT/name
            if name in metadata_alias and sha(path)!=digest:
                snapshot=metadata_alias[name];assert sha(snapshot)==digest,(phase,name)
                snapshots.append(dict(protocol=phase,source=name,snapshot=snapshot.relative_to(ROOT).as_posix(),
                    original_sha256=digest,current_metadata_sha256=sha(path),
                    reason='Citation database legitimately updated after completed algebra audit; exact previously matching bytes retained. No numerical predictor used this database.'))
            else:assert sha(path)==digest,(phase,name)
    return snapshots
snapshots=verify_all()
for name in ['input_witness_recovery.json','zero_complete.json','deployment_audit.json','comparison_report.json',
    'id_fate.json','boundary_audit.json','integration_audit.json','centering_audit.json']:
    assert read(OUT/name).get('passed',read(OUT/name).get('all_arrays_exact',False)),name
assert not read(OUT/'comparison_report.json')['accepted_flyvis_system_and_increment']
card=next(r for r in read(ROOT/'experiments/project_history.json')['experiments'] if r['id']=='full_template_native_20261109')
assert card['state']=='completed_negative' and [r['met'] for r in card['acceptance']]==[True,True,False,True]
stable={name:sha(ROOT/name) for name in ['experiments/project_history.json','experiments/metrics_glossary.json',
    'experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}
sys.argv=['scripts/validate_project_history.py']
with (OUT/'catalog_validation.log').open('x',encoding='utf8') as log:
    with contextlib.redirect_stdout(log):
        try:runpy.run_path(str(ROOT/'scripts/validate_project_history.py'),run_name='__main__')
        except SystemExit as error:
            if error.code:raise
import pytest
class CountTests:
    def __init__(self):self.passed=0;self.collected=0
    def pytest_collection_finish(self,session):self.collected=session.testscollected
    def pytest_runtest_logreport(self,report):
        if report.when=='call' and report.passed:self.passed+=1
tests=CountTests();assert pytest.main(['tests/test_project_history.py','-q'],plugins=[tests])==0
assert tests.collected>=58 and tests.passed==tests.collected
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis'
def hypotheses(path):
    return {f.relative_to(path).as_posix():sha(f) for f in path.rglob('*.md') if any('Гипотез' in p for p in f.parts)}
hyp=hypotheses(vault)
def build(path):
    sys.argv=['scripts/build_obsidian_vault.py','--vault',str(path)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
    copy=Path(temp)/'ObsidianFlyVis';shutil.copytree(vault,copy)
    build(copy);graph=audit(copy);assert hypotheses(copy)==hyp
    build(vault);assert audit(vault)==graph
assert hypotheses(vault)==hyp
assert all(sha(ROOT/name)==digest for name,digest in stable.items())
assert verify_all()==snapshots
# Closed files only. Runtime observer files and this still-open guard log are
# deliberately not advertised as immutable scientific artifacts.
artifacts={}
for path in OUT.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts:continue
    if path.name.endswith('.pending') or path.name in ['status.json','artifacts_sha256.json','final_checks.json']:continue
    if 'queue_status' in path.name:continue
    artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
write('artifacts_sha256.json',dict(created=datetime.now(MSK).isoformat(),files=artifacts,
    scope='Closed local pilot files only; data hashes identify bytes, not scientific correctness. Mutable progress excluded.'))
write('final_checks.json',dict(passed=True,catalog_tests=tests.passed,graph=graph,
    hypotheses_unchanged=True,hypothesis_files=len(hyp),stable_sources=stable,metadata_snapshots=snapshots,
    result='completed_negative',actual_ID_and_HOTA_evaluated=True,goal_achieved=False,
    manifest_sha256=sha(OUT/'artifacts_sha256.json'),
    scope='Whole20video native pilot, matched controls, direct known/unknown ID fate, numerical boundary and exact centered head decomposition. No successful new tracker claimed.'))
print('Native pilot negative result, all records and graph verified',tests.passed,flush=True)
