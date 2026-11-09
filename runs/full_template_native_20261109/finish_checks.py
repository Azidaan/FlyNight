"""Continue after successful pytest; do not repeat the58completed checks."""
from common import *
start('cpu',timeout=1200)
import re,contextlib,tempfile,shutil
assert not (OUT/'final_checks.json').exists()
old=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/123ce1a9d5a7421eaea48666469d23f1'
guard=read(old/'memory_guard.json');log=(old/'memory_guard.worker.log').read_text(encoding='utf8')
assert guard['returncode']==1 and guard['reason'] is None
assert 'assert tests.collected>=58 and tests.passed==tests.collected' in log
assert re.search(r'(?m)^58 passed in ',log)
assert "assert pytest.main(['tests/test_project_history.py','-q'],plugins=[tests])==0" in (OUT/'final_checks.py').read_text(encoding='utf8')
stable={name:sha(ROOT/name) for name in ['experiments/project_history.json','experiments/metrics_glossary.json',
    'experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}
assert all((ROOT/name).stat().st_mtime<read(old/'supervisor.json')['birth'] for name in stable)
write('checks_recovery_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
    previous_guard_sha256=sha(old/'memory_guard.json'),previous_log_sha256=sha(old/'memory_guard.worker.log'),
    previous_checker_sha256=sha(OUT/'final_checks.py'),source_sha256=sha(Path(__file__)),
    failure='After successfulpytestreturn0and58passed, custom collection_finish hook read testscollected before pytest assigned len(items). Counter assertion failed; catalogue tests themselves passed. Installed _pytest/main.py calls collection_finish at879 and assigns testscollected at882.',
    reuse='Reuse the observed58passed and zero return proved by reaching the next assertion in the unchanged source. Do not rerun tests. Root metadata was not edited during/after those tests; mtimes precede old supervisor start. Hashes below are captured NOW for the remaining graph check, not falsely advertised as earlier pretest hashes.',
    stable_sources_after_pytest=stable,pending='Preview/live graph, levels, hypothesis bytes, frozen scientific source verification and closed-file manifest.'))
phases=['protocol.json','input_recovery_protocol.json','execution_protocol.json','status_recovery_protocol.json',
    'cache_registry_protocol.json','evaluation_protocol.json','fate_recovery_protocol.json','boundary_protocol.json',
    'integration_audit_protocol.json','centering_protocol.json']
snapshots=[]
for phase in phases:
    for name,digest in read(OUT/phase)['sources'].items():
        path=ROOT/name
        if name=='experiments/useful_literature.json' and sha(path)!=digest:
            saved=OUT/'useful_literature_at_integration.json';assert sha(saved)==digest
            snapshots.append(dict(protocol=phase,source=name,snapshot=saved.relative_to(ROOT).as_posix(),
                original_sha256=digest,current_metadata_sha256=sha(path),reason='Legitimate citation database upkeep after completed audit; exact matching previous bytes retained.'))
        else:assert sha(path)==digest,(phase,name)
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis'
def hypotheses(path):return {f.relative_to(path).as_posix():sha(f) for f in path.rglob('*.md') if any('Гипотез' in p for p in f.parts)}
hyp=hypotheses(vault)
def build(path):
    sys.argv=['scripts/build_obsidian_vault.py','--vault',str(path)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
    copy=Path(temp)/'ObsidianFlyVis';shutil.copytree(vault,copy)
    build(copy);graph=audit(copy);assert hypotheses(copy)==hyp
    build(vault);assert audit(vault)==graph
assert hypotheses(vault)==hyp
assert all(sha(ROOT/name)==digest for name,digest in stable.items())
artifacts={}
for path in OUT.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts:continue
    # Exclude open outer launch logs as well as observation/status files.
    if path.suffix=='.log' or path.name.endswith('.pending'):continue
    if path.name in ['status.json','artifacts_sha256.json','final_checks.json'] or 'queue_status' in path.name:continue
    artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
write('artifacts_sha256.json',dict(created=datetime.now(MSK).isoformat(),files=artifacts,
    scope='Closed local files at this moment. All launch logs, dynamic status and still-open guard logs excluded. Hashes identify bytes, not scientific correctness.'))
write('final_checks.json',dict(passed=True,catalog_tests=58,tests_reused_not_rerun=True,
    tests_evidence=dict(log=(old/'memory_guard.worker.log').relative_to(ROOT).as_posix(),sha256=sha(old/'memory_guard.worker.log'),
        original_checker_exit=1,cause='Post-pytest counter bookkeeping, not a failing test',pytest_exit=0),
    graph=graph,hypotheses_unchanged=True,hypothesis_files=len(hyp),stable_sources=stable,
    source_fingerprint_timing='Root metadata hashes captured after successful pytest; no edits since tests, mtimes precede them. Frozen scientific protocols verified separately.',
    metadata_snapshots=snapshots,result='completed_negative',actual_ID_and_HOTA_evaluated=True,goal_achieved=False,
    manifest_sha256=sha(OUT/'artifacts_sha256.json'),
    scope='Whole20video native pilot, matched controls, direct known/unknown ID fate, numerical boundary and centered decomposition. No successful new tracker claimed.'))
print('Remaining graph and source checks passed;58pytestresults reused',flush=True)
