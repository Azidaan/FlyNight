"""Complete graph acceptance after adding the required literature field."""
from study import *
import tempfile
import shutil

prior=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/95c44d5d2f354625abea60f8c1a4bac9'
log=(prior/'memory_guard.worker.log').read_text(encoding='utf8')
assert '58 passed in 170.72s' in log and "KeyError: 'doi'" in log
assert read(prior/'memory_guard.json')['returncode']==1
assert 'OK: каталог согласован с источниками' in (OUT/'catalog_validation.log').read_text(encoding='utf8')
p=read(OUT/'protocol.json');verify(p);assert read(OUT/'audit.json')['passed']
stable={name:sha(ROOT/name) for name in ['experiments/project_history.json','experiments/metrics_glossary.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis';hyp={f.name:f.read_bytes() for f in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(path):
    sys.argv=['scripts/build_obsidian_vault.py','--vault',str(path)]
    assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
    copy=Path(tmp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);result=audit(copy);build(vault);assert audit(vault)==result
assert hyp=={f.name:f.read_bytes() for f in (vault/'Карта проекта/Гипотезы').glob('*.md')}
assert all(sha(ROOT/name)==value for name,value in stable.items());verify(p)
write('final_checks.json',dict(passed=True,catalog_tests=58,mathematical_tests_previously_passed=8,graph=result,hypotheses_unchanged=True,stable_sources=stable,protocol_sources_unchanged=True,scientific_gate_failed_and_not_promoted=True,
    catalogue_test_log= str((prior/'memory_guard.worker.log').relative_to(ROOT)),catalogue_test_log_sha256=sha(prior/'memory_guard.worker.log'),
    previous_graph_failure='New Lin2008 source lacked the string doi required by the graph generator. Publisher gives a primary URL but no DOI in its record; the field now explicitly states that. No publication identifier invented. No catalogue, glossary, model or scores changed after the58 passing tests.'))
print('Negative scientific gate and corrected graph verified.',flush=True)
