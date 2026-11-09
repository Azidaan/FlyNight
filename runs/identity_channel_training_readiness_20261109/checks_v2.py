"""Continue graph verification after a document-byte fingerprint correction."""
from finish import *
import tempfile
import shutil
import re
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit

prior = ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/19308ddfd46e4fc0bf32eb3c6098a52f'
log = (prior/'memory_guard.worker.log').read_text(encoding='utf8')
assert re.search(r'58 passed in [\d.]+s', log)
assert 'AssertionError' in log and 'СВОДКА_v4.md' in log
assert read(prior/'memory_guard.json')['returncode'] == 1
assert 'OK: каталог согласован с источниками' in (OUT/'catalog_validation.log').read_text(encoding='utf8')
verify(read(OUT/'protocol.json'))
verify(read(OUT/'execution_protocol.json'))
verify(read(OUT/'causal_check_protocol.json'))

vault = ROOT/'ObsidianFlyVis'
hyp = {p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
stable = {p:sha(ROOT/p) for p in ['experiments/project_history.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}
def build(path):
    sys.argv = ['scripts/build_obsidian_vault.py','--vault',str(path)]
    assert main() == 0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
    copy = Path(tmp)/'ObsidianFlyVis'
    shutil.copytree(vault,copy)
    build(copy)
    result = audit(copy)
    build(vault)
    assert audit(vault) == result
assert hyp == {p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
assert all(sha(ROOT/p)==value for p,value in stable.items())
write('final_checks.json',dict(passed=True,catalog_tests=58,
    catalog_test_log=str((prior/'memory_guard.worker.log').relative_to(ROOT)),
    catalog_test_log_sha256=sha(prior/'memory_guard.worker.log'),
    catalog_validation_log_sha256=sha(OUT/'catalog_validation.log'),
    graph=result,hypotheses_unchanged=True,catalog_sha256=stable['experiments/project_history.json'],
    source_inputs_stable=True,zero_model_training=True,
    previous_graph_failure='Summary checksum originally computed on LF string before Windows CRLF file write; corrected to actual file bytes. No scientific scores changed. All 58 catalogue tests had passed before graph failure.',
    test_reuse='No catalogue, glossary or test-code change after all 58 passed; only graph source checksum and explanatory journal corrected. Graph generation and audits rerun.'))
print('Catalogue checks retained; corrected graph generation independently verified.',flush=True)
