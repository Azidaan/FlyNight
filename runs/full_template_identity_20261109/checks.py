"""Validate scientific records and graph without changing frozen inputs."""
from geometry import *
import contextlib,runpy,tempfile,shutil
p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'comparison_protocol.json'));assert read(OUT/'audit.json')['passed']
stable={name:sha(ROOT/name) for name in ['experiments/project_history.json','experiments/metrics_glossary.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}
sys.argv=['scripts/validate_project_history.py']
with (OUT/'catalog_validation.log').open('x',encoding='utf8') as log:
    with contextlib.redirect_stdout(log):
        try:runpy.run_path(str(ROOT/'scripts/validate_project_history.py'),run_name='__main__')
        except SystemExit as error:
            if error.code:raise
import pytest
assert pytest.main(['tests/test_project_history.py','-q'])==0
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis';hyp={path.name:path.read_bytes() for path in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(path):
    sys.argv=['scripts/build_obsidian_vault.py','--vault',str(path)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
    copy=Path(tmp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);graph=audit(copy);build(vault);assert audit(vault)==graph
assert hyp=={path.name:path.read_bytes() for path in (vault/'Карта проекта/Гипотезы').glob('*.md')}
assert all(sha(ROOT/name)==value for name,value in stable.items());verify(p);verify(read(OUT/'comparison_protocol.json'))
write('final_checks.json',dict(passed=True,catalog_tests=58,mathematical_tests_previously_passed=6,graph=graph,hypotheses_unchanged=True,stable_sources=stable,old_and_new_protocol_sources_unchanged=True,
    scope='Mathematical and geometric support mechanism, matched OOF comparison and concrete unrun native-ID proposal; no HOTA or actual ID success claimed.'))
print('Full-template records and graph accepted.',flush=True)
