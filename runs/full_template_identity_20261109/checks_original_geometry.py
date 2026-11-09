"""Final catalogue/graph check after the third diagnostic; keep earlier receipt."""
from geometry import *
import contextlib,runpy,tempfile,shutil
verify(read(OUT/'protocol.json'));verify(read(OUT/'comparison_protocol.json'));verify(read(OUT/'original_geometry_protocol.json'))
assert read(OUT/'original_geometry_audit.json')['passed']
stable={name:sha(ROOT/name) for name in ['experiments/project_history.json','experiments/metrics_glossary.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}
sys.argv=['scripts/validate_project_history.py']
with (OUT/'original_geometry_catalog_validation.log').open('x',encoding='utf8') as log:
    with contextlib.redirect_stdout(log):
        try:runpy.run_path(str(ROOT/'scripts/validate_project_history.py'),run_name='__main__')
        except SystemExit as error:
            if error.code:raise
import pytest
assert pytest.main(['tests/test_project_history.py','-q'])==0
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis';hyp={f.name:f.read_bytes() for f in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(path):
    sys.argv=['scripts/build_obsidian_vault.py','--vault',str(path)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
    copy=Path(tmp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);result=audit(copy);build(vault);assert audit(vault)==result
assert hyp=={f.name:f.read_bytes() for f in (vault/'Карта проекта/Гипотезы').glob('*.md')};assert all(sha(ROOT/name)==value for name,value in stable.items())
verify(read(OUT/'protocol.json'));verify(read(OUT/'comparison_protocol.json'));verify(read(OUT/'original_geometry_protocol.json'))
write('original_geometry_final_checks.json',dict(passed=True,catalog_tests=58,mathematical_tests_previously_passed=6,graph=result,hypotheses_unchanged=True,stable_sources=stable,
    three_signal_diagnostics_complete=True,actual_ID_pilot_unrun=True,scope='Support mechanism, two matched geometry conditions, last candidateC34 admission, and unrun revised native pilot. No tracker or unique-core success claimed.'))
print('All support diagnostics, catalogue and graph accepted.',flush=True)
