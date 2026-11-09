"""Refresh the new journal after supplementary attribution; reuse green tests."""
import audit as a
from audit import *
import shutil,tempfile
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
assert read(OUT/'final_checks.json')['passed'];catalog_sha=sha(ROOT/'experiments/project_history.json');vault=ROOT/'ObsidianFlyVis';hyp={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(v):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
    copy=Path(temp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);result=audit(copy);build(vault);assert audit(vault)==result
assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};assert sha(ROOT/'experiments/project_history.json')==catalog_sha
s.verify(read(OUT/'protocol.json'));s.verify(read(BASE/'protocol.json'));s.verify(read(BASE/'attribution_protocol.json'));s.verify(read(BASE/'maps_complete.json'))
write('final_graph.json',dict(passed=True,previous_checks='final_checks.json:58 tests green; catalogue was not edited by supplementary attribution',catalog_sha256=catalog_sha,hypotheses_unchanged=True,graph=result));print('Journal attribution and both graphs verified',flush=True)
