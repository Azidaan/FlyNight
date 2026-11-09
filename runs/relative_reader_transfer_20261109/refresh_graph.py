"""Only derived graph refresh after the final clarification; reuse green tests."""
import study as t
from study import *
import tempfile,shutil
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
assert read(OUT/'final_checks.json')['passed'];catalog_sha=sha(ROOT/'experiments/project_history.json');vault=ROOT/'ObsidianFlyVis';hyp={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(v):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
    copy=Path(tmp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);result=audit(copy);build(vault);assert audit(vault)==result
assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};assert sha(ROOT/'experiments/project_history.json')==catalog_sha;verify(read(OUT/'protocol.json'));verify(read(OUT/'maps_complete.json'));s.verify(read(REL/'protocol.json'))
write('final_graph.json',dict(passed=True,previous_catalog_tests=58,catalog_sha256=catalog_sha,hypotheses_unchanged=True,graph=result,only_post_test_edit='New journal clarification and human report; catalogue unchanged.'));print('Final clarification graph verified',flush=True)
