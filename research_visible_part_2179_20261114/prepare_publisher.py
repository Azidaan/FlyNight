from common import *
source=(ROOT/'research_memory_2179_20261113/publish.py').read_text(encoding='utf8')
source=source.replace("read(D/'checks.json')['passed']", "read(D/'checks_v2.json')['passed']")
source=source.replace("('history_representation_2179_20261113','без-положительного-итога')]", "('history_representation_2179_20261113','без-положительного-итога'),('visible_components_2179_20261114','выполнено')]")
source=source.replace("scope='Confirmed five-case successes remaingreen. New54-case transfer is inconclusive andpurple. No manual circle edits or closureofallidentityproblems.'", "scope='Previous five-case positive checks remain green. New one-event pixel/level causal diagnostic is green within its stated manual-mask scope; it is not independent tracking improvement. Other historical results retain their states.'")
source=source.replace("['result.json','audit.json','summary.json','trace_v2.json','descriptors.npz']", "['audit.json','summary.json','feature_outputs.npz','level_features.npz','mask_review/masks_reviewed.npz']")
source=source.replace("skip={'temp','before','execution'", "skip={'source_renderer_repeat','pytest_temp','pytest_cache','synthetic_fixture','synthetic_fixture_v2','frames','hypothesis_before_bytes','temp','before','execution'")
source=source.replace("    if any(part in skip for part in rel.parts):continue", "    if any(part in skip or part.startswith('hypothesis_preview') or part.startswith('self_test') or part.startswith('fixture') for part in rel.parts):continue")
needle="# Explicit task-owned tree; immutable cache arrays stay local, with audit hashes."
addition="""# Explicitly authorized new hypothesis and its narrow generator source fix.
integration=read(D/'mask_review/hypothesis_integration_v2.json')
assert integration['passed']
for item in integration['changed_or_new']:
    name=item['path']
    if name=='ObsidianFlyVis/.graph-visible.json' or name.startswith('ObsidianFlyVis/Карта проекта/'):
        # Generated publication tree already supplies these nodes and manifest.
        continue
    path=ROOT/name
    assert sha(path)==item['after_sha256'],name
    selected[name]=str(path)
"""
assert source.count(needle)==1;source=source.replace(needle,addition+needle)
with (OUT/'publish.py').open('x',encoding='utf8',newline='\n') as f:f.write(source)
print('SAFE_PUBLISHER_PREPARED')
