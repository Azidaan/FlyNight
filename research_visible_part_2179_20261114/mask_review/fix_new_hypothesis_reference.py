"""Fix only the new hypothesis footer through its existing source renderer."""
from pathlib import Path
import sys,json,hashlib,shutil,importlib
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
report=read(HERE/'hypothesis_integration.json')
vault=ROOT/'ObsidianFlyVis'
node=report['new_graph_node']
note=report['new_source_note']
protected=read(HERE/'hypothesis_before.json')['protected']
week=report['allowed_existing_parent_change']
for rel,h in protected.items():
    if rel!=week:assert sha(vault/rel)==h
generator=ROOT/'scripts/obsidian_deferred_identity_hypotheses.py'
generator_before=sha(generator)
backup=HERE/'hypothesis_before_bytes/scripts/obsidian_deferred_identity_hypotheses.py'
backup.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(generator,backup)
text=generator.read_text(encoding='utf8')
needle="        text += 'Полная запись: [Единый план, раздел 7.12](' + (ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md').as_uri() + ').\\n'"
replacement="        if item.get('record_source'):\n            text += 'Полная постановка и источники: [запись гипотезы](' + (ROOT / item['record_source']).as_uri() + ').\\n'\n        else:\n    "+needle
assert text.count(needle)==1
generator.write_text(text.replace(needle,replacement),encoding='utf8')
registry=ROOT/'experiments/deferred_identity_hypotheses.json'
data=read(registry)
idea=next(item for item in data['ideas'] if item['id']=='deferred_light_instance_masks_20261114')
assert not idea.get('record_source')
idea['record_source']='experiments/deferred_identity_hypotheses.json'
write(registry,data)
write(HERE/'instance_segmentation_hypothesis.json',idea)
write(HERE/'deferred_identity_hypotheses.proposed.json',data)
preview=HERE/'hypothesis_preview_v4'
source_generator=importlib.import_module('scripts.obsidian_deferred_identity_hypotheses')
source_generator.build_source_notes(preview)
# Do not refresh any old source note as a side effect of the renderer.
calendar=read(ROOT/'experiments/hypothesis_calendar.json')
for source in calendar['entries']:
    if source+'.md'!=note:shutil.copyfile(vault/(source+'.md'),preview/(source+'.md'))
gen=importlib.import_module('scripts.obsidian_level_graph')
shutil.copyfile(vault/'.graph-visible.json',preview/'.graph-visible.json')
tree=gen.Tree(preview)
tree.old={k:v for k,v in tree.old.items() if v['section']=='Гипотезы'}
tree.add('Гипотезы','Гипотезы')
gen.hypotheses(tree,preview)
tree.save()
after=read(preview/'.graph-visible.json')
changed=[]
for key,item in after['nodes'].items():
    target=vault/item['path']
    if target.read_bytes()!=(preview/item['path']).read_bytes():changed.append(item['path'])
assert changed==[node['path']],changed
new_source=(preview/note).read_text(encoding='utf8')
new_graph=(preview/node['path']).read_text(encoding='utf8')
assert 'Полная запись: [Единый план, раздел 7.12]' not in new_source
assert 'Полная постановка и источники: [запись гипотезы]' in new_source
assert 'Полная постановка и источники: [запись гипотезы]' in new_graph
for rel in [note,node['path']]:shutil.copyfile(preview/rel,vault/rel)
for rel,h in protected.items():
    if rel!=week:assert sha(vault/rel)==h
old_source=read(HERE/'hypothesis_before_bytes/experiments/deferred_identity_hypotheses.json')
assert read(registry)['ideas'][:-1]==old_source['ideas']
source_generator.build_source_notes(HERE/'source_renderer_repeat')
assert (HERE/'source_renderer_repeat'/note).read_bytes()==(vault/note).read_bytes()
for item in report['changed_or_new']:
    item['after_sha256']=sha(ROOT/item['path'])
    item['source']=item['path']
report['changed_or_new'].append(dict(path=generator.relative_to(ROOT).as_posix(),before_sha256=generator_before,after_sha256=sha(generator),source=generator.relative_to(ROOT).as_posix()))
report.update(applied=datetime.now(timezone(timedelta(hours=3))).isoformat(), reference_footer_fixed=True, generator_optional_record_source=True, new_source_render_repeat_exact=True, generation_test='Existing hypothesis branch regenerated on isolated preview; only authorized new leaf changed for footer. Older source notes restored before branch render because legacy renderer adds formatting; live old notes never touched.', preserved_previous_preview_failures=['v1: missing planned B1 source in preview','v2/v3: proposed registry path leaked into date-source links; caught before any live write','v4: preview passed; stable live source links used'], no_new_gpu=True)
write(HERE/'hypothesis_integration_v2.json',report)
print(json.dumps(dict(passed=True,changed_files=len(report['changed_or_new']),new_node=node['path'],old_hypotheses_unchanged=True),ensure_ascii=False))
