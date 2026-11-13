"""Generate and apply only the explicitly requested new hypothesis branch."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import hashlib
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import scripts.obsidian_deferred_identity_hypotheses as source_generator
import scripts.obsidian_level_graph as graph_generator

def read(path):
    return json.loads(path.read_text(encoding='utf8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

vault = ROOT/'ObsidianFlyVis'
preview = HERE/'hypothesis_preview_v4'
assert not preview.exists(), 'Do not overwrite a previous preview'
idea = read(HERE/'instance_segmentation_hypothesis.json')
source_path = ROOT/'experiments/deferred_identity_hypotheses.json'
calendar_path = ROOT/'experiments/hypothesis_calendar.json'
source_before = read(source_path)
calendar_before = read(calendar_path)
assert idea['id'] not in {item['id'] for item in source_before['ideas']}
assert idea['note'] not in calendar_before['entries']
source_after = copy.deepcopy(source_before)
source_after['purpose'] = 'Отложенные идеи по прямым поручениям автора; не очередь научных запусков'
source_after['addition_source_20261114'] = 'Поручение автора 14.11.2026 добавить гипотезу дополнительной лёгкой сети сегментации и отображения отдельных объектов разными цветами; запуск новой модели не поручен.'
source_after['ideas'].append(idea)
calendar_after = copy.deepcopy(calendar_before)
calendar_after['entries'][idea['note']] = dict(date=idea['registered'], source='experiments/deferred_identity_hypotheses.json', locator='Прямое поручение автора 14.11.2026 сохранить гипотезу лёгкой сети масок отдельных экземпляров; '+idea['id'], thematic_parent='Сохранение личности объектов — отделение видимой части от соседа и фона')
write(HERE/'deferred_identity_hypotheses.proposed.json', source_after)
write(HERE/'hypothesis_calendar.proposed.json', calendar_after)
assert source_after['ideas'][:-1] == source_before['ideas']
assert {k:v for k,v in calendar_after['entries'].items() if k!=idea['note']} == calendar_before['entries']
before = {str(p.relative_to(ROOT)):sha(p) for p in [source_path,calendar_path,vault/'.graph-visible.json',vault/'.obsidian/graph.json']}
protected = {p.relative_to(vault).as_posix():sha(p) for p in vault.rglob('*.md') if 'Гипотез' in p.as_posix()}
write(HERE/'hypothesis_before.json',dict(created=datetime.now(timezone(timedelta(hours=3))).isoformat(), files=before, protected=protected))
preview.mkdir()
for folder in ['Гипотезы — подходы','Гипотезы — трекинг','Гипотезы — ядро FlyVis','Карта проекта/Гипотезы']:
    for path in (vault/folder).glob('*.md'):
        dest=preview/path.relative_to(vault)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
for rel in ['Гипотезы.md','.graph-visible.json','.obsidian/graph.json']:
    dest=preview/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(vault/rel,dest)
for note in calendar_before['entries']:
    path=vault/(note+'.md')
    dest=preview/path.relative_to(vault)
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)

class Proxy:
    def __truediv__(self, name):
        return {'experiments/deferred_identity_hypotheses.json':HERE/'deferred_identity_hypotheses.proposed.json','experiments/hypothesis_calendar.json':HERE/'hypothesis_calendar.proposed.json'}.get(str(name),ROOT/name)
    def __getattr__(self,name):
        return getattr(ROOT,name)

source_generator.ROOT=Proxy()
source_generator.build_source_notes(preview)
# The existing source notes are not authorized for a rewrite, even if a current
# renderer would generate different formatting. Only copy the new source note.
for note in calendar_before['entries']:
    shutil.copyfile(vault/(note+'.md'),preview/(note+'.md'))
old_manifest=read(vault/'.graph-visible.json')
tree=graph_generator.Tree(preview)
tree.old={k:v for k,v in tree.old.items() if v['section']=='Гипотезы'}
tree.add('Гипотезы','Гипотезы')
graph_generator.ROOT=Proxy()
graph_generator.local_file=lambda name,label:'['+label+']('+(ROOT/name).as_uri()+')'
graph_generator.hypotheses(tree,preview)
tree.save()
generated=read(preview/'.graph-visible.json')
new_key='hyp:'+idea['note']
assert set(generated['nodes'])-set(old_manifest['nodes']) == {new_key,'hyp-day:2026-11-14'}
node=generated['nodes'][new_key]
assert node['color']=='ожидает' and node['parent']=='hyp-day:2026-11-14'
assert generated['nodes']['hyp-day:2026-11-14']['parent']=='hyp-month:2026-11:week:3'
week_path=old_manifest['nodes']['hyp-month:2026-11:week:3']['path']
changes={}
for key, entry in generated['nodes'].items():
    rel=entry['path']
    current=vault/rel
    proposed=preview/rel
    if not current.exists() or current.read_bytes()!=proposed.read_bytes():
        changes[rel]=proposed
assert set(changes)=={week_path,node['path'],generated['nodes']['hyp-day:2026-11-14']['path']},list(changes)
for rel, checksum in protected.items():
    if rel==week_path:continue
    assert sha(vault/rel)==checksum
    if rel.startswith('Карта проекта/Гипотезы/') and (preview/rel).exists():
        assert sha(preview/rel)==checksum, ('Unrequested hypothesis changed',rel)
merged=copy.deepcopy(old_manifest)
merged['nodes']={k:v for k,v in old_manifest['nodes'].items() if v['section']!='Гипотезы'}
merged['nodes'].update(generated['nodes'])
merged['files']=sorted(v['path'] for v in merged['nodes'].values())
write(HERE/'graph-visible.merged.json',merged)
# Validate complete hypothesis edges in the generated preview.
edges=[]
paths={item['path']:key for key,item in generated['nodes'].items()}
for key,item in generated['nodes'].items():
    text=(preview/item['path']).read_text(encoding='utf8')
    for match in graph_generator.LINK.finditer(text):
        target=match[1]+'.md'
        assert target in paths
        child=generated['nodes'][paths[target]]
        assert child['parent']==key
        edges.append((key,paths[target]))
assert len(edges)==len(generated['nodes'])-1
assert len({v['number'] for v in generated['nodes'].values()})==len(generated['nodes'])

targets={source_path:HERE/'deferred_identity_hypotheses.proposed.json',calendar_path:HERE/'hypothesis_calendar.proposed.json',vault/(idea['note']+'.md'):preview/(idea['note']+'.md'),vault/'.graph-visible.json':HERE/'graph-visible.merged.json'}
targets.update({vault/rel:path for rel,path in changes.items()})
for rel,checksum in before.items():
    assert sha(ROOT/rel)==checksum, ('Concurrent edit',rel)
for rel,checksum in protected.items():
    assert sha(vault/rel)==checksum, ('Concurrent hypothesis edit',rel)
backup=HERE/'hypothesis_before_bytes'
for target in targets:
    if target.exists():
        dest=backup/target.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(target,dest)
manifest=[]
for target, proposed in targets.items():
    previous=sha(target) if target.exists() else None
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(proposed.read_bytes())
    manifest.append(dict(path=target.relative_to(ROOT).as_posix(),before_sha256=previous,after_sha256=sha(target),source=proposed.relative_to(ROOT).as_posix()))
for rel,checksum in protected.items():
    if rel!=week_path:assert sha(vault/rel)==checksum
assert sha(vault/'.obsidian/graph.json')==before[str((vault/'.obsidian/graph.json').relative_to(ROOT))]
live=read(vault/'.graph-visible.json')
assert live==merged
assert read(source_path)['ideas'][:-1]==source_before['ideas']
assert {k:v for k,v in read(calendar_path)['entries'].items() if k!=idea['note']}==calendar_before['entries']
write(HERE/'hypothesis_integration.json',dict(passed=True, applied=datetime.now(timezone(timedelta(hours=3))).isoformat(), changed_or_new=manifest, new_graph_node=node, new_source_note=idea['note']+'.md', prior_hypothesis_markdown_preserved=len(protected)-1, allowed_existing_parent_change=week_path, other_manifest_nodes_unchanged=all(merged['nodes'][key]==entry for key,entry in old_manifest['nodes'].items()), old_ideas_exact=True, old_calendar_entries_exact=True, graph_settings_unchanged=True, preview_tree_edges=len(edges), preview_tree_nodes=len(generated['nodes']), scope='Explicitly authorized addition only; full catalog/plan graph synchronization belongs to parent. No new model download or inference.'))
print(json.dumps(dict(passed=True, changed_files=len(manifest), new_node=node),ensure_ascii=False))
