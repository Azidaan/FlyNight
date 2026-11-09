"""Publish only this task's evidence, preserving unrelated shared drafts."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'runs/identity_channel_training_readiness_20261109'
TEMP = ROOT / 'tmp/identity_synthesis_publication'
TEMP.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from scripts.obsidian_level_graph import Tree, journal, plain

def head(path):
    return subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT)

assert json.loads((OUT / 'final_checks.json').read_text(encoding='utf8'))['passed']
published = head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8')
current = (ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
first = published.split('## Журнал изменений\n', 1)[1].lstrip().split('\n', 1)[0]
prefix = current.split('## Журнал изменений\n', 1)[1].split(first, 1)[0].strip() + '\n\n'
assert sum(x.startswith('**') for x in prefix.splitlines()) == 6
assert all('09.11.2026,' in x for x in prefix.splitlines() if x.startswith('**'))
plan = published.replace('## Журнал изменений\n', '## Журнал изменений\n\n' + prefix, 1)
(TEMP / 'plan.md').write_bytes(plan.encode('utf8'))
manifest = json.loads(head('ObsidianFlyVis/.graph-visible.json'))
live = json.loads((ROOT / 'ObsidianFlyVis/.graph-visible.json').read_text(encoding='utf8'))
paths = []
affected = set()
for key, node in live['nodes'].items():
    if node['section'] not in ('Эксперименты', 'Исследования', 'Журнал'):
        continue
    previous = manifest['nodes'].get(key)
    if previous == node:
        continue
    affected.add(key)
    if node['parent']:
        affected.add(node['parent'])
    if previous and previous['path'] != node['path']:
        paths.append('ObsidianFlyVis/' + previous['path'])
    manifest['nodes'][key] = node
affected.update(k for k in live['nodes'] if k.startswith('literature:'))
affected.add('sys:С2')
for key in affected:
    paths.append('ObsidianFlyVis/' + live['nodes'][key]['path'])
manifest['files'] = sorted(n['path'] for n in manifest['nodes'].values())
(TEMP / 'manifest.json').write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf8'))
tree = Tree(ROOT / 'ObsidianFlyVis')
tree.add('Журнал', 'Журнал')
journal(tree, plan)
node = tree.nodes['day:2026-11-09']
assert node['path'] == manifest['nodes']['day:2026-11-09']['path']
body = '---\nпункт: "' + node['number'] + '"\nуровень: 3\nдата: "2026-11-09"\ntags: [граф-проекта/видно]\n---\n\n# ' + node['number'] + '. ' + plain(node['title']) + '\n\n' + node['body']
body = '\n'.join(x.expandtabs(4).rstrip() for x in body.strip().splitlines()) + '\n'
(TEMP / 'day.md').write_bytes(body.encode('utf8'))
paths += ['experiments/project_history.json', 'experiments/obsidian_experiment_topics.json',
    'experiments/system_problems.json', 'experiments/useful_literature.json',
    'experiments/PROBLEM_SYNTHESIS_RULE.md', '.gitattributes']
folders = ['runs/identity_channel_training_readiness_20261109', 'research_identity_synthesis_20261109']
for folder in folders:
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'gemini_answer_accessibility.txt']
assert not any('accessibility' in p for p in paths)
overrides = {'dissertation/ЕДИНЫЙ_ПЛАН.md': str(TEMP / 'plan.md'),
    'ObsidianFlyVis/.graph-visible.json': str(TEMP / 'manifest.json'),
    'ObsidianFlyVis/' + node['path']: str(TEMP / 'day.md')}
(TEMP / 'paths.txt').write_bytes(('\n'.join(sorted(set(paths))) + '\n').encode('utf8'))
(TEMP / 'overrides.json').write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding='utf8')
stage = (ROOT / 'tmp/relative_reader_publication/stage.py').read_text(encoding='utf8')
stage = stage.replace("any(p.startswith('runs/'+folder+'/') for folder in ['relative_joint_reader_20261109', 'relative_reader_transfer_20261109'])", "any(p.startswith(folder+'/') for folder in " + repr(folders) + ')')
stage = stage.replace('len(nodes)==1236', 'len(nodes)==' + str(len(manifest['nodes'])))
stage = stage.replace('Whole relative-template comparison and fixed three-video descriptive transfer, with matched image controls; no extra FlyVis or ID benefit claimed.', 'Living evidence synthesis, one critically verified Gemini response and causal labeled-data readiness; no learned model or ID improvement claimed.')
(TEMP / 'stage.py').write_text(stage, encoding='utf8')
print('Prepared', len(set(paths)), 'paths;', len(manifest['nodes']), 'publication graph nodes')
