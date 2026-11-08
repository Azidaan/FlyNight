"""Publish this comparison while preserving other live drafts."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TEMP=ROOT/'tmp/singleton_neighbor_publication'
TEMP.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from scripts.obsidian_level_graph import Tree,journal,plain

def head(path): return subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
live_plan=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8')
first=published.split('## Журнал изменений\n',1)[1].lstrip().split('\n',1)[0]
prefix=live_plan.split('## Журнал изменений\n',1)[1].split(first,1)[0].strip()+'\n\n'
assert prefix.count('**08.11.2026,')==2 and 'singleton_neighbor_input_20261108' in prefix
published_plan=published.replace('## Журнал изменений\n','## Журнал изменений\n\n'+prefix,1)
(TEMP/'plan.md').write_bytes(published_plan.encode())
manifest=json.loads(head('ObsidianFlyVis/.graph-visible.json'))
live=json.loads((ROOT/'ObsidianFlyVis/.graph-visible.json').read_text(encoding='utf8'))
key='exp:'+OUT.name
added=[]
while key not in manifest['nodes']:
    manifest['nodes'][key]=live['nodes'][key]
    added.append(key)
    key=live['nodes'][key]['parent']
parents={live['nodes'][key]['parent'] for key in added}
paths=['ObsidianFlyVis/'+live['nodes'][key]['path'] for key in set(added)|parents]
for key,node in live['nodes'].items():
    if node['parent'] not in parents:
        continue
    previous=manifest['nodes'].get(key)
    paths.append('ObsidianFlyVis/'+node['path'])
    if previous and previous['path']!=node['path']:
        paths.append('ObsidianFlyVis/'+previous['path'])
    manifest['nodes'][key]=node
paths.append('ObsidianFlyVis/'+live['nodes']['sys:С2']['path'])
# Inserting a title can renumber neighboring thematic branches as well as leaves.
# Carry every affected experiment node and its parent, keeping former files archived.
affected=set()
original_nodes=json.loads(head('ObsidianFlyVis/.graph-visible.json'))['nodes']
for key,node in live['nodes'].items():
    if node['section']!='Эксперименты':
        continue
    original=original_nodes.get(key)
    if original==node:
        continue
    affected.add(key)
    if node['parent']:
        affected.add(node['parent'])
    if original and original['path']!=node['path']:
        paths.append('ObsidianFlyVis/'+original['path'])
    manifest['nodes'][key]=node
for key in affected:
    paths.append('ObsidianFlyVis/'+live['nodes'][key]['path'])
manifest['files']=sorted(n['path'] for n in manifest['nodes'].values())
(TEMP/'manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
tree=Tree(ROOT/'ObsidianFlyVis');tree.add('Журнал','Журнал');journal(tree,published_plan)
node=tree.nodes['day:2026-11-08']
assert node['path']==manifest['nodes']['day:2026-11-08']['path']
body='---\nпункт: "'+node['number']+'"\nуровень: 3\nдата: "2026-11-08"\ntags: [граф-проекта/видно]\n---\n\n# '+node['number']+'. '+plain(node['title'])+'\n\n'+node['body']
body='\n'.join(line.expandtabs(4).rstrip() for line in body.strip().splitlines())+'\n'
assert 'Отдельный календарь гипотез' not in body
(TEMP/'day.md').write_bytes(body.encode())
paths += ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','.gitattributes']
paths += [p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file()]
overrides={'dissertation/ЕДИНЫЙ_ПЛАН.md':str(TEMP/'plan.md'),
    'ObsidianFlyVis/.graph-visible.json':str(TEMP/'manifest.json'),
    'ObsidianFlyVis/'+node['path']:str(TEMP/'day.md')}
(TEMP/'paths.txt').write_bytes(('\n'.join(sorted(set(paths)))+'\n').encode())
(TEMP/'overrides.json').write_text(json.dumps(overrides,ensure_ascii=False,indent=2),encoding='utf8')
print('Explicit paths',len(set(paths)),'index-only overrides',len(overrides))
