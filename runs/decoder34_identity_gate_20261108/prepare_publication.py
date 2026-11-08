"""Publish the two completed stages without unrelated live calendar drafts."""
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
MAPS=ROOT/'runs/full_decoder_outputs_20261108';TEMP=ROOT/'tmp/decoder34_publication'
TEMP.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT))
from scripts.obsidian_level_graph import Tree,journal,plain
def head(path):return subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
assert json.loads((OUT/'final_checks.json').read_text(encoding='utf8'))['passed']
published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8')
liveplan=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
first=published.split('## Журнал изменений\n',1)[1].lstrip().split('\n',1)[0]
prefix=liveplan.split('## Журнал изменений\n',1)[1].split(first,1)[0].strip()+'\n\n'
assert prefix.count('**08.11.2026,')==4 and OUT.name in prefix and MAPS.name in prefix
plan=published.replace('## Журнал изменений\n','## Журнал изменений\n\n'+prefix,1)
(TEMP/'plan.md').write_bytes(plan.encode())
manifest=json.loads(head('ObsidianFlyVis/.graph-visible.json'))
live=json.loads((ROOT/'ObsidianFlyVis/.graph-visible.json').read_text(encoding='utf8'))
paths=[];affected=set()
for key,node in live['nodes'].items():
    if node['section'] not in ('Эксперименты','Исследования'):continue
    previous=manifest['nodes'].get(key)
    if previous==node:continue
    affected.add(key)
    if node['parent']:affected.add(node['parent'])
    if previous and previous['path']!=node['path']:paths.append('ObsidianFlyVis/'+previous['path'])
    manifest['nodes'][key]=node
affected.add('sys:С2')
affected.update(k for k in live['nodes'] if k.startswith('literature:'))
for key in affected:paths.append('ObsidianFlyVis/'+live['nodes'][key]['path'])
manifest['files']=sorted(n['path'] for n in manifest['nodes'].values())
(TEMP/'manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
tree=Tree(ROOT/'ObsidianFlyVis');tree.add('Журнал','Журнал');journal(tree,plan)
node=tree.nodes['day:2026-11-08'];assert node['path']==manifest['nodes']['day:2026-11-08']['path']
body='---\nпункт: "'+node['number']+'"\nуровень: 3\nдата: "2026-11-08"\ntags: [граф-проекта/видно]\n---\n\n# '+node['number']+'. '+plain(node['title'])+'\n\n'+node['body']
body='\n'.join(line.expandtabs(4).rstrip() for line in body.strip().splitlines())+'\n'
assert 'Отдельный календарь гипотез' not in body
(TEMP/'day.md').write_bytes(body.encode())
paths+=['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','.gitattributes']
for folder in [OUT,MAPS]:
    paths += [p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
overrides={'dissertation/ЕДИНЫЙ_ПЛАН.md':str(TEMP/'plan.md'),'ObsidianFlyVis/.graph-visible.json':str(TEMP/'manifest.json'),'ObsidianFlyVis/'+node['path']:str(TEMP/'day.md')}
(TEMP/'paths.txt').write_bytes(('\n'.join(sorted(set(paths)))+'\n').encode())
(TEMP/'overrides.json').write_text(json.dumps(overrides,ensure_ascii=False,indent=2),encoding='utf8')
source=(ROOT/'tmp/background_patch_gate_publication/stage.py').read_text(encoding='utf8')
source=source.replace("p.startswith('runs/background_patch_gate_20261108/')","p.startswith(('runs/decoder34_identity_gate_20261108/','runs/full_decoder_outputs_20261108/'))")
source=source.replace('len(nodes)==1207','len(nodes)==1209')
source=source.replace('Background patch filter, frozen fit gate and source verification.','Recovered26 standard decoder-input maps and frozen matched34 gate.')
(TEMP/'stage.py').write_text(source,encoding='utf8')
print('Prepared',len(set(paths)),'explicit files; publication nodes',len(manifest['nodes']))
