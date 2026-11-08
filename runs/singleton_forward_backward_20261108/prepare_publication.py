"""Prepare explicit publication without including earlier unrelated drafts."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TEMP=ROOT/'tmp/singleton_forward_backward_publication'
TEMP.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from scripts.obsidian_level_graph import Tree,journal,plain

def head(path):return subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
live_plan=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8')
first=published.split('## Журнал изменений\n',1)[1].lstrip().split('\n',1)[0]
prefix=live_plan.split('## Журнал изменений\n',1)[1].split(first,1)[0].strip()+'\n\n'
assert prefix.count('**08.11.2026,')==2 and OUT.name in prefix
published_plan=published.replace('## Журнал изменений\n','## Журнал изменений\n\n'+prefix,1)
(TEMP/'plan.md').write_bytes(published_plan.encode())
manifest=json.loads(head('ObsidianFlyVis/.graph-visible.json'))
live=json.loads((ROOT/'ObsidianFlyVis/.graph-visible.json').read_text(encoding='utf8'))
paths=[]
affected=set()
for key,node in live['nodes'].items():
    if node['section'] not in ('Эксперименты','Исследования','Термины'):continue
    previous=manifest['nodes'].get(key)
    if previous==node:continue
    affected.add(key)
    if node['parent']:affected.add(node['parent'])
    if previous and previous['path']!=node['path']:
        paths.append('ObsidianFlyVis/'+previous['path'])
    manifest['nodes'][key]=node
affected|={'sys:С2','exp:singleton_neighbor_input_20261108'}
for key in affected:paths.append('ObsidianFlyVis/'+live['nodes'][key]['path'])
manifest['files']=sorted(n['path'] for n in manifest['nodes'].values())
(TEMP/'manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
tree=Tree(ROOT/'ObsidianFlyVis');tree.add('Журнал','Журнал');journal(tree,published_plan)
node=tree.nodes['day:2026-11-08']
assert node['path']==manifest['nodes']['day:2026-11-08']['path']
body='---\nпункт: "'+node['number']+'"\nуровень: 3\nдата: "2026-11-08"\ntags: [граф-проекта/видно]\n---\n\n# '+node['number']+'. '+plain(node['title'])+'\n\n'+node['body']
body='\n'.join(line.expandtabs(4).rstrip() for line in body.strip().splitlines())+'\n'
assert 'Отдельный календарь гипотез' not in body
(TEMP/'day.md').write_bytes(body.encode())
# These files contain earlier calendar work. Publish only the current additions.
path='scripts/obsidian_level_graph.py'
base=head(path).decode('utf8')
working=(ROOT/path).read_text(encoding='utf8')
base=base.replace('        self.children=defaultdict(list)\n','        self.children=defaultdict(list)\n        self.stealth=defaultdict(list)\n',1)
block=working.split('            for target,label in self.stealth.get(key,[]):',1)[1].split('            rendered[node[',1)[0]
base=base.replace('            rendered[node[','            for target,label in self.stealth.get(key,[]):'+block+'            rendered[node[',1)
function='def used_literature(tree):'+working.split('def used_literature(tree):',1)[1].split('def build(vault):',1)[0]
base=base.replace('def build(vault):',function+'def build(vault):',1)
base=base.replace('    problems(tree,vault)\n','    problems(tree,vault)\n    used_literature(tree)\n',1)
compile(base,path,'exec')
(TEMP/'generator.py').write_bytes(base.encode())
audit_path='scripts/audit_obsidian_levels.py'
base=head(audit_path).decode('utf8')
working=(ROOT/audit_path).read_text(encoding='utf8')
assert 'from urllib.parse import quote' not in base
base=base.replace('from pathlib import Path\n','from pathlib import Path\nfrom urllib.parse import quote\n',1)
block='    literature=ROOT/'+working.split('    literature=ROOT/',1)[1].split('    return dict(',1)[0]
base=base.replace('    return dict(',block+'    return dict(',1)
compile(base,audit_path,'exec')
(TEMP/'audit.py').write_bytes(base.encode())
paths += ['experiments/project_history.json','experiments/obsidian_experiment_topics.json',
          'experiments/system_problems.json','experiments/metrics_glossary.json',
          'experiments/useful_literature.json','tests/test_obsidian_used_literature.py','.gitattributes']
paths += [p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file()]
overrides={'dissertation/ЕДИНЫЙ_ПЛАН.md':str(TEMP/'plan.md'),
 'ObsidianFlyVis/.graph-visible.json':str(TEMP/'manifest.json'),
 'ObsidianFlyVis/'+node['path']:str(TEMP/'day.md'),
 path:str(TEMP/'generator.py'),audit_path:str(TEMP/'audit.py')}
(TEMP/'paths.txt').write_bytes(('\n'.join(sorted(set(paths)))+'\n').encode())
(TEMP/'overrides.json').write_text(json.dumps(overrides,ensure_ascii=False,indent=2),encoding='utf8')
print('Prepared',len(set(paths)),'explicit paths,',len(overrides),'isolated overrides')
