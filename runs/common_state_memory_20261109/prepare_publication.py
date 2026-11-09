"""Stage only this result, keeping previous calendar/plugin drafts outside Git."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;TEMP=ROOT/'tmp/common_state_memory_publication';TEMP.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT))
from scripts.obsidian_level_graph import Tree,journal,plain
def head(p):return subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)
assert json.loads((OUT/'final_checks.json').read_text(encoding='utf8'))['passed']
published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8');current=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');first=published.split('## Журнал изменений\n',1)[1].lstrip().split('\n',1)[0];prefix=current.split('## Журнал изменений\n',1)[1].split(first,1)[0].strip()+'\n\n';assert prefix.count('**09.11.2026,')==2 and OUT.name in prefix
plan=published.replace('## Журнал изменений\n','## Журнал изменений\n\n'+prefix,1);(TEMP/'plan.md').write_bytes(plan.encode())
manifest=json.loads(head('ObsidianFlyVis/.graph-visible.json'));live=json.loads((ROOT/'ObsidianFlyVis/.graph-visible.json').read_text(encoding='utf8'));paths=[];affected=set()
for key,node in live['nodes'].items():
    if node['section'] not in ('Эксперименты','Исследования','Журнал'):continue
    previous=manifest['nodes'].get(key)
    if previous==node:continue
    affected.add(key)
    if node['parent']:affected.add(node['parent'])
    if previous and previous['path']!=node['path']:paths.append('ObsidianFlyVis/'+previous['path'])
    manifest['nodes'][key]=node
affected.update(k for k in live['nodes'] if k.startswith('literature:'))
affected.add('sys:С2')
for key in affected:paths.append('ObsidianFlyVis/'+live['nodes'][key]['path'])
manifest['files']=sorted(n['path'] for n in manifest['nodes'].values());(TEMP/'manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
tree=Tree(ROOT/'ObsidianFlyVis');tree.add('Журнал','Журнал');journal(tree,plan);node=tree.nodes['day:2026-11-09'];assert node['path']==manifest['nodes']['day:2026-11-09']['path'];body='---\nпункт: "'+node['number']+'"\nуровень: 3\nдата: "2026-11-09"\ntags: [граф-проекта/видно]\n---\n\n# '+node['number']+'. '+plain(node['title'])+'\n\n'+node['body'];body='\n'.join(x.expandtabs(4).rstrip() for x in body.strip().splitlines())+'\n';(TEMP/'day.md').write_bytes(body.encode())
paths+=['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','.gitattributes']
paths += [p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
# The account-specific Gemini chat URL and UI accessibility capture remain local.
consult=ROOT/'research_gemini_memory_review_20261109';capture=json.loads((consult/'capture_01_verification.json').read_text(encoding='utf8'));capture.pop('chat');(consult/'capture_public.json').write_text(json.dumps(capture,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
paths += [(consult/n).relative_to(ROOT).as_posix() for n in ['request_sent.txt','answer_01.txt','capture_public.json','ПРОВЕРКА.md']]
overrides={'dissertation/ЕДИНЫЙ_ПЛАН.md':str(TEMP/'plan.md'),'ObsidianFlyVis/.graph-visible.json':str(TEMP/'manifest.json'),'ObsidianFlyVis/'+node['path']:str(TEMP/'day.md')};(TEMP/'paths.txt').write_bytes(('\n'.join(sorted(set(paths)))+'\n').encode());(TEMP/'overrides.json').write_text(json.dumps(overrides,ensure_ascii=False,indent=2),encoding='utf8')
stage=(ROOT/'tmp/direct_memory_anchor_publication/stage.py').read_text(encoding='utf8').replace("p.startswith('runs/direct_memory_anchor_inventory_20261109/')","p.startswith('runs/common_state_memory_20261109/')").replace('len(nodes)==1221','len(nodes)=='+str(len(manifest['nodes']))).replace("'experiments/obsidian_experiment_topics.json','.gitattributes'","'experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','.gitattributes'").replace('Exact-annotation earlier-memory inventory and common-state lineage; no model or performance claim.','Common-state memory comparison, negative result, matched controls and exact candidate ownership; no new ID assignment.')
(TEMP/'stage.py').write_text(stage,encoding='utf8');print('Prepared',len(set(paths)),'explicit paths;',len(manifest['nodes']),'publication nodes')
