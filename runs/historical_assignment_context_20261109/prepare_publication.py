"""Publish only owned results; retain foreign calendar/plugin/worktree drafts."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;TEMP=ROOT/'tmp/historical_queries_publication';TEMP.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT))
from scripts.obsidian_level_graph import Tree,journal,plain
def head(p):return subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)
assert json.loads((OUT/'final_graph.json').read_text(encoding='utf8'))['passed']
published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8');current=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');first=published.split('## Журнал изменений\n',1)[1].lstrip().split('\n',1)[0];prefix=current.split('## Журнал изменений\n',1)[1].split(first,1)[0].strip()+'\n\n';assert prefix.count('**09.11.2026,')==9
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
affected.update(k for k in live['nodes'] if k.startswith('literature:'));affected.add('sys:С2')
for key in affected:paths.append('ObsidianFlyVis/'+live['nodes'][key]['path'])
manifest['files']=sorted(n['path'] for n in manifest['nodes'].values());(TEMP/'manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
tree=Tree(ROOT/'ObsidianFlyVis');tree.add('Журнал','Журнал');journal(tree,plan);node=tree.nodes['day:2026-11-09'];assert node['path']==manifest['nodes']['day:2026-11-09']['path'];body='---\nпункт: "'+node['number']+'"\nуровень: 3\nдата: "2026-11-09"\ntags: [граф-проекта/видно]\n---\n\n# '+node['number']+'. '+plain(node['title'])+'\n\n'+node['body'];body='\n'.join(x.expandtabs(4).rstrip() for x in body.strip().splitlines())+'\n';(TEMP/'day.md').write_bytes(body.encode())
paths+=['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','.gitattributes']
folders=['historical_direct_query_inventory_20261109','historical_two_candidate_comparison_20261109','candidate_relative_alignment_20261109','historical_assignment_context_20261109']
paths += [p.relative_to(ROOT).as_posix() for name in folders for p in (ROOT/'runs'/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
overrides={'dissertation/ЕДИНЫЙ_ПЛАН.md':str(TEMP/'plan.md'),'ObsidianFlyVis/.graph-visible.json':str(TEMP/'manifest.json'),'ObsidianFlyVis/'+node['path']:str(TEMP/'day.md')};(TEMP/'paths.txt').write_bytes(('\n'.join(sorted(set(paths)))+'\n').encode());(TEMP/'overrides.json').write_text(json.dumps(overrides,ensure_ascii=False,indent=2),encoding='utf8')
stage=(ROOT/'tmp/interpolation_constant_publication/stage.py').read_text(encoding='utf8').replace("(p.startswith('runs/interpolation_constant_audit_20261109/') or p.startswith('runs/free_candidate_identity_inventory_20261109/'))","any(p.startswith('runs/'+folder+'/') for folder in "+repr(folders)+")").replace('len(nodes)==1226','len(nodes)=='+str(len(manifest['nodes']))).replace('Exact-constant interpolation repair, real-map numerical regression and failed data-readiness gate; no ID benefit claimed.','Frozen direct two-candidate comparison, negative primary gate, coordinate-readiness stop and original assignment context; no new ID assignment.')
(TEMP/'stage.py').write_text(stage,encoding='utf8');print('Prepared',len(set(paths)),'paths',len(manifest['nodes']),'publication nodes')
