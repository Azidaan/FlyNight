"""Stage only the draft, one journal addition and its generated graph patch."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TEMP=ROOT/'tmp/single_candidate_draft_publication';TEMP.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT))
from scripts.obsidian_level_graph import Tree,journal,plain
def head(p):return subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)
assert json.loads((OUT/'doc_checks.json').read_text(encoding='utf8'))['passed'];published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8');current=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');first=published.split('## Журнал изменений\n',1)[1].lstrip().split('\n',1)[0];prefix=current.split('## Журнал изменений\n',1)[1].split(first,1)[0].strip()+'\n\n';assert prefix.count('**09.11.2026,')==1;plan=published.replace('## Журнал изменений\n','## Журнал изменений\n\n'+prefix,1);(TEMP/'plan.md').write_bytes(plan.encode())
manifest=json.loads(head('ObsidianFlyVis/.graph-visible.json'));live=json.loads((ROOT/'ObsidianFlyVis/.graph-visible.json').read_text(encoding='utf8'));paths=[];affected=set()
for key,node in live['nodes'].items():
    if node['section']!='Журнал':continue
    previous=manifest['nodes'].get(key)
    if previous==node:continue
    affected.add(key)
    if node['parent']:affected.add(node['parent'])
    if previous and previous['path']!=node['path']:paths.append('ObsidianFlyVis/'+previous['path'])
    manifest['nodes'][key]=node
for key in affected:paths.append('ObsidianFlyVis/'+live['nodes'][key]['path'])
manifest['files']=sorted(n['path'] for n in manifest['nodes'].values());(TEMP/'manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
tree=Tree(ROOT/'ObsidianFlyVis');tree.add('Журнал','Журнал');journal(tree,plan);node=tree.nodes['day:2026-11-09'];body='---\nпункт: "'+node['number']+'"\nуровень: 3\nдата: "2026-11-09"\ntags: [граф-проекта/видно]\n---\n\n# '+node['number']+'. '+plain(node['title'])+'\n\n'+node['body'];body='\n'.join(x.expandtabs(4).rstrip() for x in body.strip().splitlines())+'\n';(TEMP/'day.md').write_bytes(body.encode())
paths += [p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts];overrides={'dissertation/ЕДИНЫЙ_ПЛАН.md':str(TEMP/'plan.md'),'ObsidianFlyVis/.graph-visible.json':str(TEMP/'manifest.json'),'ObsidianFlyVis/'+node['path']:str(TEMP/'day.md')};(TEMP/'paths.txt').write_bytes(('\n'.join(sorted(set(paths)))+'\n').encode());(TEMP/'overrides.json').write_text(json.dumps(overrides,ensure_ascii=False,indent=2),encoding='utf8')
stage=(ROOT/'tmp/relative_reader_publication/stage.py').read_text(encoding='utf8');start=stage.index('required={p for p in paths');end=stage.index('assert required<=staged',start);stage=stage[:start]+"required={p for p in paths if p.startswith('research_single_candidate_relative_protocol_20261109/')}|set(overrides)\n"+stage[end:];stage=stage.replace('len(nodes)==1236','len(nodes)=='+str(len(manifest['nodes']))).replace('Whole relative-template comparison and fixed three-video descriptive transfer, with matched image controls; no extra FlyVis or ID benefit claimed.','Draft singleton refusal protocol and two published monotonic-threshold counterexamples; no new science run.');(TEMP/'stage.py').write_text(stage,encoding='utf8');print('Prepared draft only',len(paths),len(manifest['nodes']))
