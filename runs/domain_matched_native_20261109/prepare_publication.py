"""Prepare own completed records with the published protected graph version."""
from evaluation_common import *
start(1800)
import subprocess,shutil,re
TEMP=ROOT/'tmp/domain_native_publication';TEMP.mkdir(parents=True,exist_ok=True)
def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=ROOT,**kw)
def head(path):return git('show','HEAD:'+path)
def digest(data):return hashlib.sha256(data).hexdigest()
def save(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
checks=read(OUT/'final_checks.json');assert checks['passed'] and checks['catalog_tests']==58
assert git('rev-parse','HEAD').decode().strip()=='42ba6c1b337f21bb3c2dcc479130426d094ab971'
assert not git('diff','--cached','--name-only').strip()
for name,value in checks['stable_sources'].items():assert sha(ROOT/name)==value,name
published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8').replace('\r\n','\n');current=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
marker='## Журнал изменений\n';first=published.split(marker,1)[1].lstrip().split('\n',1)[0]
prefix=current.split(marker,1)[1].split(first,1)[0].strip()+'\n\n';headers=[s for s in prefix.splitlines() if s.startswith('**')]
assert len(headers)>=4 and all(s.startswith('**09.11.2026,') for s in headers),headers
save(TEMP/'plan.md',published.replace(marker,marker+'\n'+prefix,1).encode())
vault=ROOT/'ObsidianFlyVis';preview=TEMP/'vault';assert not preview.exists();shutil.copytree(vault,preview)
paths=git('ls-tree','-r','--name-only','-z','HEAD','--','ObsidianFlyVis').decode().strip('\0').split('\0');paths=[p for p in paths if p.endswith('.md')]
batch=git('cat-file','--batch',input=('\n'.join('HEAD:'+p for p in paths)+'\n').encode());offset=0;oldfiles={}
for path in paths:
    end=batch.index(b'\n',offset);_,kind,size=batch[offset:end].decode().split();assert kind=='blob';offset=end+1
    data=batch[offset:offset+int(size)];offset+=int(size)+1;oldfiles[path]=data
assert offset==len(batch)
old=json.loads(head('ObsidianFlyVis/.graph-visible.json'));oldactive=set()
for path,data in oldfiles.items():
    rel=path.removeprefix('ObsidianFlyVis/')
    if rel.startswith('Карта проекта/'):
        save(preview/rel,data);front=data.decode().split('---\n',2)
        if len(front)==3 and 'граф-проекта/видно' in front[1]:oldactive.add(rel)
    if any('Гипотез' in part for part in Path(rel).parts):save(preview/rel,data)
augmented=json.loads(json.dumps(old,ensure_ascii=False));known=set(old['files'])
for file in (preview/'Карта проекта').rglob('*.md'):
    front=file.read_text(encoding='utf8').split('---\n',2);rel=file.relative_to(preview).as_posix()
    if len(front)!=3 or 'граф-проекта/видно' not in front[1] or rel in known:continue
    number=re.search(r'^пункт: "([\d.]+)"$',front[1],re.M)[1]
    augmented['nodes']['publication-retired:'+rel]=dict(path=rel,number=number,parent='publication-retired',section=Path(rel).parts[1],title=file.stem,color=None)
save(preview/'.graph-visible.json',(json.dumps(augmented,ensure_ascii=False,indent=2)+'\n').encode())
class RootProxy:
    def __truediv__(self,name):return TEMP/'plan.md' if str(name)=='dissertation/ЕДИНЫЙ_ПЛАН.md' else ROOT/name
    def __fspath__(self):return str(ROOT)
    def __getattr__(self,name):return getattr(ROOT,name)
ns={'__name__':'publication_generator','__file__':str(ROOT/'scripts/obsidian_level_graph.py')}
exec(compile(head('scripts/obsidian_level_graph.py'),'published_level_generator.py','exec'),ns);ns['ROOT']=RootProxy();ns['local_file']=lambda name,label:'['+label+']('+(ROOT/name).as_uri()+')'
ns['build'](preview);manifest=read(preview/'.graph-visible.json');nodes=manifest['nodes'];protected={}
for key,node in old['nodes'].items():
    if node['section']=='Гипотезы':
        path='ObsidianFlyVis/'+node['path'];assert (preview/node['path']).read_bytes()==oldfiles[path] and nodes[key]==node;protected[path]=digest(oldfiles[path])
auditor={'__name__':'publication_audit','__file__':str(ROOT/'scripts/audit_obsidian_levels.py')}
exec(compile(head('scripts/audit_obsidian_levels.py'),'published_level_audit.py','exec'),auditor);auditor['ROOT']=RootProxy();graph=auditor['audit'](preview)
assert graph['passed'] and len(nodes)==checks['graph']['note_count']-7 and graph['journal_entries']==checks['graph']['journal_entries']-2
selected={name:str(ROOT/name) for name in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','research_identity_synthesis_20261109/АКТУАЛЬНОЕ.md','research_identity_synthesis_20261109/ПОРЯДОК_НАКОПЛЕНИЯ.md']}
for path in (ROOT/'research_identity_synthesis_20261109').glob('СВОДКА_v*.md'):
    selected[path.relative_to(ROOT).as_posix()]=str(path)
for path in OUT.iterdir():
    if path.is_file() and path.suffix in ('.py','.json','.md') and 'status' not in path.name and not path.name.endswith(('.started.json','.ended.json')) and 'identity' not in path.name:selected[path.relative_to(ROOT).as_posix()]=str(path)
for folder in ['metrics','origin_receipts','rollouts']:
    for path in (OUT/folder).rglob('*.json'):
        if folder=='rollouts' and not path.name.endswith('.receipt.json'):continue
        selected[path.relative_to(ROOT).as_posix()]=str(path)
for node in nodes.values():
    if node['section']=='Гипотезы':continue
    path='ObsidianFlyVis/'+node['path'];data=(preview/node['path']).read_bytes()
    if oldfiles.get(path)!=data:selected[path]=str(preview/node['path'])
retired=[]
for rel in oldactive-set(manifest['files']):
    path='ObsidianFlyVis/'+rel;data=(preview/rel).read_bytes();before=oldfiles[path]
    assert data==before.decode().replace('граф-проекта/видно','граф-проекта/архив',1).encode(),path
    selected[path]=str(preview/rel);retired.append(path)
selected.update({'dissertation/ЕДИНЫЙ_ПЛАН.md':str(TEMP/'plan.md'),'ObsidianFlyVis/.graph-visible.json':str(preview/'.graph-visible.json')})
assert not any('/Гипотезы/' in p or p=='AGENTS.md' or p.startswith('scripts/') or p.endswith('.obsidian/graph.json') for p in selected)
working={p:sha(ROOT/p) for p in selected if (ROOT/p).exists()}
prepared=dict(passed=True,head=git('rev-parse','HEAD').decode().strip(),graph=graph,protected_published_hypotheses=protected,working_sources=working,
    files={name:dict(source=source,sha256=digest(Path(source).read_bytes())) for name,source in sorted(selected.items())},retired_generated_notes=retired,
    scope='Completed domain-matched native development comparison with matched geometry and image controls. Publish compact primary records/code; arrays,fulloutputs,compressedrawlogs and metricdetails remain local. Foreign calendar/plugin and two foreign journal entries excluded. Goal stays open until independent actual-ID success.')
save(TEMP/'prepared.json',(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n').encode());print('PREPARED_NATIVE_PUBLICATION',len(selected),len(nodes),flush=True)
