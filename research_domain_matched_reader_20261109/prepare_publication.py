"""Generate the full publication graph from published code and own staged plan.

Audit all physical notes, including archived paths, rather than only manifest
members. Never change the live vault or shared calendar/plugin drafts.
"""
import os,sys,json,hashlib,subprocess,shutil,re,sysconfig
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=OUT]
import queue
assert Path(queue.__file__).resolve()==Path(sysconfig.get_path('stdlib'),'queue.py').resolve()
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
TEMP=ROOT/'tmp/domain_reader_publication';TEMP.mkdir(parents=True,exist_ok=True)
def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=ROOT,**kw)
def head(path):return git('show','HEAD:'+path)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def save(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
assert git('rev-parse','HEAD').decode().strip()=='1c67d9d9b51f545871f1f6679d23d25609819277'
assert not git('diff','--cached','--name-only').strip()
checks=read(OUT/'final_checks.json');assert checks['passed'] and checks['catalog_tests']==58
for name,digest in checks['stable_sources'].items():assert sha((ROOT/name).read_bytes())==digest,name
published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8').replace('\r\n','\n');current=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
marker='## Журнал изменений\n';first=published.split(marker,1)[1].lstrip().split('\n',1)[0]
prefix=current.split(marker,1)[1].split(first,1)[0].strip()+'\n\n';headers=[s for s in prefix.splitlines() if s.startswith('**')]
assert len(headers)==7 and all(s.startswith('**09.11.2026,') for s in headers),headers
plan=published.replace(marker,marker+'\n'+prefix,1);save(TEMP/'plan.md',plan.encode())
vault=ROOT/'ObsidianFlyVis';preview=TEMP/'vault';assert not preview.exists();shutil.copytree(vault,preview)
paths=git('ls-tree','-r','--name-only','-z','HEAD','--','ObsidianFlyVis').decode().strip('\0').split('\0')
paths=[p for p in paths if p.endswith('.md')]
batch=git('cat-file','--batch',input=('\n'.join('HEAD:'+p for p in paths)+'\n').encode());offset=0;oldfiles={}
for path in paths:
    end=batch.index(b'\n',offset);_,kind,size=batch[offset:end].decode().split();assert kind=='blob';offset=end+1
    data=batch[offset:offset+int(size)];offset+=int(size)+1;oldfiles[path]=data
assert offset==len(batch)
old=read(vault/'.graph-visible.json');old=json.loads(head('ObsidianFlyVis/.graph-visible.json'))
oldactive=set()
for path,data in oldfiles.items():
    rel=path.removeprefix('ObsidianFlyVis/')
    if rel.startswith('Карта проекта/'):
        save(preview/rel,data)
        front=data.decode().split('---\n',2)
        if len(front)==3 and 'граф-проекта/видно' in front[1]:oldactive.add(rel)
    if any('Гипотез' in p for p in Path(rel).parts):save(preview/rel,data)
extras=oldactive-set(old['files'])
assert not any('/Гипотезы/' in p for p in extras),extras
# Make the published generator's own archive cleanup see EVERY physical active
# note, including stale paths omitted by earlier publication manifests.
augmented=json.loads(json.dumps(old,ensure_ascii=False));known_paths=set(old['files'])
for file in (preview/'Карта проекта').rglob('*.md'):
    text=file.read_text(encoding='utf8');front=text.split('---\n',2)
    rel=file.relative_to(preview).as_posix()
    if len(front)!=3 or 'граф-проекта/видно' not in front[1] or rel in known_paths:continue
    number=re.search(r'^пункт: "([\d.]+)"$',front[1],re.M)[1]
    augmented['nodes']['publication-retired:'+rel]=dict(path=rel,number=number,parent='publication-retired',section=Path(rel).parts[1],title=file.stem,color=None)
save(preview/'.graph-visible.json',(json.dumps(augmented,ensure_ascii=False,indent=2)+'\n').encode())
class RootProxy:
    def __truediv__(self,name):return TEMP/'plan.md' if str(name)=='dissertation/ЕДИНЫЙ_ПЛАН.md' else ROOT/name
    def __fspath__(self):return str(ROOT)
    def __getattr__(self,name):return getattr(ROOT,name)
ns={'__name__':'publication_generator','__file__':str(ROOT/'scripts/obsidian_level_graph.py')}
exec(compile(head('scripts/obsidian_level_graph.py'),'published_level_generator.py','exec'),ns)
ns['ROOT']=RootProxy()
ns['local_file']=lambda name,label:'['+label+']('+(ROOT/name).as_uri()+')'
generated=ns['build'](preview)
manifest=read(preview/'.graph-visible.json');nodes=manifest['nodes']
assert len(nodes)==checks['graph']['note_count']-7 and all(not k.startswith('publication-retired:') for k in nodes)
protected={}
for key,node in old['nodes'].items():
    if node['section']=='Гипотезы':
        path='ObsidianFlyVis/'+node['path'];assert (preview/node['path']).read_bytes()==oldfiles[path],path
        assert nodes[key]==node;protected[path]=sha(oldfiles[path])
auditor={'__name__':'publication_audit','__file__':str(ROOT/'scripts/audit_obsidian_levels.py')}
exec(compile(head('scripts/audit_obsidian_levels.py'),'published_level_audit.py','exec'),auditor);auditor['ROOT']=RootProxy()
audit=auditor['audit'](preview);assert audit['passed'] and audit['journal_entries']==checks['graph']['journal_entries']-2
selected={}
for path in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json',
             'tracking/geometric_template_correspondence.py','tests/test_geometric_template_correspondence.py',
             'research_identity_synthesis_20261109/АКТУАЛЬНОЕ.md']+[f'research_identity_synthesis_20261109/СВОДКА_v{i}.md' for i in range(11,15)]:selected[path]=str(ROOT/path)
folders=['research_geometry_consistency_20261109','research_working_domain_identity_20261109','research_coordinate_reader_20261109','research_domain_matched_reader_20261109']
for folder in folders:
    for path in (ROOT/folder).rglob('*'):
        if path.is_file() and path.suffix in ('.py','.json','.md') and '__pycache__' not in path.parts:selected[path.relative_to(ROOT).as_posix()]=str(path)
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
working={p:sha((ROOT/p).read_bytes()) for p in selected if (ROOT/p).exists()}
prepared=dict(passed=True,head=git('rev-parse','HEAD').decode().strip(),graph=audit,protected_published_hypotheses=protected,
    working_sources=working,files={path:dict(source=source,sha256=sha(Path(source).read_bytes())) for path,source in sorted(selected.items())},
    retired_generated_notes=retired,previous_manifest_omitted_active_notes=sorted(extras),
    scope='Coordinate candidate negative; recovered true-domain data; fixed domain head admitted only to an unrun native pilot. Publish code, compact records and metadata, leaving array caches local. Shared calendar/plugin and foreign journal entries excluded.',
    verification_correction='Previous publication checked a manifest-only materialization. This full physical-vault audit also checks retired paths and restores generated archive tags. Local vault had already passed full physical audit; no scientific result changes.')
save(TEMP/'prepared.json',(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n').encode())
print('PREPARED_COMPLETE_VAULT',len(selected),'files',len(nodes),'nodes','old-active-extras',len(extras),'retired',len(retired),flush=True)
