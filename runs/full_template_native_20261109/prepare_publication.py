"""Prepare an audited publication overlay without touching shared draft bytes."""
import os,sys,json,hashlib,subprocess,re
from pathlib import Path
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TEMP=ROOT/'tmp/native_template_publication'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
TEMP.mkdir(parents=True,exist_ok=True)
def git(*args,**kwargs):return subprocess.check_output(['git',*args],cwd=ROOT,**kwargs)
def head(path):return git('show','HEAD:'+path)
def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
assert git('rev-parse','HEAD').decode().strip()=='c1bd2e59b3858fcbdccf61eda42e5ae248735a9d'
assert not git('diff','--cached','--name-only').strip()
checks=read(OUT/'final_checks.json');assert checks['passed'] and checks['catalog_tests']==58
for path,digest in checks['stable_sources'].items():assert sha((ROOT/path).read_bytes())==digest,path
assert sha((ROOT/'experiments/obsidian_experiment_topics.json').read_bytes())==checks['graph_topic_source_sha256']
published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8').replace('\r\n','\n')
current=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
marker='## Журнал изменений\n'
first=published.split(marker,1)[1].lstrip().split('\n',1)[0]
prefix=current.split(marker,1)[1].split(first,1)[0].strip()+'\n\n'
headers=[line for line in prefix.splitlines() if line.startswith('**')]
assert len(headers)==11 and all(line.startswith('**09.11.2026,') for line in headers)
plan=published.replace(marker,marker+'\n'+prefix,1)
save(TEMP/'plan.md',plan.encode('utf8'))
old=json.loads(head('ObsidianFlyVis/.graph-visible.json'))
live=read(ROOT/'ObsidianFlyVis/.graph-visible.json')
nodes={k:n for k,n in live['nodes'].items() if n['section'] not in ('Гипотезы','Журнал')}
nodes.update({k:n for k,n in old['nodes'].items() if n['section'] in ('Гипотезы','Журнал')})
nodes['day:2026-11-09']=live['nodes']['day:2026-11-09']
manifest={**old,'nodes':nodes,'files':sorted(n['path'] for n in nodes.values())}
assert len(nodes)==1250
replacements={}
for source in (old['nodes'],live['nodes']):
    for key,n in source.items():
        if key in nodes and nodes[key]['path']!=n['path']:
            a=n['path'][:-3];b=nodes[key]['path'][:-3]
            replacements[a]=b
            replacements[quote(n['path'],safe='')]=quote(nodes[key]['path'],safe='')
            replacements[quote(n['path'][:-3],safe='')]=quote(nodes[key]['path'][:-3],safe='')
preview=TEMP/'vault';preview.mkdir(exist_ok=True)
save(preview/'.graph-visible.json',(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
save(preview/'.obsidian/graph.json',head('ObsidianFlyVis/.obsidian/graph.json'))
overrides={};protected={}
for key,n in nodes.items():
    path='ObsidianFlyVis/'+n['path']
    historical=n['section']=='Гипотезы' or (n['section']=='Журнал' and key!='day:2026-11-09')
    data=head(path) if historical else (ROOT/path).read_bytes()
    if n['section']=='Гипотезы':protected[path]=sha(data)
    else:
        text=data.decode('utf8')
        for a,b in sorted(replacements.items(),key=lambda item:-len(item[0])):text=text.replace(a,b)
        data=text.encode('utf8')
    dest=preview/n['path'];save(dest,data)
    if n['section']!='Гипотезы' and (not (ROOT/path).exists() or data!=(ROOT/path).read_bytes()):overrides[path]=str(dest)
# The only new journal text is today's own verified result prefix.
today=preview/nodes['day:2026-11-09']['path']
assert '12:06' in today.read_text(encoding='utf8') and 'full_template_native_20261109' in today.read_text(encoding='utf8')
class RootProxy:
    def __truediv__(self,p):return TEMP/'plan.md' if str(p)=='dissertation/ЕДИНЫЙ_ПЛАН.md' else ROOT/p
namespace={'__name__':'publication_audit','__file__':str(ROOT/'scripts/audit_obsidian_levels.py')}
exec(compile(head('scripts/audit_obsidian_levels.py'),'published_level_audit.py','exec'),namespace)
namespace['ROOT']=RootProxy()
audit=namespace['audit'](preview)
assert audit['passed'] and audit['note_count']==1250 and audit['journal_entries']==1105,audit
paths=set()
for path in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json',
             'research_full_template_native_pilot_20261109/АКТУАЛЬНОЕ.md','research_identity_synthesis_20261109/СВОДКА_v9.md',
             'research_identity_synthesis_20261109/СВОДКА_v10.md','research_identity_synthesis_20261109/АКТУАЛЬНОЕ.md',
             'research_geometry_consistency_20261109/ОСНОВАНИЕ.md']:paths.add(path)
for p in OUT.iterdir():
    if p.is_file() and p.suffix in ('.py','.json','.md'):paths.add(p.relative_to(ROOT).as_posix())
for folder,pattern in [('metrics','*.json'),('rollouts','*.receipt.json')]:
    paths.update(p.relative_to(ROOT).as_posix() for p in (OUT/folder).rglob(pattern))
for key,n in nodes.items():
    if n['section']=='Гипотезы':continue
    path='ObsidianFlyVis/'+n['path'];chosen=(preview/n['path']).read_bytes()
    try:before=head(path)
    except subprocess.CalledProcessError:before=None
    if before!=chosen:
        paths.add(path)
        if chosen!=(ROOT/path).read_bytes():overrides[path]=str(preview/n['path'])
overrides.update({'dissertation/ЕДИНЫЙ_ПЛАН.md':str(TEMP/'plan.md'),'ObsidianFlyVis/.graph-visible.json':str(preview/'.graph-visible.json')})
paths.update(overrides)
assert not any('/Гипотезы/' in p or p.startswith(('scripts/','tests/')) or p=='AGENTS.md' or p.endswith('.obsidian/graph.json') for p in paths)
receipt=dict(passed=True,head=git('rev-parse','HEAD').decode().strip(),graph=audit,
    protected_published_hypotheses=protected,working_sources={p:sha((ROOT/p).read_bytes()) for p in paths if (ROOT/p).exists()},
    files={p:dict(source=overrides.get(p,str(ROOT/p)),sha256=sha(Path(overrides.get(p,str(ROOT/p))).read_bytes())) for p in sorted(paths)},
    scope='Completed negative native pilot and centered explanatory audits; preserve shared calendar, plugin, two foreign journal drafts and working graph. Published thematic hypotheses retained exactly.',
    tests='Reused exact58 passing catalog tests; publication overlay independently checked against published graph auditor. No inference or metrics repeated.')
save(TEMP/'prepared.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode())
print('PREPARED',len(paths),'files',sum(Path(x['source']).stat().st_size for x in receipt['files'].values()),'bytes',flush=True)
print(json.dumps(audit,ensure_ascii=False),flush=True)
