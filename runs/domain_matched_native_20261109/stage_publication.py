"""Stage an explicit prepared list and audit the entire physical graph index."""
import json,subprocess,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'tmp/domain_native_publication'
def git(*args,**kw):
    r=subprocess.run(['git',*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
    if r.returncode:raise RuntimeError(r.stderr.decode('utf8','replace'))
    return r.stdout
def sha(data):return hashlib.sha256(data).hexdigest()
p=json.loads((HERE/'prepared.json').read_text(encoding='utf8'));assert p['passed']
assert git('rev-parse','HEAD').decode().strip()==p['head'] and git('branch','--show-current').decode().strip()=='codex/system-integrity-20261106-e8f674ce'
assert not git('diff','--cached','--name-only').strip()
for name,digest in p['working_sources'].items():assert sha((ROOT/name).read_bytes())==digest,name
foreign={}
for name in ['AGENTS.md','ObsidianFlyVis/ИНСТРУКЦИЯ — как вести записи.md','scripts/obsidian_level_graph.py','scripts/audit_obsidian_levels.py','tests/test_obsidian_level_graph.py','experiments/OBSIDIAN_GRAPH_RULE.md','dissertation/План целей по Трекеру.md','experiments/hypothesis_calendar.json','ObsidianFlyVis/.obsidian/graph.json']:
    path=ROOT/name
    if path.exists():foreign[name]=sha(path.read_bytes())
for path in (ROOT/'ObsidianFlyVis').rglob('*.md'):
    if any('Гипотез' in part for part in path.parts):foreign[path.relative_to(ROOT).as_posix()]=sha(path.read_bytes())
for path in (ROOT/'ObsidianFlyVis/.obsidian/plugins').rglob('*'):
    if path.is_file():foreign[path.relative_to(ROOT).as_posix()]=sha(path.read_bytes())
blobs={}
for i,(name,item) in enumerate(p['files'].items()):
    data=Path(item['source']).read_bytes();assert sha(data)==item['sha256'],name
    blobs[name]=git('hash-object','-w','--path='+name,'--stdin',input=data).decode().strip()
    if i%100==0:print('HASHED',i+1,'/',len(p['files']),flush=True)
assert git('rev-parse','HEAD').decode().strip()==p['head'] and not git('diff','--cached','--name-only').strip()
git('update-index','--add','-z','--index-info',input=b''.join(('100644 '+blob+'\t'+name+'\0').encode() for name,blob in blobs.items()))
staged=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'));assert staged<=set(p['files']) and len(staged)>100
assert not git('diff','--cached','--diff-filter=D','--name-only').strip()
entries={}
for entry in git('ls-files','--stage','-z').decode().strip('\0').split('\0'):
    meta,path=entry.split('\t',1);mode,blob,stage=meta.split()
    if stage=='0':entries[path]=blob
for name,blob in blobs.items():assert entries[name]==blob,name
manifest=json.loads(git('show',':ObsidianFlyVis/.graph-visible.json'));nodes=manifest['nodes'];by_path={n['path']:key for key,n in nodes.items()}
notes=[path for path in entries if path.startswith('ObsidianFlyVis/Карта проекта/') and path.endswith('.md')]
batch=git('cat-file','--batch',input=('\n'.join(entries[path] for path in notes)+'\n').encode());offset=0;active=set();edges=set()
for path in notes:
    end=batch.index(b'\n',offset);_,kind,size=batch[offset:end].decode().split();assert kind=='blob';offset=end+1
    text=batch[offset:offset+int(size)].decode();offset+=int(size)+1;front=text.split('---\n',2)
    if len(front)!=3 or 'граф-проекта/видно' not in front[1]:continue
    rel=path.removeprefix('ObsidianFlyVis/');assert rel in by_path,('unexpected physical active note',rel);active.add(rel);key=by_path[rel]
    for target in re.findall(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]',front[2]):assert target+'.md' in by_path,(rel,target);edges.add((key,by_path[target+'.md']))
assert active==set(manifest['files']) and edges=={(n['parent'],k) for k,n in nodes.items() if n['parent']}
assert offset==len(batch) and len(nodes)==p['graph']['note_count']
for path,digest in p['protected_published_hypotheses'].items():assert sha(git('show',':'+path))==digest,path
for path,digest in foreign.items():assert sha((ROOT/path).read_bytes())==digest,path
git('diff','--cached','--check')
(HERE/'staging_receipt.json').write_text(json.dumps(dict(passed=True,staged_files=len(staged),publication_graph_nodes=len(nodes),physical_active_notes=len(active),physical_tracked_notes_audited=len(notes),publication_edges=len(edges),
    graph=p['graph'],prepared_sha256=sha((HERE/'prepared.json').read_bytes()),preserved_working_files=foreign,archived_without_content_change=p['retired_generated_notes'],blobs=blobs,scope=p['scope']),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('STAGED_NATIVE_PUBLICATION',len(staged),len(active),len(foreign),flush=True)
