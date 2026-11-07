"""Keep authored notes, exclude test scratch data, and seal current evidence."""
from pathlib import Path
import json,hashlib,subprocess
from final_union import ROOT,U,OUT,git,sha,names
m=json.loads((OUT/'manifest.json').read_bytes())
if (OUT/'manifest_initial.json').exists():
    assert (OUT/'manifest_initial.json').read_bytes()==(OUT/'manifest.json').read_bytes()
else:
    (OUT/'manifest_initial.json').write_bytes((OUT/'manifest.json').read_bytes())
prefix=U.relative_to(ROOT).as_posix()+'/per_file/'
removed=[x['path'] for x in m['files'] if x['path'].startswith(prefix) and '/' in x['path'][len(prefix):]]
if removed:git('update-index','--force-remove','-z','--stdin',data=b''.join(p.encode()+b'\0' for p in removed),private=True)
m['files']=[x for x in m['files'] if x['path'] not in removed]
roots=['notes','reports','skills','figs','СВОДНЫЙ_ПЛАН1.md']+[p.name for p in ROOT.glob('research*') if p.is_dir()]
extra=names(git('ls-files','--others','--exclude-standard','-z',*roots))
extra|={p.relative_to(ROOT).as_posix() for p in U.glob('*.py')}
extra.add((U/'final_union_build.log').relative_to(ROOT).as_posix())
rows={x['path']:x for x in m['files']}
for name in sorted(extra):
    p=ROOT/name
    if '__pycache__' in p.parts or p.stat().st_size>16*1024**2:continue
    raw=p.read_bytes();h=sha(raw);(OUT/'raw'/h).write_bytes(raw)
    exact=name.startswith(U.relative_to(ROOT).as_posix()+'/')
    args=['hash-object','-w','--stdin'] if exact else ['hash-object','-w','--path='+name,'--stdin']
    oid=git(*args,data=raw).decode().strip()
    git('update-index','--add','--cacheinfo','100644',oid,name,private=True)
    rows[name]=dict(path=name,sha256=h,oid=oid,bytes=len(raw),worktree_sha256=h)
m.update(tree=git('write-tree',private=True).decode().strip(),files=list(rows.values()),excluded_test_scratch=removed)
(OUT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(tree=m['tree'],files=len(rows),removed_scratch=len(removed),extra_paths=len(extra))))
