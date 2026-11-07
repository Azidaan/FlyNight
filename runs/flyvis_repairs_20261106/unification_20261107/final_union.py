"""Snapshot current sources and evidence into a private union index."""
from pathlib import Path
import hashlib,json,os,subprocess

ROOT=Path(__file__).resolve().parents[3]
U=Path(__file__).resolve().parent
OUT=U/'final_union'
BASE='ef828655834f41b40b1f91f7302e8832fe5bf81b'
PARENTS=['c8c8e7056eefedaa722374b8347165f34c25369c','d74a49cb1e09f3bee608e2f107ef872361acbdb8','72b6ed5472bf7d6b121879e694309da52ae709d0']
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args,data=None,private=False):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    if private:env['GIT_INDEX_FILE']=str(OUT/'index')
    p=subprocess.run(['git','-c','gc.auto=0',*args],cwd=ROOT,env=env,input=data,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    return p.stdout
def names(raw):return set(filter(None,raw.decode().split('\0')))

def main():
    OUT.mkdir(exist_ok=False);(OUT/'raw').mkdir()
    assert git('rev-parse','HEAD').decode().strip()==PARENTS[-1]
    assert not git('diff','--cached','--name-only')
    git('read-tree',BASE,private=True)
    roots=['tracking','scripts','tests','experiments','evaluation','dissertation','ObsidianFlyVis']
    selected=names(git('ls-files','--others','--exclude-standard','-z',*roots))|names(git('diff','--name-only','-z','HEAD'))
    # Publish small primary reports, not the 1.4 GB per-case arrays or caches.
    other=ROOT/'runs/flyvis_state_subgroups_ram_stream4_20261107'
    excluded=[]
    for p in other.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if ('__pycache__' not in p.parts and p.suffix in ('.json','.md','.log','.py','.png','.zip')) or p.parent==other and p.name in ('indices.npz','textures.npz','node_types.npy'):
            selected.add(rel)
        else:excluded.append(dict(path=rel,bytes=p.stat().st_size))
    # Retain our final control scripts and small verification evidence.
    for p in U.iterdir():
        if p.is_file() and p.suffix in ('.py','.log','.json','.txt'):
            selected.add(p.relative_to(ROOT).as_posix())
    for name in ['integration','integration_light','native','native4','remaining','per_file','memory_policy_previous','memory_guard_validation','memory_guard_validation_95cddc11','uncapped_checks']:
        for p in (U/name).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.log','.py') and not any('temp' in a or a=='__pycache__' for a in p.relative_to(U/name).parts):
                selected.add(p.relative_to(ROOT).as_posix())
    # Track exact raw report bytes and retain all pre-existing Git attributes.
    attrpath='runs/.gitattributes'
    remote=git('show',BASE+':'+attrpath).decode()
    local=(ROOT/attrpath).read_text(encoding='utf-8')
    lines=list(dict.fromkeys(remote.splitlines()+local.splitlines()+[
        'flyvis_state_subgroups_ram_stream4_20261107/** -text',
        'flyvis_repairs_20261106/unification_20261107/** -text']))
    attrs=('\n'.join(lines)+'\n').encode()
    selected.add(attrpath)
    rows=[];entries=[]
    modes={}
    for item in git('ls-tree','-r','-z',BASE).split(b'\0'):
        if item:
            meta,n=item.split(b'\t',1);modes[n.decode()]=meta.split()[0].decode()
    for name in sorted(selected):
        p=ROOT/name;assert p.is_file(),name
        raw=attrs if name==attrpath else p.read_bytes()
        assert len(raw)<16*1024**2,(name,len(raw))
        digest=sha(raw);(OUT/'raw'/digest).write_bytes(raw)
        exact=name.startswith('runs/flyvis_state_subgroups_ram_stream4_20261107/') or name.startswith(U.relative_to(ROOT).as_posix()+'/')
        args=['hash-object','-w','--stdin'] if exact else ['hash-object','-w','--path='+name,'--stdin']
        oid=git(*args,data=raw).decode().strip()
        entries.append((modes.get(name,'100644')+' '+oid+'\t'+name).encode()+b'\0')
        rows.append(dict(path=name,sha256=digest,oid=oid,bytes=len(raw),worktree_sha256=sha(p.read_bytes())))
    git('update-index','-z','--index-info',data=b''.join(entries),private=True)
    tree=git('write-tree',private=True).decode().strip()
    assert not git('diff','--diff-filter=D','--name-only',PARENTS[-1],tree)
    index=ROOT/'.git/index'
    report=dict(tree=tree,parents=PARENTS,files=rows,excluded_large_or_derived_outputs=excluded,
                shared_index_sha256=sha(index.read_bytes()),branch='main',source_roots=roots)
    (OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(tree=tree,files=len(rows),bytes=sum(r['bytes'] for r in rows),excluded_count=len(excluded))))
if __name__=='__main__':main()
