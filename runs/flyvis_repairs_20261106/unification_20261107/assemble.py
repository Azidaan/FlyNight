"""Build a reviewable union tree with a private Git index; never check out files."""
from pathlib import Path
import os,json,subprocess,hashlib,time
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    manifest=json.loads((OUT/'inventory.json').read_bytes())
    private=OUT/'candidate_index'
    assert not private.exists()
    env=dict(os.environ,GIT_INDEX_FILE=str(private),GIT_OPTIONAL_LOCKS='0')
    def git(*args,data=None):
        p=subprocess.run(['git','-c','gc.auto=0',*args],cwd=ROOT,env=env,input=data,capture_output=True)
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
        return p.stdout
    assert git('rev-parse','HEAD').decode().strip()==manifest['head']
    assert not git('diff','--cached','--name-only') if private.exists() else True
    git('read-tree',manifest['merged_tree'])
    rows=[];entries=[]
    for x in manifest['files']:
        raw=(ROOT/x['snapshot']).read_bytes()
        assert sha(raw)==x['sha256'],x['path']
        # Git applies the existing path-specific normalization; original bytes
        # remain independently preserved and pinned in working_snapshot/.
        oid=git('hash-object','-w','--path='+x['path'],'--stdin',data=raw).decode().strip()
        mode=x['merged']['mode'] if x['merged'] else (x['head']['mode'] if x['head'] else '100644')
        assert mode in ('100644','100755'),x['path']
        entries.append((mode+' '+oid+'\t'+x['path']).encode()+b'\0')
        rows.append(dict(path=x['path'],oid=oid,working_sha256=x['sha256']))
    git('update-index','-z','--index-info',data=b''.join(entries))
    tree=git('write-tree').decode().strip()
    actual={}
    for line in git('ls-tree','-r','-z',tree).split(b'\0'):
        if line:
            meta,name=line.split(b'\t',1);actual[name.decode()]=meta.decode().split()[2]
    assert all(actual[x['path']]==x['oid'] for x in rows)
    result=dict(tree=tree,parents=['c8c8e7056eefedaa722374b8347165f34c25369c','d74a49cb1e09f3bee608e2f107ef872361acbdb8',manifest['head']],
        inventory_sha256=sha((OUT/'inventory.json').read_bytes()),files=rows,
        private_index=str(private),shared_worktree_written=False,shared_index_written=False)
    (OUT/'candidate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(tree=tree,files=len(rows))))
if __name__=='__main__':main()
