"""Install the reviewed union locally, preserving all existing source bytes."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];U=Path(__file__).resolve().parent;D=U/'final_union'
BRANCH='codex/system-integrity-20261106-e8f674ce'
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args,data=None,private=False,check=True):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    if private:env['GIT_INDEX_FILE']=str(D/'index')
    p=subprocess.run(['git','-c','gc.auto=0',*args],cwd=ROOT,env=env,input=data,capture_output=True)
    if check and p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    return p.stdout if check else p
def main():
    m=json.loads((D/'manifest.json').read_bytes())
    drift=[x['path'] for x in m['files'] if sha((ROOT/x['path']).read_bytes())!=x['worktree_sha256']]
    print(json.dumps({'drift':drift},ensure_ascii=False),flush=True)
    assert set(drift)<= {'runs/flyvis_repairs_20261106/unification_20261107/final_union_build.log',Path(__file__).relative_to(ROOT).as_posix()},drift
    assert git('rev-parse','HEAD').decode().strip()==m['parents'][-1]
    assert git('branch','--show-current').decode().strip()=='main'
    assert not git('diff','--cached','--name-only')
    assert sha((ROOT/'.git/index').read_bytes())==m['shared_index_sha256']
    extra=drift+[Path(__file__).relative_to(ROOT).as_posix()]
    for name in extra:
        raw=(ROOT/name).read_bytes();oid=git('hash-object','-w','--stdin',data=raw).decode().strip()
        git('update-index','--add','--cacheinfo','100644',oid,name,private=True)
    # Only the union of Git attributes is allowed to change an existing file.
    attr=next(x for x in m['files'] if x['path']=='runs/.gitattributes')
    rows={}
    for line in git('ls-tree','-r','-z',git('write-tree',private=True).decode().strip()).split(b'\0'):
        if line:
            meta,name=line.split(b'\t',1);rows[name.decode()]=meta.decode().split()[2]
    missing=[n for n in rows if not (ROOT/n).exists()]
    git('update-index','--refresh',private=True,check=False)
    differences=git('diff-files','--name-only','-z',private=True).decode().split('\0')
    differences=[n for n in differences if n and n not in missing and n!='runs/.gitattributes']
    (D/'preflight.json').write_text(json.dumps(dict(missing=missing,differences=differences,files=len(rows)),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(missing=len(missing),difference_count=len(differences),first_differences=differences[:12]),ensure_ascii=False),flush=True)
    if len(sys.argv)==1:return
    assert sys.argv[1]=='install'
    assert not differences, differences
    assert git('show-ref','--verify','--quiet','refs/heads/'+BRANCH,check=False).returncode==1
    # Persist recoverable originals before the tiny index/HEAD transition.
    (D/'original_index').write_bytes((ROOT/'.git/index').read_bytes())
    (D/'original_HEAD').write_bytes((ROOT/'.git/HEAD').read_bytes())
    (D/'original_runs_attributes').write_bytes((ROOT/'runs/.gitattributes').read_bytes())
    (ROOT/'runs/.gitattributes').write_bytes((D/'raw'/attr['sha256']).read_bytes())
    for name in missing:
        p=(ROOT/name).resolve();p.relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
        raw=git('cat-file','blob',rows[name])
        with p.open('xb') as f:f.write(raw)
    git('update-index','--refresh',private=True,check=False)
    assert not git('diff-files','--name-only',private=True),git('diff-files','--name-only',private=True)[:2000]
    tree=git('write-tree',private=True).decode().strip()
    message=('Consolidate current FlyVis sources, repairs and research into the primary local branch\n\n'
             'Preserve both remote research histories, seven repair packages and current working sources. '
             'Validate real detector/cache/FlyVis/tracker execution locally; retain explicit test limitations. '
             'GitLab remains frozen by user instruction.\n\nCo-authored-by: Codex <noreply@openai.com>\n')
    args=['commit-tree',tree]
    for parent in m['parents']:args+=['-p',parent]
    commit=git(*args,data=message.encode()).decode().strip()
    git('update-ref','refs/heads/'+BRANCH,commit,'0'*40)
    git('config','branch.'+BRANCH+'.remote','origin')
    git('config','branch.'+BRANCH+'.merge','refs/heads/'+BRANCH)
    lock=ROOT/'.git/index.lock'
    with lock.open('xb') as f:
        assert sha((ROOT/'.git/index').read_bytes())==m['shared_index_sha256']
        f.write((D/'index').read_bytes());f.flush();os.fsync(f.fileno())
        git('symbolic-ref','-m','Use the consolidated primary local branch','HEAD','refs/heads/'+BRANCH)
    os.replace(lock,ROOT/'.git/index')
    git('symbolic-ref','refs/remotes/origin/HEAD','refs/remotes/origin/'+BRANCH)
    assert git('rev-parse','HEAD').decode().strip()==commit
    assert not git('diff','--cached','--name-only')
    assert not git('diff','--name-only')
    for p in m['parents']:git('merge-base','--is-ancestor',p,'HEAD')
    changed=[x['path'] for x in m['files'] if x['path'] not in extra+['runs/.gitattributes'] and sha((ROOT/x['path']).read_bytes())!=x['worktree_sha256']]
    assert not changed,changed
    result=dict(complete=True,branch=BRANCH,commit=commit,tree=tree,parents=m['parents'],existing_source_bytes_preserved=True,
                missing_files_materialized=missing,old_local_main=git('rev-parse','main').decode().strip(),
                upstream=git('rev-parse','--abbrev-ref','@{upstream}').decode().strip(),gitlab_frozen=True,pushed=False)
    (D/'installed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
