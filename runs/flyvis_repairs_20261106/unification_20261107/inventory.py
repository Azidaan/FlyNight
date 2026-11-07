"""Preserve a bounded source/document snapshot before branch consolidation."""
from pathlib import Path
import hashlib,json,subprocess,collections
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
MERGED='ef828655834f41b40b1f91f7302e8832fe5bf81b'
def git(*args):
    return subprocess.check_output(['git','-c','core.quotePath=false','-c','gc.auto=0',*args],cwd=ROOT,stderr=subprocess.DEVNULL)
def names(raw):return set(filter(None,raw.decode('utf-8').split('\0')))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def tree(ref):
    result={}
    for line in git('ls-tree','-r','-z',ref).split(b'\0'):
        if not line:continue
        meta,name=line.split(b'\t',1);mode,kind,oid=meta.decode().split()
        result[name.decode()]=dict(mode=mode,kind=kind,oid=oid)
    return result
def main():
    destination=OUT/'working_snapshot'
    destination.mkdir(exist_ok=False)
    roots=['tracking','scripts','tests','experiments','evaluation','dissertation','ObsidianFlyVis']
    new=names(git('ls-files','--others','--exclude-standard','-z',*roots))
    changed=names(git('diff','--name-only','-z','HEAD'))
    baseline=tree(MERGED);head=tree('HEAD')
    rows=[]
    for name in sorted(new|changed):
        path=(ROOT/name).resolve();path.relative_to(ROOT)
        assert path.is_file(), 'Deletion requires individual resolution: '+name
        raw=path.read_bytes();digest=sha(raw)
        assert len(raw)<16*1024**2, 'Large input requires separate review: '+name
        target=destination/digest
        if not target.exists():target.write_bytes(raw)
        previous=baseline.get(name)
        rows.append(dict(path=name,sha256=digest,bytes=len(raw),snapshot=target.relative_to(ROOT).as_posix(),
            head=head.get(name),merged=previous,
            overlaps_remote=bool(previous and head.get(name)!=previous)))
    before_index=Path(git('rev-parse','--git-path','index').decode().strip())
    if not before_index.is_absolute():before_index=ROOT/before_index
    report=dict(head=git('rev-parse','HEAD').decode().strip(),branch=git('branch','--show-current').decode().strip(),
        merged_tree=MERGED,files=rows,shared_index_sha256=sha(before_index.read_bytes()),
        staged_paths=sorted(names(git('diff','--cached','--name-only','-z'))),
        roots_for_new_files=roots,excluded_new_run_outputs=True)
    (OUT/'inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),
        new=len(new),changed=len(changed),overlaps=[x['path'] for x in rows if x['overlaps_remote']]),ensure_ascii=True))
if __name__=='__main__':main()
