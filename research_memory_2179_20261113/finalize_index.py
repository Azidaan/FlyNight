from common import *
import subprocess
def git(*args,**kwargs):return subprocess.check_output(['git',*args],cwd=ROOT,**kwargs)
p=read(OUT/'publication.json'); additions=['recover_publication_check.py','verify_publication_v2.py','publication_check_recovery.json','publication_verification_v2.json','finalize_index.py']
entries=[]
for name in additions:
    source=OUT/name;rel=source.relative_to(ROOT).as_posix();raw=source.read_bytes().replace(b'\r\n',b'\n');oid=git('hash-object','-w','--stdin',input=raw).decode().strip();entries.append(('100644 '+oid+'\t'+rel+'\0').encode());p['files'][rel]=dict(source=str(source),sha256=sha(source))
git('update-index','-z','--index-info',input=b''.join(entries))
staged={x for x in git('diff','--cached','--name-only','-z').decode().split('\0') if x};assert staged<=set(p['files'])
for name,item in p['files'].items():
    source=Path(item['source']);assert sha(source)==item['sha256'],name
    expected=source.read_bytes()
    if source.suffix.lower() not in ['.jpg','.png','.npz','.zip']:expected=expected.replace(b'\r\n',b'\n')
    assert git('show',':'+name)==expected,name
git('diff','--cached','--check')
save('publication_final.json',dict(passed=True,files=p['files'],staged_file_count=len(staged),all_index_bytes_verified=True))
print('FINAL_INDEX_VERIFIED',len(staged))
