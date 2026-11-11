from common import *
import subprocess
def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=ROOT,**kw)
p=read(OUT/'prepared.json');assert p['passed'];assert git('rev-parse','HEAD').decode().strip()==p['base_head'];assert not git('diff','--cached','--name-only').strip()
rows=[]
for name,item in p['files'].items():
    path=Path(item['source']);assert sha(path)==item['sha256'];raw=path.read_bytes();raw.decode('utf8')
    oid=git('hash-object','-w','--stdin',input=raw.replace(b'\r\n',b'\n')).decode().strip();rows.append(('100644 '+oid+'\t'+name+'\0').encode())
git('update-index','-z','--index-info',input=b''.join(rows))
assert set(filter(None,git('diff','--cached','--name-only','-z').decode().split('\0')))==set(p['files'])
git('diff','--cached','--check');print('STAGED',len(rows))
