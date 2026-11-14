from common import *
import subprocess
def git(*args,**kwargs):return subprocess.check_output(['git',*args],cwd=ROOT,**kwargs)
p=read(OUT/'publication.json')
assert git('rev-parse','HEAD').decode().strip()==p['head']
staged={x.decode('utf8') for x in git('diff','--cached','--name-only','-z').split(b'\0') if x}
assert staged<=set(p['files']),staged-set(p['files'])
checked={}
for name,item in p['files'].items():
    path=Path(item['source']);assert sha(path)==item['sha256'],name
    raw=path.read_bytes()
    if path.suffix.lower() not in ['.jpg','.png','.npz','.zip']:raw=raw.replace(b'\r\n',b'\n')
    actual=git('show',':'+name);assert actual==raw,name
    checked[name]=hashlib.sha256(actual).hexdigest()
git('diff','--cached','--check')
assert 'AGENTS.md' not in staged and 'ObsidianFlyVis/.obsidian/graph.json' not in staged
save('staging_verification.json',dict(passed=True,staged_count=len(staged),requested_files_checked=len(checked),blobs=checked,excluded_foreign_paths=['AGENTS.md','ObsidianFlyVis/.obsidian/graph.json'],note='Every requested index blob checked; unchanged requested files need not appear in staged diff. Live uncommitted plan was not staged wholesale.'))
print('STAGED_BYTES_VERIFIED',len(staged),len(checked))
