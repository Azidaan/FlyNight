"""Stage only approved blobs, canonicalize text newlines and preserve binary ZIP."""
from common import *
import subprocess
def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=ROOT,**kw)
p=read(OUT/'publication.json');assert git('rev-parse','HEAD').decode().strip()==p['head']
staged=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'));assert staged<=set(p['files'])
save('publication_line_ending_recovery.json',dict(recorded=now(),reason='Windows write_text reintroduced CRLF into formatted publication copies. Restore canonical LF at blob creation, same convention as original publisher. Frozen scientific files, parsed values and criteria unchanged; ZIP remains binary.',previous_repair='publication_whitespace_receipt.json'))
for name in ['stage_publication_v2.py','publication_line_ending_recovery.json']:
 f=OUT/name;p['files'][f.relative_to(ROOT).as_posix()]=dict(source=str(f),sha256=sha(f))
(OUT/'publication.json').write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode())
entries=[];text_suffixes={'.py','.md','.svg','.txt','.json','.log','.sha256'}
for name,entry in p['files'].items():
 src=Path(entry['source']).resolve();src.relative_to(ROOT.resolve());assert sha(src)==entry['sha256'],name
 blob=src.read_bytes()
 if src.suffix.lower() in text_suffixes or src.name=='.gitignore':blob=blob.replace(b'\r\n',b'\n')
 oid=git('hash-object','-w','--stdin',input=blob).decode().strip();entries.append(('100644 '+oid+'\t'+name+'\0').encode())
git('update-index','-z','--index-info',input=b''.join(entries))
r=subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,capture_output=True)
(OUT/'publication_diff_check_v2.log').write_bytes(r.stdout+r.stderr)
assert r.returncode==0,('Whitespace check failed; inspect saved log',r.returncode)
print('STAGED_AND_CHECKED',len(entries))
