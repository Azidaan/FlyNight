"""Keep frozen files exact; publish syntax-equivalent clean copies plus originals."""
from common import *
import subprocess,zipfile,ast,re,xml.etree.ElementTree as ET
def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=ROOT,**kw)
p=read(OUT/'publication.json');assert git('rev-parse','HEAD').decode().strip()==p['head']
staged=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'));assert staged<=set(p['files'])
save('publication_before_whitespace.json',p)
dest=OUT/'publication_text';dest.mkdir(exist_ok=False);repairs=[];archive=OUT/'frozen_source_bytes.zip'
def svg_tokens(blob):
 tree=ET.fromstring(blob)
 return [(e.tag,sorted((k,tuple(v.split())) for k,v in e.attrib.items()),(e.text or '').strip(),(e.tail or '').strip()) for e in tree.iter()]
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for name,entry in list(p['files'].items()):
  src=Path(entry['source']);assert sha(src)==entry['sha256'];raw=src.read_bytes()
  if src.suffix.lower() not in ['.py','.md','.svg','.txt','.json','.log','.sha256']:continue
  s=raw.decode('utf8').replace('\r\n','\n');clean='\n'.join(line.rstrip(' \t') for line in s.split('\n')).rstrip('\n');clean=clean+'\n' if clean else ''
  if clean==s:continue
  if src.suffix=='.py':assert ast.dump(ast.parse(s),include_attributes=False)==ast.dump(ast.parse(clean),include_attributes=False)
  if src.suffix=='.json':assert json.loads(s)==json.loads(clean)
  if src.suffix=='.svg':assert svg_tokens(raw)==svg_tokens(clean.encode())
  z.writestr(name,raw);target=dest/Path(name).relative_to(OUT.relative_to(ROOT)) if name.startswith(OUT.relative_to(ROOT).as_posix()+'/') else dest/name
  target.parent.mkdir(parents=True,exist_ok=True);target.write_text(clean,encoding='utf8')
  repairs.append(dict(file=name,frozen_source=str(src),frozen_sha256=entry['sha256'],published_sha256=sha(target),reason='Trailing whitespace/blank EOF only; frozen working file untouched. Python AST/JSON values/SVG element-and-attribute tokens checked where applicable.'))
  p['files'][name]=dict(source=str(target),sha256=sha(target))
assert repairs
with zipfile.ZipFile(archive) as z:
 for row in repairs:assert hashlib.sha256(z.read(row['file'])).hexdigest()==row['frozen_sha256']
save('publication_whitespace_receipt.json',dict(passed=True,recorded=now(),repairs=repairs,archive_sha256=sha(archive),original_frozen_files_unchanged=True,no_recomputed_scientific_results=True))
for name in ['repair_publication_whitespace.py','frozen_source_bytes.zip','publication_whitespace_receipt.json']:
 f=OUT/name;p['files'][f.relative_to(ROOT).as_posix()]=dict(source=str(f),sha256=sha(f))
(OUT/'publication.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
entries=[]
for row in repairs:
 name=row['file'];src=Path(p['files'][name]['source']);oid=git('hash-object','-w','--stdin',input=src.read_bytes()).decode().strip();entries.append(('100644 '+oid+'\t'+name+'\0').encode())
for name in ['repair_publication_whitespace.py','frozen_source_bytes.zip','publication_whitespace_receipt.json']:
 f=OUT/name;oid=git('hash-object','-w','--stdin',input=f.read_bytes()).decode().strip();entries.append(('100644 '+oid+'\t'+f.relative_to(ROOT).as_posix()+'\0').encode())
git('update-index','-z','--index-info',input=b''.join(entries));git('diff','--cached','--check')
print('PUBLICATION_WHITESPACE_FIXED',len(repairs),'FROZEN_ORIGINALS_ARCHIVED',sha(archive))
