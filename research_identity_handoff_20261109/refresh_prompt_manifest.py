"""Refresh only two reviewed handoff texts after replacing the draft DOCX link."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'tmp/identity_handoff_publication'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
path=HERE/'prepared.json';before=path.read_bytes();p=json.loads(before)
allowed={'research_identity_handoff_20261109/ЦЕЛЬ_НОВОГО_ЧАТА.txt','research_identity_handoff_20261109/ПРОМПТ_НОВОГО_ЧАТА.txt'}
for n,h in p['working_sources'].items():
    if n not in allowed:assert sha(ROOT/n)==h,n
for n in allowed:
    source=ROOT/n;assert p['files'][n]['source']==str(source);p['files'][n]['sha256']=sha(source);p['working_sources'][n]=sha(source)
name=Path(__file__).relative_to(ROOT).as_posix();p['files'][name]=dict(source=str(Path(__file__)),sha256=sha(Path(__file__)));p['working_sources'][name]=sha(Path(__file__))
with (HERE/'prepared_before_pdf_link_fix.json').open('xb') as f:f.write(before)
temp=path.with_suffix('.pending');temp.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8');assert path.read_bytes()==before;temp.replace(path)
print('HANDOFF_TEXT_MANIFEST_REFRESHED',flush=True)
