"""Resume staging after Git's global PDF text-diff driver flagged PDF syntax."""
import json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'tmp/identity_handoff_publication'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
path=HERE/'prepared.json';before=path.read_bytes();p=json.loads(before)
for n,h in p['working_sources'].items():assert sha(ROOT/n)==h,n
names=['dissertation/reports/.gitattributes',Path(__file__).relative_to(ROOT).as_posix()]
for n in names:p['files'][n]=dict(source=str(ROOT/n),sha256=sha(ROOT/n));p['working_sources'][n]=sha(ROOT/n)
with (HERE/'prepared_before_binary_pdf_attributes.json').open('xb') as f:f.write(before)
temp=path.with_suffix('.binary.pending');temp.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8');assert path.read_bytes()==before;temp.replace(path)
with (HERE/'pdf_binary_receipt.json').open('x',encoding='utf8') as f:json.dump(dict(reason='Global Git diff driver astextplain treated PDF syntax/embedded font streams as text and flagged legitimate PDF spaces. A new scoped dissertation/reports/.gitattributes marks PDF binary; no report bytes, scientific sources or numerical criteria changed.',report_sha256=sha(ROOT/'dissertation/reports/Сохранение личности объектов — результаты проверок и продолжение.pdf')),f,ensure_ascii=False,indent=2)
source=ROOT/'runs/domain_matched_native_20261109/stage_publication.py';code=source.read_text(encoding='utf8');old="ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'tmp/domain_native_publication'";assert code.count(old)==1;code=code.replace(old,"ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'tmp/identity_handoff_publication'")
old="not git('diff','--cached','--name-only').strip()";assert code.count(old)==2;code=code.replace(old,"set(git('diff','--cached','--name-only','-z').decode().strip('\\0').split('\\0')) <= set(p['files'])")
code=code.replace('len(staged)>100','len(staged)>0');exec(compile(code,str(source),'exec'))
