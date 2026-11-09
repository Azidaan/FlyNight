"""Audit the actual report commit and the readable final PDF before publication."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'research_signal_scaling_20261109/audit_commit_scope.py';code=source.read_text(encoding='utf8')
code=code.replace("HERE=ROOT/'tmp/signal_scaling_publication'","HERE=ROOT/'tmp/identity_handoff_publication'").replace("'dd91b93750c7021164f384bcc54aa1af2813b82d'","'6e46b01c281b99cb0a7c05352952951dcfb0307d'")
code=code.replace("with (HERE/'actual_commit_scope_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)","""from pypdf import PdfReader
report=ROOT/'dissertation/reports/Сохранение личности объектов — результаты проверок и продолжение.pdf'
pdf=PdfReader(report);assert len(pdf.pages)==11
plain='\\n'.join(p.extract_text() for p in pdf.pages)
assert not account.search(plain.encode()) and not credential.search(plain.encode())
out['exclusions']='Foreign graph/calendar/plugin/hypothesis drafts, unverified DOCX/draft PDFs, maps/weights, browser/UI captures and unrelated data excluded. Final11-pagePDF inspected and readable text independently privacy-scanned.'
out['pdf_text_scanned']=True
with (HERE/'actual_commit_scope_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)""")
exec(compile(code,str(source),'exec'))
