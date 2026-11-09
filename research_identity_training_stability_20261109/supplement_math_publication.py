"""Explicitly add the just-verified mathematical preparation to the whitelist."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'tmp/identity_reader_review_publication'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
path=HERE/'prepared.json';before=path.read_bytes();p=json.loads(before)
math_path=ROOT/'research_partitioned_correspondence_20261109/math_checks.json';check=json.loads(math_path.read_text(encoding='utf8'));assert check['passed']
for name,h in check['sources'].items():assert sha(ROOT/name)==h,name
names=['tracking/partitioned_channel_correspondence.py','research_partitioned_correspondence_20261109/math_check.py','research_partitioned_correspondence_20261109/math_checks.json','research_partitioned_correspondence_20261109/ОСНОВАНИЕ.md',Path(__file__).relative_to(ROOT).as_posix()]
for name in names:
    assert name not in p['files'];file=ROOT/name;digest=sha(file);p['files'][name]=dict(source=str(file),sha256=digest);p['working_sources'][name]=digest
with (HERE/'prepared_before_math_supplement.json').open('xb') as f:f.write(before)
temporary=HERE/'prepared.math.pending';temporary.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8');assert path.read_bytes()==before;temporary.replace(path)
with (HERE/'math_supplement_receipt.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,added=names,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=sha(path),science_scope='Only mathematical preparation, no real-pair training or deployment claimed.'),f,ensure_ascii=False,indent=2)
print('EXPLICIT_MATH_PUBLICATION_SUPPLEMENT',len(names),len(p['files']),flush=True)
