import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path,digest in read(OUT/'protocol.json')['sources'].items():assert sha(ROOT/path)==digest,path
for path,digest in read(OUT/'execution_receipt.json')['checksums'].items():assert sha(OUT/path)==digest,path
g=read(OUT/'graph_audit.json');assert g['passed'] and g['hypotheses_unchanged']
assert 'OK: каталог согласован с источниками' in (OUT/'catalog_validation.log').read_text(encoding='utf8')
assert read(OUT/'independent_verification.json')['passed']
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,exact_frame_audit_only=True,evidence={n:sha(OUT/n) for n in ['summary.json','graph_audit.json','catalog_validation.log','independent_verification.json']}),f,indent=2)
print('Final integrity passed')
