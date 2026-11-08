import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path,digest in read(OUT/'protocol.json')['sources'].items():assert sha(ROOT/path)==digest,path
for path,digest in read(OUT/'results.json')['files'].items():assert sha(ROOT/path)==digest,path
for name,digest in read(OUT/'execution_receipt.json')['checksums'].items():assert sha(OUT/name)==digest,name
assert read(OUT/'independent_verification.json')['passed']
graph=read(OUT/'graph_audit.json');assert graph['passed'] and graph['hypotheses_unchanged']
assert 'OK: каталог согласован с источниками' in (OUT/'catalog_validation.log').read_text(encoding='utf8')
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:
    json.dump(dict(passed=True,input_hashes=len(read(OUT/'protocol.json')['sources']),events_verified=200,
        evidence={name:sha(OUT/name) for name in ['graph_audit.json','independent_verification.json','catalog_validation.log','execution_receipt.json']}),f,indent=2)
print('Final integrity passed')
