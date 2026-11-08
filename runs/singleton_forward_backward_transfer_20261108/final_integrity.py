"""Verify saved inputs and reports after all computation has ended."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checked=0
for record in ['protocol.json','maps_complete.json']:
    for path,digest in read(OUT/record)['sources'].items():
        assert sha(ROOT/path)==digest,path
        checked+=1
for file,digest in read(OUT/'execution_receipt.json')['checksums'].items():assert sha(OUT/file)==digest,file
for video,path in read(OUT/'maps_complete.json')['retinal_files'].items():
    assert sha(ROOT/path)==read(OUT/(video+'.retinal.json'))['sha256']
assert read(OUT/'graph_audit.json')['passed']
assert read(OUT/'graph_audit.json')['hypotheses_unchanged']
assert 'OK: каталог согласован с источниками' in (OUT/'catalog_validation.log').read_text(encoding='utf8')
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:
    json.dump(dict(passed=True,checked_input_hashes=checked,archived_retinal_hashes_stable=True,
        evidence={name:sha(OUT/name) for name in ['execution_receipt.json','graph_audit.json','catalog_validation.log']},
        scientific_sources_unchanged=True),f,ensure_ascii=False,indent=2)
print('Final integrity passed',checked,'source hashes')
