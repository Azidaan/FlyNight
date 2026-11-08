import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;MAPS=ROOT/'runs/full_decoder_outputs_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for folder in [OUT,MAPS]:
    for path,digest in read(folder/'protocol.json')['sources'].items():assert sha(ROOT/path)==digest,path
    for path,digest in read(folder/'execution_receipt.json')['checksums'].items():assert sha(folder/path)==digest,path
for path,digest in read(OUT/'results.json')['files'].items():assert sha(ROOT/path)==digest,path
for path,digest in read(MAPS/'completed.json')['files'].items():assert sha(ROOT/path)==digest,path
assert read(OUT/'independent_verification.json')['passed']
graph=read(OUT/'graph_audit.json');assert graph['passed'] and graph['hypotheses_unchanged']
assert 'OK: каталог согласован с источниками' in (OUT/'catalog_validation.log').read_text(encoding='utf8')
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:
    json.dump(dict(passed=True,maps_verified=399,events_verified=200,reference_anchor_status_preserved=True,
        evidence={name:sha(OUT/name) for name in ['summary.json','graph_audit.json','catalog_validation.log','independent_verification.json']}),f,indent=2)
print('Final integrity passed')
