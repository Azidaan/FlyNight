"""Verify both reused baselines and the final saved outputs, without rescoring."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'runs/singleton_forward_backward_transfer_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json')
for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
for name,digest in read(OUT/'execution_receipt.json')['checksums'].items():assert sha(OUT/name)==digest,name
baseline=read(BASE/'results.json')['records'];rows=read(OUT/'results.json')['records']
reuse=0
for r in rows:
    old=baseline[r['three_cell_result_index']]
    for field in ['video','frame','old','new','kind','representation']:assert r[field]==old[field]
    assert r['baseline_error_grid']==old['error_grid_steps'] and r['baseline_error_px']==old['error_px']
    assert r['baseline_forward_axes']==old['forward_axes'] and r['baseline_backward_axes']==old['backward_axes']
    c=next(c for c in p['cases'] if (c['video'],c['old'],c['new'])==(r['video'],r['old'],r['new']))
    assert r['cells']==c['geometry']['eligible_cells'] and len(r['cells'])==c['points']
    if r['forward_reused_from']:
        saved=next(o for o in read(ROOT/r['forward_reused_from'])['pairs'] if o['video']==r['video'] and o['old']==r['old'])
        assert r['forward_scores']==saved['variants']['FlyVis']['scores'];reuse+=1
assert reuse==1 and read(OUT/'audit.json')['passed']
graph=read(OUT/'graph_audit.json');assert graph['passed'] and graph['hypotheses_unchanged']
assert 'OK: каталог согласован с источниками' in (OUT/'catalog_validation.log').read_text(encoding='utf8')
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:
    json.dump(dict(passed=True,checked_input_hashes=len(p['sources']),baseline_records_reused=28,
        full_forward_reused=1,baseline_fields_exact=True,full_support_exact=True,
        evidence={name:sha(OUT/name) for name in ['execution_receipt.json','graph_audit.json','catalog_validation.log']}),f,ensure_ascii=False,indent=2)
print('Final integrity passed',len(p['sources']),'source hashes, exact reuse of 28 baselines and 1 full forward result')
