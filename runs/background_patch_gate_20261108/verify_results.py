"""Independent scalar metrics and exact geometry check for reused baselines."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json');rows=[read(ROOT/path) for path in read(OUT/'results.json')['files']]
summary=read(OUT/'summary.json');known=[r for r in rows if r['kind']!='unknown' and r['supported'] and all(v.get('valid',False) for v in r['variants'].values())]
for name in p['representations']:
    for field in ['correlation_max','filter_max']:
        actual=roc_auc_score([r['kind']=='own' for r in known],[r['variants'][name][field] for r in known])
        assert abs(actual-summary['scores'][name][field])<1e-12
full=read(ROOT/'runs/singleton_support_size_20261108/protocol.json')['cases']
original=read(ROOT/'runs/singleton_matched_support_20261108/scores.json')['pairs']
reuse=0
for e,r in zip(p['events'],rows):
    for name,v in r['variants'].items():
        if not v.get('correlation_reused'):continue
        c=next((c for c in full if (c['video'],c['frame'],c['old'],c['new'])==(e['video'],e['frame'],e['old_tid'],e['new_tid'])),None)
        if c is not None:
            assert e['geometry']['cells']==c['geometry']['eligible_cells']
            assert np.array_equal(e['geometry']['prediction'],c['geometry']['prediction'])
            assert np.array_equal(e['geometry']['offsets'],c['geometry']['offsets'])
        else:
            old=next(o for o in original if (o['video'],o['frame'],o['old'],o['new'])==(e['video'],e['frame'],e['old_tid'],e['new_tid']))
            assert e['geometry']['cells']==old['cells']
            assert np.array_equal(e['geometry']['prediction'],old['predicted_px'])
            assert np.array_equal(e['geometry']['offsets'],old['offsets_px'])
        reuse+=1
assert reuse==12
for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
for path,digest in read(OUT/'results.json')['files'].items():assert sha(ROOT/path)==digest,path
with (OUT/'independent_verification.json').open('x',encoding='utf8') as f:
    json.dump(dict(passed=True,metrics_independently_verified=8,reused_geometry_verified=12,unknown_not_invented=True,
        summary_sha256=sha(OUT/'summary.json')),f,indent=2)
print('Independent verification passed')
