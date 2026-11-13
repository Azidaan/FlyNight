"""Independent saved-vector comparison: each reference keeps its own car core."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
source = HERE.parent
files = [source/'feature_outputs.json', source/'feature_outputs.npz']
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p): digest(p) for p in files}
metadata = json.loads(files[0].read_text(encoding='utf8'))
bank = {}
with np.load(files[1], allow_pickle=False) as arrays:
    for rec in metadata['records']:
        vector = tuple(float(x) for x in arrays[rec['key']])
        assert len(vector) == 256 and all(math.isfinite(x) for x in vector)
        assert abs(math.sqrt(math.fsum(x*x for x in vector))-1) < 1e-5
        bank[rec['fill'],rec['role'],rec['subset']] = vector

def distance(a,b):
    cosine = math.fsum(x*y for x,y in zip(a,b))/math.sqrt(
        math.fsum(x*x for x in a)*math.fsum(x*x for x in b))
    return (1-max(-1,min(1,cosine)))/2

rows = []
for fill in ['imagenet_mean','black']:
    for query_subset in range(32):
        target = bank[fill,'target_ref',1]
        competitor = bank[fill,'competitor_ref',2]
        query = bank[fill,'query',query_subset]
        dt = distance(target,query)
        dc = distance(competitor,query)
        rows.append(dict(fill=fill,target_reference_subset=1,
            competitor_reference_subset=2,query_subset=query_subset,
            target_distance=dt,competitor_distance=dc,gap=dt-dc))
for path, expected in before.items(): assert digest(Path(path)) == expected
out = dict(passed=True,inputs=before,evaluator_sha256=digest(Path(__file__)),
    definition='Gap=d(target red-core reference, query)-d(competitor black-core reference, query). Positive favors competitor reference. Core masks are conservative manual visible regions; unselected own-car pixels may exist in uncertain group. A physical-car label still requires annotation audit.',
    warning='Joint subset1 from pixel_audit retains the same red-car group inside both original reference canvases; it is NOT the clean own-car reference comparison. This file uses target subset1 and competitor subset2.',
    rows=rows)
with (HERE/'cross_component.json').open('x',encoding='utf8',newline='\n') as stream:
    json.dump(out,stream,ensure_ascii=False,indent=2,allow_nan=False)
    stream.write('\n')
print(json.dumps([r for r in rows if r['query_subset'] in (0,1,2,31)],indent=2))
