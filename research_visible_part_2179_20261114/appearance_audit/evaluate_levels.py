"""Independent scalar audit of diagnostic feature-pyramid-level intervention."""
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = [SOURCE/'level_features.json', SOURCE/'level_features.npz',
         SOURCE/'feature_outputs.json', SOURCE/'feature_outputs.npz']
inputs = {str(p):digest(p) for p in paths}
metadata = json.loads(paths[0].read_text(encoding='utf8'))
records = metadata['records']
fills = ['imagenet_mean','black']
modes = ['blank','full','red_query_clean_refs','black_query_clean_refs']
roles = ['target_ref','competitor_ref','query']
subsets = {'blank':(0,0,0),'full':(31,31,31),
           'red_query_clean_refs':(1,2,1),'black_query_clean_refs':(1,2,2)}
levels = [-1,0,1]
bank = {}; norms=[]
with np.load(paths[1],allow_pickle=False) as arrays:
    used = set()
    for row in records:
        key = row['key']
        identity = (row['fill'],row['mode'],row['role'],row['level'])
        assert row['fill'] in fills and row['mode'] in modes and row['role'] in roles and row['level'] in levels
        assert identity not in bank and key not in used
        vector = arrays[key];assert vector.shape == (256,) and vector.dtype.kind == 'f'
        vector = tuple(float(x) for x in vector)
        assert all(math.isfinite(x) for x in vector)
        norm = math.sqrt(math.fsum(x*x for x in vector))
        assert abs(norm-1)<1e-5,(key,norm)
        norms.append(dict(key=key,norm=norm))
        bank[identity]=vector;used.add(key)
    assert set(arrays.files) == used
assert len(bank) == 72,len(bank)
old_metadata=json.loads(paths[2].read_text(encoding='utf8'))
old_lookup={(r['fill'],r['role'],r['subset']):r['key'] for r in old_metadata['records']}
reproduction=[]
with np.load(paths[3],allow_pickle=False) as old:
    for fill in fills:
        for mode in modes:
            for role,subset in zip(roles,subsets[mode]):
                current=bank[fill,mode,role,-1]
                reference=old[old_lookup[fill,role,subset]]
                error=max(abs(x-float(y)) for x,y in zip(current,reference))
                assert error<1e-5,(fill,mode,role,error)
                reproduction.append(dict(fill=fill,mode=mode,role=role,error=error))

def distance(a,b):
    cosine=math.fsum(x*y for x,y in zip(a,b))/math.sqrt(
        math.fsum(x*x for x in a)*math.fsum(x*x for x in b))
    assert -1-1e-12<=cosine<=1+1e-12
    return (1-max(-1,min(1,cosine)))/2

scores=[];comparison=[]
for fill in fills:
    for mode in modes:
        scored={}
        for level in levels:
            target,competitor,query=[bank[fill,mode,role,level] for role in roles]
            dt=distance(target,query);dc=distance(competitor,query)
            item=dict(fill=fill,mode=mode,level=level,target_distance=dt,
                      competitor_distance=dc,gap=dt-dc)
            scored[level]=item;scores.append(item)
        comparison.append(dict(fill=fill,mode=mode,native_gap=scored[-1]['gap'],
            level0_gap=scored[0]['gap'],level1_gap=scored[1]['gap'],
            native_minus_level0=scored[-1]['gap']-scored[0]['gap'],
            native_minus_level1=scored[-1]['gap']-scored[1]['gap']))
for path,expected in inputs.items():assert digest(Path(path))==expected,path
output=dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),
    inputs=inputs,evaluator_sha256=digest(Path(__file__)),records_checked=len(bank),
    scores_computed=len(scores),maximum_norm_error=max(abs(r['norm']-1) for r in norms),
    maximum_native_reproduction_error=max(r['error'] for r in reproduction),
    vector_checks=norms,reproduction=reproduction,scores=scores,comparisons=comparison,
    interpretation='Gap=d(target_ref,query)-d(competitor_ref,query); positive prefers competitor reference, not necessarily wrong physical identity. -1 is native area-based routing; 0/1 force all roles to the same feature-pyramid level. Reduced gap establishes a causal contribution of routing in this diagnostic, not a tracker improvement or universal best scale.')
with (HERE/'level_audit.json').open('x',encoding='utf8',newline='\n') as stream:
    json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(passed=True,maximum_norm_error=output['maximum_norm_error'],
    maximum_native_reproduction_error=output['maximum_native_reproduction_error'],
    comparisons=comparison),indent=2))
