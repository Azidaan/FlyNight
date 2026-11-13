"""Independent scalar comparison of saved pixel-intervention descriptors.

No model or root evaluator import. Metadata never determines semantic identity:
positive gap means only that the descriptor of competitor_ref is closer.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--metadata', type=Path, default=HERE.parent/'feature_outputs.json')
parser.add_argument('--features', type=Path, default=HERE.parent/'feature_outputs.npz')
parser.add_argument('--output', type=Path, default=HERE/'pixel_audit.json')
args = parser.parse_args()
for p in [args.metadata,args.features,args.output]:p.resolve().relative_to(ROOT.resolve())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

before={str(p.resolve()):digest(p) for p in [args.metadata,args.features]}
metadata=json.loads(args.metadata.read_text(encoding='utf8'))
records=metadata['records'] if isinstance(metadata,dict) else metadata
roles={'target_ref':1159,'competitor_ref':1164,'query':1165}
fills=['imagenet_mean','black']
components={1:'red',2:'black',4:'occluders',8:'remaining_scene'}
bank={}; checks=[]
with np.load(args.features,allow_pickle=False) as archive:
    used=set()
    for rec in records:
        ident=(rec['fill'],rec['subset'],rec['role'])
        assert rec['fill'] in fills and type(rec['subset']) is int and 0<=rec['subset']<=15,rec
        assert rec['role'] in roles and rec['frame']==roles[rec['role']],rec
        assert ident not in bank,ident
        key=rec['key'];assert key in archive,key
        a=archive[key];assert a.shape==(256,) and a.dtype.kind=='f',(key,a.shape,str(a.dtype))
        vector=tuple(float(x) for x in a)
        assert all(math.isfinite(x) for x in vector),key
        norm=math.sqrt(math.fsum(x*x for x in vector))
        assert norm>0 and abs(norm-1)<1e-5,(key,norm)
        bank[ident]=vector;used.add(key)
        checks.append(dict(key=key,fill=rec['fill'],subset=rec['subset'],role=rec['role'],norm=norm))
    assert set(archive.files)==used,('Unaccounted feature arrays',set(archive.files)-used)
assert len(bank)==96,len(bank)
for fill in fills:
    for subset in range(16):
        for role in roles:assert (fill,subset,role) in bank

def distance(a,b):
    denominator=math.sqrt(math.fsum(x*x for x in a)*math.fsum(x*x for x in b))
    cosine=math.fsum(x*y for x,y in zip(a,b))/denominator
    assert -1-1e-12<=cosine<=1+1e-12,cosine
    return (1-min(1,max(-1,cosine)))/2

modes={
    'joint':lambda s:(s,s,s),
    'references_only':lambda s:(s,s,15),
    'query_only':lambda s:(15,15,s),
    'target_reference_only':lambda s:(s,15,15),
    'competitor_reference_only':lambda s:(15,s,15),
}
scores=[]; factors=[]
for fill in fills:
    for mode,selector in modes.items():
        values={}
        for subset in range(16):
            st,sc,sq=selector(subset)
            dt=distance(bank[(fill,st,'target_ref')],bank[(fill,sq,'query')])
            dc=distance(bank[(fill,sc,'competitor_ref')],bank[(fill,sq,'query')])
            values[subset]=dt-dc
            scores.append(dict(fill=fill,mode=mode,subset=subset,
                applied_subsets=dict(target_ref=st,competitor_ref=sc,query=sq),
                target_distance=dt,competitor_distance=dc,gap=dt-dc,
                nearer_reference='competitor_ref' if dt>dc else 'target_ref' if dt<dc else 'equal'))
        attribution=[]
        for bit,name in components.items():
            conditional=[];shapley=0.
            for subset in range(16):
                if subset&bit:continue
                size=subset.bit_count()
                difference=values[subset|bit]-values[subset]
                weight=math.factorial(size)*math.factorial(3-size)/math.factorial(4)
                shapley+=weight*difference
                conditional.append(dict(other_components=subset,gap_without=values[subset],
                    gap_with=values[subset|bit],difference=difference,shapley_weight=weight))
            attribution.append(dict(bit=bit,component=name,
                full_context_deletion_effect=values[15]-values[15^bit],
                mean_conditional_effect=math.fsum(x['difference'] for x in conditional)/8,
                shapley_effect=shapley,conditional=conditional))
        efficiency_error=abs(math.fsum(x['shapley_effect'] for x in attribution)-(values[15]-values[0]))
        assert efficiency_error<1e-12,efficiency_error
        factors.append(dict(fill=fill,mode=mode,full_gap=values[15],empty_gap=values[0],
            full_minus_empty=values[15]-values[0],components=attribution,
            shapley_efficiency_error=efficiency_error))

# Full images must be independent of fill. Descriptor exactness is more direct
# than merely agreeing similarity scores. The check also catches mask leakage.
full_fill_max_error=max(abs(x-y) for role in roles
    for x,y in zip(bank[(fills[0],15,role)],bank[(fills[1],15,role)]))
assert full_fill_max_error<1e-5,full_fill_max_error
for path,h in before.items():assert digest(Path(path))==h,path
result=dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),
    inputs=before,evaluator_sha256=digest(Path(__file__)),
    records_checked=len(checks),scores_computed=len(scores),
    maximum_norm_error=max(abs(x['norm']-1) for x in checks),
    full_fill_max_descriptor_error=full_fill_max_error,
    vector_checks=checks,scores=scores,factorial=factors,
    scope='Scalar math.fsum distances, no root scoring helper; saved descriptors are shared input. Positive gap favors competitor_ref, not automatically a wrong physical identity.',
    interpretation='Deletion effect=gap(full)-gap(full without component). Positive effect means including component promotes competitor preference under that intervention. Conditional/Shapley effects may interact and depend on fill. They are not intrinsic pixel importance or dataset identity proof.')
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x',encoding='utf8',newline='\n') as f:
    json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print('PIXEL_SCALAR_AUDIT_PASSED',len(checks),len(scores),full_fill_max_error)
