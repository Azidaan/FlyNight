from pathlib import Path
from collections import Counter
import json,hashlib,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
j=lambda p:json.loads(p.read_text(encoding='utf-8'))
inv=j(P/'inventory.json');eligible=j(P/'eligible_pairs.json');maths=j(P/'mathematical_checks.json');ident=j(P/'identifiability.json')
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in inv['sources'].items())
assert len(eligible)==inv['unique_pairs']==254 and inv['unique_feature_keys']==507
ready=[x for x in eligible if x['main'] and x['age']<=2];assert len(ready)==62
fit={e['video'] for e in j(ROOT/'runs/bbox_fit_maps_20261108/inventory_recovery1.json')['events']}
counter=Counter();label_hash={}
for r in ready:
    path=ROOT/'runs/bbox_singleton_pilot_20261108/labels'/f"{r['video']}.json"
    l=j(path)['rows'][f"{r['frame']}|{r['old']}|{r['new']}"];label_hash[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    kind='unknown' if not l['known'] else 'correct' if l['positive'] else 'foreign'
    counter[('fit20' if r['video'] in fit else 'evaluate26',kind)]+=1
assert counter=={('fit20','correct'):2,('fit20','foreign'):1,('fit20','unknown'):24,('evaluate26','correct'):8,('evaluate26','unknown'):27}
assert maths['roundtrip_error']<1e-12 and maths['affine_field_interpolation_error']<1e-12
assert ident['passed'] and ident['pattern_best_shift']==[13.,6.5] and ident['pattern_margin']>0
report=dict(passed=True,files_verified=len(inv['sources']),unique_pairs=254,primary_age2_pairs=62,evaluation_foreign=0,acceptance_protocol_ready=False,reason='No evaluation negatives under mathematically cleaner same-scale/common-support domain.',labels=label_hash,artifacts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file()})
with (P/'audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print({k:v for k,v in report.items() if k not in ('labels','artifacts_sha256')})
