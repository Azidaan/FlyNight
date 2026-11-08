from pathlib import Path
import json,hashlib
from collections import Counter
P=Path(__file__).resolve().parent;R=P.parents[1]
def j(p):return json.loads(p.read_text(encoding='utf-8'))
e=j(P/'evidence.json');s=j(R/'runs/singleton_id_fate_117_20261108/recovery1/summary.json')
assert {(r['video'],r['edge']) for r in e['rows']}=={tuple(x) for x in s['arms']['B']['improved_vs_B0']}
assert all(r['revocation'] is None for r in e['rows'])
for r in e['rows']:
    if r.get('next_fragment') and r['next_fragment']['exported_old_row']:
        q=r['next_fragment'];assert q['exported_old_row']['age']==q['age_from_last_public']
    if r['first_bad_evidence']:
        x=r['first_bad_evidence'];assert all(y['iou']>=.5 for y in x['raw_hits']+x['public_hits'])
ownership=j(P/'ownership.json');rejected=[r for r in ownership['rows'] if not r['accepted']]
assert len(ownership['rows'])==96 and len(rejected)==1 and rejected[0]['key']=='74|107|136' and rejected[0]['reasons']==['canonical_owner_changed']
assert all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in e['sources'].items())
with (P/'audit.json').open('x',encoding='utf-8') as f:json.dump(dict(passed=True,events=6,eligible_edges=96,accepted_edges=95,rejection=rejected[0],source_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ('evidence.json','ownership.json','protocol.md','trace.py','ownership.py')}),f,indent=2)
