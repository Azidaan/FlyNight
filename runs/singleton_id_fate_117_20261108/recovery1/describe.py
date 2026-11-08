from pathlib import Path
import json,csv,hashlib
from collections import Counter
P=Path(__file__).resolve().parent
r=json.loads((P/'events.json').read_text(encoding='utf-8'));out={}
for arm in ('B0','H','V','B'):
    c=Counter()
    for e in r:
        a=e['arms'][arm];obs=a['observations']
        if a['state']!='saved_both':continue
        c['saved_first_two']+=1
        if all(x['status']=='saved' and not x['hijacked'] for x in obs):c['saved_all_remaining_annotations']+=1
        elif any(x['status']=='changed_id' or x['hijacked'] for x in obs[2:]):c['later_known_change_or_hijack']+=1
        else:c['later_missing_match_only']+=1
    out[arm]=dict(c)
ret=Counter()
with (P/'returns.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['video','edge','prior_target','return_frame','delay','class','raw_index','x1','y1','x2','y2','raw_x1','raw_y1','raw_x2','raw_y2','return_direct_gt','nearest_future_labels','warning_lineage'])
    for e in r:
        a=e['return_info']
        if not a:continue
        ret['returned']+=1;ret['prior_unknown']+=e['target'] is None;ret['direct_return_identity_known']+=a['gt_identity'] is not None
        labels=[x['identity'] for x in a['nearby_annotations'] if x['frame']>=a['frame'] and x['identity']]
        if e['target'] is not None:
            state='no_future_match' if not labels else 'same_on_available_future' if set(labels)=={e['target']} else 'other_on_available_future' if e['target'] not in labels else 'mixed_future'
            ret[state]+=1
        w.writerow([e['video'],e['key'],e['target'],a['frame'],a['delay'],a['cls'],a['raw_index'],*a['box'],*a['raw_box'],a['gt_identity'],json.dumps(a['nearby_annotations']),e['lineage']])
with (P/'extended_summary.json').open('x',encoding='utf-8') as f:json.dump(dict(arms=out,returns=ret,events_sha256=hashlib.sha256((P/'events.json').read_bytes()).hexdigest()),f,indent=2)
