from pathlib import Path
import json
from collections import Counter
root=Path(__file__).resolve().parent.parent
out=Path(__file__).resolve().parent
def read(folder,name):
    return json.loads((root/folder/name).read_bytes())
mm='research_memory_gap_transfer_20261114'
ap='research_gap_appearance_recovery_20261114'
z=read(mm,'actions.json')
t={(x['case'],x['frame']):x for x in z['traces']}
s=read(mm,'result.json')['episodes']
base={r['case']:r for r in s if r['method']=='fixed'}
v=read(ap,'actions.json')
rank={r['case']:r for r in v['ranks'] if r['slot']==0}
episodes=[]
for r in z['records']:
    if r['method']!='ordinary' or r['slot']!=0 or not r['first_return']:continue
    c=r['case'];mat=t[c,r['frame']];score=mat['scores']['ordinary'];idx=mat['rids']
    valid=[i for i,a in enumerate(mat['overlap_with_truth'][0]) if a>.5]
    old=base[c];new=next(q for q in s if q['case']==c and q['method']=='ordinary')
    item=dict(case=c,source=r['prior_available'],current=bool(valid),fixed=old['full_sequence_pass'],ordinary=new['full_sequence_pass'],selected=r['action'],state=r['status'],appearance_rank=rank[c]['first_correct_rank'])
    assert bool(valid)==r['available_current']
    if not item['source'] and not valid:reason='no_source_or_current'
    elif not item['source']:reason='no_source'
    elif not valid:reason='no_adequate_current'
    elif r['status']=='correct':reason='returned'
    else:
        i=max(valid,key=score[0].__getitem__)
        item.update(correct_candidate=idx[i],target_overlap=score[0][i],rival_overlap=score[1][i],max_overlap=max(score[0]))
        if max(score[0])==0:reason='zero_overlap_all_candidates'
        elif score[0][i]>.5 and score[1][i]>score[0][i] and dict(mat['actions']['ordinary']).get(1)==idx[i]:reason='rival_won_target'
        else:reason='other'
    item['reason']=reason;episodes.append(item)
assert len(episodes)==20 and len(set(q['case'] for q in episodes))==20
counts=Counter(q['reason'] for q in episodes)
assert counts==dict(returned=6,no_source_or_current=1,no_source=1,no_adequate_current=7,zero_overlap_all_candidates=4,rival_won_target=1)
assert [sum(x[a] for x in episodes) for a in ['fixed','ordinary']]==[4,6]
with (out/'twenty_gap_outcomes.json').open('x',encoding='utf8') as f:json.dump(dict(inputs=[mm+'/actions.json',mm+'/result.json',ap+'/actions.json'],counts=counts,episodes=episodes),f,indent=2);f.write('\n')
