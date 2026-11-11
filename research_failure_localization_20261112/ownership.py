"""Resolve competitors' direct identity, without assuming an old ID is canonical."""
from common import *
import numpy as np
p=verify();stage1=read(OUT/'stage1.json');stage2=read(OUT/'stage2.json');records=[]
for v in sorted({r['video'] for r in p['selected']}):
    d=events(v);lookup={e['frame']:e for e in d['events']};ann=annotations(v)
    with np.load(NATIVE/'rollouts/C34'/(v+'.npz'),allow_pickle=False) as z:public=z['outputs']
    for r in [r for r in stage1['records'] if r['video']==v]:
        target=r['reference_hits'][0]['identity'];rf=r['reference_frame'];e=lookup[r['frame']]
        owners=[]
        for f in [rf,r['frame']]:
            rows=[]
            for a in public[(public[:,0]==f)&(public[:,7]==2)]:
                h=hits(a[1:5],ann[f])
                if len(h)==1 and h[0]['identity']==target:
                    rows.append(dict(tid=int(a[5]),box=a[1:5].tolist(),iou=h[0]['iou']))
            owners.append(dict(frame=f,public_tracks_on_same_GT=rows))
        s=next(x for x in stage2['records'] if (x['video'],x['frame'],x['old_id'],x['raw_id'])==(v,r['frame'],r['old_id'],r['raw_id']))
        result=dict(video=v,frame=r['frame'],old_id=r['old_id'],raw_id=r['raw_id'],kind=r['kind'],target=target,owners=owners,
                    candidate_winner_history=r['competitors'],diagnostic_selected=s['new_selected'])
        if r['kind']=='own' and not s['new_selected']:
            result['remaining_mechanism']='cost_above_unchanged_threshold' if s['diagnostic_cost']>.8 else 'competition_in_full_matrix'
        elif r['kind']=='own' and not r['selected']:
            result['remaining_mechanism']='permission_sufficient_for_this_matrix_only'
        else:result['remaining_mechanism']='no_required_assignment_change'
        result['canonical_ID_proven']=False
        records.append(result)
save('ownership.json',dict(passed=True,records=records,source_sha256=sha(OUT/'stage1.json'),script_sha256=sha(__file__),
 limitation='Multiple public boxes with unique IoU>=0.5 match to one GT prove ambiguity of pair-based ID restoration target. These are not a one-to-one TrackEval match and do not by themselves designate a canonical tracker ID.'))
print(json.dumps([r for r in records if r['old_id']==1102],ensure_ascii=False,indent=2))
