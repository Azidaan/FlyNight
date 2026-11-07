"""Read-only witnesses for a possible ambiguity-dependent write intervention.

No tracker replay, feature extraction, threshold selection, or scientific retest.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import numpy as np
    folder=RUN/'memory_write_first_divergence';folder.mkdir(exist_ok=True)
    service.write(folder/'protocol.json',dict(created=service.now(),scope='One offline diagnostic for possible ambiguity-dependent memory write: earliest output divergence with same prefix, compare recorded original and appearance costs. No GT, inference, replay, parameter selection, or quality acceptance.',limitations='A witness localizes the first changed output; it does not establish ground-truth identity, harmfulness of that event, or a common cause. No new policy is published.'))
    p=service.read(RUN/'memory_write_val/protocol.json');rows=[]
    for source in p['sources']:
        video=source['video'];old=RUN/'masa_query_fp32_recovery1/videos'/video;new=RUN/'memory_write_val/videos'/video
        oldraw=np.load(old/'raw.npz');newraw=np.load(new/'raw.npz')
        frames=service.read(old/'tracking_completed.json')['frames']
        query=set(service.read(old/'tracking_completed.json')['query_frames'])
        for encoder,key in [('r50','r50_1333'),('detic','detic_1333')]:
            a=oldraw[key];b=newraw[encoder+'_cautious'];first=None
            # ID order itself is not a different set of tracked objects.
            def ordered(x):return x[np.lexsort(tuple(x[:,k] for k in reversed(range(x.shape[1]))))] if len(x) else x
            for frame in range(frames):
                if not np.array_equal(ordered(a[a[:,0]==frame]),ordered(b[b[:,0]==frame])):
                    first=frame;break
            item=dict(video=video,encoder=encoder,first_different_output_frame=first)
            if first is not None:
                item['first_is_query_frame']=first in query
                item['same_output_prefix']=True
                e1=[e for e in service.read(old/(key+'_events.json')) if e['frame']==first]
                e2=[e for e in service.read(new/(encoder+'_cautious_events.json')) if e['frame']==first]
                item['last_events']=e1;item['cautious_events']=e2
                item['comparable_recorded_costs']=[]
                for left,right in zip(e1,e2):
                    same=(left['track_ids']==right['track_ids'] and left['candidate_boxes']==right['candidate_boxes'])
                    item['comparable_recorded_costs'].append(dict(same_track_and_candidate_keys=same,
                        original_cost_exact=same and left['original']==right['original'],
                        eligible_exact=same and left['eligible']==right['eligible'],
                        appearance_cost_different=same and left['appearance']!=right['appearance']))
                relevant={int(t) for e in e2 for t in e['track_ids']}
                writes=service.read(new/(encoder+'_cautious_writes.json'))
                item['last_memory_write_before_divergence']={str(t):next((dict(frame=w['frame'],score=w['score'],alpha=w['alpha'],previous_frame=w['previous_frame']) for w in reversed(writes) if w['track_id']==t and w['frame']<first),None) for t in relevant}
            rows.append(item)
    changed=[x for x in rows if x['first_different_output_frame'] is not None]
    pairs=[v for x in changed for v in x['comparable_recorded_costs']]
    service.write(folder/'report.json',dict(complete=True,diagnostic_only=True,rows=rows,
        totals=dict(comparisons=len(rows),comparisons_with_output_difference=len(changed),all_first_changes_at_query=all(x['first_is_query_frame'] for x in changed),
            comparable_cost_groups=sum(x['same_track_and_candidate_keys'] for x in pairs),
            groups_with_same_original_and_eligible_but_changed_appearance=sum(x['original_cost_exact'] and x['eligible_exact'] and x['appearance_cost_different'] for x in pairs)),
        scientific_verdict=None,limitation='No identity labels: these witnesses do not say which first decision is correct or prove pollution/staleness. Association cost versus write reliability remains a proposed future action, not an accepted improvement.',updated=service.now()))

if __name__=='__main__':main()
