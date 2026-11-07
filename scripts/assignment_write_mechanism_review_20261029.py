"""One offline check to prepare a distinct cost-use intervention; no rerun."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import numpy as np
    folder=RUN/'assignment_write_mechanism_review';folder.mkdir(exist_ok=True)
    service.write(folder/'protocol.json',dict(created=service.now(),scope='Read exact cost-event logs to test whether a common positive appearance penalty can remove geometrically admitted candidates, and measure actual age of read memory. Candidate next action: center the appearance correction while preserving original geometry; requires separate plan agreement before any new tracking run.',
        limitation='Counts are candidate-edge events, not realized errors or a proof of the root cause. No GT/policy inference, quality-based tuning, or tracker replay.'))
    p=service.read(RUN/'assignment_write_val/protocol.json');threshold=p['config']['match_thresh'];rows=[]
    for source in p['sources']:
        video=source['video']
        for encoder,key in [('r50','r50_1333'),('detic','detic_1333')]:
            for policy,base,name in [('last',RUN/'masa_query_fp32_recovery1',key),('guarded',RUN/'assignment_write_val',encoder+'_guarded')]:
                events=service.read(base/'videos'/video/(name+'_events.json'));ages=[];pairs=crossings=all_excluded_frames=0
                for e in events:
                    original=np.asarray(e['original']);updated=np.asarray(e['updated']);eligible=np.asarray(e['eligible'],bool)
                    pairs+=int(eligible.sum());crossings+=int((eligible & (original<=threshold) & (updated>threshold)).sum())
                    all_excluded_frames+=int(any(np.any(mask) and np.all(row[mask]>threshold) for row,mask in zip(updated,eligible)))
                    ages.extend(e['frame']-f for f in e['past_frames'])
                rows.append(dict(video=video,encoder=encoder,policy=policy,cost_events=len(events),eligible_pairs=pairs,
                    crossed_threshold_pair_events=crossings,events_with_any_row_all_candidates_over_threshold=all_excluded_frames,
                    memory_reads=len(ages),memory_age_sum=sum(ages),max_memory_age=max(ages,default=0),reads_older60=sum(x>60 for x in ages)))
    totals={}
    for encoder in ['r50','detic']:
        for policy in ['last','guarded']:
            group=[x for x in rows if x['encoder']==encoder and x['policy']==policy]
            counts={k:sum(x[k] for x in group) for k in ['cost_events','eligible_pairs','crossed_threshold_pair_events','events_with_any_row_all_candidates_over_threshold','memory_reads','memory_age_sum','reads_older60']}
            counts['mean_memory_age_frames']=counts['memory_age_sum']/counts['memory_reads'] if counts['memory_reads'] else None
            counts['max_memory_age_frames']=max(x['max_memory_age'] for x in group)
            totals[encoder+'_'+policy]=counts
    service.write(folder/'report.json',dict(complete=True,diagnostic_only=True,threshold=threshold,totals=totals,per_video=rows,
        scientific_verdict=None,limitation='Different policies have different trajectories and event sets. Do not interpret unpaired aggregate differences as a causal effect of age or threshold crossings. An edge crossing is not necessarily the selected edge or a missed ground-truth target. No new cost rule run.',updated=service.now()))

if __name__=='__main__':main()
