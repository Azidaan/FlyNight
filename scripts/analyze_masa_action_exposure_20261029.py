"""One diagnostic from completed audited artifacts; no new tuning or inference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import numpy as np
    folder=RUN/'masa_action_exposure';gpu=RUN/'masa_query_fp32_recovery1'
    p=service.read(gpu/'protocol.json')
    assert service.read(RUN/'masa_audit_fp32_recovery1/report.json')['passed']
    final=service.read(RUN/'masa_final_checked_record_recovery1/controller_result.json')
    assert final['exit_code']==0 and final['all_owned_ended']
    rows=[]
    for source in p['sources']:
        video=source['video'];out=gpu/'videos'/video;raw=np.load(out/'raw.npz')
        saved=service.read(out/'tracking_completed.json')
        assert all(service.sha(ROOT/f)==h for f,h in saved['sha256'].items())
        frames=saved['frames'];queries=len(saved['query_frames']);item=dict(video=video,frames=frames,queries=queries,variants={})
        def changed(a,b):
            fa,fb=a[:,0].astype(int),b[:,0].astype(int)
            return sum(not np.array_equal(a[fa==frame],b[fb==frame]) for frame in range(frames))
        for name in ['r50_1024','r50_1333','detic_1333']:
            events=service.read(out/(name+'_events.json'))
            eligible=sum(int(np.asarray(e['eligible'],bool).sum()) for e in events)
            deltas=[]
            for e in events:
                mask=np.asarray(e['eligible'],bool)
                delta=np.asarray(e['updated'])-np.asarray(e['original'])
                deltas.extend(delta[mask].tolist())
            item['variants'][name]=dict(cost_event_frames=len({e['frame'] for e in events}),eligible_pair_events=eligible,
                changed_raw_frames_vs_native=changed(raw[name],raw['native']),
                mean_applied_cost_delta=float(np.mean(deltas)) if deltas else None,
                max_applied_cost_delta=float(max(deltas)) if deltas else None)
        item['detic_vs_matched_r50_changed_raw_frames']=changed(raw['detic_1333'],raw['r50_1333'])
        rows.append(item)
        service.write(folder/'status.json',dict(video=video,completed=len(rows),total=20,updated=service.now()))
    totals=dict(frames=sum(x['frames'] for x in rows),query_frames=sum(x['queries'] for x in rows),
        detic_vs_matched_r50_changed_raw_frames=sum(x['detic_vs_matched_r50_changed_raw_frames'] for x in rows),
        variants={n:{k:sum(x['variants'][n][k] for x in rows) for k in ['cost_event_frames','eligible_pair_events','changed_raw_frames_vs_native']} for n in ['r50_1024','r50_1333','detic_1333']})
    service.write(folder/'report.json',dict(complete=True,diagnostic_only=True,scientific_verdict=None,totals=totals,per_video=rows,
        question='How often does the changed appearance signal reach a different full tracker output under the frozen query/ambiguity rule?',
        limitation='Raw-frame differences can propagate from earlier assignments; not a count of identity mistakes, not proof of a common root cause or standalone encoder quality. No parameter selection from this historical val analysis.',
        next_action_boundary='Use as evidence to choose an isolated action; do not start another broad diagnostic series or change weights/thresholds on these videos.',updated=service.now()))

if __name__=='__main__':main()
