"""Explain return-event counter changes by corresponding native output rows."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_identity_assignment_run_20261108.common import *
FULL=ROOT/'runs/bbox_normalized_signals_20261108/full'
RUN=ROOT/'runs/identity_evidence_20261108'


def main():
    import numpy as np
    assert read(FULL/'queue_result.json')['passed'];source=Path(__file__);write(RUN/'return_count_diagnostic_protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources={source.relative_to(ROOT).as_posix():sha(source),'runs/bbox_normalized_signals_20261108/full/comparison_report.json':sha(FULL/'comparison_report.json'),'runs/bbox_normalized_signals_20261108/full/independent_audit.json':sha(FULL/'independent_audit.json')},scope='Read-only native-row correspondence for all20videos. Decompose correct-return count difference; do not change acceptance, labels, tracker or outcomes. Gap closure is not automatically an identity-quality proof.'),new=True)
    p=read(FULL/'protocol.json');totals=Counter();cases=[];sources={}
    for video in p['videos']:
        paths={a:FULL/'rollouts'/a/(video+'.npz') for a in ['B0','IMAGE']};arrays={}
        for a,path in paths.items():
            with np.load(path,allow_pickle=False) as z:arrays[a]=z['outputs']
            sources[path.relative_to(ROOT).as_posix()]=sha(path)
        a,b=arrays['B0'],arrays['IMAGE'];assert np.array_equal(a[:,[0,1,2,3,4,6,7]],b[:,[0,1,2,3,4,6,7]])
        event={arm:read(FULL/'additional'/arm/(video+'.json'))['events'] for arm in ['B0','IMAGE']};lookup={arm:{(e['frame'],e['tid']):e for e in events} for arm,events in event.items()};base_correct={(e['frame'],e['tid']):e for e in event['B0'] if e['kind']=='correct'};image_correct={(e['frame'],e['tid']):e for e in event['IMAGE'] if e['kind']=='correct'};mapped=set()
        for key,e in base_correct.items():
            f,tid=key;ix=np.flatnonzero((a[:,0]==f)&(a[:,5]==tid));assert len(ix)==1;j=int(ix[0]);canonical=int(b[j,5]);other=lookup['IMAGE'].get((f,canonical));mapped.add((f,canonical))
            if other and other['kind']=='correct':totals['correct_events_retained']+=1;continue
            previous=np.flatnonzero((b[:,0]==f-1)&(b[:,5]==canonical));past=np.flatnonzero((b[:,0]<f)&(b[:,5]==canonical));kind='became_'+other['kind'] if other else 'gap_closed_by_continuous_canonical_ID' if len(previous) else 'became_first_public_appearance_of_this_ID' if not len(past) else 'missing_event_requires_investigation'
            assert kind!='missing_event_requires_investigation';totals['removed_correct_events']+=1;totals[kind]+=1;cases.append(dict(video=video,frame=f,native_id=tid,canonical_id=canonical,counter_change=-1,classification=kind,baseline_event=e,image_event=other,previous_native_owners=[int(a[k,5]) for k in previous],interpretation='A missing return event can mean that the public ID was continuous across the native fragment change. This fact alone does not establish the identity of every unannotated physical observation.'))
        for key,e in image_correct.items():
            if key in mapped:continue
            f,tid=key;ix=np.flatnonzero((b[:,0]==f)&(b[:,5]==tid));assert len(ix)==1;j=int(ix[0]);native=int(a[j,5]);original=lookup['B0'].get((f,native));assert original is None or original['kind']!='correct';totals['added_correct_events']+=1;cases.append(dict(video=video,frame=f,native_id=native,canonical_id=tid,counter_change=1,classification='added_correct_return_event',baseline_event=original,image_event=e))
    delta=totals['added_correct_events']-totals['removed_correct_events'];report=read(FULL/'comparison_report.json');assert delta==report['identity_gain']['IMAGE']['correct_returns'];write(RUN/'return_count_diagnostic.json',dict(passed=True,counts=dict(totals),delta_correct_return_count=delta,cases=cases,source_checksums=sources,acceptance_unchanged=True,scope='Exact decomposition of the counter change, not a new quality metric or an excuse to accept a failed trial.'),new=True);print(dict(totals),delta)


if __name__=='__main__':main()
