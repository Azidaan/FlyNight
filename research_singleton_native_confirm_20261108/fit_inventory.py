"""Read existing fit-only snapshots: native first emission available for old-ID linking?"""
import os,sys,pickle,zipfile
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_identity_assignment_run_20261108.common import read,write,sha,datetime,MSK
from research_singleton_verify_20261108.common import guard
RUN=ROOT/'runs/singleton_native_confirm_20261108'


def main(obs):
    import numpy as np
    sources={n:sha(ROOT/n) for n in ['research_singleton_native_confirm_20261108/fit_inventory.py','runs/identity_assignment_20261108/export_complete.json','research_identity_assignment_protocol_20261108/queries.json','runs/singleton_verify_20261108/calibration_rows.json','runs/singleton_verify_20261108/selection.json']}
    protocol=dict(registered=datetime.now(MSK).isoformat(),sources=sources,
        question='Among already archived fit20 query frames, how often is the sole valid old-track candidate actually emitted by a previously unconfirmed native track?',
        scope='Read-only feasibility inventory, no tracker/core/detector/evaluator rerun. Only fit20. This is a lower bound: frames not in the query export are absent, so zero exposure does not reject a full native-confirmation policy.',
        proposal='Only after actual native first public confirmation, link the never-public new native identity to a currently non-emitted old canonical identity. All native physics and all physical outputs must remain exact. No previous labels are rewritten. Two already public working tracks are outside this first-problem check.',
        controls='Same old geometry/support and frozen thresholds for IMAGE/R1/DIS/FLYVIS/RANDOM; geometry and recency without additional image features required. Unknown identity never negative.',
        alternative='Prior offline stitch attempts already existed. A causal label link must outperform the same causal no-FlyVis link; preserving physical boxes does not itself prove correct identity or unchanged official CLEAR counts.',
        literature=[dict(title='Simple Online and Realtime Tracking',url='https://arxiv.org/pdf/1602.00763',section='3.4',supports='Probationary association to prevent false positives; does not prove this canonical-link mechanism.'),dict(local='experiments/project_history.json: stitch_appearance_models_v1',supports='Offline all-candidate stitching already tested; geometry-only claim from the earlier label-filtered experiment cannot be reused.')],
        acceptance='Count native-confirmed exposures and retain known positive/foreign/unknown separately. Verify source hashes. This inventory alone cannot accept a tracking fix.',
        new_core_calls=0,new_detector_calls=0,new_tracker_updates=0)
    write(RUN/'fit_inventory_protocol.json',protocol,new=True)
    qs=read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows'];fit={q['video'] for q in qs if q['split']=='fit20'}
    cal={(r['video'],r['frame'],r['row'],r['column']):r for r in read(ROOT/'runs/singleton_verify_20261108/calibration_rows.json')['rows']}
    exports=read(ROOT/'runs/identity_assignment_20261108/export_complete.json')['videos'];rows=[];totals=Counter();inputs={}
    for video in sorted(fit):
        p=ROOT/exports[video]['path'];assert sha(p)==exports[video]['sha256'];d=read(p);zp=p.with_suffix('.states.zip');assert sha(zp)==d['states_sha256'];inputs[p.relative_to(ROOT).as_posix()]=sha(p);inputs[zp.relative_to(ROOT).as_posix()]=sha(zp)
        scores_path=ROOT/f'runs/singleton_verify_20261108/scores/{video}.json';scores=read(scores_path)['frames'] if scores_path.exists() else {};inputs[scores_path.relative_to(ROOT).as_posix()]=sha(scores_path) if scores_path.exists() else None
        with zipfile.ZipFile(zp) as archive, np.load(p.with_suffix('.npz'),allow_pickle=False) as matrices:
            for frame,snap in d['snapshots'].items():
                if not any(q['split']=='fit20' for q in snap['queries']):continue
                pre=pickle.loads(archive.read(frame+'.pkl'));unconfirmed={int(t['track_id']) for t in pre['pools']['tracked_stracks'] if not t['is_activated']};emitted={int(o[4]) for o in snap['native_output']};confirmed=unconfirmed & emitted
                raw_to_new={int(snap['native_final_assignments'][str(t)]):t for t in confirmed};used_i={i for i,j in snap['native_matches']};used_j={j for i,j in snap['native_matches']};C=matrices['cost_'+frame]
                for i,j in snap['votes']:
                    if i in used_i or j in used_j or C[i,j]>=1:continue
                    r=dict(video=video,frame=int(frame),row=i,column=j,old_tid=snap['rows'][i]['tid'],raw_id=snap['raw_ids'][j],confirmed_new_tid=raw_to_new.get(snap['raw_ids'][j]))
                    sc=scores.get(frame,{}).get(f'{i},{j}');r['support']=None if sc is None else sc['support'];r['scores']=None if sc is None else sc['scores']
                    label=cal.get((video,int(frame),i,j));r['known']=False if label is None else label['primary_known'];r['positive']=None if label is None else label['positive'];rows.append(r);totals['archived_free_sole_edges']+=1
                    if r['confirmed_new_tid'] is not None:
                        totals['native_confirmed_edges']+=1;totals['known_correct' if r['known'] and r['positive'] else 'known_foreign' if r['known'] else 'unknown']+=1
    for n,h in sources.items():assert sha(ROOT/n)==h,n
    for n,h in inputs.items():
        if h is not None:assert sha(ROOT/n)==h,n
    write(RUN/'fit_inventory.json',dict(passed=True,totals=dict(totals),rows=rows,inputs=inputs,protocol_sha256=sha(RUN/'fit_inventory_protocol.json'),scope=protocol['scope']),new=True)
    print(dict(totals))


if __name__=='__main__':main(guard())
