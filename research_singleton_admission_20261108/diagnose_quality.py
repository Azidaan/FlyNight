"""Separate identity-unknown from geometrically unmatchable annotated outputs."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_singleton_age_20261108.common import *
AGE=RUN;RUN=ROOT/'runs/singleton_admission_20261108';ERROR=ROOT/'runs/singleton_error_audit_20261108'


def main(obs):
    import numpy as np
    from datasets.bdd100k_loader import CLASS_NAMES
    from research_singleton_free_20261108.full_audit import overlaps
    assert read(OWN_FULL/'independent_audit.json')['passed'] and read(AGE/'full/independent_audit.json')['passed'] and read(ERROR/'queue_result.json')['passed']
    p=read(OWN_FULL/'protocol.json');events=read(ERROR/'report.json')['rows'];arms=read(ERROR/'input_manifest.json')['arms'];sources={}
    names=['research_singleton_admission_20261108/diagnose_quality.py','runs/singleton_error_audit_20261108/report.json','runs/singleton_error_audit_20261108/input_manifest.json',
        'runs/singleton_verify_20261108/full/independent_audit.json','runs/singleton_age_20261108/full/independent_audit.json','research_singleton_free_20261108/full_audit.py','datasets/bdd100k_loader.py']
    for video,entry in p['videos'].items():
        names.extend([entry['detections'],entry['raw_array'],f'data/bdd100k/labels/box_track_20/val/{video}.json',(OWN_FULL/'runtime_inputs'/(video+'.json')).relative_to(ROOT).as_posix()])
        for arm,folder in arms.items():names.append((Path(folder)/'rollouts'/arm/(video+'.npz')).as_posix())
    sources={n:sha(ROOT/n) for n in names}
    protocol=dict(registered=datetime.now(MSK).isoformat(),sources=sources,question='Do the directly added annotated outputs lack any admissible GTbox, or does a good raw box become geometrically bad after updating?',
        definitions=dict(current_annotated='Only actual sparse-label anchors; no interpolation.',geometric_unmatchable='Maximum same-class validGT IoU<0.5. This is a box-quality property, not a different-identity label.',
            ignore_overlap='Any positive-area intersection with a crowd or non-evaluated-category GTbox; retain separately, do not label it a certain FP.',
            confidence_rejection='Native cost>0.8 but inferred unfused predicted-box IoU>0.2+1e-6; removing confidence fusion alone would make that edge locally admissible. Not a recommendation to do so.',
            baseline_presence='Same-class native B0emitted box overlaps this raw candidate at IoU>=0.5; numeric track IDs never compared across variants.'),
        limitation='Descriptive local diagnosis on already studied20val. Unmatchable output is not automatically a measured incremental CLEARFP: ignored regions, duplicate matching and later feedback matter. No threshold fitting on these outcomes. Identity-unknown remains unknown.',
        new_detector_calls=0,new_core_calls=0,new_tracker_updates=0)
    write(RUN/'quality_protocol.json',protocol,new=True);totals={a:Counter() for a in arms};rows=[];groups=defaultdict(list)
    for r in events:
        if r['current_annotated']:groups[r['video']].append(r)
    for video,group in groups.items():
        entry=p['videos'][video]
        with np.load(ROOT/entry['detections'],allow_pickle=False) as z:det={k:z[k] for k in z.files}
        gt={int(f['frameIndex']):f for f in read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')};clock=read(OWN_FULL/'runtime_inputs'/(video+'.json'))['anchors']
        with np.load(ROOT/entry['raw_array'],allow_pickle=False) as z:base=z['outputs']
        outputs={}
        for arm,folder in arms.items():
            with np.load(ROOT/folder/'rollouts'/arm/(video+'.npz'),allow_pickle=False) as z:outputs[arm]=z['outputs']
        for r in group:
            frame=r['frame'];anchor=clock[str(frame)];f=gt[anchor['frameIndex']];assert f['name']==anchor['name'];raw_id=r['raw_id'];assert int(det['frame_idx'][raw_id])==frame
            category=CLASS_NAMES[int(det['cls'][raw_id])];raw_box=det['xyxy'][raw_id];valid=[];ignored=[]
            for obj in f.get('labels') or []:
                box=[obj['box2d'][k] for k in ['x1','y1','x2','y2']]
                if (obj.get('attributes') or {}).get('crowd',False) or obj['category'] not in CLASS_NAMES.values():ignored.append(box)
                elif obj['category']==category:valid.append(box)
            maximum=lambda box:float(overlaps([box],valid).max(initial=0))
            def ignore_overlap(box):
                if not ignored:return False
                b=np.array(ignored);lo=np.maximum(box[:2],b[:,:2]);hi=np.minimum(box[2:],b[:,2:]);return bool(np.any(np.maximum(0,hi-lo).prod(-1)>0))
            raw_max=maximum(raw_box);trajectory=outputs[r['arm']];current=trajectory[(trajectory[:,0]==frame)&(trajectory[:,5]==r['tid'])];assert len(current)==int(r['emitted'])
            old=base[(base[:,0]==frame)&(base[:,7]==int(det['cls'][raw_id]))];native_near=bool(len(old) and overlaps([raw_box],old[:,1:5]).max()>=.5)
            confidence=float(det['conf'][raw_id]);inferred_iou=float((1-r['cost'])/confidence);assert -1e-5<=inferred_iou<=1+1e-5
            item=dict(r,raw_max_valid_gt_iou=raw_max,raw_overlaps_ignore=ignore_overlap(raw_box),native_emits_near_raw_candidate=native_near,detector_confidence=confidence,inferred_native_predicted_iou=inferred_iou)
            c=totals[r['arm']];c['annotated_added_pairs']+=1;c['raw_unmatchable' if raw_max<.5 else 'raw_has_possible_gt']+=1
            if inferred_iou>.2+1e-6:c['locally_rejected_only_after_confidence_fusion']+=1
            if len(current):
                box=current[0,1:5];value=maximum(box);ignored_overlap=ignore_overlap(box);item.update(output_max_valid_gt_iou=value,output_overlaps_ignore=ignored_overlap)
                c['output_unmatchable' if value<.5 else 'output_has_possible_gt']+=1
                if value<.5 and not ignored_overlap:
                    c['output_unmatchable_without_ignore_overlap']+=1
                    if not native_near:c['unmatchable_output_and_no_native_box_near_raw']+=1
                if raw_max>=.5 and value<.5:c['good_raw_became_unmatchable_after_update']+=1
                if raw_max<.5 and value<.5:c['raw_and_output_both_unmatchable']+=1
                if ignored_overlap:c['output_overlaps_ignore']+=1
            rows.append(item)
    for name,h in sources.items():assert sha(ROOT/name)==h,name
    assert len(rows)==sum(r['current_annotated'] for r in events)
    write(RUN/'quality_report.json',dict(passed=True,totals={a:dict(v) for a,v in totals.items()},rows=rows,protocol_sha256=sha(RUN/'quality_protocol.json'),limitations=protocol['limitation'],
        interpretation='A direct annotated output can have no admissible same-class GTbox while its object identity is unknown. Keep these two labels separate; this diagnosis does not relabel unknown identities as foreign.'),new=True)


if __name__=='__main__':main(guard())
