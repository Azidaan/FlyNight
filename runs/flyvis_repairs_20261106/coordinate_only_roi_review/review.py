"""Independent NumPy replay of a fixed point-only diagnostic; no model or tuning."""
from pathlib import Path
import ast
import hashlib
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
ORIGINAL = ROOT/'runs/flyvis_repairs_20261106/coordinate_only_roi'
SOURCE = ROOT/'runs/video_rate/b1c66a42-6f7d68ca'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tracks(frame):
    result={}
    for label in frame.get('labels') or []:
        if label.get('box2d') is None:
            continue
        b=label['box2d'];key=(label['category'],str(label['id']))
        assert key not in result
        box=np.array([b[k] for k in ('x1','y1','x2','y2')],dtype=np.float32)
        assert np.isfinite(box).all() and np.all(box[2:]>box[:2])
        result[key]=box
    return result


def replay(truth,matches,points,vectors):
    errors=[]
    for i in range(1,len(truth)):
        previous,current=tracks(truth[i-1]),tracks(truth[i])
        start,stop=matches[i-1]['video_frame_index'],matches[i]['video_frame_index']
        for key in sorted(previous.keys()&current.keys()):
            box=previous[key].copy();velocity=np.zeros(2,np.float32)
            for frame in range(start+1,stop+1):
                center=(box[:2]+box[2:])*.5;half=(box[2:]-box[:2])*.5
                inside=np.all((points>=box[:2])&(points<=box[2:]),axis=1)
                chosen=np.flatnonzero(inside)
                if len(chosen)<3:
                    distance=np.linalg.norm((points-center)/np.maximum(half,1.),axis=1)
                    chosen=np.unique(np.concatenate([chosen,np.argsort(distance)[:3]]))
                samples=vectors[frame,chosen]
                movement=np.median(samples,axis=0).astype(np.float32)
                dispersion=float(np.median(np.linalg.norm(samples-movement,axis=1)))
                confidence=float(np.clip(min(1.,len(chosen)/3.)/(1.+dispersion),0.,1.))
                velocity=confidence*movement+(1.-confidence)*velocity
                box=box+np.tile(velocity,2)
            predicted=(box[:2]+box[2:])*.5
            target=(current[key][:2]+current[key][2:])*.5
            errors.append(float(np.linalg.norm(predicted-target)))
    return np.asarray(errors,np.float64)


def main():
    paths=dict(metadata=SOURCE/'motion/flyvis_full.json',motion=SOURCE/'motion/flyvis_full.npz',
        alignment=SOURCE/'alignment.json',truth=ROOT/'data/bdd100k/labels/box_track_20/val/b1c66a42-6f7d68ca.json')
    expected=json.loads((ORIGINAL/'report.json').read_text())
    assert {k:sha(p) for k,p in paths.items()}==expected['protocol']['source_sha256']
    metadata=json.loads(paths['metadata'].read_text());alignment=json.loads(paths['alignment'].read_text())
    truth=json.loads(paths['truth'].read_text());matches=alignment['matches']
    assert metadata['source']=='flyvis_full' and metadata['flow_gain_in_cache']==1.
    assert metadata.get('geometry_version','legacy_v1')=='legacy_v1'
    assert metadata['fps']==alignment['video']['fps']
    assert len(truth)==len(matches)==alignment['reference_frames']
    assert all(m['reference_index']==i and m['reference_name']==t['name'] for i,(m,t) in enumerate(zip(matches,truth)))
    raw=np.array([m['video_frame_index'] for m in matches])
    assert np.all(np.diff(raw)>0)
    with np.load(paths['motion'],allow_pickle=False) as z:
        points=z['points'].copy();vectors=z['vectors']*expected['protocol']['flow_gain']
    assert vectors.shape==(metadata['frame_count'],len(points),2)
    assert metadata['frame_count']==alignment['video']['decoded_frame_count']
    assert raw.min()>=0 and raw.max()<len(vectors)
    assert np.isfinite(points).all() and np.isfinite(vectors).all() and np.all(vectors[0]==0)
    width,height=alignment['video']['width'],alignment['video']['height']
    # From the actual BoxEye._receptor_centers implementation and _meta.yaml:
    # Torch long conversion truncates these half-integer centres toward zero.
    centers=np.array([(13*(u+v/2),13*v) for u in range(-15,16)
                      for v in range(max(-15,-15-u),min(15,15-u)+1)],dtype=np.int64)
    size=centers.max(0)-centers.min(0)+1
    assert size.tolist()==[391,391]
    indices=(centers+size//2).astype(np.float32)
    old=np.column_stack((indices[:,1]/390*width,indices[:,0]/390*height)).astype(np.float32)
    assert np.array_equal(old,points), 'Cached point order or eye geometry differs'
    direct=np.column_stack(((indices[:,1]+.5)/391*width-.5,
                            (indices[:,0]+.5)/391*height-.5)).astype(np.float32)
    inverted=points.astype(np.float64)/[width,height]*390
    from_inverse=((inverted+.5)/391*[width,height]-.5).astype(np.float32)
    values={};seconds={}
    for label,p in [('legacy',points),('root_inverse',from_inverse),('direct_indices',direct)]:
        started=time.monotonic();values[label]=replay(truth,matches,p,vectors);seconds[label]=time.monotonic()-started
    np.savez_compressed(OUT/'endpoint_errors.npz',**values)
    result=dict(passed_review=True,source_hashes={k:sha(p) for k,p in paths.items()},
        metadata_and_index_checks_passed=True,reference_count=len(truth),raw_indices=[int(raw.min()),int(raw.max())],
        raw_frames=len(vectors),eye_size=size.tolist(),legacy_reconstructed_exact=True,
        root_vs_direct_coordinates_max_abs=float(np.abs(direct-from_inverse).max()),
        root_vs_direct_differing_coordinate_count=int(np.count_nonzero(direct!=from_inverse)),
        samples={k:len(v) for k,v in values.items()},means={k:float(v.mean()) for k,v in values.items()},
        root_original_means={k:expected[k] for k in ('old_mean_endpoint_error_pixels','new_mean_endpoint_error_pixels')},
        direct_delta=float(values['direct_indices'].mean()-values['legacy'].mean()),
        independent_matches_reported_old=float(values['legacy'].mean())==expected['old_mean_endpoint_error_pixels'],
        independent_matches_reported_new=float(values['root_inverse'].mean())==expected['new_mean_endpoint_error_pixels'],
        root_vs_direct_error_max_abs=float(np.abs(values['direct_indices']-values['root_inverse']).max()),seconds=seconds,
        script_sha256=sha(__file__),source_references={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
            (ROOT/'tools/flyvis-runtime/flyvis/datasets/rendering/eye.py',ROOT/'runs/flyvis/results/flow/0000/000/_meta.yaml',
             ROOT/'scripts/run_bdd_video_rate_experiment.py',ROOT/'tracking/flyvis_tracker.py')},
        limitations='Retrospective single-video common-object interval diagnostic, not HOTA or statistical evidence of general harm. Uses saved alignment rather than recomputing image matches. Tests a 391x391 full-eye mapping only. Gain is fixed beforehand; no fitting. Root inverse has minor float32 roundtrip differences, so direct renderer-index replay is reported separately.')
    assert result['independent_matches_reported_old'] and result['independent_matches_reported_new']
    assert all(sha(p)==expected['protocol']['source_sha256'][k] for k,p in paths.items())
    (OUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
