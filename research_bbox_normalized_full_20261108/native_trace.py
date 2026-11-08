"""Derive the new alignment only; never repeat the already recovered native run."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_normalized_full_20261108.common import *


def main(obs):
    import numpy as np,shutil
    from tracking.bbox_singleton_gate import proposals
    from tracking.box_normalized_correspondence import box_transform
    from tracking.own_template_verification import correspondence
    from tracking.spatial_identity_assignment import frame_points
    from research_identity_assignment_run_20261108.full_planner import dis_domain
    full_phase();p=read(FULL/'protocol.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');receipts={}
    for video,entry in p['videos'].items():
        data=read(INVENTORY/'native'/(video+'.json'))
        with np.load(INVENTORY/'native'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/entry['raw_array'],allow_pickle=False) as z:assert np.array_equal(native[:,:8],z['outputs']) and int(z['frame_count'])==data['frames']
        events=[]
        for d in data['events']:
            edges=[]
            for e in proposals(d,cap=2,threshold=.2,bidirectional=False):
                if np.float32(e['cost']-.025)>.8:continue
                c=next(c for r in d['rows'] if r['old_tid']==e['old_tid'] for c in r['candidates'] if c['new_tid']==e['new_tid']);row=e['row_data'];transform=box_transform(row['history'][-1]['box'],c['box']);cv=dis_domain(frame_points(g['points'],row['current_crop'],g['eye_size']));rv=dis_domain(frame_points(g['points'],row['history'][-1]['crop'],g['eye_size']));pairs=correspondence(transform,row['history'],row['current_crop'],c['box'],g,cv,rv);unavailable='reference_sequence_shorter_than_two_observations' if len(row['history'])<2 else 'fewer_than_three_reference_cells' if pairs is None else None
                edges.append(dict(e,frame=d['frame'],box=c['box'],transform=transform,support=0 if unavailable else len(pairs['unique']),unavailable=unavailable))
            if edges:events.append(dict(frame=d['frame'],newly_confirmed=d['newly_confirmed'],edges=edges))
        folder=FULL/'native_trace';folder.mkdir(exist_ok=True);path=folder/(video+'.npz');shutil.copyfile(INVENTORY/'native'/path.name,path);meta=folder/(video+'.json');write(meta,dict(passed=True,video=video,frames=data['frames'],events=events,complete_cached_B0_exact=True,new_native_tracker_calls=0),new=True);receipts[video]=dict(meta_sha256=sha(meta),npz_sha256=sha(path));status('normalized_full_missing_alignment',video=video,complete=len(receipts),total=len(p['videos']),edges=sum(len(d['edges']) for d in events),guard=obs.read())
    full_phase();write(FULL/'native_trace_complete.json',dict(passed=True,videos=receipts,tracker_detector_core_calls=0),new=True)


if __name__=='__main__':main(guard())
