"""CPU-only cross-check of new pure ROI inputs against the independent gate."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_region_assignment_20261025 import RUN,GATE,queries_and_crops,geometry
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from tracking.z13_region_assignment import regions


def main():
    queries,_=queries_and_crops();g=geometry();p=read(RUN/'protocol.json');count=0
    for v in p['evaluate26']:
        path=GATE/'videos'/(v+'.json');assert sha(path)==read(GATE/'report.json')['video_sha256'][v]
        rows={x['uid']:x for x in read(path)['rows']}
        with np.load(ROOT/f'runs/train_cache/{v}/detections/raw_video.npz') as z:d={k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
        for q in queries[v]:
            row=rows[q['uid']];l=q['label'];ix=(d['frame_idx']==q['frame'])&(d['conf']>=.25)
            choices,_,_=regions(g,l['crop'],l['pred'],d['xyxy'][ix],d['conf'][ix],d['cls'][ix],row['regions']['track_class'])
            assert choices==row['regions']['choices'][1:]
            assert len(choices)==len(row['arms']['regions'])-1;count+=1
    assert count==8326
    write(RUN/'roi_preflight.json',dict(at=now().isoformat(),rows=count,all_choices_equal_independent_gate=True,known_unknown_labels_not_network_inputs=True,source_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
