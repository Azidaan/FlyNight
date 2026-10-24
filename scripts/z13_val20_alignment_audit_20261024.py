"""Read-only audit of the stopped input guard; never writes an accepted alignment."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now


def main():
    import cv2,numpy as np
    from tracking.video_alignment import decode_compact_video,load_compact_images
    cv2.setNumThreads(2);v='b23b2649-8349d2a1';run=ROOT/'runs/z13_val20_matched_20261024/inputs';p=read(run/'protocol.json');assert v==p['videos'][1]
    vf,meta=decode_compact_video(ROOT/f'data/bdd100k/videos/val/{v}.mov');images=sorted((ROOT/f'data/bdd100k/images/track/val/{v}').glob('*.jpg'));refs=load_compact_images(images);rows=[];start=time.monotonic()
    for i,ref in enumerate(refs):
        expected=round(i*(len(vf)-1)/max(1,len(refs)-1));lo=max(0,expected-24);hi=min(len(vf),expected+25);cost=np.array([cv2.absdiff(ref,x).mean() for x in vf[lo:hi]]);order=np.argsort(cost);j=int(order[0]);rows.append(dict(i=i,name=images[i].name,expected=expected,best=lo+j,mae=float(cost[j]),second_mae=float(cost[order[1]]),image_sha256=sha(images[i])))
    bad=[i for i in range(1,len(rows)) if rows[i]['best']<=rows[i-1]['best']]
    details=[]
    for i in sorted(set(j for k in bad for j in (k-1,k))):
        cost=np.array([cv2.absdiff(refs[i],x).mean() for x in vf]);order=np.argsort(cost);details.append(dict(**rows[i],global_best=int(order[0]),global_mae=float(cost[order[0]]),equal_best_indices=np.flatnonzero(cost==cost.min()).tolist(),global_second_mae=float(cost[order[1]])))
    write(run/'alignment_failure_audit.json',dict(at=now().isoformat(),video=v,metadata=meta,reference_count=len(refs),bad_positions=bad,local_rows=rows,details=details,seconds=time.monotonic()-start,accepted_alignment_written=False,scientific_thresholds_changed=False));print(dict(bad=bad,details=details))


if __name__=='__main__':main()
