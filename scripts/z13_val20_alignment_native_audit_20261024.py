"""Check whether compact thumbnails explain a reversed pair; no accepted mapping."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now


def main():
    import cv2,numpy as np
    cv2.setNumThreads(2);run=ROOT/'runs/z13_val20_matched_20261024/inputs';p=read(run/'alignment_failure_audit.json');v=p['video'];images=ROOT/f'data/bdd100k/images/track/val/{v}';refs=[(x,cv2.imread(str(images/x['name']))) for x in p['details']];cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/val/{v}.mov'));costs=[[] for _ in refs];f=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        for i,(r,ref) in enumerate(refs):
            if abs(f-r['expected'])<=24:
                assert ref.shape==frame.shape;costs[i].append((float(cv2.absdiff(ref,frame).mean()),f))
        f+=1
    cap.release();rows=[]
    for (r,ref),values in zip(refs,costs):
        values.sort();rows.append(dict(reference=r['i'],native_best=values[0][1],native_mae=values[0][0],top10=values[:10],compact_best=r['best']))
    write(run/'alignment_native_audit.json',dict(at=now().isoformat(),rows=rows,accepted_alignment_written=False,original24_window_unchanged=True));print(rows)


if __name__=='__main__':main()
