"""Короткая проверка последовательного чтения против произвольного перехода в видео."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now

def main():
    import cv2,numpy as np
    from tracking.pixel_box_refinement import region
    v='015f23ad-5049df48';folder=RUN/'pixel_box_dataset/train'/v;r=read(folder/'records.json')[0];stored=np.load(folder/'images.npy',mmap_mode='r')[0,1]
    path=ROOT/f'data/bdd100k/videos/train/{v}.mov';cap=cv2.VideoCapture(str(path));sequential=None
    for frame in range(r['frame']+1):ok,image=cap.read();assert ok
    sequential=region(image,r['box']);cap.release()
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame']);ok,image=cap.read();assert ok;seek=region(image,r['box']);cap.release()
    result=dict(updated=now(),video=v,frame=r['frame'],stored_matches_sequential=bool(np.array_equal(stored,sequential)),stored_matches_seek=bool(np.array_equal(stored,seek)),
        sequential_seek_max_difference=int(np.max(abs(sequential.astype(int)-seek.astype(int)))),sequential_seek_mean_difference=float(np.mean(abs(sequential.astype(int)-seek.astype(int)))),
        stored_sha256=__import__('hashlib').sha256(stored.tobytes()).hexdigest(),sequential_sha256=__import__('hashlib').sha256(sequential.tobytes()).hexdigest(),seek_sha256=__import__('hashlib').sha256(seek.tobytes()).hexdigest(),source_sha256=sha(__file__))
    write(RUN/'pixel_box_decode_diagnostic.json',result);print(result)

if __name__=='__main__':main()
