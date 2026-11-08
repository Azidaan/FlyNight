"""Reuse exact retinal views and render only absent own-track causal inputs."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from types import SimpleNamespace
from research_singleton_verify_20261108.common import *


def main(obs):
    import numpy as np,cv2,torch
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.foveated_eyes import EyeBank
    from tracking.retina import make_retina
    phase('input_phase.json');inventory=read(RUN/'inventory_recovery1.json');torch.set_num_threads(2);activate_flyvis(device='cpu')
    from flyvis.datasets.rendering import BoxEye
    eye=BoxEye(extent=15,kernel_size=13);g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    centers=eye.receptor_centers.detach().cpu().numpy();assert eye.min_frame_size.tolist()==[g['eye_size']]*2
    assert np.array_equal(np.column_stack([centers[:,1]+g['eye_size']//2,centers[:,0]+g['eye_size']//2]),g['points'])
    bank=SimpleNamespace(torch=torch,eye_size=g['eye_size'],eye=eye,estimator=SimpleNamespace(flyvis=SimpleNamespace(device=torch.device('cpu'))))
    grouped=defaultdict(dict)
    for key,r in inventory['requests'].items():grouped[r['video']][key]=r
    all_requests={};manifest={};totals=Counter()
    for number,(video,requests) in enumerate(grouped.items()):
        folder=RUN/'retinal';folder.mkdir(exist_ok=True);marker=folder/(video+'.json');npz=folder/(video+'.npz')
        if marker.exists():
            item=read(marker);assert sha(npz)==item['npz_sha256'];all_requests.update(item['requests']);manifest[video]=sha(marker);totals.update(item['counts']);continue
        old=read(BASE/'retinal'/(video+'.json'));oldnpz=BASE/'retinal'/(video+'.npz');assert sha(oldnpz)==old['npz_sha256']
        archive=ROOT/f'runs/z13_data/{video}.npz';assert sha(archive)==inventory['sources'][archive.relative_to(ROOT).as_posix()]
        values={};witness=None;witness_value=None
        with np.load(archive,allow_pickle=False) as z:
            for f,c,h in zip(z['fr_frame'],z['fr_crop'],z['fr_hex']):
                point=(int(f),tuple(map(int,c)))
                if point in values:assert np.array_equal(values[point],h)
                else:values[point]=h.copy()
            witness=min(values);witness_value=values[witness].copy()
        with np.load(oldnpz,allow_pickle=False) as z:
            for f,c,h in zip(z['frames'],z['crops'],z['hex']):
                point=(int(f),tuple(map(int,c)))
                if point in values:assert np.array_equal(values[point],h)
                else:values[point]=h.copy()
        required=set()
        for r in requests.values():
            if 'reuse_parent_sha256' in r:continue
            seq=r['sequence'][-1:] if 'reuse_current_uid' in r else r['sequence']
            required.update((int(f),tuple(c)) for f,c in seq)
        missing=defaultdict(list)
        for point in sorted(required-values.keys()):missing[point[0]].append(point[1])
        movie=ROOT/f'data/bdd100k/videos/train/{video}.mov';movie_hash=None;error=0
        if missing:
            fps=float(read(ROOT/f'runs/train_cache/{video}/motion/flyvis_tiled.json')['fps']);retina=make_retina('lamina',fps,blur_downscale=4)
            movie_hash=sha(movie);render={f:list(c) for f,c in missing.items()}
            if witness[1] not in render.setdefault(witness[0],[]):render[witness[0]].append(witness[1])
            cap=cv2.VideoCapture(str(movie));assert cap.isOpened()
            for frame in range(max(render)+1):
                ok,image=cap.read();assert ok,(video,frame)
                processed=retina.process(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
                if frame in render:
                    batch=render[frame]
                    with torch.inference_mode():result=EyeBank._render(bank,processed,batch)[:,0].cpu().numpy().reshape(len(batch),721).astype(np.float16)
                    for crop,value in zip(batch,result):
                        point=(frame,tuple(crop))
                        if point==witness:
                            error=float(np.max(np.abs(value.astype(float)-witness_value)));assert np.array_equal(value,witness_value),(video,error)
                        if point in required:values[point]=value
                if frame%200==0:status('missing_own_retinal_views',video=video,frame=frame,total=max(render)+1,guard=obs.read())
            cap.release();assert sha(movie)==movie_hash
        keys=sorted(required);index={v:i for i,v in enumerate(keys)}
        updated={k:dict(r) for k,r in requests.items()}
        for r in updated.values():
            if 'reuse_parent_sha256' not in r:
                seq=r['sequence'][-1:] if 'reuse_current_uid' in r else r['sequence']
                r['input_indices']=[index[(int(f),tuple(c))] for f,c in seq]
        with npz.open('xb') as f:np.savez_compressed(f,frames=np.array([p[0] for p in keys]),crops=np.array([p[1] for p in keys]).reshape(-1,4),
            hex=np.stack([values[p] for p in keys]) if keys else np.empty((0,721),np.float16))
        counts=dict(retinal_views=len(keys),new_views=sum(map(len,missing.values())),reused_views=len(keys)-sum(map(len,missing.values())))
        item=dict(passed=True,requests=updated,counts=counts,npz_sha256=sha(npz),old_npz_sha256=old['npz_sha256'],movie_sha256=movie_hash,witness_max_error=error)
        write(marker,item,new=True);all_requests.update(updated);manifest[video]=sha(marker);totals.update(counts)
        status('own_retinal_video_complete',complete=number+1,total=len(grouped),video=video,counts=dict(totals),guard=obs.read())
    assert not torch.cuda.is_initialized();phase('input_phase.json')
    write(RUN/'input_complete.json',dict(passed=True,requests=all_requests,videos=manifest,counts=dict(totals)),new=True)


if __name__=='__main__':main(guard())
