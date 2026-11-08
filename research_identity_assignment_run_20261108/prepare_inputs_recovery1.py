"""Reuse saved retinal inputs; render only previously absent causal views on CPU."""
import os,sys,time
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.export_states import sequence_key


def main(obs):
    import numpy as np,cv2,torch
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.foveated_eyes import EyeBank
    from tracking.retina import make_retina
    assert read(RUN/'export_complete.json')['passed'];verify_phase('input_recovery1_phase.json')
    torch.set_num_threads(2);activate_flyvis(device='cpu')
    from flyvis.datasets.rendering import BoxEye
    eye=BoxEye(extent=15,kernel_size=13);g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    assert eye.min_frame_size.tolist()==[g['eye_size'],g['eye_size']]
    centers=eye.receptor_centers.detach().cpu().numpy()
    assert np.array_equal(np.column_stack([centers[:,1]+g['eye_size']//2,centers[:,0]+g['eye_size']//2]),g['points'])
    bank=SimpleNamespace(torch=torch,eye_size=g['eye_size'],eye=eye,estimator=SimpleNamespace(flyvis=SimpleNamespace(device=torch.device('cpu'))))
    exports=read(RUN/'export_complete.json')['videos'];old_features=read(ROOT/'runs/spatial_identity_maps_20261107/features_completed.json')
    folder=RUN/'retinal';folder.mkdir(exist_ok=True);manifest={};all_requests={};summary=[]
    for number,(video,entry) in enumerate(exports.items()):
        dest=folder/(video+'.json')
        if dest.exists():
            item=read(dest);assert sha(folder/(video+'.npz'))==item['npz_sha256'];all_requests.update(item['requests']);summary.append(item['counts']);manifest[video]=sha(dest);continue
        source=ROOT/entry['path'];assert sha(source)==entry['sha256'];data=read(source)
        required=set()
        for snapshot in data['snapshots'].values():
            for col in snapshot['columns'].values():
                if not col['geometry_valid']:continue
                required.add(col['current_key']);required.update(x['reference_key'] for x in col['items'])
        requests={k:dict(data['requests'][k]) for k in sorted(required)}
        refs={};old_path=ROOT/f'runs/spatial_identity_maps_20261107/tracker/{video}.json'
        if old_path.exists():
            for row in read(old_path)['rows']:
                key=sequence_key(video,[[h['frame'],h['crop']] for h in row['history']])
                file=ROOT/f"runs/spatial_identity_maps_20261107/features/{row['uid']}.npz"
                if file.exists():
                    assert sha(file)==old_features['files'][file.name];refs.setdefault(key,file.relative_to(ROOT).as_posix())
        archived=ROOT/f'runs/z13_data/{video}.npz'
        expected_archive=read(ROOT/'runs/z13_temporal_pair_readiness_20261025/protocol.json')['sha256'][archived.relative_to(ROOT).as_posix()]
        assert sha(archived)==expected_archive
        with np.load(archived,allow_pickle=False) as z:
            fs=z['fr_frame'];crops=z['fr_crop'];hexes=z['fr_hex']
        archive_index={ (int(f),tuple(int(x) for x in c)):i for i,(f,c) in enumerate(zip(fs,crops))}
        meta={r['uid']:r for r in read(ROOT/f'runs/z13_temporal_pair_readiness_20261025/inputs/{video}.json')['rows']}
        needed=set();reused_refs=0;current_bindings=0
        for key,r in requests.items():
            if key in refs:r['reuse_reference']=refs[key];reused_refs+=1;continue
            uid=r['uid'];cache_ok=False
            if uid is not None:
                m=meta[uid];lo=m['state_prefix_start'];hi=m['current_array_index']+1
                saved=[[int(f),list(map(int,c))] for f,c in zip(fs[lo:hi],crops[lo:hi])]
                cache_ok=saved==r['sequence']
            r['reuse_current_uid']=uid if cache_ok else None
            if cache_ok:needed.add((r['sequence'][-1][0],tuple(r['sequence'][-1][1])));current_bindings+=1
            else:needed.update((int(f),tuple(c)) for f,c in r['sequence'])
        values={};missing=defaultdict(list)
        for key in sorted(needed):
            if key in archive_index:values[key]=hexes[archive_index[key]].copy()
            else:missing[key[0]].append(key[1])
        max_error=0.;witness=None
        video_sha256=None
        if missing:
            # A cached witness proves the same preprocessing path before relying on new views.
            witness=min(archive_index);render={f:list(cs) for f,cs in missing.items()}
            if witness[1] not in render.setdefault(witness[0],[]):render[witness[0]].append(witness[1])
            fps=float(read(ROOT/f'runs/train_cache/{video}/motion/flyvis_tiled.json')['fps'])
            retina=make_retina('lamina',fps,blur_downscale=4)
            video_path=ROOT/f'data/bdd100k/videos/train/{video}.mov';video_sha256=sha(video_path)
            cap=cv2.VideoCapture(str(video_path));assert cap.isOpened(),video
            for frame in range(max(render)+1):
                ok,image=cap.read();assert ok,(video,frame)
                processed=retina.process(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
                if frame not in render:continue
                batch=render[frame]
                with torch.inference_mode():v=EyeBank._render(bank,processed,batch)[:,0].cpu().numpy().reshape(len(batch),721).astype(np.float16)
                for crop,a in zip(batch,v):
                    point=(frame,tuple(crop))
                    if point==witness:
                        expected=hexes[archive_index[point]];max_error=float(np.max(np.abs(a.astype(float)-expected.astype(float))))
                        assert np.array_equal(a,expected),('retinal provenance',video,max_error)
                    if point in needed:values[point]=a
            cap.release()
            assert sha(video_path)==video_sha256
        keys=sorted(values);index={k:i for i,k in enumerate(keys)}
        for key,r in requests.items():
            if 'reuse_reference' in r:continue
            if r['reuse_current_uid'] is not None:r['input_indices']=[index[(r['sequence'][-1][0],tuple(r['sequence'][-1][1]))]]
            else:r['input_indices']=[index[(f,tuple(c))] for f,c in r['sequence']]
        npz=folder/(video+'.npz')
        with npz.open('xb') as stream:np.savez_compressed(stream,frames=np.array([k[0] for k in keys]),
            crops=np.array([k[1] for k in keys]).reshape(-1,4),hex=np.stack([values[k] for k in keys]) if keys else np.empty((0,721),np.float16))
        counts=dict(video=video,requests=len(requests),reused_references=reused_refs,current_cache_bindings=current_bindings,
            retinal_views=len(keys),new_rendered_views=sum(len(x) for x in missing.values()),render_witness_max_error=max_error)
        assert sha(archived)==expected_archive
        item=dict(passed=True,requests=requests,counts=counts,npz_sha256=sha(npz),episode_sha256=expected_archive,video_sha256=video_sha256,export_sha256=entry['sha256'])
        write(dest,item,new=True);all_requests.update(requests);summary.append(counts);manifest[video]=sha(dest)
        status('retinal_recovery',complete=number+1,total=len(exports),video=video,counts=counts,guard=obs.read())
    assert not torch.cuda.is_initialized();verify_phase('input_recovery1_phase.json')
    write(RUN/'input_complete.json',dict(passed=True,requests=all_requests,videos=manifest,summary=summary),new=True)


from collections import defaultdict
if __name__=='__main__':main(guard())
