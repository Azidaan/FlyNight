"""Recover only missing representations; reuse original current and reference maps."""
import os,sys,time
from collections import defaultdict
from pathlib import Path
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def interpolate_plan(points,locations):
    import numpy as np
    from scipy.spatial import Delaunay
    tri=Delaunay(points);simp=tri.find_simplex(locations);valid=simp>=0
    vertices=np.zeros((len(locations),3),int);weights=np.zeros((len(locations),3))
    vertices[valid]=tri.simplices[simp[valid]]
    affine=tri.transform[simp[valid]]
    xy=np.einsum('nij,nj->ni',affine[:,:2],locations[valid]-affine[:,2])
    weights[valid]=np.column_stack([xy,1-xy.sum(1)])
    return vertices,weights,valid


def main(obs):
    import numpy as np,torch
    from scripts.z13_target_model import build_arm,receptor_features_rich
    from research_spatial_identity_20261107.extract_reference import image_features
    from tracking.z13_readout import HexGeometry
    from tracking.gpu_lock import GpuLock
    verify_phase('feature_phase.json');inputs=read(RUN/'input_complete.json');assert inputs['passed']
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.array(g['points'],np.float32)
    geom=HexGeometry(points,g['eye_size'])
    c_norm=read(ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json')
    r_norm=read(ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json')
    cm=np.array(c_norm['mean'])[:34];cs=np.array(c_norm['sd'])[:34]
    rm=np.array(r_norm['mean'])[:34];rs=np.array(r_norm['sd'])[:34]
    # UID lookup reads compressed identifier arrays only, not already completed maps.
    uid_index={};cache_base=ROOT/'runs/z13_spatial_set_readout_20261025/recovery2/features'
    master=read(cache_base/'report.json')['manifest'];verified_chunks=set()
    for split in ['fit84','fit20','train26']:
        for path in sorted((cache_base/'C'/split).glob('chunk_*.npz')):
            with np.load(path,allow_pickle=False) as z:uids=z['uid']
            for i,u in enumerate(uids):uid_index[int(u)]=(path,i)
    old_reference_manifest=read(ROOT/'runs/spatial_identity_maps_20261107/features_completed.json')['files']
    requests=inputs['requests'];by_video=defaultdict(list);motion_sources={};motion_checks={}
    convention_sources={n:sha(ROOT/n) for n in ['tracking/classical_motion.py','scripts/run_bdd_video_rate_experiment.py']}
    for key,r in requests.items():by_video[r['video']].append(key)
    folder=RUN/'features';folder.mkdir(exist_ok=True);manifest={};counts=defaultdict(int);first_new=None
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        _,_,backbone=build_arm('C',torch,None)
        weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
        for number,(video,keys) in enumerate(by_video.items()):
            retinal=RUN/'retinal'/(video+'.npz');item=read(RUN/'retinal'/(video+'.json'))
            assert sha(retinal)==item['npz_sha256']
            with np.load(retinal,allow_pickle=False) as z:hx_all=z['hex']
            motion=ROOT/f'runs/train_cache/{video}/motion/dis_quarter.npz';motion_meta=read(motion.with_suffix('.json'))
            assert motion_meta['source']=='dis_quarter' and motion_meta.get('flow_gain_in_cache',1)==1
            motion_hash=sha(motion);motion_sources[motion.relative_to(ROOT).as_posix()]=motion_hash
            meta_hash=sha(motion.with_suffix('.json'));motion_sources[motion.with_suffix('.json').relative_to(ROOT).as_posix()]=meta_hash
            with np.load(motion,allow_pickle=False) as z:motion_points=z['points'];vectors=z['vectors']
            assert motion_points.shape[1]==2 and vectors.shape[1:]==(len(motion_points),2)
            assert len(vectors)==motion_meta['frame_count'] and np.isfinite(vectors).all() and np.count_nonzero(vectors[0])==0
            source_fps=read(ROOT/f'runs/train_cache/{video}/motion/flyvis_tiled.json')['fps']
            assert np.isclose(motion_meta['fps'],source_fps,rtol=1e-9,atol=0)
            # The historical quarter-resolution DIS grid uses source pixel coordinates.
            offset=0 if motion_meta.get('geometry_version','legacy_v1')=='legacy_v1' else 1.5
            xx,yy=np.meshgrid(np.arange(4,320,8)*4+offset,np.arange(4,180,8)*4+offset)
            expected_points=np.column_stack([xx.ravel(),yy.ravel()]);assert np.array_equal(motion_points,expected_points)
            probes=np.array([[100.,100.],[640.,360.],[1100.,600.]])
            pv,pw,valid=interpolate_plan(motion_points,probes);assert valid.all()
            linear=np.column_stack([.01*motion_points[:,0]-.002*motion_points[:,1]+1,.004*motion_points[:,0]+.02*motion_points[:,1]-2])
            expected=np.column_stack([.01*probes[:,0]-.002*probes[:,1]+1,.004*probes[:,0]+.02*probes[:,1]-2])
            error=float(np.max(np.abs(np.sum(linear[pv]*pw[:,:,None],axis=1)-expected)));assert error<2e-6
            motion_checks[video]=dict(passed=True,frame_count=len(vectors),fps=source_fps,geometry=motion_meta.get('geometry_version','legacy_v1'),
                analytic_interpolation_max_error=error,historical_video_identity_present='video_identity' in motion_meta)
            grouped=defaultdict(list);cache_chunks={};pending={}
            for key in keys:
                dest=folder/(key+'.npz')
                if dest.exists():manifest[key]=sha(dest);counts['resumed']+=1;continue
                r=requests[key];frame,crop=r['sequence'][-1]
                crop=np.array(crop,float);locations=crop[:2]+points*(crop[2:]-crop[:2])/g['eye_size']
                vi,vw,dv=interpolate_plan(motion_points,locations)
                assert 0<=frame<len(vectors),(video,frame)
                flow=np.sum(vectors[frame][vi]*vw[:,:,None],axis=1).astype(np.float32);flow[~dv]=0
                if 'reuse_reference' in r:
                    assert sha(ROOT/r['reuse_reference'])==old_reference_manifest[Path(r['reuse_reference']).name]
                    with np.load(ROOT/r['reuse_reference'],allow_pickle=False) as z:
                        values=dict(C=z['fly_ref'],R1=z['r1_ref'],IMAGE=z['image_ref'])
                    counts['reference_reused']+=1
                elif r.get('reuse_current_uid') in uid_index:
                    uid=r['reuse_current_uid'];path,index=uid_index[uid]
                    if path not in cache_chunks:
                        other=Path(str(path).replace('/C/','/R1/').replace('\\C\\','\\R1\\'))
                        for source in [path,other]:
                            marker=source.with_suffix('.json');name=marker.relative_to(ROOT).as_posix()
                            assert sha(marker)==master[name] and sha(source)==read(marker)['sha256']
                        with np.load(path,allow_pickle=False) as z:cu=z['uid'];cf=z['full'][:,:,4:12]
                        with np.load(other,allow_pickle=False) as z:ru=z['uid'];rf=z['full'][:,:,:34]
                        assert np.array_equal(cu,ru);cache_chunks[path]=(cf,rf)
                    cf,rf=cache_chunks[path]
                    last=hx_all[r['input_indices'][-1]].astype(float)
                    values=dict(C=cf[index],R1=rf[index],IMAGE=image_features(last,points,geom.neighbors,crop,g['eye_size']))
                    counts['current_reused']+=1
                else:
                    assert r.get('reuse_current_uid') is None,('Missing cached UID',r.get('reuse_current_uid'))
                    sequence=hx_all[r['input_indices']].astype(np.float32)
                    grouped[len(sequence)].append((key,sequence))
                    pending[key]=dict(DIS=flow,DIS_VALID=dv,frame=frame,crop=crop)
                    continue
                with dest.open('xb') as stream:np.savez_compressed(stream,**values,DIS=flow,DIS_VALID=dv)
                manifest[key]=sha(dest)
            for length,batches in grouped.items():
                for start in range(0,len(batches),4):
                    batch=batches[start:start+4];sequence=np.stack([v for _,v in batch])
                    tensor=torch.from_numpy(sequence).to(backbone.dev);state=backbone.init(tensor[:,:1])
                    for t in range(1,length):state,feature=backbone.step(state,tensor[:,t-1],tensor[:,t])
                    arr=((feature.cpu().numpy().transpose(0,2,1).astype(float)-cm)/cs).astype(np.float32)[:,:,4:12]
                    rich=receptor_features_rich(sequence.astype(np.float64))[-1].transpose(0,2,1)
                    r1=((rich-rm)/rs).astype(np.float32)
                    for i,(key,hx) in enumerate(batch):
                        p=pending[key];image=image_features(hx[-1].astype(float),points,geom.neighbors,p['crop'],g['eye_size'])
                        dest=folder/(key+'.npz')
                        with dest.open('xb') as stream:np.savez_compressed(stream,C=arr[i],R1=r1[i],IMAGE=image,DIS=p['DIS'],DIS_VALID=p['DIS_VALID'])
                        manifest[key]=sha(dest);counts['new_core_sequences']+=1
                        if first_new is None:
                            # Audit a newly recovered sequence in an independent batch shape.
                            h=torch.from_numpy(hx[None]).to(backbone.dev);st=backbone.init(h[:,:1])
                            for t in range(1,len(hx)):st,ff=backbone.step(st,h[:,t-1],h[:,t])
                            check=((ff[0].cpu().numpy().T.astype(float)-cm)/cs)[:,4:12]
                            error=float(np.max(np.abs(check-arr[i])));assert error<=.002
                            first_new=dict(key=key,max_normalized_map_error=error,tolerance=.002,passed=True)
                            write(RUN/'first_new_feature_check.json',first_new,new=True)
                            del h,st,ff
                    del tensor,state,feature,arr,rich,r1
            assert sha(motion)==motion_hash and sha(motion.with_suffix('.json'))==meta_hash
            status('feature_recovery',complete=number+1,total=len(by_video),video=video,files=len(manifest),counts=dict(counts),guard=obs.read())
        for name,value in backbone.net.named_parameters():assert torch.equal(value,weights[name]),name
    verify_phase('feature_phase.json')
    for name,h in convention_sources.items():assert sha(ROOT/name)==h
    write(RUN/'features_complete.json',dict(passed=True,files=manifest,counts=dict(counts),requests=len(requests),first_new_check=first_new,motion_sources=motion_sources,
        motion_checks=motion_checks,flow_convention=dict(source_checksums_now=convention_sources,
            units='Source pixels per adjacent video frame; x right, y down; vectors[f] from f-1 to f, first zero.',
            provenance_limit='Legacy caches without video_identity lack a contemporaneous cryptographic movie binding; present checks do not invent it retroactively.')),
        new=True)


if __name__=='__main__':main(guard())
