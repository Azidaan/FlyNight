"""Same frozen maps and controls; no recomputation of ready causal sequences."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from research_singleton_verify_20261108.common import *


def main(obs):
    import numpy as np,torch,shutil
    from tracking.gpu_lock import GpuLock
    from tracking.flyvis_adapter import activate_flyvis
    from scripts.z13_target_model import build_arm,receptor_features_rich
    from research_spatial_identity_20261107.extract_reference import image_features
    from research_identity_assignment_run_20261108.extract_features import interpolate_plan
    from tracking.z13_readout import HexGeometry
    phase('feature_phase.json');inputs=read(RUN/'input_complete.json');assert inputs['passed'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    points=np.array(g['points'],np.float32);geometry=HexGeometry(points,g['eye_size']);stats={}
    for name in ['C','R1']:
        norm=read(ROOT/f'runs/z13_target_center_dev_20261018/{name}/normalization.json');stats[name]=(np.array(norm['mean'])[:34],np.array(norm['sd'])[:34])
    cache=ROOT/'runs/z13_spatial_set_readout_20261025/recovery2/features';master=read(cache/'report.json')['manifest'];uid_index={}
    for split in ['fit84','fit20','train26']:
        for path in sorted((cache/'C'/split).glob('chunk_*.npz')):
            with np.load(path,allow_pickle=False) as z:uids=z['uid']
            for i,u in enumerate(uids):uid_index[int(u)]=(path,i)
    grouped=defaultdict(dict)
    for key,r in inputs['requests'].items():grouped[r['video']][key]=r
    folder=RUN/'features';folder.mkdir(exist_ok=True);counts=Counter();manifest={};reused={};first=None;backbone=None;weights=None
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        for number,(video,requests) in enumerate(grouped.items()):
            retinal=RUN/'retinal'/(video+'.npz');assert sha(retinal)==read(RUN/'retinal'/(video+'.json'))['npz_sha256']
            with np.load(retinal,allow_pickle=False) as z:hexes=z['hex']
            motion=ROOT/f'runs/train_cache/{video}/motion/dis_quarter.npz';motion_hash=sha(motion);motion_meta=read(motion.with_suffix('.json'))
            assert motion_meta['source']=='dis_quarter' and motion_meta.get('flow_gain_in_cache',1)==1
            with np.load(motion,allow_pickle=False) as z:motion_points=z['points'];vectors=z['vectors']
            assert not np.count_nonzero(vectors[0]) and np.isfinite(vectors).all();chunks={};pending=defaultdict(list)
            for key,r in requests.items():
                dest=folder/(key+'.npz')
                if dest.exists():manifest[key]=sha(dest);counts['resumed']+=1;continue
                if 'reuse_parent_sha256' in r:
                    source=BASE/'features'/(key+'.npz');assert sha(source)==r['reuse_parent_sha256'];shutil.copyfile(source,dest)
                    manifest[key]=sha(dest);reused[key]=dict(source=source.relative_to(ROOT).as_posix(),sha256=sha(source));counts['parent_reused']+=1;continue
                frame,crop=r['sequence'][-1];crop=np.array(crop,float);locations=crop[:2]+points*(crop[2:]-crop[:2])/g['eye_size']
                vi,vw,valid=interpolate_plan(motion_points,locations);flow=np.sum(vectors[frame][vi]*vw[:,:,None],axis=1).astype(np.float32);flow[~valid]=0
                if 'reuse_current_uid' in r:
                    path,index=uid_index[r['reuse_current_uid']]
                    if path not in chunks:
                        other=Path(str(path).replace('/C/','/R1/').replace('\\C\\','\\R1\\'))
                        for source in [path,other]:
                            marker=source.with_suffix('.json');assert sha(marker)==master[marker.relative_to(ROOT).as_posix()] and sha(source)==read(marker)['sha256']
                        with np.load(path,allow_pickle=False) as z:cu=z['uid'];cf=z['full'][:,:,4:12]
                        with np.load(other,allow_pickle=False) as z:ru=z['uid'];rf=z['full'][:,:,:34]
                        assert np.array_equal(cu,ru);chunks[path]=(cf,rf)
                    cf,rf=chunks[path];image=image_features(hexes[r['input_indices'][-1]].astype(float),points,geometry.neighbors,crop,g['eye_size'])
                    with dest.open('xb') as f:np.savez_compressed(f,C=cf[index],R1=rf[index],IMAGE=image,DIS=flow,DIS_VALID=valid)
                    manifest[key]=sha(dest);reused[key]=dict(source=path.relative_to(ROOT).as_posix(),uid=r['reuse_current_uid'],index=index,sha256=sha(path));counts['current_reused']+=1
                else:
                    hx=hexes[r['input_indices']].astype(np.float32);pending[len(hx)].append((key,hx,flow,valid,crop))
            for length,items in pending.items():
                if backbone is None:
                    assert torch.cuda.is_available();activate_flyvis(device='cuda');_,_,backbone=build_arm('C',torch,None)
                    assert backbone.dev.type=='cuda' and not backbone.net.training
                    weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
                for start in range(0,len(items),4):
                    batch=items[start:start+4];hx=np.stack([r[1] for r in batch]);x=torch.from_numpy(hx).to(backbone.dev);state=backbone.init(x[:,:1])
                    for t in range(1,length):state,features=backbone.step(state,x[:,t-1],x[:,t])
                    cm,cs=stats['C'];rm,rs=stats['R1'];maps=(((features.cpu().numpy().transpose(0,2,1).astype(float)-cm)/cs).astype(np.float32))[:,:,4:12]
                    rich=receptor_features_rich(hx.astype(np.float64))[-1].transpose(0,2,1);r1=((rich-rm)/rs).astype(np.float32)
                    for k,(key,seq,flow,valid,crop) in enumerate(batch):
                        image=image_features(seq[-1].astype(float),points,geometry.neighbors,crop,g['eye_size']);dest=folder/(key+'.npz')
                        assert np.isfinite(maps[k]).all() and np.isfinite(r1[k]).all()
                        with dest.open('xb') as f:np.savez_compressed(f,C=maps[k],R1=r1[k],IMAGE=image,DIS=flow,DIS_VALID=valid)
                        manifest[key]=sha(dest);counts['new_core_sequences']+=1
                        if first is None:
                            h=torch.from_numpy(seq[None]).to(backbone.dev);st=backbone.init(h[:,:1])
                            for t in range(1,len(seq)):st,ff=backbone.step(st,h[:,t-1],h[:,t])
                            check=((ff[0].cpu().numpy().T.astype(float)-cm)/cs)[:,4:12];error=float(np.max(np.abs(check-maps[k])));assert error<=.002
                            first=dict(passed=True,key=key,max_error=error,device=str(backbone.dev));write(RUN/'first_new_feature_check.json',first,new=True)
                    del x,state,features,hx,maps,rich,r1
            assert sha(motion)==motion_hash;status('own_map_video_complete',complete=number+1,total=len(grouped),video=video,counts=dict(counts),guard=obs.read())
        if backbone is not None:assert all(torch.equal(v,weights[k]) for k,v in backbone.net.named_parameters())
    phase('feature_phase.json');write(RUN/'features_complete.json',dict(passed=True,files=manifest,reused=reused,counts=dict(counts),first_new=first),new=True)


if __name__=='__main__':main(guard())
