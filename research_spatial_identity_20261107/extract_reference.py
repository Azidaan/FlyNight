"""Recover missing reference maps from saved retinal prefixes, reusing current maps."""
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_spatial_identity_20261107.run_common import RUN, read, sha, write, status, verified_protocol, guard


def image_features(values, points, neighbors, crop, eye_size):
    import numpy as np
    result = np.zeros((len(points), 3), np.float64)
    result[:, 0] = values
    for i in range(len(points)):
        nb = np.unique(neighbors[i])
        nb = nb[nb != i]
        delta = points[nb] - points[i]
        result[i, 1:] = np.linalg.lstsq(delta, values[nb]-values[i], rcond=None)[0]
    result[:, 1] *= eye_size / (crop[2]-crop[0])
    result[:, 2] *= eye_size / (crop[3]-crop[1])
    return result.astype(np.float32)


def main(obs):
    import numpy as np
    import torch
    from scripts.z13_target_model import build_arm, receptor_features_rich
    from tracking.z13_readout import HexGeometry
    from tracking.gpu_lock import GpuLock

    p = verified_protocol()
    assert read(RUN / "tracker_export_completed.json")["passed"]
    phase = read(RUN / "features_phase_protocol.json")
    for name, h in phase['source_sha256'].items():
        assert sha(ROOT/name) == h
    rows = read(ROOT / "research_spatial_identity_20261107/readiness_rows.json")
    g = read(ROOT / "runs/z13_local_peak_20261021/geometry.json")
    geom = HexGeometry(np.asarray(g['points'],np.float32),g['eye_size'])
    normal = read(ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json')
    mu, sd = np.asarray(normal['mean'])[:34], np.asarray(normal['sd'])[:34]
    nr = read(ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json')
    rmu, rsd = np.asarray(nr['mean'])[:34], np.asarray(nr['sd'])[:34]
    by_video = {}
    for row in rows:
        by_video.setdefault(row['video'],[]).append(row)
    # The first fixed fit84 episode is validated before any evaluation map extraction.
    first = min((r for r in rows if r['split']=='fit84'),key=lambda r:(r['video'],r['episode'],r['uid']))
    order = [first['video']] + sorted(set(by_video)-{first['video']})
    folder = RUN/'features'; folder.mkdir(exist_ok=True)
    manifest = {}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with GpuLock('flyvis_fovea_gpu'), torch.no_grad():
        _,_,bb = build_arm('C',torch,None)
        assert not bb.net.training
        initial = {name: value.detach().clone() for name,value in bb.net.named_parameters()}
        gate = RUN/'reference_first_case.json'
        if not gate.exists():
            path = ROOT/'runs/z13_data'/(first['video']+'.npz')
            with np.load(path,allow_pickle=False) as z:
                begin = int(z['ep_offset'][first['episode']])
                hx = z['fr_hex'][begin:first['current_array_index']+1].astype(np.float32)
            started = time.perf_counter()
            sequence = torch.from_numpy(hx)[None].to(bb.dev)
            state = bb.init(sequence[:,:1])
            for t in range(1,sequence.shape[1]):
                state, f = bb.step(state,sequence[:,t-1],sequence[:,t])
            actual = ((f[0].cpu().numpy().T.astype(np.float64)-mu)/sd)[:,4:12]
            with np.load(ROOT/first['current_map'][0],allow_pickle=False) as z:
                assert int(z['uid'][first['current_map'][1]])==first['uid']
                expected = z['full'][first['current_map'][1],:,4:12]
            error = float(np.max(np.abs(actual-expected)))
            torch.cuda.synchronize()
            write(gate,dict(uid=first['uid'],video=first['video'],max_normalized_map_error=error,
                            tolerance=.002,passed=error<=.002,seconds=time.perf_counter()-started,
                            gpu_peak_bytes=torch.cuda.max_memory_allocated()),new=True)
            assert error<=.002,("Current-map provenance gate",error)
            del sequence,state,f
        else:
            assert read(gate)['passed']
        for vi,video in enumerate(order):
            cases = by_video[video]
            if all((folder/(str(r['uid'])+'.npz')).exists() for r in cases):
                for r in cases:
                    path=folder/(str(r['uid'])+'.npz');manifest[path.name]=sha(path)
                continue
            tracked = {r['uid']:r for r in read(RUN/'tracker'/(video+'.json'))['rows']}
            with np.load(ROOT/'runs/z13_data'/(video+'.npz'),allow_pickle=False) as z:
                retina=z['fr_hex'];offsets=z['ep_offset'];crops=z['fr_crop']
            episodes = sorted({r['episode'] for r in cases})
            references = {}
            for start in range(0,len(episodes),4):
                batch=episodes[start:start+4]
                prefix=np.stack([retina[int(offsets[e]):int(offsets[e])+4] for e in batch]).astype(np.float32)
                hx=torch.from_numpy(prefix).to(bb.dev);state=bb.init(hx[:,:1])
                for t in range(1,4):state,f=bb.step(state,hx[:,t-1],hx[:,t])
                data=((f.cpu().numpy().transpose(0,2,1).astype(np.float64)-mu)/sd).astype(np.float32)
                for i,e in enumerate(batch):
                    base=int(offsets[e])
                    rich=receptor_features_rich(prefix[i:i+1].astype(np.float64))[3][0].T
                    references[e]=(data[i,:,4:12],((rich-rmu)/rsd).astype(np.float32),
                        image_features(retina[base+3].astype(np.float64),geom.points,geom.neighbors,crops[base+3],geom.eye_size))
                del hx,state,f,data
            needed={r['current_map'][0] for r in cases}
            for current_file in sorted(needed):
                subset=[r for r in cases if r['current_map'][0]==current_file]
                rfile=current_file.replace('/C/','/R1/')
                for source in (current_file,rfile):
                    assert sha(ROOT/source)==read((ROOT/source).with_suffix('.json'))['sha256'],source
                with np.load(ROOT/current_file,allow_pickle=False) as z:
                    cuids=z['uid'];full=z['full']
                with np.load(ROOT/rfile,allow_pickle=False) as z:
                    ruids=z['uid'];rfull=z['full']
                assert np.array_equal(cuids,ruids)
                for r in subset:
                    path=folder/(str(r['uid'])+'.npz')
                    if path.exists():manifest[path.name]=sha(path);continue
                    i=r['current_map'][1];assert int(cuids[i])==r['uid']
                    rec=tracked[r['uid']];e=r['episode'];ref,ref_r,ref_im=references[e]
                    current_image=image_features(retina[r['current_array_index']].astype(np.float64),
                        geom.points,geom.neighbors,rec['current_crop'],geom.eye_size)
                    with path.open('xb') as stream:
                        np.savez_compressed(stream,uid=r['uid'],fly_now=full[i,:,4:12],fly_ref=ref,
                            r1_now=rfull[i,:,:34],r1_ref=ref_r,image_now=current_image,image_ref=ref_im)
                    manifest[path.name]=sha(path)
            status('reference_extraction',complete=len(manifest),total=len(rows),video=video,
                   videos_complete=vi+1,videos_total=len(order),guard=obs.read())
            print('features',vi+1,len(order),len(manifest),flush=True)
        for name,value in bb.net.named_parameters():assert torch.equal(value,initial[name]),name
    # A common fixed image normalization is fitted without identity labels on fit84 only.
    total=0; sums=np.zeros(3);squares=np.zeros(3)
    for r in rows:
        if r['split']!='fit84':continue
        with np.load(folder/(str(r['uid'])+'.npz'),allow_pickle=False) as z:
            values=np.concatenate([z['image_ref'],z['image_now']],axis=0).astype(np.float64)
        total+=len(values);sums+=values.sum(0);squares+=(values*values).sum(0)
    mean=sums/total;std=np.sqrt(np.maximum(squares/total-mean*mean,0))
    write(RUN/'image_normalization.json',dict(mean=mean.tolist(),sd=std.tolist(),samples=total,split='fit84'),new=True)
    verified_protocol()
    write(RUN/'features_completed.json',dict(passed=True,files=manifest,pairs=len(rows),
        reference_episodes=682,checkpoint_sha256=sha(ROOT/'runs/flyvis/results/flow/0000/000/best_chkpt')),
        new=True)


if __name__=='__main__':
    main(guard())
