"""Equal-support diagnostic. Each stage writes new artifacts and reuses completed stages."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with (ROOT/path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    with (OUT/name).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def verify():
    protocol = read(OUT/'protocol.json')
    for path, expected in protocol['sources'].items():
        assert sha(path) == expected, path
    return protocol


def get_row(case):
    base = 'singleton_working_domain' if case['split']=='development20' else 'bbox_confirmation_inventory'
    path = f'runs/{base}_20261108/native/{case["video"]}.json'
    event, = [e for e in read(path)['events'] if e['frame']==case['frame']]
    row, = [r for r in event['rows'] if r['old_tid']==case['old']]
    candidate, = [c for c in row['candidates'] if c['forward_iou']>=.2]
    assert candidate['new_tid']==case['new'] and candidate['raw_id']==case['raw_id']
    return row['row_data'], candidate, path


def freeze():
    inventory = read('runs/singleton_car_map_coverage_20261108/inventory.json')
    eligible = [c for c in inventory['cases'] if c['direct'] and c['same_scale'] and c['points']>=3]
    foreign = [c for c in eligible if c['kind']=='foreign']
    assert len(foreign)==1
    own = [c for c in eligible if c['kind']=='own' and c['points']==foreign[0]['points']]
    assert len(own)==1
    cases = foreign+own
    files = ['runs/singleton_car_map_coverage_20261108/inventory.json',
             'runs/singleton_two_map_scores_20261108/scores.json',
             'runs/singleton_two_map_scores_20261108/protocol.json',
             'runs/z13_local_peak_20261021/geometry.json',
             'runs/z13_target_center_dev_20261018/C/normalization.json',
             'runs/flyvis/results/flow/0000/000/best_chkpt',
             'runs/flyvis/results/flow/0000/000/_meta.yaml',
             'runs/flyvis_local_shift_feasibility_20261108/МАТЕМАТИКА_И_ДАННЫЕ.md']
    files += [str(Path(__file__).relative_to(ROOT)).replace('\\','/')]
    files += ['scripts/z13_target_model.py', 'tracking/z13_readout.py', 'tracking/foveated_eyes.py',
              'tracking/retina.py', 'tracking/flyvis_motion.py', 'tracking/flyvis_checkpoint.py',
              'tracking/flyvis_adapter.py', 'research_spatial_identity_20261107/extract_reference.py']
    for case in cases:
        track, candidate, path = get_row(case)
        files.append(path)
        base = 'singleton_working_domain' if case['split']=='development20' else 'bbox_singleton_pilot'
        files += [f'runs/{base}_20261108/labels/{case["video"]}.json',
                  f'data/bdd100k/labels/box_track_20/train/{case["video"]}.json',
                  f'runs/train_cache/{case["video"]}/alignment.json']
        reference = [[h['frame'],h['crop']] for h in track['history']]
        case['sequences'] = [reference, reference+track['search']]
        for seq, key in zip(case['sequences'],case['map_keys']):
            assert hashlib.sha256(json.dumps([case['video'],seq],separators=(',',':')).encode()).hexdigest()[:32]==key
        if case['kind']=='own':
            files += [f'runs/z13_data/{case["video"]}.npz',
                      f'data/bdd100k/videos/train/{case["video"]}.mov',
                      f'runs/train_cache/{case["video"]}/motion/flyvis_tiled.json']
        else:
            files += [p for paths in case['map_paths'] for p in paths]
    write('protocol.json',dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        deadline='2026-11-09T07:40:00+03:00', cases=cases,
        selection='All direct same-scale singleton car events age<=2 from the saved 86-video inventory. Match cell count exactly to the only foreign case. Selection reads no scores.',
        method='Unchanged fixed Q, 19 offsets, per-channel centering and joint normalized dot product. C uses eight saved T4/T5 channels. Brightness and gradients use the same retinal input. Seed42 joint spatial permutation.',
        reuse='Read foreign scores verbatim. Restore only absent own causal maps, shared prefix evaluated once. No threshold fitting.',
        acceptance='Equal Q count, exact retinal witness, immutable weights and inputs; independent interpolation and scalar products agree 1e-9. Descriptive comparison, no tracker acceptance.',
        alternative='At three cells, maximizing image correlation may fit unrelated structure. Own and foreign have equal counts but may differ in content, geometry and elapsed time.',
        sources={p:sha(p) for p in sorted(set(files))}))


def restore():
    if (OUT/'maps_complete.json').exists():
        receipt=read(OUT/'maps_complete.json')
        assert all(sha(p)==h for p,h in receipt['files'].items())
        return
    protocol=verify()
    case, = [c for c in protocol['cases'] if c['kind']=='own']
    video=case['video']
    sequence=case['sequences'][1]
    required={(f,tuple(c)) for f,c in sequence}
    values={}
    with np.load(ROOT/f'runs/z13_data/{video}.npz',allow_pickle=False) as z:
        for f,c,h in zip(z['fr_frame'],z['fr_crop'],z['fr_hex']):
            key=(int(f),tuple(map(int,c)))
            if key in required: values[key]=h.copy()
        witness=(int(z['fr_frame'][0]),tuple(map(int,z['fr_crop'][0])))
        witness_value=z['fr_hex'][0].copy()
    missing=required-values.keys()
    print('Required retinal views',len(required),'missing',len(missing),flush=True)
    import torch
    import cv2
    from types import SimpleNamespace
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.foveated_eyes import EyeBank
    from tracking.retina import make_retina
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    activate_flyvis(device='cpu')
    from flyvis.datasets.rendering import BoxEye
    g=read('runs/z13_local_peak_20261021/geometry.json')
    points=np.asarray(g['points'],np.float32)
    eye=BoxEye(extent=15,kernel_size=13)
    centers=eye.receptor_centers.detach().cpu().numpy()
    assert np.array_equal(np.column_stack([centers[:,1]+g['eye_size']//2,centers[:,0]+g['eye_size']//2]),points)
    bank=SimpleNamespace(torch=torch,eye_size=g['eye_size'],eye=eye,
        estimator=SimpleNamespace(flyvis=SimpleNamespace(device=torch.device('cpu'))))
    render={}
    for f,c in missing|{witness}: render.setdefault(f,[]).append(c)
    fps=read(f'runs/train_cache/{video}/motion/flyvis_tiled.json')['fps']
    retina=make_retina('lamina',float(fps),blur_downscale=4)
    capture=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/train/{video}.mov'))
    assert capture.isOpened()
    witness_error=None
    with torch.inference_mode():
        for frame in range(max(render)+1):
            ok,image=capture.read()
            assert ok,frame
            processed=retina.process(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
            if frame in render:
                crops=render[frame]
                batch=EyeBank._render(bank,processed,crops)[:,0].cpu().numpy().reshape(len(crops),721).astype(np.float16)
                for crop,h in zip(crops,batch):
                    key=(frame,crop)
                    if key==witness:
                        witness_error=float(np.max(np.abs(h.astype(float)-witness_value)))
                        assert np.array_equal(h,witness_value),witness_error
                    if key in required: values[key]=h
    capture.release()
    assert required<=values.keys() and witness_error==0
    hexes=np.stack([values[(f,tuple(c))] for f,c in sequence]).astype(np.float32)
    with (OUT/'retinal.npz').open('xb') as stream:
        np.savez_compressed(stream,hex=hexes,frames=np.array([f for f,c in sequence]),crops=np.array([c for f,c in sequence]))
    from tracking.gpu_lock import GpuLock
    from scripts.z13_target_model import build_arm
    from research_spatial_identity_20261107.extract_reference import image_features
    norm=read('runs/z13_target_center_dev_20261018/C/normalization.json')
    mu,sd=np.array(norm['mean'])[:34],np.array(norm['sd'])[:34]
    files={}
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        activate_flyvis(device='cuda')
        _,geom,backbone=build_arm('C',torch,None)
        assert backbone.dev.type=='cuda' and not backbone.net.training
        weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
        x=torch.from_numpy(hexes[None]).to(backbone.dev)
        state=backbone.init(x[:,:1])
        ends={len(s)-1:k for s,k in zip(case['sequences'],case['map_keys'])}
        for t in range(1,len(sequence)):
            state,features=backbone.step(state,x[:,t-1],x[:,t])
            if t in ends:
                C=((features[0].cpu().numpy().T.astype(float)-mu)/sd).astype(np.float32)[:,4:12]
                IMAGE=image_features(hexes[t].astype(float),points,geom.neighbors,sequence[t][1],g['eye_size'])
                assert C.shape==(721,8) and np.isfinite(C).all() and np.isfinite(IMAGE).all()
                path=OUT/(ends[t]+'.npz')
                with path.open('xb') as stream: np.savez_compressed(stream,C=C,IMAGE=IMAGE)
                files[path.relative_to(ROOT).as_posix()]=sha(path)
        assert all(torch.equal(v,weights[k]) for k,v in backbone.net.named_parameters())
    verify()
    write('maps_complete.json',dict(files=files,sequence=sequence,missing_views=len(missing),
        reused_views=len(required)-len(missing),witness_max_error=witness_error,
        shared_prefix_once=True,weights_unchanged=True,retinal_sha256=sha(OUT/'retinal.npz')))


def score():
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator
    protocol=verify()
    receipt=read(OUT/'maps_complete.json')
    assert all(sha(p)==h for p,h in receipt['files'].items())
    results=[]
    for case in protocol['cases']:
        if case['kind']=='foreign':
            old=read('runs/singleton_two_map_scores_20261108/scores.json')
            result, = [p for p in old['pairs'] if p['video']==case['video'] and p['old']==case['old']]
            results.append(dict(**result,reused_from='runs/singleton_two_map_scores_20261108/scores.json'))
            continue
        track,candidate,_=get_row(case)
        g=read('runs/z13_local_peak_20261021/geometry.json')
        points=np.array(g['points'],float)
        mesh=Delaunay(points)
        axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])
        zero=int(np.flatnonzero((axes==0).all(1))[0])
        offsets=axes@np.array([[13.,6.5],[0.,13.]])
        h=track['history'][-1]
        crop=np.array(track['current_crop'])
        extent=crop[2:]-crop[:2]
        assert all(np.array_equal(np.subtract(c[2:],c[:2]),extent) for f,c in [case['sequences'][0][-1]]+track['search'])
        scale=extent/g['eye_size']
        box0=np.array(h['box'])
        pred=np.array(track['mean'][:2])-(box0[:2]+box0[2:])/2
        old=points*scale+h['crop'][:2]
        xy=old[:,None,:]+pred+offsets*scale
        query=(xy-crop[:2])/scale
        box=np.array(candidate['box'])
        keep=(((old>=box0[:2])&(old<=box0[2:])).all(1)&((xy>=box[:2])&(xy<=box[2:])).all((1,2))&(mesh.find_simplex(query)>=0).all(1))
        idx=np.flatnonzero(keep)
        assert len(idx)==case['points']==3
        query=query[keep]
        simplex=mesh.find_simplex(query)
        tr=mesh.transform[simplex]
        w=np.einsum('...ij,...j->...i',tr[:,:,:2],query-tr[:,:,2])
        w=np.concatenate((w,1-w.sum(-1,keepdims=True)),axis=-1)
        vertices=mesh.simplices[simplex]
        arrays=[]
        for key in case['map_keys']:
            with np.load(OUT/(key+'.npz'),allow_pickle=False) as z: arrays.append({k:z[k].astype(float) for k in ['C','IMAGE']})
        variants={}
        for name,k,sl in [('FlyVis','C',slice(None)),('brightness','IMAGE',slice(0,1)),('gradient','IMAGE',slice(1,3)),('permuted','C',slice(None))]:
            a=arrays[0][k][idx,sl]
            if name=='permuted': a=a[np.random.default_rng(42).permutation(len(idx))]
            b=arrays[1][k][:,sl]
            sampled=np.einsum('qsv,qsvc->qsc',w,b[vertices])
            other=LinearNDInterpolator(mesh,b)(query)
            assert np.allclose(other,sampled,rtol=0,atol=1e-9)
            ref=a-a.mean(0)
            cur=sampled-sampled.mean(0)
            scores=[]
            for j in range(len(axes)):
                den=np.linalg.norm(ref)*np.linalg.norm(cur[:,j,:])
                value=float(np.sum(ref*cur[:,j,:])/den) if den else None
                if value is not None:
                    independent=other[:,j,:]-other[:,j,:].mean(0)
                    assert abs(np.dot(ref.ravel(),independent.ravel())/(np.linalg.norm(ref)*np.linalg.norm(independent))-value)<1e-9
                scores.append(value)
            valid=[j for j,s in enumerate(scores) if s is not None]
            best=max(valid,key=lambda j:scores[j]) if valid else None
            variants[name]=dict(fixed=scores[zero],local=scores[best] if valid else None,
                gain=scores[best]-scores[zero] if valid and scores[zero] is not None else None,
                winning_offset_px=(offsets[best]*scale).tolist() if valid else None,scores=scores)
        results.append(dict(video=case['video'],frame=case['frame'],old=case['old'],new=case['new'],
            split=case['split'],kind=case['kind'],cells=idx.tolist(),predicted_px=pred.tolist(),
            offsets_px=(offsets*scale).tolist(),variants=variants))
    verify()
    write('scores.json',dict(protocol_sha256=sha(OUT/'protocol.json'),pairs=results,
        verification='New own scores independently checked within 1e-9; foreign scores reused verbatim.',
        limitation='One own and one foreign event, different videos. Equal count does not equal identical conditions. No thresholds or tracker outcomes.'))


{'freeze':freeze,'restore':restore,'score':score}[sys.argv[1]]()
