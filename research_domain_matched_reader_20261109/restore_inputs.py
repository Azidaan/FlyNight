"""Recover only missing exact eye views, with archived-view witnesses per video."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/full_template_native_20261109'))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(3600,(STOP-datetime.now(MSK)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,obj):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
def key(frame,crop):return hashlib.sha256(json.dumps([int(frame),list(crop)],separators=(',',':')).encode()).hexdigest()[:32]
def seqkey(video,seq):return hashlib.sha256(json.dumps([video,seq],separators=(',',':')).encode()).hexdigest()[:32]
import numpy as np
mode=sys.argv[1]
if mode=='register':
    p=read(OUT/'protocol.json');assert read(OUT/'input_inventory.json')['passed'];verify(p)
    files=[Path(__file__),OUT/'protocol.json',OUT/'input_inventory.json',
        ROOT/'tracking/retina.py',ROOT/'tracking/foveated_eyes.py',ROOT/'scripts/z13_data.py',
        ROOT/'runs/full_template_native_20261109/sparse_views.py',ROOT/'runs/full_template_native_20261109/common.py',
        ROOT/'runs/z13_local_peak_20261021/geometry.json']
    videos={};requests={}
    for video in p['videos']:
        data=ROOT/f'runs/z13_data/{video}.npz'
        with np.load(data,allow_pickle=False) as z:frames=z['fr_frame'];crops=z['fr_crop']
        locations={}
        for i,(f,c) in enumerate(zip(frames,crops)):locations.setdefault(key(int(f),c.tolist()),[]).append(i)
        needed={}
        for e in [e for e in p['edges'] if e['video']==video]:
            for seq in [e['sequence'][:4],e['sequence']]:
                requests[seqkey(video,seq)]=dict(video=video,sequence=seq)
                for f,c in seq:needed[key(f,c)]=dict(frame=f,crop=c,archive_indices=locations.get(key(f,c),[]))
        missing=[k for k,v in needed.items() if not v['archive_indices']]
        witnesses=[];movie=None;fpsfile=ROOT/f'runs/train_cache/{video}/motion/flyvis_tiled.json'
        files+=[data,fpsfile]
        if missing:
            assert len(set(map(int,frames)))>=2
            for i in [int(np.argmin(frames)),int(np.argmax(frames))]:witnesses.append(dict(frame=int(frames[i]),crop=crops[i].tolist(),archive_index=i))
            movie=ROOT/f'data/bdd100k/videos/train/{video}.mov';assert movie.exists();files.append(movie)
        videos[video]=dict(data=data.relative_to(ROOT).as_posix(),fps_file=fpsfile.relative_to(ROOT).as_posix(),
            video_path=None if movie is None else movie.relative_to(ROOT).as_posix(),views=needed,missing=missing,witnesses=witnesses)
    save('input_recovery_protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in files},videos=videos,requests=requests,deadline=STOP.isoformat(),
        reason='Whole exact unperturbed episodes exist for only11of26pairs. Do not reduce the26case sample. Recover individual EXACT(frame,crop)views; a missing whole episode does not imply every view is absent.',
        derivation='rho_t=Retina(I0..It); eye_t,c=Sample_c(rho_t). In scripts/z13_data.py retina advances once per video frame outside the render(crops) loop. Eye rendering is stateless in crop identity. Hence an exact(frame,crop)view can be reused across episode IDs/perturbation flags, while the core receives the original causal sequence in its original order. This does NOT authorize reusing a recurrent core state across histories.',
        operation='First reuse exact archived fr_hex views, requiring all duplicate indices to hold identical float16 arrays. For absent exact views, decode source video strictly sequentially from0, same lamina retina/fps/blur_downscale4, and call the already witnessed SparseViews.render CPU implementation. No cap.seek, nearby crops, detector, FlyVis, fitting or metric calls.',
        witness='On each video requiring decoding, exactly reproduce earliest and latest archived eye view before accepting ANY newly decoded view. They bracket the archived clock; decode through max(witnessframes,requestedframes). Witnesses are validation only and never enter an earlier core input. Stop on any float16 mismatch; preserve diagnostic failure.',
        source_limit='Original raw-video hash was not captured by the old z13_data extraction. This protocol hashes the actual current source and proves exact archived-view witnesses; never present the new hash as an original pre-run hash.',
        parity='Ready12core/12ordinarysnapshots and all original arrays remain unchanged. Restore ONLY missing eye inputs/maps in new folders; ordinary processing and frozen core snapshot restoration require separate subsequent execution passports.'))
    print('REGISTERED_INPUT_RECOVERY',len(requests),'snapshots',sum(len(v['missing']) for v in videos.values()),'missing views',flush=True)
    raise SystemExit(0)
assert mode=='run';p=read(OUT/'input_recovery_protocol.json');verify(p)
import torch,cv2
from types import SimpleNamespace
from tracking.retina import make_retina
from sparse_views import SparseViews
torch.set_num_threads(2)
points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int)
renderer=object.__new__(SparseViews);renderer.torch=torch
renderer.px=torch.from_numpy(points[:,0].copy());renderer.py=torch.from_numpy(points[:,1].copy())
renderer.kernel=torch.ones((1,1,13,13),dtype=torch.float32,device='cpu');renderer.bias=torch.zeros(1,dtype=torch.float32,device='cpu')
receipts=[]
def write_view(video,name,value,origin):
    path=OUT/'retinal'/video/(name+'.npy');meta=path.with_suffix('.json');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert read(meta)['sha256']==sha(path) and np.array_equal(np.load(path,allow_pickle=False),value);return
    with path.open('xb') as f:np.save(f,value,allow_pickle=False)
    save(meta.relative_to(OUT),dict(sha256=sha(path),protocol_sha256=sha(OUT/'input_recovery_protocol.json'),origin=origin))
for video,info in p['videos'].items():
    marker=OUT/'input_receipts'/(video+'.json')
    if marker.exists():
        receipt=read(marker);assert receipt['protocol_sha256']==sha(OUT/'input_recovery_protocol.json')
        for name,digest in receipt['views'].items():assert sha(ROOT/name)==digest,name
        receipts.append(receipt);continue
    with np.load(ROOT/info['data'],allow_pickle=False) as z:hexes=z['fr_hex']
    assert hexes.dtype==np.float16 and hexes.shape[1]==721
    for name,view in info['views'].items():
        indices=view['archive_indices']
        if not indices:continue
        value=hexes[indices[0]].copy()
        for i in indices:np.testing.assert_array_equal(value,hexes[i])
        write_view(video,name,value,dict(kind='exact_archived_frame_crop',data=info['data'],indices=indices,frame=view['frame'],crop=view['crop']))
    witnesses=[];decoded=0
    if info['missing']:
        fps=read(ROOT/info['fps_file'])['fps'];retina=make_retina('lamina',fps,blur_downscale=4)
        cap=cv2.VideoCapture(str(ROOT/info['video_path']));assert cap.isOpened()
        targets={}
        for name in info['missing']:
            view=info['views'][name];targets.setdefault(view['frame'],[]).append(('new',name,view))
        for w in info['witnesses']:targets.setdefault(w['frame'],[]).append(('witness',key(w['frame'],w['crop']),w))
        last=max(targets);pending={}
        try:
            for frame in range(last+1):
                ok,image=cap.read();assert ok,(video,frame,'decode')
                gray=retina.process(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY));decoded+=1
                if frame not in targets:continue
                def get(frame_arg,now):assert frame_arg==frame and now==frame;return gray
                renderer.frames=SimpleNamespace(get=get)
                for kind,name,view in targets[frame]:
                    value=renderer.render(frame,view['crop'],frame)
                    if kind=='witness':
                        expected=hexes[view['archive_index']];error=float(abs(value.astype(float)-expected.astype(float)).max())
                        if not np.array_equal(value,expected):
                            save('input_failure_'+video+'.json',dict(video=video,frame=frame,maxerror=error,source_limit=p['source_limit']))
                            raise AssertionError((video,frame,'archived eye witness differs',error))
                        witnesses.append(dict(frame=frame,crop=view['crop'],maxerror=error,exact=True))
                    else:pending[name]=value.copy()
        finally:cap.release()
        assert len(witnesses)==2 and len(pending)==len(info['missing'])
        for name,value in pending.items():write_view(video,name,value,dict(kind='missing_exact_view_restored_sequentially',
            frame=info['views'][name]['frame'],crop=info['views'][name]['crop'],video_path=info['video_path'],fps=fps,archived_witnesses=witnesses))
    paths=[OUT/'retinal'/video/(name+'.npy') for name in info['views']]
    receipt=dict(passed=True,video=video,protocol_sha256=sha(OUT/'input_recovery_protocol.json'),
        views={path.relative_to(ROOT).as_posix():sha(path) for path in paths},
        archived_views=len(paths)-len(info['missing']),new_views=len(info['missing']),decoded_frames=decoded,witnesses=witnesses)
    save(marker.relative_to(OUT),receipt);receipts.append(receipt)
    print('INPUT_RECOVERY',len(receipts),'/20',video,'archived',receipt['archived_views'],'new',receipt['new_views'],'witnesses',len(witnesses),flush=True)
snapshots={}
for name,item in p['requests'].items():
    views=[OUT/'retinal'/item['video']/(key(f,c)+'.npy') for f,c in item['sequence']]
    hx=np.stack([np.load(path,allow_pickle=False) for path in views]).astype(np.float32)
    assert hx.ndim==2 and hx.shape[1]==721 and np.isfinite(hx).all()
    path=OUT/'inputs'/(name+'.npy');path.parent.mkdir(exist_ok=True)
    with path.open('xb') as f:np.save(f,hx,allow_pickle=False)
    save('inputs/'+name+'.json',dict(sha256=sha(path),protocol_sha256=sha(OUT/'input_recovery_protocol.json'),video=item['video'],sequence=item['sequence'],
        sources={q.relative_to(ROOT).as_posix():sha(q) for q in views}))
    snapshots[name]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
verify(p);assert not torch.cuda.is_initialized() and not any(x=='flyvis' or x.startswith('flyvis.') for x in sys.modules)
save('inputs_complete.json',dict(passed=True,snapshots=snapshots,total_snapshots=len(snapshots),
    archived_views=sum(r['archived_views'] for r in receipts),new_views=sum(r['new_views'] for r in receipts),
    decoded_frames=sum(r['decoded_frames'] for r in receipts),exact_witnesses=sum(len(r['witnesses']) for r in receipts),
    no_detector_core_fit_score_or_metric_calls=True,source_limit=p['source_limit']))
print('INPUTS_COMPLETE',len(snapshots),flush=True)
