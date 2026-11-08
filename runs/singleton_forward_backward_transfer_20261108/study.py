"""Frozen transfer diagnostic. Geometry and selection precede all new scores."""
import os
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=1800)
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
AXES=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])
BASIS=np.array([[13.,6.5],[0.,13.]])
NAMES=['FlyVis','center8','neighbor8','input8_temporal']
SELECT=[0,5,11,16,17,22,28,33]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def stamp():return datetime.now(timezone(timedelta(hours=3))).isoformat()
def log(message):print(stamp(),message,flush=True)
def verify(p):
    for file,digest in p['sources'].items():assert sha(ROOT/file)==digest,file
def corr(a,b):
    a=a-a.mean(0);b=b-b.mean(0);den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/den) if den else None
def best(values):
    valid=[j for j,v in enumerate(values) if v is not None]
    return max(valid,key=lambda j:values[j]) if valid else None
def sample(mesh,points,values):
    si=mesh.find_simplex(points)
    assert (si>=0).all()
    tr=mesh.transform[si]
    w=np.einsum('...ij,...j->...i',tr[...,:2,:],points-tr[...,2,:])
    w=np.concatenate([w,1-w.sum(-1,keepdims=True)],axis=-1)
    out=np.einsum('...v,...vc->...c',w,values[mesh.simplices[si]])
    other=LinearNDInterpolator(mesh,values)(points)
    assert np.allclose(out,other,rtol=0,atol=1e-9)
    return out,other
def row(case):
    base='singleton_working_domain' if case['split']=='development20' else 'bbox_confirmation_inventory'
    p=ROOT/f'runs/{base}_20261108/native/{case["video"]}.json'
    event,=[e for e in read(p)['events'] if e['frame']==case['frame']]
    r,=[r for r in event['rows'] if r['old_tid']==case['old']]
    candidate,=[c for c in r['candidates'] if c['forward_iou']>=.2]
    assert candidate['new_tid']==case['new']
    return r['row_data'],candidate,p
def geometry(case,track,candidate,g):
    points=np.array(g['points'],float);mesh=Delaunay(points)
    b0=np.array(track['history'][-1]['box']);b=np.array(candidate['box'])
    c0=np.array(track['history'][-1]['crop']);ct=np.array(track['current_crop'])
    s0=(c0[2:]-c0[:2])/g['eye_size'];st=(ct[2:]-ct[:2])/g['eye_size']
    step=np.maximum(s0,st);off=(AXES@BASIS)*step
    x=points*s0+c0[:2];d=np.array(track['mean'][:2])-(b0[:2]+b0[2:])/2
    trial=x[:,None,:]+d+off
    inside=((x>=b0[:2])&(x<=b0[2:])).all(1)&((trial>=b[:2])&(trial<=b[2:])).all((1,2))
    inside&=(mesh.find_simplex((trial-ct[:2])/st)>=0).all(1)
    eligible=np.flatnonzero(inside)
    assert len(eligible)==case['points']
    # Three spread-out points chosen from geometry alone. Stable index tie break.
    chosen=[int(eligible[0])]
    while len(chosen)<3:
        distance=((points[eligible,None]-points[np.array(chosen)][None])**2).sum(-1).min(1)
        distance[np.isin(eligible,chosen)]=-1
        chosen.append(int(eligible[np.argmax(distance)]))
    return dict(cells=chosen,eligible_cells=eligible.tolist(),prediction=d.tolist(),
        crop0=c0.tolist(),crop=ct.tolist(),scale0=s0.tolist(),scale=st.tolist(),
        step_scale=step.tolist(),offsets=off.tolist(),old_box=b0.tolist())
def register():
    inventory_path=ROOT/'runs/singleton_car_map_coverage_20261108/inventory.json'
    inv=read(inventory_path)
    previous=read(ROOT/'runs/singleton_forward_backward_20261108/protocol.json')
    used={tuple(c) for c in previous['cases']}
    direct=[c for c in inv['cases'] if c['direct'] and (c['video'],c['frame'],c['old'],c['new']) not in used]
    excluded=[c for c in direct if c['points']<3]
    cases=[c for c in direct if c['points']>=3]
    assert len(cases)==7 and sum(c['kind']=='foreign' for c in cases)==3
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    files=[inventory_path,Path(__file__),ROOT/'runs/singleton_forward_backward_20261108/protocol.json',
        ROOT/'runs/singleton_forward_backward_20261108/mathematical_checks.json',
        ROOT/'runs/singleton_forward_backward_20261108/recovery1_geometry_check.json',
        ROOT/'runs/z13_local_peak_20261021/geometry.json']
    files += [ROOT/p for p in ['scripts/z13_target_model.py','tracking/z13_readout.py','tracking/foveated_eyes.py',
        'tracking/retina.py','tracking/flyvis_adapter.py','tracking/flyvis_motion.py','tracking/flyvis_checkpoint.py',
        'runs/flyvis/results/flow/0000/000/best_chkpt','runs/flyvis/results/flow/0000/000/_meta.yaml',
        'runs/z13_target_center_dev_20261018/C/normalization.json',
        'runs/z13_target_center_dev_20261018/R1/normalization.json']]
    for c in cases:
        t,can,path=row(c);files.append(path)
        c['sequences']=[[[h['frame'],h['crop']] for h in t['history']]]
        c['sequences'].append(c['sequences'][0]+t['search'])
        for key,seq in zip(c['map_keys'],c['sequences']):
            assert hashlib.sha256(json.dumps([c['video'],seq],separators=(',',':')).encode()).hexdigest()[:32]==key
        c['geometry']=geometry(c,t,can,g)
        c['ready_maps']={}
        for key,paths in zip(c['map_keys'],c['map_paths']):
            if paths:
                assert len(paths)==1 and paths[0].endswith('.npz')
                with np.load(ROOT/paths[0],allow_pickle=False) as z:
                    assert z['C'].shape==(721,8) and z['R1'].shape==(721,34)
                c['ready_maps'][key]=paths[0];files.append(ROOT/paths[0])
        base='singleton_working_domain' if c['split']=='development20' else 'bbox_singleton_pilot'
        labelpath=ROOT/f'runs/{base}_20261108/labels/{c["video"]}.json';files.append(labelpath)
        label=read(labelpath)['rows'][f'{c["frame"]}|{c["old"]}|{c["new"]}']
        assert label['target']==c['target'] and label['identity']==c['identity']
        assert label['target_anchors']==c['target_anchors'] and label['fragment_anchors']==c['fragment_anchors']
        truth=ROOT/f'data/bdd100k/labels/box_track_20/train/{c["video"]}.json'
        align=ROOT/f'runs/train_cache/{c["video"]}/alignment.json'
        files += [truth,align]
        byframe={int(a['video_frame_index']):f for a,f in zip(read(align)['matches'],read(truth))}
        hits=[];box=np.array(can['box'])
        for obj in byframe[c['frame']].get('labels') or []:
            if obj['category']!='car' or (obj.get('attributes') or {}).get('crowd',False):continue
            bb=np.array([obj['box2d'][k] for k in ['x1','y1','x2','y2']])
            inter=np.maximum(0,np.minimum(box[2:],bb[2:])-np.maximum(box[:2],bb[:2])).prod()
            iou=inter/((box[2:]-box[:2]).prod()+(bb[2:]-bb[:2]).prod()-inter)
            if iou>=.5:hits.append(str(obj['id']))
        assert hits==[c['identity']] and [c['frame'],c['identity']] in c['fragment_anchors']
        c['direct_raw_annotation_verified']=True
        if len(c['ready_maps'])<2:
            files += [ROOT/f'runs/z13_data/{c["video"]}.npz',ROOT/f'data/bdd100k/videos/train/{c["video"]}.mov',
                      ROOT/f'runs/train_cache/{c["video"]}/motion/flyvis_tiled.json']
            c['retinal_archives']=[]
            for p in sorted((ROOT/'runs').glob(f'*/retinal/{c["video"]}.npz')):
                c['retinal_archives'].append(p.relative_to(ROOT).as_posix());files.append(p)
    # Pixel/eye transforms invert exactly even with unequal crop sizes.
    p=np.array([[101.,67.],[120.,90.],[80.,44.]])
    c0=np.array([13.,21.]);ct=np.array([31.,19.]);s0=np.array([.3,.3]);st=np.array([.5,.5])
    d=np.array([7.,-2.]);delta=np.array([3.,4.]);y=p+d+delta
    back=((y-d-delta-c0)/s0)*s0+c0
    assert np.allclose(back,p,rtol=0,atol=1e-12)
    assert np.allclose(((y-ct)/st)*st+ct,y,rtol=0,atol=1e-12)
    assert read(ROOT/'runs/singleton_forward_backward_20261108/mathematical_checks.json')['passed']
    write('mathematical_checks.json',dict(passed=True,unequal_crop_roundtrip_max_error=float(np.max(abs(back-p))),
        prior_math_reused='runs/singleton_forward_backward_20261108/mathematical_checks.json',
        interpretation='Return remains norm(delta_f+delta_b) in physical pixels. A changing crop alters receptive-field support and is an explicit scientific confound.'))
    protocol=dict(registered=stamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        deadline='2026-11-09T07:40:00+03:00',cases=cases,excluded=excluded,
        selection='All other directly annotated singleton-car pairs in the existing 86-video inventory with >=3 fixed admissible cells. No score-dependent selection. Previous two cases excluded.',
        support='Exactly three geometry-selected cells: smallest eligible index, then farthest from the chosen set with stable lowest-index ties. Same cells for all four methods.',
        forward='x=eye0*scale0+crop0_origin; y=x+d0+delta_f. 19 old hex offsets scaled by max(scale0,scale_t), as in existing feasibility geometry. Per-channel centering and joint norm unchanged.',
        backward='Current template at y is independently matched to old(y-d0+delta_b). Same physical offsets both directions. Return residual delta_f+delta_b. Whole archived old eye allowed. Unsupported backward grid is reported, no replacement cells.',
        controls='Same four representations, prior R1 channel indices [0,5,11,16,17,22,28,33], fixed training normalization. center8 duplicates first four temporal means; neighbor8 appends their six-neighbor means.',
        reuse='Existing C and R1 maps read verbatim. Only missing retinal views, maps and scores computed. The earlier eight-cell own forward result is a different support and is retained separately.',
        acceptance='Immutable source hashes; direct raw annotation rechecked; all methods share three cells; independent interpolation and scalar products agree <=1e-9; no score fitting; unknown/unsupported cases retained explicitly.',
        limitation='All three new foreign cases change eye scale; all four own cases keep it. Cannot attribute group differences exclusively to identity. No fitted threshold or tracker-performance claim.',
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))})
    write('protocol.json',protocol)
    catalogpath=ROOT/'experiments/project_history.json';cat=read(catalogpath)
    assert not any(c['id']==OUT.name for c in cat['experiments'])
    cat['experiments'].append(dict(id=OUT.name,title='Проверка возврата пространственных карт на других размеченных своих и чужих парах',
        area='flyvis',state='running',started='2026-11-08',question='Повторяется ли различие ошибки возврата на других прямо размеченных парах?',
        hypothesis='Несогласованный возврат по T4/T5 может отличать чужие продолжения от своих.',expected='Ошибки семи новых пар с четырьмя представлениями и ограничениями по масштабу.',
        acceptance=[dict(criterion=protocol['acceptance'],met=None)],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Условия до расчёта')]))
    catalogpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    journal('Зарегистрирован перенос проверки возврата на другие подтверждённые пары.',
        'Все 7 других прямо размеченных пар с не менее чем 3 допустимыми ячейками: 4 свои, 3 чужие. Две пары без трёх ячеек исключены до оценок. Три общие опорные ячейки выбраны только геометрией. У всех чужих меняется масштаб глаза, у своих он постоянный; смешение факторов записано до расчёта. Готовые C/R1 используются без повторной сети. Условия и математическая проверка `runs/'+OUT.name+'/protocol.json`, `runs/'+OUT.name+'/mathematical_checks.json`, карточка `'+OUT.name+'`.')
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf8')
    now=datetime.now(timezone(timedelta(hours=3)))
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — {title}\n\n'+text+'\n\n'
    path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')

def restore():
    protocol=read(OUT/'protocol.json');verify(protocol)
    if (OUT/'maps_complete.json').exists():
        verify(read(OUT/'maps_complete.json'));return
    pending=[c for c in protocol['cases'] if len(c['ready_maps'])<2]
    import torch,cv2
    from types import SimpleNamespace
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.foveated_eyes import EyeBank
    from tracking.retina import make_retina
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    activate_flyvis(device='cpu')
    from flyvis.datasets.rendering import BoxEye
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.asarray(g['points'],np.float32)
    eye=BoxEye(extent=15,kernel_size=13);centers=eye.receptor_centers.detach().cpu().numpy()
    assert np.array_equal(np.column_stack([centers[:,1]+g['eye_size']//2,centers[:,0]+g['eye_size']//2]),points)
    bank=SimpleNamespace(torch=torch,eye_size=g['eye_size'],eye=eye,estimator=SimpleNamespace(flyvis=SimpleNamespace(device=torch.device('cpu'))))
    retinal_files={};retinal_counts={}
    for video in sorted({c['video'] for c in pending}):
        cases=[c for c in pending if c['video']==video]
        dest=OUT/(video+'.retinal.npz')
        if dest.exists():
            item=read(OUT/(video+'.retinal.json'));assert sha(dest)==item['sha256']
            retinal_files[video]=dest.relative_to(ROOT).as_posix();retinal_counts[video]=item;continue
        required={(f,tuple(crop)) for c in cases for f,crop in c['sequences'][1]}
        values={}
        with np.load(ROOT/f'runs/z13_data/{video}.npz',allow_pickle=False) as z:
            witness=(int(z['fr_frame'][0]),tuple(map(int,z['fr_crop'][0])));witness_value=z['fr_hex'][0].copy()
            for f,c,h in zip(z['fr_frame'],z['fr_crop'],z['fr_hex']):
                key=(int(f),tuple(map(int,c)))
                if key in required:values[key]=h.copy()
        for path in cases[0]['retinal_archives']:
            with np.load(ROOT/path,allow_pickle=False) as z:
                if not {'frames','crops','hex'}<=set(z.files):continue
                for f,c,h in zip(z['frames'],z['crops'],z['hex']):
                    key=(int(f),tuple(map(int,c)))
                    if key in required:
                        if key in values:assert np.array_equal(values[key],h)
                        values[key]=h.copy()
        missing=required-values.keys();error=None
        if missing:
            render={}
            for f,c in missing|{witness}:render.setdefault(f,[]).append(c)
            fps=read(ROOT/f'runs/train_cache/{video}/motion/flyvis_tiled.json')['fps']
            retina=make_retina('lamina',float(fps),blur_downscale=4)
            cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/train/{video}.mov'));assert cap.isOpened()
            with torch.inference_mode():
                for frame in range(max(render)+1):
                    ok,im=cap.read();assert ok,(video,frame)
                    processed=retina.process(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
                    if frame in render:
                        batch=EyeBank._render(bank,processed,render[frame])[:,0].cpu().numpy().reshape(len(render[frame]),721).astype(np.float16)
                        for crop,h in zip(render[frame],batch):
                            key=(frame,crop)
                            if key==witness:
                                error=float(np.max(abs(h.astype(float)-witness_value)));assert np.array_equal(h,witness_value)
                            if key in required:values[key]=h
                    if frame%200==0:log(f'retina {video} frame {frame}/{max(render)}')
            cap.release()
        assert required<=values.keys()
        keys=sorted(required)
        with dest.open('xb') as f:np.savez_compressed(f,frames=np.array([k[0] for k in keys]),crops=np.array([k[1] for k in keys]),hex=np.stack([values[k] for k in keys]))
        item=dict(sha256=sha(dest),new_views=len(missing),reused_views=len(required)-len(missing),witness_max_error=error)
        write(video+'.retinal.json',item);retinal_counts[video]=item;retinal_files[video]=dest.relative_to(ROOT).as_posix()
    from tracking.gpu_lock import GpuLock
    from scripts.z13_target_model import build_arm,receptor_features_rich
    stats={name:read(ROOT/f'runs/z13_target_center_dev_20261018/{name}/normalization.json') for name in ['C','R1']}
    files={};reused={};backbone=None;weights=None
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        for c in protocol['cases']:
            needed=[key for key in c['map_keys'] if key not in c['ready_maps'] and not (OUT/(key+'.npz')).exists()]
            if needed:
                if backbone is None:
                    activate_flyvis(device='cuda');_,_,backbone=build_arm('C',torch,None)
                    assert backbone.dev.type=='cuda' and not backbone.net.training
                    weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
                with np.load(ROOT/retinal_files[c['video']],allow_pickle=False) as z:
                    values={(int(f),tuple(map(int,crop))):h.copy() for f,crop,h in zip(z['frames'],z['crops'],z['hex'])}
                sequence=c['sequences'][1];hx=np.stack([values[(f,tuple(crop))] for f,crop in sequence]).astype(np.float32)
                rich=receptor_features_rich(hx[None].astype(float));x=torch.from_numpy(hx[None]).to(backbone.dev)
                state=backbone.init(x[:,:1]);ends={len(seq)-1:key for seq,key in zip(c['sequences'],c['map_keys'])}
                for t in range(1,len(sequence)):
                    state,feat=backbone.step(state,x[:,t-1],x[:,t])
                    if t in ends and ends[t] in needed:
                        cm,cs=np.array(stats['C']['mean'])[:34],np.array(stats['C']['sd'])[:34]
                        rm,rs=np.array(stats['R1']['mean'])[:34],np.array(stats['R1']['sd'])[:34]
                        C=((feat[0].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32)[:,4:12]
                        R1=((rich[t][0].T-rm)/rs).astype(np.float32)
                        with (OUT/(ends[t]+'.npz')).open('xb') as f:np.savez_compressed(f,C=C,R1=R1)
                        log('saved map '+ends[t])
            for key in c['map_keys']:
                path=c['ready_maps'].get(key,f'runs/{OUT.name}/{key}.npz')
                files[path]=sha(ROOT/path)
                if key in c['ready_maps']:reused[key]=path
            log('maps ready '+c['video']+' '+str(c['old']))
        if backbone is not None:assert all(torch.equal(v,weights[k]) for k,v in backbone.net.named_parameters())
    verify(protocol)
    write('maps_complete.json',dict(passed=True,sources=files,retinal_files=retinal_files,retinal_counts=retinal_counts,
        reused_maps=reused,new_maps=len(files)-len(reused),shared_prefix_once=True,weights_unchanged=True))

def score():
    if (OUT/'results.json').exists():raise SystemExit('Ready results exist, no calculation repeated.')
    p=read(OUT/'protocol.json');verify(p);maps=read(OUT/'maps_complete.json');verify(maps)
    from scripts.z13_target_model import neighbor_aug
    from tracking.z13_readout import HexGeometry
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.asarray(g['points'],float)
    nb=HexGeometry(points,g['eye_size']).neighbors;mesh=Delaunay(points)
    records=[];pool_error=0.
    for c in p['cases']:
        t=c['geometry'];idx=np.array(t['cells']);crop0=np.array(t['crop0']);ct=np.array(t['crop'])
        s0=np.array(t['scale0']);st=np.array(t['scale']);off=np.array(t['offsets']);d=np.array(t['prediction'])
        x=points[idx]*s0+crop0[:2];forward_q=(x[:,None]+d+off-ct[:2])/st
        arrays={name:[] for name in NAMES}
        for key in c['map_keys']:
            path=c['ready_maps'].get(key,f'runs/{OUT.name}/{key}.npz')
            with np.load(ROOT/path,allow_pickle=False) as z:
                C=z['C'].astype(float);temporal=z['R1'][:,SELECT].astype(float)
            central=temporal[:,:4];pooled=neighbor_aug(central.T[None],nb)[0].T
            independent=np.stack([central[row[1:]].sum(0)/6 for row in nb])
            error=float(np.max(abs(pooled[:,4:]-independent)));pool_error=max(pool_error,error);assert error<=1e-9
            for name,a in zip(NAMES,[C,np.concatenate([central,central],1),pooled,temporal]):
                assert a.shape==(721,8) and np.isfinite(a).all();arrays[name].append(a)
        for name,(old,current) in arrays.items():
            sampled,independent=sample(mesh,forward_q,current)
            forward=[corr(old[idx],sampled[:,j]) for j in range(19)]
            for j,v in enumerate(forward):
                check=corr(old[idx],independent[:,j]);assert v is None and check is None or abs(v-check)<=1e-9
            fi=best(forward);assert fi is not None
            y=x+d+off[fi];template,_=sample(mesh,(y-ct[:2])/st,current)
            backward_xy=y[:,None]-d+off;query=(backward_xy-crop0[:2])/s0
            supported=bool((mesh.find_simplex(query)>=0).all())
            record=dict(video=c['video'],frame=c['frame'],old=c['old'],new=c['new'],kind=c['kind'],
                same_scale=c['same_scale'],representation=name,cells=idx.tolist(),
                forward_scores=forward,forward_axes=AXES[fi].tolist(),forward_score=forward[fi],fixed_score=forward[9],
                geometry_only_return_grid=int(max(abs(AXES[fi][0]),abs(AXES[fi][1]),abs(sum(AXES[fi])))),
                geometry_only_return_px=float(np.linalg.norm(off[fi])),backward_supported=supported)
            if supported:
                b,other=sample(mesh,query,old);scores=[]
                for j in range(19):
                    v=corr(template,b[:,j]);check=corr(template,other[:,j]);assert v is None and check is None or abs(v-check)<=1e-9
                    if v is not None:
                        a=template-template.mean(0);bb=other[:,j]-other[:,j].mean(0)
                        check=float(np.dot(a.ravel(),bb.ravel())/(np.linalg.norm(a)*np.linalg.norm(bb)));assert abs(v-check)<=1e-9
                    scores.append(v)
                bi=best(scores)
                if bi is not None:
                    res=AXES[fi]+AXES[bi];physical=off[fi]+off[bi]
                    inside=((backward_xy>=np.array(t['old_box'])[:2])&(backward_xy<=np.array(t['old_box'])[2:])).all(-1).all(0)
                    record.update(backward_scores=scores,backward_axes=AXES[bi].tolist(),backward_score=scores[bi],
                        forward_ties=[j for j,v in enumerate(forward) if v==forward[fi]],backward_ties=[j for j,v in enumerate(scores) if v==scores[bi]],
                        residual_axes=res.tolist(),error_grid_steps=int(max(abs(res[0]),abs(res[1]),abs(sum(res)))),
                        error_px=float(np.linalg.norm(physical)),residual_px=physical.tolist(),
                        winner_inside_old_box=bool(inside[bi]),backward_all_inside_old_box=inside.tolist())
            records.append(record)
            log(f'{c["kind"]} {c["old"]}->{c["new"]} {name} {record.get("error_grid_steps","unsupported")} steps')
    verify(p);verify(maps)
    write('results.json',dict(protocol_sha256=sha(OUT/'protocol.json'),records=records))
    write('audit.json',dict(passed=True,source_hashes_stable=True,same_three_cells_all_methods=True,
        interpolation_dot_tolerance=1e-9,pool_max_error=pool_error,prior_math_reused=True,
        unsupported_records=sum(not r['backward_supported'] for r in records),result_sha256=sha(OUT/'results.json')))

{'register':register,'restore':restore,'score':score}[sys.argv[1]]()
