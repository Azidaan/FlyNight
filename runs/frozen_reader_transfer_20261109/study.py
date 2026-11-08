"""Transfer the already frozen reader; never tune it on transfer labels."""
import os
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=1800)
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug,receptor_features_rich
OTHER=[0,1,2,3]+list(range(12,34));SELECT=[0,5,11,16,17,22,28,33]
NAMES=['FlyVis','C34','center8','neighbor8','input8_temporal','R134_temporal','R134_spatial']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [cast(x) for x in v]
    return v
def write(name,v):
    p=OUT/name;p.parent.mkdir(exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
def stamp():return datetime.now(timezone(timedelta(hours=3))).isoformat()
def journal(title,body):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
    p.write_text(s.replace('## Журнал изменений\n',f'## Журнал изменений\n\n**{now:%d.%m.%Y, %H:%M} MSK** — {title}.\n\n{body}\n\n',1),encoding='utf8')
def load_functions(path,names,ns):
    nodes=[n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
def raw_observation(native,raw,tid,frame,cfg):
    ns=dict(np=np);load_functions(ROOT/'runs/clipped_anchor_readout_20261109/study.py',['raw_observation'],ns)
    return ns['raw_observation'](native,raw,tid,frame,cfg)
def register():
    invpath=ROOT/'runs/singleton_car_map_coverage_20261108/inventory.json';inv=read(invpath)
    fit={e['video'] for e in read(ROOT/'runs/background_patch_gate_20261108/protocol.json')['events']}
    direct=[c for c in inv['cases'] if c['direct'] and c['video'] not in fit]
    cases=[c for c in direct if c['points']>=3];excluded=[c for c in direct if c['points']<3]
    assert len(cases)==6 and len(excluded)==2
    paths=[Path(__file__),invpath,ROOT/'runs/background_patch_gate_20261108/protocol.json',ROOT/'runs/singleton_support_size_20261108/protocol.json',ROOT/'runs/singleton_support_size_20261108/results.json',ROOT/'runs/channel_balance_attribution_20261109/transfer.json',ROOT/'runs/singleton_matched_support_20261108/maps_complete.json',ROOT/'runs/singleton_forward_backward_transfer_20261108/maps_complete.json',ROOT/'runs/clipped_anchor_readout_20261109/study.py',ROOT/'runs/clipped_anchor_readout_20261109/runtime_verification.json',ROOT/'runs/z13_local_peak_20261021/geometry.json']
    paths += [ROOT/p for p in ['tracking/observed_spatial_reader.py','tracking/channel_balanced_correspondence.py','tracking/clipping_aware_correspondence.py','tracking/z13_readout.py','scripts/z13_target_model.py','tracking/flyvis_activity.py','tracking/flyvis_motion.py','tracking/flyvis_adapter.py','tracking/flyvis_checkpoint.py','runs/flyvis/results/flow/0000/000/best_chkpt','runs/flyvis/results/flow/0000/000/_meta.yaml','runs/z13_target_center_dev_20261018/C/normalization.json','runs/z13_target_center_dev_20261018/R1/normalization.json']]
    geometry=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geometry['points'],geometry['eye_size'],[1280,720])
    cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config'];inputs={};mapinfo={};readiness=[]
    manifests={}
    for folder in ['singleton_forward_backward_transfer_20261108','singleton_matched_support_20261108']:
        m=read(ROOT/f'runs/{folder}/maps_complete.json');manifests.update(m.get('sources',m.get('files',{})))
    for c in cases:
        base='singleton_working_domain' if c['split']=='development20' else 'bbox_confirmation_inventory'
        nativepath=ROOT/f'runs/{base}_20261108/native/{c["video"]}.npz';eventpath=nativepath.with_suffix('.json');data=read(eventpath)
        e,=[e for e in data['events'] if e['frame']==c['frame']];row,=[r for r in e['rows'] if r['old_tid']==c['old']];can,=[q for q in row['candidates'] if q['forward_iou']>=.2]
        assert can['new_tid']==c['new'] and can['raw_id']==c['raw_id'];t=row['row_data'];h=t['history'][-1]
        c.update(track=t,candidate=can,sequences=[[[v['frame'],v['crop']] for v in t['history']],[[v['frame'],v['crop']] for v in t['history']]+t['search']])
        rawpath=ROOT/f'runs/train_cache/{c["video"]}/detections/raw_video.npz'
        with np.load(nativepath,allow_pickle=False) as z:native=z['outputs']
        with np.load(rawpath,allow_pickle=False) as z:raw={n:z[n] for n in ['frame_idx','conf','cls','xyxy']}
        old=raw_observation(native,raw,c['old'],h['frame'],cfg);now=raw_observation(native,raw,c['new'],c['frame'],cfg)
        assert np.array_equal(np.asarray(h['box'],np.float32),np.asarray(old['state_box'],np.float32));assert np.array_equal(now['box'],can['box'])
        c.update(reference_observation=cast(old),current_observation=cast(now))
        pred=np.array(t['mean'][:2])-(np.array(h['box'][:2])+h['box'][2:])/2
        scale=np.maximum((np.array(h['crop'][2:])-h['crop'][:2])/geometry['eye_size'],(np.array(t['current_crop'][2:])-t['current_crop'][:2])/geometry['eye_size'])
        axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2]);offsets=(axes@np.array([[13.,6.5],[0.,13.]]))*scale
        c['plan_old']=cast(reader.plan(h['box'],h['crop'],can['box'],t['current_crop'],pred,offsets,use_visible_edge=False))
        c['plan_joint']=cast(reader.plan(h['box'],h['crop'],can['box'],t['current_crop'],pred,offsets,reference_observation=old['box']))
        assert len(c['plan_old']['cells'])==c['points']
        c['clipped']=any(b[0]<=0 or b[1]<=0 or b[2]>=1280 or b[3]>=720 for b in [old['box'],now['box']])
        paths += [nativepath,eventpath,rawpath,ROOT/f'data/bdd100k/labels/box_track_20/train/{c["video"]}.json',ROOT/f'runs/train_cache/{c["video"]}/alignment.json']
        for key,seq,possible in zip(c['map_keys'],c['sequences'],c['map_paths']):
            assert hashlib.sha256(json.dumps([c['video'],seq],separators=(',',':')).encode()).hexdigest()[:32]==key
            candidates=[ROOT/p for p in possible if p.endswith('.npz')]+[ROOT/p for p in manifests if Path(p).stem==key]
            assert candidates,(key,'No ready C8/R134 map')
            path=candidates[0];paths.append(path)
            if path.relative_to(ROOT).as_posix() in manifests:assert sha(path)==manifests[path.relative_to(ROOT).as_posix()]
            mapinfo[key]=dict(path=path.relative_to(ROOT).as_posix(),old_sha256=sha(path),sequence=seq,video=c['video'])
        readiness.append(dict(video=c['video'],frame=c['frame'],old=c['old'],new=c['new'],kind=c['kind'],clipped=c['clipped'],old_cells=len(c['plan_old']['cells']),joint_cells=len(c['plan_joint']['cells']),anchor_changed=c['plan_joint']['anchor']['changed']))
    # Restore retinal inputs from archives only. No image decode or old-map replay.
    normal=read(ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json');mu=np.array(normal['mean'])[:34];sd=np.array(normal['sd'])[:34];parity={};missing_r1={}
    for video in sorted({c['video'] for c in cases}):
        needed={key:r for key,r in mapinfo.items() if r['video']==video};required={(f,tuple(c)) for r in needed.values() for f,c in r['sequence']};values={}
        candidates=sorted((ROOT/'runs').glob(f'*/retinal/{video}.npz'))
        candidates += [p for p in [ROOT/f'runs/singleton_forward_backward_transfer_20261108/{video}.retinal.npz',ROOT/f'runs/singleton_matched_support_20261108/{video}.retinal.npz'] if p.exists()]
        if video=='02fa8491-2a4967e3':candidates.append(ROOT/'runs/singleton_matched_support_20261108/retinal.npz')
        for path in candidates:
            with np.load(path,allow_pickle=False) as z:
                if not {'frames','crops','hex'}<=set(z.files):continue
                hits=0
                for f,c,h in zip(z['frames'],z['crops'],z['hex']):
                    item=(int(f),tuple(map(int,c)))
                    if item in required:
                        if item in values:assert np.array_equal(values[item],h)
                        values[item]=h.copy();hits+=1
                if hits:paths.append(path)
        assert required<=values.keys(),(video,'Missing archived views',required-values.keys())
        for key,r in needed.items():
            hx=np.stack([values[(f,tuple(c))] for f,c in r['sequence']]);inputs[key]=hx
            actual=((receptor_features_rich(hx[None].astype(float))[-1][0].T-mu)/sd).astype(np.float32)
            with np.load(ROOT/r['path'],allow_pickle=False) as z:
                if 'R1' in z.files:err=float(np.max(abs(actual-z['R1'])));assert err<=1e-4,(key,err);parity[key]=err
                else:missing_r1[key]=actual
    with (OUT/'inputs.npz').open('xb') as f:np.savez_compressed(f,**inputs)
    with (OUT/'missing_input_features.npz').open('xb') as f:np.savez_compressed(f,**missing_r1)
    write('readiness.json',dict(passed=True,cases=readiness,excluded=excluded,sequences=len(inputs),r134_max_error=max(parity.values()),r134_errors=parity,new_r134_keys=list(missing_r1),no_missing_retinal_views=True,ready_maps=mapinfo))
    paths += [OUT/'inputs.npz',OUT/'missing_input_features.npz',OUT/'readiness.json']
    p=dict(registered=stamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline='2026-11-09T07:40:00+03:00',cases=cases,maps=mapinfo,
        selection='All directly current-annotated singleton-car pairs with original >=3 common-support cells from the archived86-video inventory, excluding ALL20 reader-development videos. Six events, five videos,5own/1foreign. Two original-unsupported events retained as excluded untested, not asserted unsupported by the new geometry. No score-based selection. Prior trajectory identity anchors remain conditional, last reference may not be directly annotated.',
        reader='Frozen observed_spatial_reader.py from825a59de: same19 shifts, full common support, equal spatial correlation per reference-active channel and visible raw-detection edge only on an image-clipped axis. No new masks, thresholds, scales, learned parameters or core weights.',
        comparison='Old global normalization and prediction; equal-channel normalization alone; visible-edge geometry alone; joint reader. Apply identically to C8,C34 and five fixed input controls, including equal34-channel temporal and spatial controls. Report fixed predicted zero-shift and maximum, support and unsupported scores.',
        reuse='Read all12 ready C8 maps and10 ready R134 maps. Derive only2 absent R134 controls from archived retinal input using the fixed normalization. Recover ONLY26 missing decoder-input maps, with oldC8 parity<=1e-4. Reuse all existing score grids when exact support/query/normalization matches. Existing equal-channel scores on five events reused where anchor unchanged. No old inference, detector, video decode or ready score grids repeated.',
        math='Unchanged13-test reader proof reused by exact source SHA; affine coordinate plan independently checked against saved support/query. New cases do not justify changing math or picking thresholds.',
        metrics='Descriptive pooledAUROC, and all own-minus-foreign rankings, separately for clipped/unclipped and all matched supported. Paired5000 video bootstrap seed42 only if resample has both classes; missing class yields null, never imputed. With one foreign/video, intervals cannot establish broad transfer.',
        criteria=[dict(criterion='Frozen sources, exact direct-current annotation, raw-box provenance, shared geometry and independent interpolation agree; old scores exactly reused wherever available.',met=None),dict(criterion='C34 joint pooled AUROC improves over old C34 on this fixed other-video set, and neither available clip stratum worsens; report all point values and uncertainty.',met=None),dict(criterion='C34 joint exceeds both34-channel matched input controls; report strongest prior geometry/normalization-only control without selecting a deployment threshold.',met=None)],
        stop='No deployment or real ID claim. If transfer fails or coverage is inadequate, retain the failed criterion; do not retune on these labels. Real ID fate requires an independent full association test later.',
        sources={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(paths))})
    write('protocol.json',p)
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);assert not any(c['id']==OUT.name for c in cat['experiments'])
    cat['experiments'].append(dict(id=OUT.name,title='Перенос совместного пространственного считывателя на другие видео с обрезкой и без неё',area='flyvis',state='running',started='2026-11-09',question='Сохраняется ли выигрыш зафиксированной нормировки и привязки края на других подтверждённых парах?',hypothesis='Совместная правка может исправлять чтение усечённых объектов, но равный вес каналов и шумная кромка могут ухудшать перенос.',expected='Шесть пар вне20видео разработки, все старые карты и оценки переиспользованы, новые26каналов восстановлены только при совпадении8старых.',acceptance=p['criteria'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Зафиксированное сравнение до новых оценок')]));catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    journal('Зафиксирован перенос совместного считывателя на другие видео',f'*Постановка (`{OUT.name}`).* Все6прямо подтверждённых текущей разметкой пар с прежней поддержкой≥3 из каталога86видео вне20видео разработки:5своих и1чужая,5видео. Отдельно обрезка исходной детекции кадром. Последний шаблон не всегда размечен прямо; прошлую цель подтверждают прежние траекторные опоры. Ещё2случая прежней нулевой поддержки не объявляются непригодными для новой геометрии.12карт C8/R134 готовы, все входы восстановлены из архивов; требуется только26недостающих каналов. Те же19сдвигов, обе нормировки и геометрии, семь представлений, без подбора порогов. При одной чужой паре вывод обобщения ограничен. Источник `runs/{OUT.name}/protocol.json`.')
    print('Registered',readiness,'R1 parity',max(parity.values()),flush=True)
def restore():
    p=read(OUT/'protocol.json');verify(p)
    if (OUT/'maps_complete.json').exists():verify(read(OUT/'maps_complete.json'));return
    import torch,psutil,time
    from tracking.gpu_lock import GpuLock
    from scripts.z13_target_model import build_arm
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    normal=read(ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json');mu=np.array(normal['mean'])[:34];sd=np.array(normal['sd'])[:34];files={};maximum=0.
    (OUT/'maps').mkdir(exist_ok=True)
    with np.load(OUT/'inputs.npz',allow_pickle=False) as z,GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        _,_,bb=build_arm('C',torch,None);assert bb.dev.type=='cuda' and not bb.net.training;weights={k:v.detach().clone() for k,v in bb.net.named_parameters()}
        for key,r in p['maps'].items():
            dest=OUT/'maps'/(key+'.npz');marker=dest.with_suffix('.json')
            if marker.exists():
                receipt=read(marker);assert sha(dest)==receipt['sha256'];files[dest.relative_to(ROOT).as_posix()]=receipt['sha256'];maximum=max(maximum,receipt['old_c8_error']);continue
            assert not (ROOT/f'runs/full_decoder_outputs_20261108/maps/{key}.npz').exists(),'Ready full map must be reused, not recomputed'
            x=torch.from_numpy(z[key][None].astype(np.float32)).to(bb.dev);state=bb.init(x[:,:1])
            for t in range(1,x.shape[1]):state,f=bb.step(state,x[:,t-1],x[:,t])
            full=((f[0].cpu().numpy().T.astype(float)-mu)/sd).astype(np.float32)
            with np.load(ROOT/r['path'],allow_pickle=False) as old:error=float(np.max(abs(full[:,4:12]-old['C'])))
            assert error<=1e-4,(key,error);extra=full[:,OTHER];assert np.isfinite(extra).all()
            with dest.open('xb') as out:np.savez_compressed(out,C_OTHER=extra)
            receipt=dict(sha256=sha(dest),old_c8_error=error,protocol_sha256=sha(OUT/'protocol.json'));write(Path('maps')/marker.name,receipt);files[dest.relative_to(ROOT).as_posix()]=sha(dest);maximum=max(maximum,error)
            print('Recovered missing26',len(files),'/',len(p['maps']),'oldC8error',error,flush=True);time.sleep(.05)
        assert all(torch.equal(v,weights[k]) for k,v in bb.net.named_parameters())
    verify(p);write('maps_complete.json',dict(passed=True,sources=files,sequences=len(files),old_c8_max_error=maximum,weights_unchanged=True,zero_new_input_views=True))

if __name__=='__main__':{'register':register,'restore':restore}[sys.argv[1]]()
