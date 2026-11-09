"""Frozen A/B/C memory diagnostic. No detector, identity assignment or tuning.

A: original episodic path; B: common causal path, latest snapshot;
C: that exact common path/current snapshot, earlier annotated snapshot.
The timestamp is evaluation construction, never a runtime memory policy.
"""
import os
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import ast,hashlib,json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,6,35,tzinfo=MSK)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(STOP-datetime.now(MSK)).total_seconds()
if remaining<=0:raise SystemExit('Scientific save/stop reached')
guard_entrypoint(timeout=min(2400,remaining))
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.foveated_eyes import crop_box
OTHER=[0,1,2,3]+list(range(12,34));SELECT=[0,5,11,16,17,22,28,33]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v
def write(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for path,h in p['sources'].items():assert sha(ROOT/path)==h,path
def key(video,seq):return hashlib.sha256(json.dumps([video,seq],separators=(',',':')).encode()).hexdigest()[:32]
def stamp():return datetime.now(MSK).isoformat()
def status(stage,**kw):(OUT/'status.json').write_text(json.dumps(dict(updated=stamp(),stage=stage,**cast(kw)),ensure_ascii=False,indent=2),encoding='utf8')
def journal(title,body):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf8');now=datetime.now(MSK)
    p.write_text(s.replace('## Журнал изменений\n',f'## Журнал изменений\n\n**{now:%d.%m.%Y, %H:%M} MSK** — {title}.\n\n{body}\n\n',1),encoding='utf8')
def crop(b):return list(crop_box(((b[0]+b[2])/2,(b[1]+b[3])/2,b[2]-b[0],b[3]-b[1]),(1280,720)))
def movie(video):
    paths=list((ROOT/'data/bdd100k/videos').glob(f'*/{video}.mov'));assert len(paths)==1,(video,paths)
    return paths[0]
def metadata(video):
    candidates=[ROOT/f'runs/{folder}/{video}/motion/flyvis_tiled.json' for folder in ['train_cache','z13_val20_matched_20261024/cache','video_rate','newval_cache']]
    candidates=[p for p in candidates if p.exists()];assert candidates,video
    return candidates[0]
def register():
    parent=ROOT/'runs/direct_memory_anchor_inventory_20261109';old=read(parent/'protocol.json');verify(old)
    data=read(parent/'results.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.array(g['points']);reader=ObservedSpatialReader(points,g['eye_size'],[1280,720])
    axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2]);cases=[];requests={};paths=[Path(__file__),parent/'protocol.json',parent/'results.json',parent/'audit.json',parent/'literature_check.json']
    for r in data['records']:
        if r['status']!='both_unique':continue
        t=r['track'];h=t['history'][-1];video=r['video'];info=old['videos'][video]
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs']
        # Append real public native observations, not hidden annotations/interpolation.
        prefix=r['reference_recipe'];known={v['frame']:v['crop'] for v in t['history']};tail=native[(native[:,5]==r['old'])&(native[:,0]>r['reference_frame'])&(native[:,0]<=r['last_observation'])]
        assert len(set(map(int,tail[:,0])))==len(tail)
        common=prefix+[[int(v[0]),known.get(int(v[0]),crop(v[1:5]))] for v in tail]+t['search']
        original=r['current_recipe'];assert common[-1]==original[-1] and common[-len(t['search']):]==t['search']
        last_index=len(common)-len(t['search'])-1;early_index=len(prefix)-1
        assert common[last_index]==[h['frame'],h['crop']] and early_index>0 and last_index>=early_index
        c0=np.array(h['crop']);ct=np.array(t['current_crop']);scale=np.maximum((c0[2:]-c0[:2])/g['eye_size'],(ct[2:]-ct[:2])/g['eye_size']);offsets=(axes@np.array([[13.,6.5],[0.,13.]]))*scale
        prediction=np.array(t['mean'][:2])-(np.array(h['box'][:2])+h['box'][2:])/2
        # Recover last raw observation only from frozen caches (same helper as audited parent).
        ns=dict(np=np);source=ROOT/'runs/clipped_anchor_readout_20261109/study.py';node=next(n for n in ast.parse(source.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='raw_observation');exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={n:z[n] for n in ['frame_idx','conf','cls','xyxy']}
        cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config'];last=ns['raw_observation'](native,raw,r['old'],h['frame'],cfg)
        assert np.array_equal(np.array(last['state_box'],np.float32),np.array(h['box'],np.float32))
        bp=reader.plan(h['box'],h['crop'],r['candidate']['box'],t['current_crop'],prediction,offsets,reference_observation=last['box'])
        early=r['reference_raw'];ec=np.array(prefix[-1][1]);es=(ec[2:]-ec[:2])/g['eye_size'];ep=np.array(t['mean'][:2])-(np.array(early['state_box'][:2])+early['state_box'][2:])/2
        cp=reader.plan(early['state_box'],ec,r['candidate']['box'],t['current_crop'],ep,offsets,reference_observation=early['box'])
        # One common current sampling grid and cell count. C is sampled at the
        # corresponding earlier physical location; B's current grid never moves.
        cells=np.array(bp['cells'],int);xb=points[cells]*(c0[2:]-c0[:2])/g['eye_size']+c0[:2];xe=xb+bp['prediction']-cp['prediction'];eq=(xe-ec[:2])/es
        eb=np.array(early['state_box']);si=reader.mesh.find_simplex(eq);keep=((xe>=eb[:2])&(xe<=eb[2:])).all(1)&(si>=0)
        bquery=(points*es+ec[:2]+cp['prediction']-bp['prediction']-c0[:2])/((c0[2:]-c0[:2])/g['eye_size'])
        valid_b=reader.mesh.find_simplex(bquery)>=0
        keep&=valid_b[reader.mesh.simplices[np.maximum(si,0)]].all(1)
        commonplan={**bp,'cells':cells[keep],'query':np.array(bp['query'])[keep]}
        assert np.allclose((eq[keep]*es+ec[:2]+cp['prediction'])[:,None]+offsets,(points[np.array(commonplan['cells'])]*(c0[2:]-c0[:2])/g['eye_size']+c0[:2]+bp['prediction'])[:,None]+offsets,atol=1e-10)
        c=dict(video=video,frame=r['frame'],old=r['old'],new=r['new'],kind=r['kind'],clipped=r['clipped'],reference_age=r['reference_age'],reference_frame=r['reference_frame'],last_frame=h['frame'],early_equals_last=r['reference_frame']==h['frame'],original=original,common=common,early_index=early_index,last_index=last_index,bplan=bp,commonplan=commonplan,early_query=eq[keep],roundtrip_b_query=bquery,roundtrip_valid_nodes=valid_b,early_native_plan=cp,reference_gt=r['reference_gt_hits'],current_gt=r['current_gt_hits'],last_reference_identity='unknown unless same timestamp as directly annotated earlier reference')
        c['snapshots']={}
        for label,seq in [('A_last',original[:len(t['history'])]),('A_current',original),('B_early',common[:early_index+1]),('B_last',common[:last_index+1]),('B_current',common)]:
            k=key(video,seq);c['snapshots'][label]=k
            if len(commonplan['cells'])>=3:requests[k]=dict(video=video,sequence=seq)
        cases.append(c)
        paths.extend([ROOT/info['native'],ROOT/info['raw'],movie(video),metadata(video)])
        if (ROOT/f'runs/z13_data/{video}.npz').exists():paths.append(ROOT/f'runs/z13_data/{video}.npz')
    assert len(cases)==22
    # Find exact ready snapshots before any inference. A ready result remains read-only.
    for k,v in requests.items():
        candidates=sorted((ROOT/'runs').glob(f'*/maps/{k}.npz'))+sorted((ROOT/'runs').glob(f'*/features/{k}.npz'))+sorted((ROOT/'runs').glob(f'*/{k}.npz'))
        v['ready']={}
        for p in candidates:
            with np.load(p,allow_pickle=False) as z:
                for name in ['C','C_OTHER','R1']:
                    if name in z.files:
                        a=z[name];assert a.shape in [(721,8),(721,26),(721,34)]
                        slot='C8' if name=='C' and a.shape[1]==8 else ('C34' if name=='C' else name)
                        if slot not in v['ready']:v['ready'][slot]=dict(path=p.relative_to(ROOT).as_posix(),array=name);paths.append(p)
    for video in sorted({r['video'] for r in requests.values()}):
        paths+=list((ROOT/'runs').glob(f'*/retinal/{video}.npz'))
        paths+=list((ROOT/'runs').glob(f'*/{video}.retinal.npz'))
    paths += [ROOT/p for p in ['tracking/observed_spatial_reader.py','tracking/channel_balanced_correspondence.py','tracking/clipping_aware_correspondence.py','tracking/z13_readout.py','tracking/foveated_eyes.py','tracking/retina.py','scripts/z13_target_model.py','tracking/flyvis_adapter.py','tracking/flyvis_activity.py','tracking/flyvis_motion.py','tracking/flyvis_checkpoint.py','runs/z13_local_peak_20261021/geometry.json','runs/flyvis/results/flow/0000/000/best_chkpt','runs/flyvis/results/flow/0000/000/_meta.yaml','runs/z13_target_center_dev_20261018/C/normalization.json','runs/z13_target_center_dev_20261018/R1/normalization.json','runs/clipped_anchor_readout_20261109/runtime_verification.json']]
    p=dict(registered=stamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline='2026-11-09T06:40:00+03:00',scientific_save_stop=STOP.isoformat(),cases=cases,requests=requests,
        scope='All22 directly unique earlier/current raw-GT pairs from pre-score inventory; unsupported remain explicit. Binary identity refers to earlier annotated observation. Last unannotated observation is not assumed correct. GT timestamps are evaluation construction, never runtime policy.',
        arms='A original short path/latest template; B common causal path/latest template; C SAME common path/current map, early template. Common path starts at frozen reference recipe, adds all real public native observations through last, then original searches. Uses original fixed per-observation integration, no clock change or fake frames. This is a contiguous state trajectory over saved observations, not a claim of dense all-video-frame replay.',
        geometry='B native support reported. Primary B/C comparison uses identical B current query locations for all19 shifts, restricts to points whose earlier aligned reference location is inside its archived statebox and eye hull. B reads its native template point; C linearly interpolates earlier template there. Same number/current physical positions, same19 offsets and channel-balancing math; A_common uses same final grid. This controls sampling coverage, not viewpoint, subcell phase or interpolation blur.',
        controls='C34, C8, fixed34-channel temporal image EMA/transients, fixed34-channel spatial input (17EMA plus6neighbor mean), fixed8temporal, center8 and neighbor8 from same retinal views and full A/B/C paths. Ordinary EMA is stateful; A/B image controls need not be identical. No learned mask, threshold or runtime selection.',
        math='For each common point xB, xE=xB+dB-dE, hence xE+dE+offset=xB+dB+offset. hCurrent_B is byte-identical for B/C by construction. B/C only replaces sampled reference content at fixed current samples and support; age/visibility/neural phase remain interpretations to audit, not removed. An extra interpolation pass may smooth C, so smooth B with the exact same earlier-grid roundtrip as a non-memory interpolation control.',
        interpolation_control='B_roundtrip: project B feature map onto all earlier-eye grid points via xB=xE+dE-dB, then sample it back at C reference queries; preserve domain, report any undefined nodes and restrict all common variants to fully valid roundtrip triangles. This changes only interpolation, not template age; no threshold.',
        metrics='Descriptive own-vs-foreign AUROC/ranking counts before (offset9) and after maximum over unchanged19 shifts; same supported events across A/B/C and representations, unknowns/undefined never imputed. Per-video paired5000 bootstrap seed42 only with both classes, report valid resamples. Show own/foreign individual changes and clipped stratum separately. No IDSW/HOTA measured.',
        acceptance=[dict(criterion='Exact shared current samples, immutable input/weights, all available archived retinal witnesses exact (new videos without an archived witness explicitly disclosed), independent interpolations and similarities agree<=1e-9; all ready maps reused.',met=None),dict(criterion='At fixed common support C34 early C improves AUROC over latest B and interpolation-only Broundtrip after search, including available clip strata without observed worsening.',met=None),dict(criterion='C34 C exceeds matched34-channel temporal and spatial input C; uncertainty and sparse coverage reported, no full tracker claim.',met=None)],
        decision='Stop this hypothesis if fixed comparison fails; no retuning/automatic promotion. If sparse support/classes preclude inference, inconclusive. Even a passing diagnostic requires causal template policy and independent real ID-fate evaluation.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))})
    # A genuinely new coordinate check, not a repetition of the reader's13 tests.
    affine=points@np.array([[.7,-.2],[.3,.8]])+np.array([4.,-2.]);error=0.
    for c in cases:
        q=np.array(c['early_query']);bq=np.array(c['roundtrip_b_query']);valid=np.array(c['roundtrip_valid_nodes'])
        if not len(q):continue
        sampled=LinearNDInterpolator(reader.mesh,affine)(bq[valid]);back=np.zeros_like(affine);back[valid]=sampled
        twice=LinearNDInterpolator(reader.mesh,back)(q)
        c0=np.array(c['common'][c['last_index']][1]);ec=np.array(c['common'][c['early_index']][1]);es=(ec[2:]-ec[:2])/g['eye_size'];bs=(c0[2:]-c0[:2])/g['eye_size'];mapped=(q*es+ec[:2]+np.array(c['early_native_plan']['prediction'])-np.array(c['bplan']['prediction'])-c0[:2])/bs
        expected=mapped@np.array([[.7,-.2],[.3,.8]])+np.array([4.,-2.]);error=max(error,float(np.max(abs(twice-expected))))
    assert error<1e-9
    write('mathematical_checks.json',dict(passed=True,new_common_grid_affine_roundtrip_max_error=error,existing_reader_proof='runs/clipped_anchor_readout_20261109/runtime_verification.json',shared_current_isolation='B/C consume one exact current map object and the same query tensor; only reference content changes. Matched roundtrip control bounds interpolation explanations, not actual visibility/age.',counterexample='With equal norm feature pairs [appearance,dynamic], appearance cos=1-epsilon, old target dynamic +v and current target -v while distractor +v, early wrong-minus-own=(2*gamma^2-epsilon)/(1+gamma^2)>0 for gamma^2>epsilon/2. This only disproves unconditional earlier-is-better, not an observed cause in FlyVis.'))
    p['sources'][(OUT/'mathematical_checks.json').relative_to(ROOT).as_posix()]=sha(OUT/'mathematical_checks.json')
    write('protocol.json',p)
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);assert not any(x['id']==OUT.name for x in cat['experiments'])
    cat['experiments'].append(dict(id=OUT.name,title='Ранняя и последняя память на одной истории состояния — проверка личности',area='flyvis',state='running',started='2026-11-09',question='Меняет ли выбор раннего шаблона порядок своих и чужих связей при одной текущей карте и общей пространственной поддержке?',hypothesis='Ранний шаблон может сохранить различия цели, но возраст, ракурс и нейронная фаза могут ухудшить соответствие; польза не предполагается.',expected='Фиксированные A/B/C, C34/C8 и одинаковые обычные временные и пространственные признаки, только недостающие данные.',acceptance=p['acceptance'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Протокол до новых карт и оценок')]))
    catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    journal('Зафиксировано сравнение ранней и последней памяти на общей истории',f'*Постановка (`{OUT.name}`).* Все22пары прошлого аудита,18своих/4чужих. A — прежняя короткая история; B — общая причинная цепочка сохранённых наблюдений с последним шаблоном; C — та же цепочка и текущая карта, ранний шаблон. Точки текущего поиска и число ячеек B/C совпадают; дополнительный контроль отделяет размытие от интерполяции. Обе34-канальные обычные ветви также имеют временную память. Часы, веса,19сдвигов и пороги неизменны; готовое не повторяется. Неразмеченное последнее наблюдение не объявляется доказанной своей машиной. Это диагностика считывателя, не новый рабочий трекер. Литературное основаниеSTARK/KeepTrack/MeMOTR проверено ранее в `runs/direct_memory_anchor_inventory_20261109/literature_check.json`; простые запреты памяти и часы уже проверялись отрицательно. До06:35сохранение/остановка, срок06:40. Источник `runs/{OUT.name}/protocol.json`.')
    status('registered',events=len(cases),common_support=[len(c['commonplan']['cells']) for c in cases],snapshots=len(requests));print('REGISTERED',len(cases),len(requests),flush=True)

def retinal():
    p=read(OUT/'protocol.json');verify(p)
    if (OUT/'retinal_complete.json').exists():verify(read(OUT/'retinal_complete.json'));return
    import torch,cv2
    from types import SimpleNamespace
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.foveated_eyes import EyeBank
    from tracking.retina import make_retina
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);activate_flyvis(device='cpu')
    from flyvis.datasets.rendering import BoxEye
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');eye=BoxEye(extent=15,kernel_size=13);centers=eye.receptor_centers.cpu().numpy();assert np.array_equal(np.column_stack([centers[:,1]+g['eye_size']//2,centers[:,0]+g['eye_size']//2]),np.array(g['points']))
    bank=SimpleNamespace(torch=torch,eye_size=g['eye_size'],eye=eye,estimator=SimpleNamespace(flyvis=SimpleNamespace(device=torch.device('cpu'))));files={};counts={}
    for video in sorted({r['video'] for r in p['requests'].values()}):
        dest=OUT/'retinal'/(video+'.npz');marker=dest.with_suffix('.json')
        if marker.exists():r=read(marker);assert sha(dest)==r['sha256'];files[dest.relative_to(ROOT).as_posix()]=sha(dest);counts[video]=r;continue
        required={(f,tuple(c)) for r in p['requests'].values() if r['video']==video for f,c in r['sequence']};values={}
        def collect(z,fn,cn,hn):
            for f,c,h in zip(z[fn],z[cn],z[hn]):
                k=int(f),tuple(map(int,c))
                if k in required:
                    if k in values:assert np.array_equal(values[k],h),k
                    values[k]=h.copy()
        witness=None;wval=None
        if (ROOT/f'runs/z13_data/{video}.npz').exists():
            with np.load(ROOT/f'runs/z13_data/{video}.npz',allow_pickle=False) as z:
                witness=int(z['fr_frame'][0]),tuple(map(int,z['fr_crop'][0]));wval=z['fr_hex'][0].copy();collect(z,'fr_frame','fr_crop','fr_hex')
        candidates=sorted((ROOT/'runs').glob(f'*/retinal/{video}.npz'))+sorted((ROOT/'runs').glob(f'*/{video}.retinal.npz'))
        for path in candidates:
            assert path.relative_to(ROOT).as_posix() in p['sources']
            with np.load(path,allow_pickle=False) as z:
                if {'frames','crops','hex'}<=set(z.files):
                    if witness is None:witness=int(z['frames'][0]),tuple(map(int,z['crops'][0]));wval=z['hex'][0].copy()
                    collect(z,'frames','crops','hex')
        missing=required-values.keys();error=None
        if missing:
            render={}
            for f,c in missing|({witness} if witness is not None else set()):render.setdefault(f,[]).append(c)
            fps=read(metadata(video))['fps'];retina=make_retina('lamina',float(fps),blur_downscale=4);cap=cv2.VideoCapture(str(movie(video)));assert cap.isOpened()
            with torch.inference_mode():
                for frame in range(max(render)+1):
                    if datetime.now(MSK)>=STOP:raise RuntimeError('Save/stop deadline')
                    ok,im=cap.read();assert ok,(video,frame);gray=retina.process(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
                    if frame in render:
                        batch=EyeBank._render(bank,gray,render[frame])[:,0].cpu().numpy().reshape(len(render[frame]),721).astype(np.float16)
                        for c,h in zip(render[frame],batch):
                            k=frame,c
                            if k==witness:error=float(np.max(abs(h.astype(float)-wval)));assert np.array_equal(h,wval)
                            if k in required:values[k]=h
                    if frame%200==0:status('retina',video=video,frame=frame,maximum=max(render),videos_done=len(files));print('RETINA',video,frame,max(render),flush=True)
            cap.release();assert witness is None or error==0
        assert required<=values.keys();keys=sorted(required);dest.parent.mkdir(exist_ok=True)
        with dest.open('xb') as f:np.savez_compressed(f,frames=np.array([k[0] for k in keys]),crops=np.array([k[1] for k in keys]),hex=np.stack([values[k] for k in keys]))
        r=dict(sha256=sha(dest),new_views=len(missing),reused_views=len(required)-len(missing),archived_witness_available=witness is not None,witness_error=error);write(Path('retinal')/marker.name,r);files[dest.relative_to(ROOT).as_posix()]=sha(dest);counts[video]=r
    verify(p);write('retinal_complete.json',dict(passed=True,sources=files,counts=counts));status('retina_complete',videos=len(files))

if __name__=='__main__':{'register':register,'retinal':retinal}[sys.argv[1]]()
