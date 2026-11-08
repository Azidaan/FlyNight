"""One frozen CPU gate for a spatial background-aware patch filter."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import ast,hashlib,json,sys,subprocess
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'runs/bbox_fit_maps_20261108'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=1200)
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from tracking.hex_background_filter import BackgroundPatchFilter,background_offsets,unit_patch
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
AXES=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])
BASIS=np.array([[13.,6.5],[0.,13.]])
NAMES=['FlyVis','center8','neighbor8','input8_temporal'];SELECT=[0,5,11,16,17,22,28,33]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify(p):
    for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
def stamp():return datetime.now(timezone(timedelta(hours=3))).isoformat()
def sample(mesh,query,values):
    si=mesh.find_simplex(query);assert (si>=0).all()
    tr=mesh.transform[si];w=np.einsum('...ij,...j->...i',tr[...,:2,:],query-tr[...,2,:])
    w=np.concatenate([w,1-w.sum(-1,keepdims=True)],axis=-1)
    a=np.einsum('...v,...vc->...c',w,values[mesh.simplices[si]])
    b=LinearNDInterpolator(mesh,values)(query);assert np.allclose(a,b,rtol=0,atol=1e-9)
    return a
def geometry(e,g,mesh):
    t=e['row_data'];h=t['history'][-1];c0=np.asarray(h['crop']);ct=np.asarray(t['current_crop'])
    s0=(c0[2:]-c0[:2])/g['eye_size'];st=(ct[2:]-ct[:2])/g['eye_size']
    oldbox=np.asarray(h['box']);box=np.asarray(e['box']);points=np.asarray(g['points'])
    pred=np.asarray(t['mean'][:2])-(oldbox[:2]+oldbox[2:])/2;off=(AXES@BASIS)*np.maximum(s0,st)
    x=points*s0+c0[:2];trial=x[:,None]+pred+off;query=(trial-ct[:2])/st
    keep=((x>=oldbox[:2])&(x<=oldbox[2:])).all(1)&((trial>=box[:2])&(trial<=box[2:])).all((1,2))
    keep&=(mesh.find_simplex(query)>=0).all(1);idx=np.flatnonzero(keep)
    box_eye=np.r_[(oldbox[:2]-c0[:2])/s0,(oldbox[2:]-c0[:2])/s0]
    neg=background_offsets(points,idx,box_eye,mesh) if len(idx)>=3 else np.empty((0,2))
    return dict(cells=idx.tolist(),prediction=pred.tolist(),offsets=off.tolist(),query=query[idx].tolist(),negative_offsets=neg.tolist())
def register():
    inv=read(BASE/'inventory.json');manifest=read(BASE/'features_complete.json')['files']
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');mesh=Delaunay(g['points'])
    files=[Path(__file__),ROOT/'tracking/hex_background_filter.py',ROOT/'tests/test_hex_background_filter.py',
        ROOT/'tracking/z13_readout.py',ROOT/'scripts/z13_target_model.py',BASE/'inventory.json',BASE/'features_complete.json',
        ROOT/'runs/z13_local_peak_20261021/geometry.json',
        ROOT/'runs/singleton_support_size_20261108/results.json',ROOT/'runs/singleton_support_size_20261108/protocol.json',
        ROOT/'runs/singleton_forward_backward_20261108/results.json',ROOT/'runs/singleton_forward_backward_20261108/protocol.json']
    events=[]
    for e in inv['events']:
        e['geometry']=geometry(e,g,mesh);events.append(e)
        for key in [e['reference_key'],e['current_key']]:
            path=BASE/'features'/(key+'.npz');assert sha(path)==manifest[key];files.append(path)
        files.append(ROOT/f'runs/bbox_singleton_pilot_20261108/labels/{e["video"]}.json')
    sizes=[len(e['geometry']['cells']) for e in events]
    write('mathematical_checks.json',dict(passed=True,method='Weighted direct spatial ridge: one positive target patch, mean negative-patch loss, fixed unit regularizer. Self-normalize score.',
        target_only_identity='With no negatives the calibrated score equals the old normalized centered dot product exactly.',
        conditioning='Normalized samples give tr(X^T X)<=2, so the unit-regularized matrix has condition<=3 and target response>=1/3.',
        tests='tests/test_hex_background_filter.py: independent primal/dual equality, target-only identity, real-patch geometry, degenerate and stale-state rejection.',
        events=len(events),supported=sum(n>=3 for n in sizes),with_background=sum(n>=3 and bool(e['geometry']['negative_offsets']) for n,e in zip(sizes,events))))
    p=dict(registered=stamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        deadline='2026-11-09T07:40:00+03:00',events=events,representations=NAMES,
        intervention='Replace equal-weight target correlation with a spatial patch ridge filter learned from the old reference target and at most64 real nonoverlapping background patches of that same old eye. Fixed unit ridge, average negative loss. No global training, model/core changes or thresholds.',
        matched_control='Normalized centered correlation on identical eligible cells and19 shifts, four identical representations. Context-free limit is algebraically the baseline. No circular shifts or wrap, no new raster rendering.',
        older_attempts='z13_target_model_probe_v1 learned one feature-channel weight for every independent cell. z13_spatial_derivatives_v1 added local slopes. Here each position/channel in the full target patch has a distinct weight and training samples are entire displaced real patches. This is not full BACF.',
        selection='All200 stored fit20 singleton proposals, all classes. Geometry support selected without labels. Unknown labels remain unknown. All14 recent case-study maps and7 evaluation pairs are not the evaluation population here.',
        gate='On identical supported known fit20 pairs, require AUROC(FlyVis background)>AUROC(FlyVis correlation) and AUROC(FlyVis background)>max(AUROC(simple background controls)). Point direction gate only for advancing to frozen independent evaluation, not acceptance of tracker benefit.',
        alternative='The filter may only learn past background, or T4/T5 may lack stable instance cues. Simple-feature controls and the old spatial readouts distinguish the mechanism; a failed gate stops this exact filter rather than changing lambda on evaluation examples.',
        next_action='Only if the directional gate passes: freeze it and prepare independent evaluation, then integrate its evidence before the native global assignment with a zero-intervention baseline. No post-hoc public-ID copying.',
        source=dict(title='Learning Background-Aware Correlation Filters for Visual Tracking',authors='Hamed Kiani Galoogahi, Ashton Fagg, Simon Lucey',year=2017,doi='10.1109/ICCV.2017.129',
            url='https://arxiv.org/pdf/1703.04590',sections='Sections3–4, Eq3, real negative patches in Figure2.',
            code_access='Official author-page MATLAB ZIP link returned502; CVF PDF returned403, arXiv primary manuscript opened. No claim of executing official BACF code.'),
        acceptance='Input hashes stable; math tests pass; same support and sampling for both readers and all representations; independent interpolation and filter-score algebra agree1e-9; completed per-event files reused on resume.',
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))})
    write('protocol.json',p)
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath)
    assert not any(c['id']==OUT.name for c in cat['experiments'])
    cat['experiments'].append(dict(id=OUT.name,title='Пространственный фильтр цели с реальными фоновыми примерами',area='flyvis',state='running',started='2026-11-08',
        question='Улучшает ли учёт реального прошлого фона различение своих и чужих продолжений?',hypothesis='Общий фон и движение могут маскировать различия экземпляров; пространственный фильтр может их подавить.',
        expected='Один зафиксированный фильтр против той же корреляции и простых признаков на готовых fit20 запросах.',acceptance=[dict(criterion=p['acceptance'],met=None)],
        evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Условия до расчёта')]))
    catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Зарегистрирован пространственный фильтр с реальными отрицательными участками прошлого кадра.\n\n*Постановка.* По активной цели автора заменить равномерную корреляцию шаблона дискриминативным пространственным фильтром. Готовые 200 fit20 запросов, четыре прежних представления, одинаковая поддержка и19сдвигов. Отличие от старого точечного ridge и локальных производных записано в протоколе. Единственный фиксированный вариант, без изменения ядра и настройки порога. Первичный BACF2017, раздел4, уравнение3; переносится принцип целых реальных отрицательных участков, не официальный трекер. Четыре математических теста пройдены. Карточка `{OUT.name}`, `runs/{OUT.name}/protocol.json`.\n\n'
    path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
def run():
    if (OUT/'results.json').exists():raise SystemExit('Ready fit results exist; no calculation repeated.')
    p=read(OUT/'protocol.json');verify(p);g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.asarray(g['points'],float)
    mesh=Delaunay(points);nb=HexGeometry(points,g['eye_size']).neighbors
    ready={}
    for file in ['singleton_support_size_20261108','singleton_forward_backward_20261108']:
        for r in read(ROOT/f'runs/{file}/results.json')['records']:
            ready[(r['video'],r.get('frame',r.get('current_frame')),r['old'],r['new'],r['representation'])]=r
    directory=OUT/'events';directory.mkdir(exist_ok=True);outputs={}
    for number,e in enumerate(p['events']):
        dest=directory/f'{e["video"]}_{e["frame"]}_{e["old_tid"]}_{e["new_tid"]}.json'
        if dest.exists():
            r=read(dest);assert r['protocol_sha256']==sha(OUT/'protocol.json');outputs[dest.relative_to(ROOT).as_posix()]=sha(dest);continue
        geom=e['geometry'];idx=np.asarray(geom['cells'],int)
        r=dict(protocol_sha256=sha(OUT/'protocol.json'),video=e['video'],frame=e['frame'],old=e['old_tid'],new=e['new_tid'],cls=e['row_data']['cls'],cells=idx.tolist(),supported=len(idx)>=3,variants={})
        if r['supported']:
            arrays={n:[] for n in NAMES}
            for key in [e['reference_key'],e['current_key']]:
                with np.load(BASE/'features'/(key+'.npz'),allow_pickle=False) as z:C=z['C'].astype(float);temporal=z['R1'][:,SELECT].astype(float)
                central=temporal[:,:4];pool=neighbor_aug(central.T[None],nb)[0].T
                for n,a in zip(NAMES,[C,np.concatenate([central,central],1),pool,temporal]):arrays[n].append(a)
            neg=np.asarray(geom['negative_offsets']).reshape(-1,2);query=np.asarray(geom['query'])
            for name,(old,current) in arrays.items():
                negative=sample(mesh,points[idx][None]+neg[:,None],old) if len(neg) else np.empty((0,len(idx),8))
                model=BackgroundPatchFilter()
                if not model.fit(old[idx],negative):r['variants'][name]=dict(valid=False,reason='constant_reference');continue
                patches=sample(mesh,query,current).transpose(1,0,2)
                key=(e['video'],e['frame'],e['old_tid'],e['new_tid'],name);prior=ready.get(key)
                if prior is not None and prior['cells']==idx.tolist():
                    baseline=prior.get('forward_scores',prior.get('forward_reused',{}).get('scores'));reused=True
                else:baseline=[float(unit_patch(old[idx])@unit_patch(v)) if unit_patch(v) is not None else None for v in patches];reused=False
                filtered=[model.score(v) for v in patches]
                # Independent Woodbury solution without the positive training row.
                z0=unit_patch(old[idx]);validneg=[unit_patch(v) for v in negative];validneg=[v for v in validneg if v is not None]
                if validneg:
                    B=np.stack(validneg)/np.sqrt(len(validneg));q=np.linalg.solve(np.eye(len(B))+B@B.T,B@z0);w=z0-B.T@q
                else:w=z0
                for v,s in zip(patches,filtered):
                    z=unit_patch(v)
                    if z is not None:assert abs(float(z@w/(z0@w))-s)<=1e-9
                valid=[i for i,s in enumerate(filtered) if s is not None];winner=max(valid,key=lambda i:filtered[i]) if valid else None
                validbase=[s for s in baseline if s is not None]
                r['variants'][name]=dict(valid=winner is not None,negative_count=model.negative_count,correlation_scores=baseline,
                    correlation_reused=reused,correlation_max=max(validbase) if validbase else None,filter_scores=filtered,
                    filter_max=filtered[winner] if winner is not None else None,filter_axes=AXES[winner].tolist() if winner is not None else None,
                    self_response=model.normalizer)
        label=read(ROOT/f'runs/bbox_singleton_pilot_20261108/labels/{e["video"]}.json')['rows'][f'{e["frame"]}|{e["old_tid"]}|{e["new_tid"]}']
        r['kind']='own' if label['known'] and label['positive'] else 'foreign' if label['known'] else 'unknown'
        with dest.open('x',encoding='utf8') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False)
        outputs[dest.relative_to(ROOT).as_posix()]=sha(dest)
        if number%10==0:print('completed',number+1,'/',len(p['events']),flush=True)
    verify(p);write('results.json',dict(protocol_sha256=sha(OUT/'protocol.json'),files=outputs))
    write('audit.json',dict(passed=True,input_hashes_stable=True,independent_interpolation_and_filter_algebra_tolerance=1e-9,
        future_or_identity_labels_in_filter=False,ready_event_results_reused=True,result_sha256=sha(OUT/'results.json')))

{'register':register,'run':run}[sys.argv[1]]()
