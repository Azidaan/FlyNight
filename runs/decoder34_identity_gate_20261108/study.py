"""Fixed standard-decoder34 representation versus matched34 input controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import ast,hashlib,json,sys,subprocess
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PARENT=ROOT/'runs/background_patch_gate_20261108';MAPS=ROOT/'runs/full_decoder_outputs_20261108';BASE=ROOT/'runs/bbox_fit_maps_20261108'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from tracking.hex_background_filter import BackgroundPatchFilter,unit_patch
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
OTHER=[0,1,2,3]+list(range(12,34));NAMES=['C34','R134_temporal','R134_spatial']
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify(p):
    for file,digest in p['sources'].items():assert sha(ROOT/file)==digest,file
def sampler():
    path=PARENT/'study.py';tree=ast.parse(path.read_text(encoding='utf8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample')
    namespace=dict(np=np,LinearNDInterpolator=LinearNDInterpolator)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace);return namespace['sample']
def register():
    completed=read(MAPS/'completed.json');assert completed['passed'] and completed['sequences']==399 and completed['old_c8_max_error']<=1e-4
    for file,digest in completed['files'].items():assert sha(ROOT/file)==digest,file
    # A wider tensor alone cannot improve either reader when new channels are zero.
    rng=np.random.default_rng(102);a=rng.normal(size=(9,8));b=rng.normal(size=(9,8));negative=rng.normal(size=(11,9,8))
    m=BackgroundPatchFilter();z=BackgroundPatchFilter();assert m.fit(a,negative)
    assert z.fit(np.pad(a,((0,0),(0,26))),np.pad(negative,((0,0),(0,0),(0,26))))
    error=abs(m.score(b)-z.score(np.pad(b,((0,0),(0,26)))));assert error<1e-12
    write('mathematical_checks.json',dict(passed=True,zero_padding8_to34_filter_error=error,
        zero_padding_correlation='Per-channel centering, squared norms and scalar product are unchanged by26 zero channels.',
        premise='The extra channels are recovered standard decoder inputs, not new fitted features or a scan over65 cell types.'))
    sources=[Path(__file__),PARENT/'protocol.json',PARENT/'study.py',PARENT/'results.json',PARENT/'summary.json',PARENT/'independent_verification.json',
        MAPS/'protocol.json',MAPS/'completed.json',MAPS/'input_readiness.json',ROOT/'tracking/hex_background_filter.py',
        ROOT/'tracking/z13_readout.py',ROOT/'scripts/z13_target_model.py',ROOT/'runs/z13_local_peak_20261021/geometry.json',OUT/'mathematical_checks.json']
    sources += [ROOT/f for f in completed['files']]
    sources += [ROOT/f for f in read(PARENT/'results.json')['files']]
    sources += [BASE/'features'/(key+'.npz') for key in read(MAPS/'protocol.json')['requests']]
    p=dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        deadline='2026-11-09T07:40:00+03:00',parent='runs/background_patch_gate_20261108/protocol.json',
        representations=dict(C34='All34 fixed standard decoder cell-type maps, normalized with existing C statistics. Original C8 is preserved exactly as subset4:12.',
            R134_temporal='Ready34 R1 channels:17 fixed EMA and17 residuals, old normalization.',
            R134_spatial='Same17 normalized EMA channels plus their six-neighbor means,34 channels.'),
        readers=['correlation_max','filter_max'],
        unchanged='All200 fit20 requests, old eligible support and19 shifts, same reference/current maps, same background offsets and fixed unit-ridge model. Preserve all unknowns and unsupported events.',
        reuse='Read C8 and simple8 reader results from parent per-event files, no new8-channel scoring, no repeats of the ready core outputs. Zero-padding equivalence is mathematical, not another data run.',
        choice='Choose the C34 reader with highest AUROC on matched known fit20 pairs. Exact tie chooses correlation. Gate requires it strictly exceeds C8 with that reader and both34-channel simple controls with that reader, as well as the best previous8-channel simple result. No threshold fitting.',
        acceptance=[dict(criterion='Stable inputs, identical support, original8 subset exact, independent interpolation/filter algebra<=1e-9, same45-known pair comparison.',met=None),
            dict(criterion='The frozen directional representation gate in choice passes before any independent/native-tracker evaluation.',met=None)],
        limitation='Conditional training gate only. Selecting between two fixed readers on fit20 is not evidence of independent ID preservation. Actual ID fate and native global assignment remain required.',
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(sources))})
    write('protocol.json',p)
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments'])
    cat['experiments'].append(dict(id=OUT.name,title='Все34 стандартные карты FlyVis против равных временных и пространственных признаков входа',area='flyvis',state='running',started='2026-11-08',
        question='Мешало ли отбрасывание26 стандартных каналов различать экземпляры объектов?',hypothesis='В дополнительных стандартных выходах могут быть сведения сверх выбранных8T4/T5.',expected='Один зафиксированный34-канальный гейт с двумя уже заданными считывателями и двумя равными контролями.',
        acceptance=p['acceptance'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Критерии и выбор до оценок')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Зарегистрирован34-канальный гейт сохранения представления.\n\n*Постановка.* Те же200fit20запросов и45известных допустимых пар. Полные34стандартных выхода против34простых временных каналов и17средних+17соседей. Два фиксированных считывателя, выбор только наfit20; проверочные пары и пороги не подбираются. Восьмиканальные результаты читаются готовыми. Нулевая добавка26каналов математически ничего не меняет. Карточка `{OUT.name}`, `runs/{OUT.name}/protocol.json`. Принятие полного трекера этим гейтом не заменяется.\n\n'
    path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
def run():
    if (OUT/'results.json').exists():raise SystemExit('Ready34-channel scores exist; no calculation repeated.')
    p=read(OUT/'protocol.json');verify(p);parent=read(PARENT/'protocol.json');oldfiles=read(PARENT/'results.json')['files'];oldrows=[read(ROOT/f) for f in oldfiles]
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.asarray(g['points'],float);mesh=Delaunay(points);nb=HexGeometry(points,g['eye_size']).neighbors;sample=sampler()
    directory=OUT/'events';directory.mkdir(exist_ok=True);files={}
    for number,(e,oldrow) in enumerate(zip(parent['events'],oldrows)):
        dest=directory/f'{e["video"]}_{e["frame"]}_{e["old_tid"]}_{e["new_tid"]}.json'
        if dest.exists():
            r=read(dest);assert r['protocol_sha256']==sha(OUT/'protocol.json');files[dest.relative_to(ROOT).as_posix()]=sha(dest);continue
        geom=e['geometry'];idx=np.asarray(geom['cells'],int);r={k:oldrow[k] for k in ['video','frame','old','new','cls','kind','cells','supported']};r.update(protocol_sha256=sha(OUT/'protocol.json'),variants={})
        if r['supported']:
            arrays={n:[] for n in NAMES}
            for key in [e['reference_key'],e['current_key']]:
                with np.load(BASE/'features'/(key+'.npz'),allow_pickle=False) as z:c8=z['C'];temporal=z['R1'].astype(float)
                with np.load(MAPS/'maps'/(key+'.npz'),allow_pickle=False) as z:other=z['C_OTHER']
                full=np.empty((721,34),float);full[:,4:12]=c8;full[:,OTHER]=other;assert np.array_equal(full[:,4:12],c8)
                spatial=neighbor_aug(temporal[:,:17].T[None],nb)[0].T
                for n,a in zip(NAMES,[full,temporal,spatial]):arrays[n].append(a)
            neg=np.asarray(geom['negative_offsets']).reshape(-1,2);query=np.asarray(geom['query'])
            for name,(old,current) in arrays.items():
                negative=sample(mesh,points[idx][None]+neg[:,None],old) if len(neg) else np.empty((0,len(idx),34))
                model=BackgroundPatchFilter();assert model.fit(old[idx],negative)
                patches=sample(mesh,query,current).transpose(1,0,2);z0=unit_patch(old[idx]);unit=[unit_patch(v) for v in patches]
                baseline=[float(z0@v) if v is not None else None for v in unit];filtered=[model.score(v) for v in patches]
                validneg=[unit_patch(v) for v in negative];validneg=[v for v in validneg if v is not None]
                if validneg:
                    B=np.stack(validneg)/np.sqrt(len(validneg));w=z0-B.T@np.linalg.solve(np.eye(len(B))+B@B.T,B@z0)
                else:w=z0
                for v,s in zip(unit,filtered):
                    if v is None:assert s is None
                    else:assert abs(float(v@w/(z0@w))-s)<1e-9
                vb=[s for s in baseline if s is not None];vf=[s for s in filtered if s is not None]
                r['variants'][name]=dict(valid=bool(vb and vf),negative_count=model.negative_count,correlation_scores=baseline,correlation_max=max(vb) if vb else None,filter_scores=filtered,filter_max=max(vf) if vf else None)
        with dest.open('x',encoding='utf8') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False)
        files[dest.relative_to(ROOT).as_posix()]=sha(dest)
        if number%10==0:print('34-channel completed',number+1,'/',len(parent['events']),flush=True)
    verify(p);write('results.json',dict(protocol_sha256=sha(OUT/'protocol.json'),files=files))
    write('audit.json',dict(passed=True,input_hashes_stable=True,original8_subset_exact=True,original8_results_reused=True,
        independent_interpolation_and_filter_tolerance=1e-9,result_sha256=sha(OUT/'results.json')))

{'register':register,'run':run}[sys.argv[1]]()
