"""Eight fixed temporal input channels, using archived maps wherever available."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=240)
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scripts.z13_target_model import R1_TAUS, receptor_features_rich

SELECT=[0,5,11,16,17,22,28,33]
BASE=ROOT/'runs/singleton_matched_support_20261108'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)

def verify(protocol):
    for path,value in protocol['sources'].items(): assert sha(ROOT/path)==value,path

def register():
    previous=read(BASE/'protocol.json')
    files=[BASE/'protocol.json',BASE/'scores.json',BASE/'maps_complete.json',BASE/'retinal.npz',
        ROOT/'scripts/z13_target_model.py',ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json',
        ROOT/'runs/z13_local_peak_20261021/geometry.json',Path(__file__).resolve()]
    files += [ROOT/p for p in previous['cases'][0]['map_paths'][0]+previous['cases'][0]['map_paths'][1]]
    files += [ROOT/'runs/bbox_fit_maps_20261108/features_complete.json',
        ROOT/'research_singleton_verify_20261108/extract_features.py']
    p=dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        channel_indices=SELECT,taus=R1_TAUS[[0,5,11,16]].tolist(),
        definition='Existing R1 EMA and h-EMA. Four log-spaced indices nearest to linspace(0,16,4), fixed before scores. Old R1 training normalization, no refitting. Filters advance once per archived causal observation, as in existing R1 extraction.',
        formula='m_j(t)=(1-1/tau_j)*m_j(t-1)+h(t)/tau_j, initialized with h(0); other four channels h(t)-m_j(t).',
        geometry='Reuse exact three reference cells, predictions and 19 pixel offsets per pair. Same interpolation, channel centering and joint norm as archived C.',
        reuse='C and image scores read verbatim. Foreign R1 rows sliced from existing 34-channel maps. Only own missing R1 computed from five saved retinal observations. No model inference.',
        acceptance='All arrays finite; own EMA agrees with closed form <=1e-12; interpolation/dot products agree <=1e-9; original hashes stable and original scores reused exactly.',
        limitation='Eight columns do not imply eight independent signals. EMA and residual channels are linearly related. Different scenes, eye scales and temporal histories remain. Diagnostic on two pairs.',
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    write('protocol.json',p)
    path=ROOT/'experiments/project_history.json'
    catalog=read(path)
    assert not any(c['id']==OUT.name for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=OUT.name,title='Восемь простых временных признаков входа глаза против T4/T5',
        area='flyvis',state='running',started='2026-11-08',
        question='Сохраняется ли различие своей и чужой машины при восьми простых каналах входа глаза?',
        hypothesis='Часть различия с одноканальной яркостью может объясняться многоканальной временной обработкой.',
        expected='Сопоставить восемь простых каналов с готовыми T4/T5 на прежних ячейках и сдвигах.',
        acceptance=[dict(criterion=p['acceptance'],met=None)],
        evidence=[dict(kind='file',ref='runs/'+OUT.name+'/protocol.json',claim='Условия до расчёта')]))
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    plan_path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    plan=plan_path.read_text(encoding='utf8')
    entry=f'**{datetime.now(timezone(timedelta(hours=3))):%d.%m.%Y, %H:%M} MSK** — Зарегистрирован контроль восьми простых каналов входа глаза.\n\n*Постановка.* По прямому поручению автора те же пары 390→448 и 3→273, по 3 ячейки, прежние 19 сдвигов. Четыре скользящих средних из прежнего R1 и четыре отклонения от них; индексы 0, 5, 11, 16 выбраны по логарифмической сетке до оценок. Прежняя обучающая нормировка. Готовые оценки FlyVis и изображения читаются из файлов, чужие признаки R1 тоже готовы. Вычисляется только отсутствующий контроль своей пары. Карточка `{OUT.name}`, условия `runs/{OUT.name}/protocol.json`.\n\n'
    plan_path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')

def run():
    if (OUT/'scores.json').exists():
        raise SystemExit('Completed scores exist; no calculation repeated.')
    p=read(OUT/'protocol.json'); verify(p)
    previous=read(BASE/'scores.json')
    cases=read(BASE/'protocol.json')['cases']
    norm=read(ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json')
    mu,sd=np.array(norm['mean'])[:34],np.array(norm['sd'])[:34]
    assert np.all(sd>0)
    with np.load(BASE/'retinal.npz',allow_pickle=False) as z: hx=z['hex'].astype(np.float64)
    own_case=cases[1]
    assert len(hx)==len(own_case['sequences'][1])==5
    rich=receptor_features_rich(hx[None])
    selected_taus=R1_TAUS[[0,5,11,16]]
    max_error=0.
    own_maps=[]
    for sequence in own_case['sequences']:
        end=len(sequence)-1
        decay=1-1/selected_taus
        closed=decay[:,None]**end*hx[0]
        for t in range(1,end+1): closed+=(1/selected_taus)[:,None]*decay[:,None]**(end-t)*hx[t]
        independent=np.concatenate([closed,hx[end]-closed]).T
        values=rich[end][0].T[:,SELECT]
        error=float(np.max(np.abs(values-independent)))
        max_error=max(max_error,error); assert error<=1e-12
        own_maps.append(((values-mu[SELECT])/sd[SELECT]).astype(np.float32))
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    points=np.asarray(g['points'],float); mesh=Delaunay(points)
    results=[]; maps={}
    ready=read(ROOT/'runs/bbox_fit_maps_20261108/features_complete.json')['files']
    for i,(case,old) in enumerate(zip(cases,previous['pairs'])):
        assert (case['video'],case['old'])==(old['video'],old['old'])
        if case['kind']=='foreign':
            arrays=[]
            for paths,key in zip(case['map_paths'],case['map_keys']):
                path=ROOT/paths[0]
                assert sha(path)==ready[key]
                with np.load(path,allow_pickle=False) as z:
                    assert z['R1'].shape==(721,34)
                    arrays.append(z['R1'][:,SELECT].copy())
        else: arrays=own_maps
        assert all(a.shape==(721,8) and np.isfinite(a).all() for a in arrays)
        for key,array in zip(case['map_keys'],arrays): maps[key]=array
        seq0=case['sequences'][0]; seqt=case['sequences'][1]
        crop0=np.array(seq0[-1][1]); crop=np.array(seqt[-1][1])
        extent=crop[2:]-crop[:2]
        assert np.array_equal(crop0[2:]-crop0[:2],extent)
        idx=np.asarray(old['cells']); assert len(idx)==3
        scale=extent/g['eye_size']
        origin=points[idx]*scale+crop0[:2]
        offsets=np.asarray(old['offsets_px']); zero=int(np.flatnonzero((offsets==0).all(1))[0])
        query=(origin[:,None,:]+old['predicted_px']+offsets-crop[:2])/scale
        simplex=mesh.find_simplex(query); assert (simplex>=0).all()
        tr=mesh.transform[simplex]
        w=np.einsum('...ij,...j->...i',tr[:,:,:2],query-tr[:,:,2])
        w=np.concatenate([w,1-w.sum(-1,keepdims=True)],axis=-1)
        b=arrays[1].astype(float)
        sampled=np.einsum('qsv,qsvc->qsc',w,b[mesh.simplices[simplex]])
        other=LinearNDInterpolator(mesh,b)(query)
        assert np.allclose(sampled,other,rtol=0,atol=1e-9)
        variants={}
        for name in ['input8','input8_permuted']:
            a=arrays[0][idx].astype(float)
            if name.endswith('permuted'): a=a[np.random.default_rng(42).permutation(len(idx))]
            a-=a.mean(0)
            cur=sampled-sampled.mean(0)
            scores=[]
            for j in range(19):
                denominator=np.linalg.norm(a)*np.linalg.norm(cur[:,j])
                score=float(np.sum(a*cur[:,j])/denominator) if denominator else None
                if score is not None:
                    independent=other[:,j]-other[:,j].mean(0)
                    check=np.dot(a.ravel(),independent.ravel())/(np.linalg.norm(a)*np.linalg.norm(independent))
                    assert abs(score-check)<=1e-9
                scores.append(score)
            valid=[j for j,s in enumerate(scores) if s is not None]
            best=max(valid,key=lambda j:scores[j]) if valid else None
            variants[name]=dict(fixed=scores[zero],local=scores[best] if valid else None,
                gain=scores[best]-scores[zero] if valid and scores[zero] is not None else None,
                winning_offset_px=offsets[best].tolist() if valid else None,scores=scores)
        variants['FlyVis']=old['variants']['FlyVis']
        variants['brightness']=old['variants']['brightness']
        variants['gradient']=old['variants']['gradient']
        variants['FlyVis_permuted']=old['variants']['permuted']
        results.append(dict(video=case['video'],old=case['old'],new=case['new'],kind=case['kind'],
            cells=old['cells'],crop_extent_px=extent.tolist(),variants=variants,
            input8_centered_reference_rank=int(np.linalg.matrix_rank(arrays[0][idx]-arrays[0][idx].mean(0)))))
    verify(p)
    with (OUT/'input8_maps.npz').open('xb') as stream: np.savez_compressed(stream,**maps)
    write('scores.json',dict(protocol_sha256=sha(OUT/'protocol.json'),pairs=results))
    write('audit.json',dict(passed=True,ema_closed_form_max_error=max_error,
        interpolation_and_score_tolerance=1e-9,original_scores_reused_exactly=True,source_hashes_stable=True,
        eight_duplicate_brightness_channels='Exact algebra: centered numerator and squared norms each multiply by 8, so correlation is identical. No duplicate calculation needed.',
        outputs={name:sha(OUT/name) for name in ['input8_maps.npz','scores.json']},
        taus=selected_taus.tolist(),channels=SELECT))
    for row in results:
        print(row['kind'],row['old'],row['new'],row['variants']['input8'],flush=True)

{'register':register,'run':run}[sys.argv[1]]()
