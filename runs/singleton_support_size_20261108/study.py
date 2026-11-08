"""Paired support-size intervention over ready maps, without model inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'runs/singleton_forward_backward_transfer_20261108'
OLD_FULL=ROOT/'runs/singleton_two_map_scores_20261108'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
AXES=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])
NAMES=['FlyVis','center8','neighbor8','input8_temporal']
SELECT=[0,5,11,16,17,22,28,33]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def stamp():return datetime.now(timezone(timedelta(hours=3))).isoformat()
def verify(p):
    for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
def helpers():
    # Reuse the exact frozen correlation, argmax and interpolation functions.
    path=BASE/'study.py';tree=ast.parse(path.read_text(encoding='utf8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['corr','best','sample']]
    assert len(nodes)==3
    namespace=dict(np=np,LinearNDInterpolator=LinearNDInterpolator)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return tuple(namespace[n] for n in ['corr','best','sample'])
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf8')
    now=datetime.now(timezone(timedelta(hours=3)))
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — {title}\n\n'+text+'\n\n'
    path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
def register():
    previous=read(BASE/'protocol.json');maps=read(BASE/'maps_complete.json');verify(maps)
    assert sha(BASE/'results.json')==read(BASE/'audit.json')['result_sha256']
    corr,_,_=helpers()
    sizes=sorted({3}|{len(c['geometry']['eligible_cells']) for c in previous['cases']})
    ranks={}
    for n in sizes:
        H=np.eye(n)-np.ones((n,n))/n
        rank=int(np.linalg.matrix_rank(H));assert rank==n-1
        ranks[str(n)]=rank
    a=np.arange(24.).reshape(3,8);b=np.cos(a)+a*.2
    value=corr(a,b);duplicate=corr(np.repeat(a,2,axis=0),np.repeat(b,2,axis=0))
    assert abs(value-duplicate)<1e-12
    write('mathematical_checks.json',dict(passed=True,centering_projection_ranks=ranks,
        repeated_cells_correlation_error=abs(value-duplicate),
        conclusion='More distinct cells increase spatial degrees of freedom after centering. Duplicating observations alone leaves correlation unchanged. Neither fact guarantees better identity or lower cycle error.',
        prior_cycle_math='runs/singleton_forward_backward_transfer_20261108/mathematical_checks.json'))
    files=[Path(__file__),BASE/'protocol.json',BASE/'study.py',BASE/'results.json',BASE/'summary.json',
        BASE/'audit.json',BASE/'maps_complete.json',BASE/'final_checks.json',BASE/'inventory_provenance.json',
        BASE/'mathematical_checks.json',OUT/'mathematical_checks.json',
        OLD_FULL/'protocol.json',OLD_FULL/'scores.json',
        ROOT/'runs/z13_local_peak_20261021/geometry.json',ROOT/'scripts/z13_target_model.py',ROOT/'tracking/z13_readout.py']
    files += [ROOT/path for path in maps['sources']]
    cases=previous['cases']
    reuse=[]
    for c in cases:
        t=c['geometry'];full=t['eligible_cells'];assert len(full)==c['points'] and set(t['cells'])<set(full)
        for r in read(OLD_FULL/'scores.json')['pairs']:
            if (r['video'],r['frame'],r['old'],r['new'])!=(c['video'],c['frame'],c['old'],c['new']):continue
            assert r['cells']==full
            assert np.array_equal(r['predicted_px'],t['prediction']) and np.array_equal(r['offsets_px'],t['offsets'])
            for path in c['ready_maps'].values():
                assert sha(ROOT/path)==read(OLD_FULL/'protocol.json')['inputs'][path]
            reuse.append(dict(video=c['video'],frame=c['frame'],old=c['old'],new=c['new'],representation='FlyVis',
                source='runs/singleton_two_map_scores_20261108/scores.json'))
    assert len(reuse)==1
    p=dict(registered=stamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        deadline='2026-11-09T07:40:00+03:00',cases=cases,representations=NAMES,
        intervention='Replace the three geometry-selected reference cells with the entire already registered eligible_cells set. This is the only intervention. Means and norms are recomputed over that support by the unchanged formula.',
        support='Same full eligible set for all four representations and all 19 forward shifts. No activity-based weighting, trimming, background removal or post-score cell selection.',
        forward='Unchanged physical coordinates, prediction, 19 offsets, centering, joint normalized dot product and tie order from the previous protocol.',
        backward='Independent backward argmax on the previous whole eye, same offsets and same enlarged current template. If a complete backward grid is unsupported, report it explicitly without reducing support.',
        reuse='Reuse all 28 three-cell results verbatim. Reuse the ready eight-cell FlyVis forward scores for 1738→1834 after checking exact support and geometry. Read all 14 ready C/R1 maps, no video processing or model inference.',
        prior_full_forward_reuse=reuse,
        scientific_question='Does enlarging the support increase exact returns among own pairs and decrease exact returns among foreign pairs for FlyVis? Exact closure is a descriptive property already used before this experiment, not a fitted operating threshold.',
        acceptance='Frozen hashes remain stable; exact three-cell baseline reuse; exact prior full-forward reuse where applicable; common full support; independent interpolation and dot products agree <=1e-9; unsupported cases retained.',
        limitation='Seven pairs, different full-support counts (8–57), all foreign crops change scale, all own crops keep it. The paired intervention controls scale within each event, not between classes. Added cells also change sampled content and background. No tracker or core-mechanism claim.',
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))})
    write('protocol.json',p)
    path=ROOT/'experiments/project_history.json';cat=read(path)
    assert not any(c['id']==OUT.name for c in cat['experiments'])
    cat['experiments'].append(dict(id=OUT.name,title='Три опорные ячейки против всей допустимой пространственной поддержки',
        area='flyvis',state='running',started='2026-11-08',question='Улучшает ли увеличение шаблона различение своих и чужих продолжений?',
        hypothesis='Трёхточечный шаблон может терять пространственные различия; полная допустимая поддержка может их вернуть.',
        expected='Сравнить ошибки возврата на прежних семи парах для всех четырёх представлений, без подбора порогов.',
        acceptance=[dict(criterion=p['acceptance'],met=None)],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Условия до расчёта')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    journal('Зарегистрировано сравнение трёх ячеек со всей допустимой поддержкой.',
        'Те же семь прямо размеченных пар, те же карты и четыре представления. Единственное изменение — 3 опоры заменены всеми 15, 28, 8, 57, 8, 26, 18 допустимыми ячейками соответственно порядку protocol.json. Готовые 28 трёхточечных результатов и прямые восемь-ячеечные оценки FlyVis для 1738→1834 используются без повтора. Математически проверен ранг центрирования N−1; рост числа ячеек не гарантирует разделение личности. Условия и проверка до оценок `runs/'+OUT.name+'/protocol.json`, `runs/'+OUT.name+'/mathematical_checks.json`, карточка `'+OUT.name+'`.')
def score():
    if (OUT/'results.json').exists():raise SystemExit('Ready results exist; no calculation repeated.')
    p=read(OUT/'protocol.json');verify(p)
    corr,best,sample=helpers()
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.asarray(g['points'],float)
    mesh=Delaunay(points);nb=HexGeometry(points,g['eye_size']).neighbors
    baseline=read(BASE/'results.json')['records'];old_full=read(OLD_FULL/'scores.json')['pairs']
    records=[];pool_error=0.;reused_forward=0
    for c in p['cases']:
        t=c['geometry'];idx=np.asarray(t['eligible_cells']);crop0=np.asarray(t['crop0']);ct=np.asarray(t['crop'])
        s0=np.asarray(t['scale0']);st=np.asarray(t['scale']);off=np.asarray(t['offsets']);d=np.asarray(t['prediction'])
        x=points[idx]*s0+crop0[:2];fq=(x[:,None]+d+off-ct[:2])/st
        arrays={n:[] for n in NAMES}
        for key in c['map_keys']:
            path=c['ready_maps'].get(key,f'runs/{BASE.name}/{key}.npz')
            with np.load(ROOT/path,allow_pickle=False) as z:
                C=z['C'].astype(float);temporal=z['R1'][:,SELECT].astype(float)
            central=temporal[:,:4];pooled=neighbor_aug(central.T[None],nb)[0].T
            check=np.stack([central[row[1:]].sum(0)/6 for row in nb]);error=float(np.max(abs(check-pooled[:,4:])))
            pool_error=max(pool_error,error);assert error<=1e-9
            for n,a in zip(NAMES,[C,np.concatenate([central,central],1),pooled,temporal]):
                assert np.isfinite(a).all();arrays[n].append(a)
        for name,(old,current) in arrays.items():
            candidates=[r for r in old_full if name=='FlyVis' and (r['video'],r['frame'],r['old'],r['new'])==(c['video'],c['frame'],c['old'],c['new'])]
            if candidates:
                source,=candidates;forward=source['variants']['FlyVis']['scores'];reused_forward+=1
                assert source['cells']==idx.tolist()
            else:
                values,independent=sample(mesh,fq,current)
                forward=[]
                for j in range(19):
                    v=corr(old[idx],values[:,j]);check=corr(old[idx],independent[:,j])
                    assert v is None and check is None or abs(v-check)<=1e-9
                    if v is not None:
                        a=old[idx]-old[idx].mean(0);b=independent[:,j]-independent[:,j].mean(0)
                        assert abs(v-float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b))))<=1e-9
                    forward.append(v)
            fi=best(forward);assert fi is not None
            y=x+d+off[fi];template,_=sample(mesh,(y-ct[:2])/st,current)
            bxy=y[:,None]-d+off;query=(bxy-crop0[:2])/s0
            valid=bool((mesh.find_simplex(query)>=0).all())
            old_index,old_row=next((i,r) for i,r in enumerate(baseline) if (r['video'],r['old'],r['new'],r['representation'])==(c['video'],c['old'],c['new'],name))
            r=dict(video=c['video'],frame=c['frame'],old=c['old'],new=c['new'],kind=c['kind'],same_scale=c['same_scale'],
                representation=name,cells=idx.tolist(),cell_count=len(idx),three_cell_result_index=old_index,
                baseline_error_grid=old_row['error_grid_steps'],baseline_error_px=old_row['error_px'],
                baseline_forward_axes=old_row['forward_axes'],baseline_backward_axes=old_row['backward_axes'],
                forward_scores=forward,forward_axes=AXES[fi].tolist(),fixed_score=forward[9],forward_score=forward[fi],
                forward_reused_from='runs/singleton_two_map_scores_20261108/scores.json' if candidates else None,
                backward_supported=valid,geometry_only_return_grid=int(max(abs(AXES[fi][0]),abs(AXES[fi][1]),abs(sum(AXES[fi])))))
            if valid:
                sampled,other=sample(mesh,query,old);backward=[]
                for j in range(19):
                    v=corr(template,sampled[:,j]);check=corr(template,other[:,j])
                    assert v is None and check is None or abs(v-check)<=1e-9
                    if v is not None:
                        a=template-template.mean(0);b=other[:,j]-other[:,j].mean(0)
                        assert abs(v-float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b))))<=1e-9
                    backward.append(v)
                bi=best(backward)
                if bi is not None:
                    residual=AXES[fi]+AXES[bi];physical=off[fi]+off[bi]
                    inside=((bxy>=np.array(t['old_box'])[:2])&(bxy<=np.array(t['old_box'])[2:])).all(-1).all(0)
                    r.update(backward_scores=backward,backward_axes=AXES[bi].tolist(),backward_score=backward[bi],
                        forward_ties=[j for j,v in enumerate(forward) if v==forward[fi]],backward_ties=[j for j,v in enumerate(backward) if v==backward[bi]],
                        residual_axes=residual.tolist(),residual_px=physical.tolist(),error_px=float(np.linalg.norm(physical)),
                        error_grid_steps=int(max(abs(residual[0]),abs(residual[1]),abs(sum(residual)))),
                        winner_inside_old_box=bool(inside[bi]))
                    r['delta_error_grid']=r['error_grid_steps']-old_row['error_grid_steps']
                    r['delta_error_px']=r['error_px']-old_row['error_px']
            records.append(r)
            print(c['kind'],str(c['old'])+'→'+str(c['new']),name,len(idx),'cells',old_row['error_grid_steps'],'→',r.get('error_grid_steps','unsupported'),flush=True)
    assert reused_forward==1;verify(p)
    write('results.json',dict(protocol_sha256=sha(OUT/'protocol.json'),baseline_source='runs/'+BASE.name+'/results.json',records=records))
    write('audit.json',dict(passed=True,input_hashes_stable=True,prior_three_cell_results_read_only=True,
        prior_full_forward_scores_reused=1,new_forward_scores=27,new_backward_scores=28,
        same_full_support_all_methods=True,pool_max_error=pool_error,interpolation_dot_tolerance=1e-9,
        unsupported_records=sum(not r['backward_supported'] for r in records),result_sha256=sha(OUT/'results.json')))

{'register':register,'score':score}[sys.argv[1]]()
