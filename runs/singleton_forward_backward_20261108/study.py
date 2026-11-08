"""Independent backward matching of fixed three-point templates on archived maps."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'runs/singleton_matched_support_20261108'
NEIGHBOR = ROOT/'runs/singleton_neighbor_input_20261108'
TEMPORAL = ROOT/'runs/singleton_eight_input_20261108'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=240)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

AXES = np.array([(u,v) for u in range(-2,3) for v in range(-2,3)
                 if max(abs(u),abs(v),abs(u+v))<=2])
NAMES = ['FlyVis','center8','neighbor8','input8_temporal']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def correlate(reference,current):
    a = reference-reference.mean(0)
    b = current-current.mean(0)
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denominator) if denominator else None


def best_index(scores):
    valid = [i for i,value in enumerate(scores) if value is not None]
    return max(valid,key=lambda i:scores[i]) if valid else None


def math_checks():
    # Exact prescribed translation is recovered in both directions.
    rng = np.random.default_rng(430)
    frequency = rng.normal(size=(2,8))
    phase = rng.normal(size=8)
    q = np.array([[0.,0.],[100.,0.],[0.,100.]])
    prediction = np.array([.31,-.27])
    true_delta = np.array([1.,0.])
    def field(x):
        return np.sin(x@frequency+phase)+.4*np.cos(x@(frequency*.73)-phase)
    a = field(q)
    forward = [correlate(a,field(q+delta-true_delta)) for delta in AXES]
    f = best_index(forward)
    assert np.array_equal(AXES[f],true_delta)
    y = q+prediction+AXES[f]
    template = field(y-prediction-true_delta)
    backward = [correlate(template,field(y-prediction+delta)) for delta in AXES]
    b = best_index(backward)
    assert np.array_equal(AXES[f]+AXES[b],[0,0])
    # Construct a different reverse winner. Negating f would fail this test.
    a = np.repeat(np.array([-1.,0.,1.])[:,None],8,axis=1)
    t = np.repeat(np.array([-1.,1.,0.])[:,None],8,axis=1)
    forward = [correlate(a,t if np.array_equal(delta,[1,0]) else -a) for delta in AXES]
    f = best_index(forward)
    backward = []
    for delta in AXES:
        location = AXES[f]+delta
        patch = -t
        if np.array_equal(location,[0,0]):
            patch = a
        if np.array_equal(location,[1,1]):
            patch = t
        backward.append(correlate(t,patch))
    b = best_index(backward)
    assert np.array_equal(AXES[f],[1,0]) and np.array_equal(AXES[b],[0,1])
    assert correlate(np.ones((3,8)),a) is None
    assert all(any(np.array_equal(-delta,other) for other in AXES) for delta in AXES)
    write('mathematical_checks.json',dict(passed=True,exact_translation_recovers_origin=True,
        backward_not_forced_inverse=dict(forward=AXES[f].tolist(),backward=AXES[b].tolist(),
            residual=(AXES[f]+AXES[b]).tolist()),constant_reference_abstains=True,
        limitation='The same exact-translation example can represent two different objects with identical features and still gives zero error. Cycle consistency does not establish identity.'))


def register():
    assert read(OUT/'mathematical_checks.json')['passed']
    cases = read(BASE/'protocol.json')['cases']
    files = [Path(__file__).resolve(), OUT/'mathematical_checks.json',BASE/'protocol.json',
        BASE/'scores.json',NEIGHBOR/'scores.json',NEIGHBOR/'neighbor8_maps.npz',NEIGHBOR/'audit.json',
        TEMPORAL/'input8_maps.npz',TEMPORAL/'audit.json',ROOT/'runs/z13_local_peak_20261021/geometry.json']
    files += [ROOT/p for case in cases for paths in case['map_paths'] for p in paths]
    files += [BASE/(key+'.npz') for key in cases[1]['map_keys']]
    for case in cases:
        base = 'singleton_working_domain' if case['split']=='development20' else 'bbox_confirmation_inventory'
        files.append(ROOT/f'runs/{base}_20261108/native/{case["video"]}.json')
    protocol = dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        representations=NAMES,cases=[[c['video'],c['frame'],c['old'],c['new']] for c in cases],
        forward='Reuse archived winner delta_f and score, x_i are the same three old-frame centers. d_f=d0+delta_f; y_i=x_i+d_f. Extract the new reference from the current map at y_i.',
        backward='Independently evaluate S(current(y_i),old(y_i-d0+delta_b)) for every delta_b in the same symmetric 19-offset set. Backward prediction is -d0. Returned point is x_i+delta_f+delta_b.',
        error='Euclidean norm(delta_f+delta_b) in original-frame pixels, also eye pixels and axial grid distance. Report residual vector. No threshold selection.',
        support='Preserve all three Q centers, channels and normalization. Require all backward query points inside archived eye hull for every method and shift. Backward lookup uses the whole old eye, including outside old box, like point tracking in the old image. Report old-box coverage for each shift.',
        tie='First exact maximum in the unchanged axes order, with all exactly tied winners recorded.',
        acceptance='Math checks pass first; every backward query supported; independent interpolation and dot products agree 1e-9; no input mutation; original forward scores and winners reused exactly.',
        reference=dict(title='Forward-Backward Error: Automatic Detection of Tracking Failures',
            authors=['Zdenek Kalal','Krystian Mikolajczyk','Jiri Matas'],year=2010,doi='10.1109/ICPR.2010.675',
            url='https://cmp.felk.cvut.cz/ftp/articles/matas/kalal-2010-fb_track-icpr.pdf',
            sections='Section 2 and Figure 1, threshold evaluation Section 3. Section 2 explicitly excludes trackers consistent by construction.',
            transfer='Independent return-distance test on archived feature maps; not a reproduction of Lucas-Kanade or a reversal of FlyVis dynamics.'),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    write('protocol.json',protocol)
    path = ROOT/'experiments/project_history.json'
    catalog = read(path)
    assert not any(c['id']==OUT.name for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=OUT.name,
        title='Проверка возврата к образцу при прямом и обратном сопоставлении карт',area='flyvis',
        state='running',started='2026-11-08',question='Различает ли ошибка независимого обратного поиска свою и чужую машину?',
        hypothesis='Несогласованный возврат может дать дополнительный признак ошибки продолжения.',
        expected='Измерить ошибки возврата на двух прежних парах с четырьмя готовыми представлениями.',
        acceptance=[dict(criterion=protocol['acceptance'],met=None)],
        evidence=[dict(kind='file',ref='runs/'+OUT.name+'/'+name,claim='Постановка и математическая проверка до расчёта') for name in ['protocol.json','mathematical_checks.json']]))
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path = ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    plan = path.read_text(encoding='utf8')
    now = datetime.now(timezone(timedelta(hours=3)))
    entry = f'**{now:%d.%m.%Y, %H:%M} MSK** — Математика и постановка независимого обратного поиска.\n\n*Постановка.* По прямому поручению автора те же две пары и 3 ячейки, готовый прямой максимум и отдельный обратный поиск по 19 сдвигам вокруг обратного прогноза Калмана. Математическая проверка точного переноса возвращает начало, специально несовместимые карты дают ненулевую ошибку и исключают автоматическую смену знака. Нулевой знаменатель даёт отказ. Обратная область — весь сохранённый глаз с явной отметкой выхода за прежнюю рамку. Карточка `{OUT.name}`, `runs/{OUT.name}/protocol.json`, `runs/{OUT.name}/mathematical_checks.json`. Литературное основание — Kalal et al. 2010, раздел 2; согласованность по построению прямо исключена авторами из области применимости ошибки возврата.\n\n'
    path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')


def verify(protocol):
    for path,expected in protocol['sources'].items():
        assert sha(ROOT/path)==expected,path


def interpolate(mesh,points,values):
    simplex = mesh.find_simplex(points)
    assert (simplex>=0).all()
    transform = mesh.transform[simplex]
    weights = np.einsum('...ij,...j->...i',transform[:,:,:2],points-transform[:,:,2])
    weights = np.concatenate([weights,1-weights.sum(-1,keepdims=True)],axis=-1)
    sampled = np.einsum('...v,...vc->...c',weights,values[mesh.simplices[simplex]])
    independent = LinearNDInterpolator(mesh,values)(points)
    assert np.allclose(sampled,independent,rtol=0,atol=1e-9)
    return sampled,independent


def run():
    if (OUT/'results.json').exists():
        raise SystemExit('Completed backward results exist; no calculation repeated.')
    protocol = read(OUT/'protocol.json')
    verify(protocol)
    cases = read(BASE/'protocol.json')['cases']
    archived = read(BASE/'scores.json')['pairs']
    ready = read(NEIGHBOR/'scores.json')['pairs']
    g = read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    points = np.asarray(g['points'],float)
    mesh = Delaunay(points)
    records = []
    with np.load(TEMPORAL/'input8_maps.npz',allow_pickle=False) as temporal, np.load(NEIGHBOR/'neighbor8_maps.npz',allow_pickle=False) as neighbor:
        for case,old,forward in zip(cases,archived,ready):
            keys = case['map_keys']
            C = []
            paths = [p[0] for p in case['map_paths']] if case['kind']=='foreign' else [str((BASE/(key+'.npz')).relative_to(ROOT)) for key in keys]
            for path in paths:
                with np.load(ROOT/path,allow_pickle=False) as source:
                    C.append(source['C'].astype(float))
            temporal_maps = [temporal[key].astype(float) for key in keys]
            neighbors = [neighbor[key+'_neighbor8'].astype(float) for key in keys]
            centers = [np.concatenate([a[:,:4],a[:,:4]],axis=1) for a in temporal_maps]
            representations = dict(FlyVis=C,center8=centers,neighbor8=neighbors,input8_temporal=temporal_maps)
            crop0 = np.asarray(case['sequences'][0][-1][1])
            crop = np.asarray(case['sequences'][1][-1][1])
            scale = (crop[2:]-crop[:2])/g['eye_size']
            assert np.array_equal(crop0[2:]-crop0[:2],crop[2:]-crop[:2])
            x = points[np.asarray(old['cells'])]*scale+crop0[:2]
            prediction = np.asarray(old['predicted_px'])
            offsets = np.asarray(old['offsets_px'])
            for name,(reference,current) in representations.items():
                f = forward['variants'][name]
                fi = best_index(f['scores'])
                assert f['scores'][fi]==f['local']
                assert np.array_equal(offsets[fi],f['winning_offset_px'])
                y = x+prediction+offsets[fi]
                template,_ = interpolate(mesh,(y-crop[:2])/scale,current)
                backward_xy = y[:,None,:]-prediction+offsets
                queries = (backward_xy-crop0[:2])/scale
                sampled,independent = interpolate(mesh,queries,reference)
                scores = []
                for j in range(len(AXES)):
                    value = correlate(template,sampled[:,j])
                    check = correlate(template,independent[:,j])
                    if value is None:
                        assert check is None
                    else:
                        a = template-template.mean(0)
                        b = independent[:,j]-independent[:,j].mean(0)
                        independent_dot = float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)))
                        assert abs(value-check)<=1e-9 and abs(value-independent_dot)<=1e-9
                    scores.append(value)
                bi = best_index(scores)
                residual = offsets[fi]+offsets[bi] if bi is not None else None
                residual_axes = AXES[fi]+AXES[bi] if bi is not None else None
                last = case['last_observation']
                base = 'singleton_working_domain' if case['split']=='development20' else 'bbox_confirmation_inventory'
                export = read(ROOT/f'runs/{base}_20261108/native/{case["video"]}.json')
                event = next(e for e in export['events'] if e['frame']==case['frame'])
                row = next(r for r in event['rows'] if r['old_tid']==case['old'])
                box = np.asarray(row['row_data']['history'][-1]['box'])
                inside = ((backward_xy>=box[:2])&(backward_xy<=box[2:])).all(-1).all(0)
                records.append(dict(video=case['video'],kind=case['kind'],old=case['old'],new=case['new'],
                    reference_frame=last,current_frame=case['frame'],representation=name,cells=old['cells'],
                    forward_reused=f,forward_axes=AXES[fi].tolist(),
                    backward_axes=AXES[bi].tolist() if bi is not None else None,
                    backward_offset_px=offsets[bi].tolist() if bi is not None else None,
                    backward_scores=scores,backward_score=scores[bi] if bi is not None else None,
                    backward_tied_indices=[j for j,v in enumerate(scores) if bi is not None and v==scores[bi]],
                    residual_px=residual.tolist() if residual is not None else None,
                    error_px=float(np.linalg.norm(residual)) if residual is not None else None,
                    error_eye_px=float(np.linalg.norm(residual/scale)) if residual is not None else None,
                    residual_axes=residual_axes.tolist() if residual_axes is not None else None,
                    error_grid_steps=int(max(abs(residual_axes[0]),abs(residual_axes[1]),abs(sum(residual_axes)))) if residual_axes is not None else None,
                    backward_all_inside_eye=True,backward_all_inside_old_box=inside.tolist(),
                    winner_inside_old_box=bool(inside[bi]) if bi is not None else None))
    verify(protocol)
    write('results.json',dict(protocol_sha256=sha(OUT/'protocol.json'),records=records))
    write('audit.json',dict(passed=True,mathematical_checks=read(OUT/'mathematical_checks.json'),
        interpolation_and_dot_tolerance=1e-9,all_queries_inside_eye=True,ready_forward_reused_exactly=True,
        input_hashes_stable=True,result_sha256=sha(OUT/'results.json')))
    for record in records:
        print(record['kind'],record['representation'],record['forward_axes'],record['backward_axes'],
              record['error_px'],record['error_grid_steps'])


{'math':math_checks,'register':register,'run':run}[sys.argv[1]]()
