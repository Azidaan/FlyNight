"""Matched one-hop spatial pooling from ready temporal input maps."""
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
TEMPORAL = ROOT/'runs/singleton_eight_input_20261108'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=240)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scripts.z13_target_model import neighbor_aug
from tracking.z13_readout import HexGeometry


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (OUT/name).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def verify(protocol):
    for path, expected in protocol['sources'].items():
        assert sha(ROOT/path) == expected, path


def register():
    files = [Path(__file__).resolve(), TEMPORAL/'input8_maps.npz', TEMPORAL/'protocol.json',
             TEMPORAL/'scores.json', TEMPORAL/'audit.json', BASE/'protocol.json', BASE/'scores.json',
             ROOT/'scripts/z13_target_model.py', ROOT/'tracking/z13_readout.py',
             ROOT/'runs/z13_local_peak_20261021/geometry.json']
    protocol = dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        channels='Four ready normalized EMA channels, taus 1.5,3.903211717149933,12.297565051134073,32. Neighbor8 concatenates these four with their six-neighbor means. Center8 repeats the same four twice.',
        normalization='Ready training normalization reused. Identical channel scale for central and pooled counterpart; no new normalization or weighting.',
        spatial='Existing HexGeometry and neighbor_aug, one hop. Grid edge padding uses center as in existing helper. No masking neighbors to box; only comparison centers Q stay fixed.',
        geometry='Same saved three cells, Kalman displacement, 19 pixel offsets per pair, interpolation and joint centered correlation.',
        question='Does ordinary neighboring-cell averaging improve identity ordering relative to the matched central-only input?',
        acceptance='Independent neighbor means, interpolation and dot products agree <=1e-9. Center8 identity to four-channel correlation checked analytically. Sources immutable; ready scores reused verbatim.',
        limitation='Two cases; neighbor8 changes the last four channels relative to the previous EMA+residual control. Attribute pooling only through center8 versus neighbor8, not through previous input8. Same number of channels does not match FlyVis receptive field or dynamics.',
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    write('protocol.json',protocol)
    path = ROOT/'experiments/project_history.json'
    catalog = read(path)
    assert not any(c['id']==OUT.name for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=OUT.name,
        title='Среднее соседних ячеек в восьми признаках входа глаза против T4/T5',area='flyvis',
        state='running',started='2026-11-08',question=protocol['question'],
        hypothesis='Обычная пространственная обработка может объяснять часть различия с T4/T5.',
        expected='Выделить вклад соседей сравнением одинаковых четырёх временных каналов с повторением либо усреднением соседей.',
        acceptance=[dict(criterion=protocol['acceptance'],met=None)],
        evidence=[dict(kind='file',ref='runs/'+OUT.name+'/protocol.json',claim='Условия до оценки')]))
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path = ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    plan = path.read_text(encoding='utf8')
    now = datetime.now(timezone(timedelta(hours=3)))
    entry = f'**{now:%d.%m.%Y, %H:%M} MSK** — Зарегистрирован контроль среднего соседних ячеек.\n\n*Постановка.* По прямому поручению автора те же две пары, 3 ячейки, прежние 19 сдвигов. Четыре готовых скользящих средних входа глаза и четыре средних по шести соседям. Согласованный контроль без соседей повторяет те же четыре центральных канала. Нормировка общая и прежняя. Прежние оценки FlyVis, яркости, градиентов и восьми временных каналов читаются готовыми. Карточка `{OUT.name}`, протокол `runs/{OUT.name}/protocol.json`.\n\n'
    path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')


def run():
    if (OUT/'scores.json').exists():
        raise SystemExit('Ready scores exist; no calculation repeated.')
    protocol = read(OUT/'protocol.json')
    verify(protocol)
    audit = read(TEMPORAL/'audit.json')
    assert sha(TEMPORAL/'input8_maps.npz') == audit['outputs']['input8_maps.npz']
    cases = read(BASE/'protocol.json')['cases']
    archived = read(BASE/'scores.json')['pairs']
    temporal = read(TEMPORAL/'scores.json')['pairs']
    g = read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    points = np.asarray(g['points'],float)
    geometry = HexGeometry(points,g['eye_size'])
    nb = geometry.neighbors
    mesh = Delaunay(points)
    results = []
    maps = {}
    max_pool_error = 0.
    with np.load(TEMPORAL/'input8_maps.npz',allow_pickle=False) as source:
        for case,old,previous in zip(cases,archived,temporal):
            assert (case['video'],case['old']) == (old['video'],old['old']) == (previous['video'],previous['old'])
            arrays = {'center8':[], 'neighbor8':[]}
            for key in case['map_keys']:
                central = source[key][:,:4].astype(float)
                center8 = np.concatenate([central,central],axis=1)
                pooled = neighbor_aug(central.T[None],nb)[0].T
                independent = np.stack([central[row[1:]].sum(0)/6 for row in nb])
                error = float(np.max(np.abs(pooled[:,4:]-independent)))
                max_pool_error = max(max_pool_error,error)
                assert error<=1e-9
                assert np.array_equal(pooled[:,:4],central)
                arrays['center8'].append(center8)
                arrays['neighbor8'].append(pooled)
                maps[key+'_neighbor8'] = pooled
            idx = np.asarray(old['cells'])
            assert len(idx)==3
            crop0 = np.asarray(case['sequences'][0][-1][1])
            crop = np.asarray(case['sequences'][1][-1][1])
            extent = crop[2:]-crop[:2]
            assert np.array_equal(crop0[2:]-crop0[:2],extent)
            scale = extent/g['eye_size']
            xy0 = points[idx]*scale+crop0[:2]
            offsets = np.asarray(old['offsets_px'])
            zero = int(np.flatnonzero((offsets==0).all(1))[0])
            query = (xy0[:,None,:]+old['predicted_px']+offsets-crop[:2])/scale
            simplex = mesh.find_simplex(query)
            assert (simplex>=0).all()
            tr = mesh.transform[simplex]
            w = np.einsum('...ij,...j->...i',tr[:,:,:2],query-tr[:,:,2])
            w = np.concatenate([w,1-w.sum(-1,keepdims=True)],axis=-1)
            vertices = mesh.simplices[simplex]
            variants = {}
            for name,(reference,current) in arrays.items():
                assert reference.shape==current.shape==(721,8)
                assert np.isfinite(reference).all() and np.isfinite(current).all()
                sampled = np.einsum('qsv,qsvc->qsc',w,current[vertices])
                independent = LinearNDInterpolator(mesh,current)(query)
                assert np.allclose(sampled,independent,rtol=0,atol=1e-9)
                ref = reference[idx]-reference[idx].mean(0)
                cur = sampled-sampled.mean(0)
                scores = []
                for j in range(19):
                    denominator = np.linalg.norm(ref)*np.linalg.norm(cur[:,j])
                    value = float(np.sum(ref*cur[:,j])/denominator) if denominator else None
                    if value is not None:
                        direct = independent[:,j]-independent[:,j].mean(0)
                        check = np.dot(ref.ravel(),direct.ravel())/(np.linalg.norm(ref)*np.linalg.norm(direct))
                        assert abs(check-value)<=1e-9
                        if name=='center8':
                            a,b = ref[:,:4],cur[:,j,:4]
                            check4 = np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))
                            assert abs(check4-value)<=1e-9
                    scores.append(value)
                valid = [j for j,s in enumerate(scores) if s is not None]
                best = max(valid,key=lambda j:scores[j]) if valid else None
                variants[name] = dict(fixed=scores[zero],local=scores[best] if valid else None,
                    gain=scores[best]-scores[zero] if valid and scores[zero] is not None else None,
                    winning_offset_px=offsets[best].tolist() if valid else None,scores=scores)
            variants['FlyVis'] = previous['variants']['FlyVis']
            variants['input8_temporal'] = previous['variants']['input8']
            variants['brightness'] = previous['variants']['brightness']
            variants['gradient'] = previous['variants']['gradient']
            variants['FlyVis_permuted'] = previous['variants']['FlyVis_permuted']
            results.append(dict(video=case['video'],kind=case['kind'],old=case['old'],new=case['new'],
                cells=old['cells'],variants=variants,
                reference_pool_cells=np.unique(nb[idx,1:]).tolist(),
                current_pool_cells_all_shifts=np.unique(nb[np.unique(vertices),1:]).tolist()))
    verify(protocol)
    with (OUT/'neighbor8_maps.npz').open('xb') as stream:
        np.savez_compressed(stream,**maps)
    write('scores.json',dict(protocol_sha256=sha(OUT/'protocol.json'),pairs=results))
    write('audit.json',dict(passed=True,neighbor_mean_max_error=max_pool_error,
        interpolation_and_dot_tolerance=1e-9,center_duplicate_identity_verified=True,
        ready_scores_reused_exactly=True,source_hashes_stable=True,
        outputs={name:sha(OUT/name) for name in ['scores.json','neighbor8_maps.npz']}))
    for pair in results:
        print(pair['kind'],pair['old'],pair['new'])
        for name in ['center8','neighbor8','FlyVis']:
            print(name,pair['variants'][name]['fixed'],pair['variants'][name]['local'])


{'register':register,'run':run}[sys.argv[1]]()
