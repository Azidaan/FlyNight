"""Two archived pairs. Run with 'freeze' first, then without arguments."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=120)
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

inventory = 'runs/singleton_car_map_coverage_20261108/inventory.json'
geometry = 'runs/z13_local_peak_20261021/geometry.json'
protocol = 'runs/flyvis_local_shift_feasibility_20261108/МАТЕМАТИКА_И_ДАННЫЕ.md'
def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def write(name, value):
    with (HERE/name).open('x', encoding='utf8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)

cases = [c for c in read(inventory)['cases'] if c['direct'] and c['eligible']]
assert {(c['frame'],c['old'],c['new']) for c in cases}=={(578,390,448),(1121,1738,1834)}
def native_path(case):
    return 'runs/bbox_confirmation_inventory_20261108/native/'+case['video']+'.json'

if 'freeze' in sys.argv:
    files=[inventory,geometry,protocol,Path(__file__).resolve().relative_to(ROOT).as_posix()]
    for case in cases:
        files.extend([native_path(case),*(p for paths in case['map_paths'] for p in paths)])
    write('protocol.json',dict(time=datetime.now(timezone(timedelta(hours=3))).isoformat(),
          head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          cases=[[c['frame'],c['old'],c['new']] for c in cases],inputs={p:sha(p) for p in files},
          grid='Delaunay over saved eye centres, same mesh for all variants; original frame offsets',
          scores='Reference correlation at Kalman prediction and max over 19 common shifts. Mean removed per channel, one joint norm. Zero energy abstains.',
          controls='Brightness primary; two image gradients separately. Shuffle reference C rows jointly on fixed support, default_rng(42) per pair; one illustration.',
          acceptance='Reproduce three and eight fixed points. Independent triangle interpolation and correlation agree for every shift. Descriptive only.'))
    sys.exit(0)

frozen=read(HERE/'protocol.json')
assert all(sha(p)==s for p,s in frozen['inputs'].items())
g=read(geometry)
points=np.array(g['points'],float)
mesh=Delaunay(points)
axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])
zero=int(np.flatnonzero((axes==0).all(1))[0])
offsets=axes@np.array([[13.,6.5],[0.,13.]])
results=[]
for case in cases:
    event,=[e for e in read(native_path(case))['events'] if e['frame']==case['frame']]
    row,=[r for r in event['rows'] if r['old_tid']==case['old']]
    cand,=[c for c in row['candidates'] if c['forward_iou']>=.2]
    assert cand['new_tid']==case['new'] and cand['raw_id']==case['raw_id']
    track=row['row_data']
    h=track['history'][-1]
    ref_seq=[[s['frame'],s['crop']] for s in track['history']]
    seqs=[ref_seq,ref_seq+track['search']]
    assert [hashlib.sha256(json.dumps([case['video'],seq],separators=(',',':')).encode()).hexdigest()[:32] for seq in seqs]==case['map_keys']
    crop=np.asarray(track['current_crop'])
    extent=crop[2:]-crop[:2]
    assert all(np.array_equal(np.subtract(c[2:],c[:2]),extent) for _,c in [ref_seq[-1]]+track['search'])
    scale=extent/g['eye_size']
    box0=np.asarray(h['box'])
    pred=np.asarray(track['mean'][:2])-(box0[:2]+box0[2:])/2
    old=points*scale+h['crop'][:2]
    xy=old[:,None,:]+pred+offsets*scale
    query=(xy-crop[:2])/scale
    box=np.asarray(cand['box'])
    keep=(((old>=box0[:2])&(old<=box0[2:])).all(1) &
          ((xy>=box[:2])&(xy<=box[2:])).all((1,2)) & (mesh.find_simplex(query)>=0).all(1))
    idx=np.flatnonzero(keep)
    assert len(idx)==case['points']
    query=query[keep]
    simplex=mesh.find_simplex(query)
    tr=mesh.transform[simplex]
    w=np.einsum('...ij,...j->...i',tr[:,:,:2],query-tr[:,:,2])
    w=np.concatenate((w,1-w.sum(-1,keepdims=True)),axis=-1)
    vertices=mesh.simplices[simplex]
    arrays=[]
    for paths in case['map_paths']:
        with np.load(ROOT/paths[0],allow_pickle=False) as z:
            arrays.append({k:z[k].astype(float) for k in ['C','IMAGE']})
    variants={}
    for name,k,sl in [('FlyVis','C',slice(None)),('brightness','IMAGE',slice(0,1)),
                      ('gradient','IMAGE',slice(1,3)),('permuted','C',slice(None))]:
        a=arrays[0][k][idx,sl]
        if name=='permuted':a=a[np.random.default_rng(42).permutation(len(idx))]
        b=arrays[1][k][:,sl]
        sampled=np.einsum('qsv,qsvc->qsc',w,b[vertices])
        # Independent interpolation via SciPy on the same fixed mesh.
        other=LinearNDInterpolator(mesh,b)(query)
        assert np.allclose(other,sampled,rtol=0,atol=1e-9)
        ref=a-a.mean(0)
        cur=sampled-sampled.mean(0)
        scores=[]
        for j in range(len(axes)):
            den=np.linalg.norm(ref)*np.linalg.norm(cur[:,j,:])
            score=float(np.sum(ref*cur[:,j,:])/den) if den else None
            if score is not None:
                check=other[:,j,:]-other[:,j,:].mean(0)
                assert abs(np.dot(ref.ravel(),check.ravel())/(np.linalg.norm(ref)*np.linalg.norm(check))-score)<1e-9
            scores.append(score)
        valid=[j for j,s in enumerate(scores) if s is not None]
        best=max(valid,key=lambda j:scores[j]) if valid else None
        peak=scores[best] if valid else None
        variants[name]=dict(fixed=scores[zero],local=peak,
            gain=peak-scores[zero] if valid and scores[zero] is not None else None,
            winning_offset_px=(offsets[best]*scale).tolist() if valid else None,scores=scores)
    results.append(dict(video=case['video'],frame=case['frame'],old=case['old'],new=case['new'],kind=case['kind'],
          split=case['split'],cells=idx.tolist(),predicted_px=pred.tolist(),offsets_px=(offsets*scale).tolist(),variants=variants))
assert all(sha(p)==s for p,s in frozen['inputs'].items())
write('scores.json',dict(protocol_sha256=sha(HERE/'protocol.json'),pairs=results,
      verification='Triangle interpolation and independent scalar products agree within 1e-9 on all defined scores.'))
