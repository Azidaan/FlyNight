"""Read-only inventory of car singleton events, annotation evidence and paired maps."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs'
sys.path.insert(0, str(RUN / 'flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=240)
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linear_sum_assignment

sources = {}
def read(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))

def overlap(boxes, truth):
    a = np.asarray(boxes, float).reshape(-1, 4)
    b = np.asarray(truth, float).reshape(-1, 4)
    area = np.maximum(0, np.minimum(a[:, None, 2:], b[:, 2:]) -
                      np.maximum(a[:, None, :2], b[:, :2])).prod(-1)
    return area / np.maximum(1e-10, (a[:, 2:] - a[:, :2]).prod(1)[:, None] +
                             (b[:, 2:] - b[:, :2]).prod(1) - area)

def key(video, sequence):
    return hashlib.sha256(json.dumps([video, sequence], separators=(',', ':')).encode()).hexdigest()[:32]

geometry = read(RUN / 'z13_local_peak_20261021/geometry.json')
points = np.asarray(geometry['points'])
grid = Delaunay(points)
offsets = np.array([(u, v) for u in range(-2, 3) for v in range(-2, 3)
                    if max(abs(u), abs(v), abs(u+v)) <= 2]) @ np.array([[13, 6.5], [0, 13]])
old_support = {(x['video'], x['frame'], x['old'], x['new']): x
               for x in read(RUN/'flyvis_local_shift_feasibility_20261108/eligible_pairs.json')}

def geometry_check(track, box, saved):
    h = track['history'][-1]
    crops = np.array([h['crop']] + [c for f, c in track['search']])
    assert np.array_equal(crops[-1], track['current_crop'])
    sizes = crops[:, 2:] - crops[:, :2]
    same = bool((sizes == sizes[0]).all())
    if saved:
        assert same == saved['constant_scale_after_ref'] and saved['prefix']
        return same, saved['fixed_cells']
    S0, St = sizes[[0,-1]] / geometry['eye_size']
    X = points * S0 + crops[0,:2]
    b = np.asarray(h['box'])
    X = X[((X >= b[:2]) & (X <= b[2:])).all(1)]
    trial = X[:,None,:] + np.array(track['mean'][:2]) - (b[:2]+b[2:])/2 + offsets * np.maximum(S0,St)
    inside = ((trial >= box[:2]) & (trial <= box[2:])).all(-1)
    hull = grid.find_simplex((trial-crops[-1,:2])/St) >= 0
    return same, int((inside & hull).all(1).sum())

packed = {}
for folder in ['bbox_fit_maps','bbox_normalized_signals','singleton_verify','native_confirm_policy','identity_assignment','spatial_identity_maps']:
    date = '20261107' if folder=='spatial_identity_maps' else '20261108'
    base = RUN / (folder+'_'+date)
    manifest = base/'features_complete.json'
    if manifest.exists():
        for k, sha in read(manifest).get('files', {}).items():
            packed.setdefault(k,(base/'features'/(k+'.npz'), sha))
full_sources = [RUN/p for p in ('bbox_normalized_signals_20261108/full','singleton_verify_20261108/full','native_confirm_risk_20261108/full','identity_assignment_20261108/full_precision')]
ready_manifest = read(full_sources[0]/'protocol.json')['ready_feature_sources']
def maps(video, k):
    if k in packed:
        path, expected = packed[k]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        with np.load(path,allow_pickle=False) as z:
            if all(t in z for t in ('C','IMAGE')):
                assert all(z[t].shape==(721,n) and np.isfinite(z[t]).all() for t,n in [('C',8),('IMAGE',3)])
                return [path.relative_to(ROOT).as_posix()]
    for base in full_sources:
        paths = [base/'features'/video/(k+'.'+t+'.npy') for t in ['C','IMAGE']]
        if all(p.is_file() for p in paths):
            receipt_path = base/'feature_receipts'/(video+'.json')
            receipt = (read(receipt_path)['artifacts'] if receipt_path.is_file() else
                       {v['path']:v['sha256'] for v in ready_manifest.get(video,{}).values()})
            if not all(p.relative_to(ROOT).as_posix() in receipt for p in paths):
                for arm in ['FLYVIS','IMAGE']:
                    arm_receipt=base/'rollouts'/arm/(video+'.receipt.json')
                    if arm_receipt.exists():receipt.update(read(arm_receipt).get('feature_files',{}))
            for path,n in zip(paths,[8,3]):
                assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt[path.relative_to(ROOT).as_posix()]
                a=np.load(path,allow_pickle=False)
                assert a.shape==(721,n) and np.isfinite(a).all()
            return [p.relative_to(ROOT).as_posix() for p in paths]
    return []

inventory = RUN/'bbox_confirmation_inventory_20261108'
working = RUN/'singleton_working_domain_20261108'
videos = {v:(spec['split'], inventory) for v,spec in read(inventory/'protocol.json')['videos'].items()}
for path in (working/'labels').glob('*.json'):
    assert path.stem not in videos
    videos[path.stem]=('development20',working)
records=[]
for video,(split,archive) in videos.items():
    data=read(archive/'native'/(video+'.json'))
    candidates=[]
    for event in data['events']:
        for r in event['rows']:
            if r['age']>2 or r['row_data']['cls']!=2:
                continue
            allowed=[c for c in r['candidates'] if c['forward_iou']>=.2]
            if len(allowed)==1 and allowed[0]['new_tid'] is not None:
                candidates.append((event['frame'],r,allowed[0]))
    if not candidates:continue
    val=split=='val20'
    annotation=read(ROOT/'data/bdd100k/labels/box_track_20'/('val' if val else 'train')/(video+'.json'))
    if val:
        anchors=read(full_sources[0]/'runtime_inputs'/(video+'.json'))['anchors']
        by_index={f['frameIndex']:f for f in annotation}
        by_frame={int(t):by_index[c['frameIndex']] for t,c in anchors.items()}
        assert all(by_frame[int(t)]['name']==c['name'] for t,c in anchors.items())
    else:
        align=read(RUN/'train_cache'/video/'alignment.json')['matches']
        assert len(align)==len(annotation)
        by_frame={int(a['video_frame_index']):f for a,f in zip(align,annotation)}
    targets={}
    for t,f in by_frame.items():
        objects=[a for a in f.get('labels') or [] if a['category']=='car'
                 and not (a.get('attributes') or {}).get('crowd',False)]
        targets[t]=([str(a['id']) for a in objects],[[a['box2d'][k] for k in ('x1','y1','x2','y2')] for a in objects])
    if not val:
        label_base=working if archive==working else RUN/'bbox_singleton_pilot_20261108'
        labels=read(label_base/'labels'/(video+'.json'))['rows']
    else:
        # Reconstruct only missing validation fragment labels on saved native outputs.
        with np.load(archive/'native'/(video+'.npz')) as z:native=z['outputs']
        needed=set()
        for t,r,c in candidates:
            end=r['row_data']['history'][-1]['frame']
            needed.update(f for f in by_frame if end-12<=f<=end or t<=f<=t+12)
        matches={}
        for f in sorted(needed):
            arr=native[(native[:,0]==f)&(native[:,7]==2)]
            ids,boxes=targets[f]
            s=overlap(arr[:,1:5],boxes)
            s[s<.5]=0
            ri,ci=linear_sum_assignment(-s)
            matches[f]={int(arr[i,5]):ids[j] for i,j in zip(ri,ci) if s[i,j]>0}
    for t,r,c in candidates:
        old,new=r['old_tid'],c['new_tid']
        track=r['row_data']
        last=track['history'][-1]['frame']
        if not val:
            l=labels[f'{t}|{old}|{new}']
            past,future=l['target_anchors'],l['fragment_anchors']
        else:
            past=[[f,m[old]] for f,m in matches.items() if last-12<=f<=last and old in m]
            future=[[f,m[new]] for f,m in matches.items() if t<=f<=t+12 and new in m]
        old_ids={x[1] for x in past}
        new_ids={x[1] for x in future}
        target=next(iter(old_ids)) if len(old_ids)==1 else None
        identity=next(iter(new_ids)) if len(new_ids)==1 else None
        if not val:assert (target,identity)==(l['target'],l['identity'])
        kind='unknown' if target is None or identity is None else ('own' if target==identity else 'foreign')
        ids,gt_boxes=targets.get(t,([],[]))
        hits=[(g,float(s)) for g,s in zip(ids,overlap([c['box']],gt_boxes)[0]) if s>=.5]
        # Require a direct match of the confirmed native output AND of the raw box
        # to the same unique annotated car, in addition to stable fragment labels.
        direct=kind!='unknown' and len(hits)==1 and hits[0][0]==identity and [t,identity] in future
        ref=[[h['frame'],h['crop']] for h in track['history']]
        keys=[key(video,seq) for seq in (ref,ref+track['search'])]
        available=[maps(video,k) for k in keys]
        saved=old_support.get((video,t,old,new))
        if saved:assert keys==[saved['reference_key'],saved['current_key']]
        scale,npoints=geometry_check(track,np.array(c['box']),saved)
        records.append(dict(video=video,split=split,frame=t,old=old,new=new,age=r['age'],raw_id=c['raw_id'],
            forward_iou=c['forward_iou'],kind=kind,direct=direct,target=target,identity=identity,
            target_anchors=past,fragment_anchors=future,raw_matches=hits,last_observation=last,
            same_scale=scale,points=npoints,map_keys=keys,map_paths=available,
            eligible=all(available) and scale and npoints>=3))

report=dict(videos=Counter(s for s,_ in videos.values()),cases=records,sources=sources,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (OUT/'inventory.json').open('x',encoding='utf8') as dest:
    json.dump(report,dest,ensure_ascii=False,indent=2)
print('car singleton cases',len(records))
for split in sorted(set(r['split'] for r in records)):
    group=[r for r in records if r['split']==split]
    print(split,Counter((r['kind'],r['direct'],r['eligible']) for r in group))
