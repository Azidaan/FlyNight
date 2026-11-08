"""Exact-frame identity and border inventory, before any transfer scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v))
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,default=default,allow_nan=False)
def match(box,objects):
    hits=[]
    for o in objects:
        if o['category']!='car' or (o.get('attributes') or {}).get('crowd',False):continue
        b=[o['box2d'][k] for k in ['x1','y1','x2','y2']];w=max(0,min(box[2],b[2])-max(box[0],b[0]));h=max(0,min(box[3],b[3])-max(box[1],b[1]));u=max(0,box[2]-box[0])*max(0,box[3]-box[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-w*h
        iou=w*h/u if u else 0
        if iou>=.5:hits.append(dict(identity=str(o['id']),iou=iou,box=b))
    return hits
def register():
    p=read(OUT/'protocol.json');old=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json');fit={e['video'] for e in read(ROOT/'runs/background_patch_gate_20261108/protocol.json')['events']};paths=[Path(__file__),OUT/'protocol.json',ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json',ROOT/'runs/clipped_anchor_readout_20261109/study.py',ROOT/'runs/z13_local_peak_20261021/geometry.json'];videos={};missing=[]
    for video,e in old['videos'].items():
        if video in fit:continue
        native=ROOT/f'runs/bbox_confirmation_inventory_20261108/native/{video}.npz'
        if not native.exists():missing.append(video);continue
        raw=ROOT/e['raw'];ann=ROOT/f'data/bdd100k/labels/box_track_20/{"val" if e["split"]=="val20" else "train"}/{video}.json';alignment=raw.parent.parent/'alignment.json'
        files=[native,native.with_suffix('.json'),raw,ann,alignment]
        assert all(f.exists() for f in files),[(str(f),f.exists()) for f in files];paths+=files;videos[video]=dict(split=e['split'],native=native.relative_to(ROOT).as_posix(),raw=raw.relative_to(ROOT).as_posix(),annotation=ann.relative_to(ROOT).as_posix(),alignment=alignment.relative_to(ROOT).as_posix())
    write('external_selection_protocol.json',dict(registered=__import__('datetime').datetime.now().isoformat(),parent_protocol_sha256=sha(OUT/'protocol.json'),reason='All6 original>=3 direct-current cases lack border clipping. Need directly annotated old and current observations from the saved max_age6 native context to test border transfer.',selection='All other-video car rows with exactly one candidate at the unchanged forward IoU>=0.2 and new_tid present, maximum age6. Require unique noncrowd IoU>=0.5 GT match of BOTH recovered raw reference and current detection on actual aligned annotation frames. No interpolation, score selection or cap by outcomes. Keep all clip/no-clip cases and support failures.',videos=videos,unavailable_native_context=missing,reader_unchanged=True,deadline=p['deadline'],sources={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(paths))}))
def run():
    p=read(OUT/'external_selection_protocol.json')
    for path,h in p['sources'].items():assert sha(ROOT/path)==h,path
    ns=dict(np=np);path=ROOT/'runs/clipped_anchor_readout_20261109/study.py';node=next(n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='raw_observation');exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2]);rows=[];counts={};sources=[]
    for video,info in p['videos'].items():
        gt=read(ROOT/info['annotation']);align=read(ROOT/info['alignment'])['matches'];assert len(gt)==len(align);ann={int(a['video_frame_index']):f.get('labels') or [] for f,a in zip(gt,align)}
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={n:z[n] for n in ['frame_idx','xyxy','conf','cls']}
        video_counts=dict(singletons=0,not_both_annotated=0,not_both_unique=0,confirmed=0)
        for event in read((ROOT/info['native']).with_suffix('.json'))['events']:
            for r in event['rows']:
                choices=[c for c in r['candidates'] if c['forward_iou']>=.2]
                if len(choices)!=1 or choices[0]['new_tid'] is None or int(r['row_data']['cls'])!=2:continue
                can=choices[0];t=r['row_data'];h=t['history'][-1];frame=event['frame'];video_counts['singletons']+=1
                if frame not in ann or h['frame'] not in ann:video_counts['not_both_annotated']+=1;continue
                a=ns['raw_observation'](native,raw,r['old_tid'],h['frame'],cfg);b=ns['raw_observation'](native,raw,can['new_tid'],frame,cfg)
                assert np.array_equal(np.array(h['box'],np.float32),np.array(a['state_box'],np.float32));assert np.array_equal(b['box'],can['box'])
                ref=match(a['box'],ann[h['frame']]);cur=match(b['box'],ann[frame])
                if len(ref)!=1 or len(cur)!=1:video_counts['not_both_unique']+=1;continue
                video_counts['confirmed']+=1;d=np.array(t['mean'][:2])-(np.array(h['box'][:2])+h['box'][2:])/2;scale=np.maximum((np.array(h['crop'][2:])-h['crop'][:2])/g['eye_size'],(np.array(t['current_crop'][2:])-t['current_crop'][:2])/g['eye_size']);offsets=(axes@np.array([[13.,6.5],[0.,13.]]))*scale
                c=dict(video=video,split=info['split'],frame=frame,old=r['old_tid'],new=can['new_tid'],kind='own' if ref[0]['identity']==cur[0]['identity'] else 'foreign',reference_gt=ref[0],current_gt=cur[0],track=t,candidate=can,reference_observation=a,current_observation=b,clipped=any(bb[0]<=0 or bb[1]<=0 or bb[2]>=1280 or bb[3]>=720 for bb in [a['box'],b['box']]))
                c['plan_old']=reader.plan(h['box'],h['crop'],can['box'],t['current_crop'],d,offsets,use_visible_edge=False);c['plan_joint']=reader.plan(h['box'],h['crop'],can['box'],t['current_crop'],d,offsets,reference_observation=a['box']);c['sequences']=[[[v['frame'],v['crop']] for v in t['history']],[[v['frame'],v['crop']] for v in t['history']]+t['search']];c['map_keys']=[hashlib.sha256(json.dumps([video,s],separators=(',',':')).encode()).hexdigest()[:32] for s in c['sequences']];rows.append(c)
        counts[video]=video_counts;print('Inventory',video,video_counts,flush=True)
    for path,h in p['sources'].items():assert sha(ROOT/path)==h,path
    write('external_inventory.json',dict(passed=True,cases=rows,counts=counts,selection_protocol_sha256=sha(OUT/'external_selection_protocol.json'),sources_stable=True));print('Confirmed',[(r['video'],r['frame'],r['old'],r['new'],r['kind'],r['clipped'],len(r['plan_old']['cells']),len(r['plan_joint']['cells'])) for r in rows],flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
