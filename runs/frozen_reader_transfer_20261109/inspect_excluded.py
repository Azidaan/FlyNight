"""Check whether the two original zero-support exclusions hide a border case."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config'];context=read(OUT/'external_selection_protocol.json')['videos'];rows=[];sources=[Path(__file__),OUT/'readiness.json',OUT/'external_selection_protocol.json']
src=ROOT/'runs/clipped_anchor_readout_20261109/study.py';node=next(n for n in ast.parse(src.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='raw_observation');ns=dict(np=np);exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
for c in read(OUT/'readiness.json')['excluded']:
    info=context[c['video']];npz=ROOT/info['native'];data=read(npz.with_suffix('.json'));event,=[e for e in data['events'] if e['frame']==c['frame']];row,=[r for r in event['rows'] if r['old_tid']==c['old']];can,=[r for r in row['candidates'] if r['forward_iou']>=.2];assert can['new_tid']==c['new'];t=row['row_data'];h=t['history'][-1]
    with np.load(npz,allow_pickle=False) as z:native=z['outputs']
    with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={n:z[n] for n in ['frame_idx','xyxy','conf','cls']}
    a=ns['raw_observation'](native,raw,c['old'],h['frame'],cfg);b=ns['raw_observation'](native,raw,c['new'],c['frame'],cfg);d=np.array(t['mean'][:2])-(np.array(h['box'][:2])+h['box'][2:])/2;scale=np.maximum((np.array(h['crop'][2:])-h['crop'][:2])/g['eye_size'],(np.array(t['current_crop'][2:])-t['current_crop'][:2])/g['eye_size']);axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2]);off=(axes@np.array([[13.,6.5],[0.,13.]]))*scale
    old=reader.plan(h['box'],h['crop'],can['box'],t['current_crop'],d,off,use_visible_edge=False);new=reader.plan(h['box'],h['crop'],can['box'],t['current_crop'],d,off,reference_observation=a['box']);assert len(old['cells'])==c['points']==0
    rows.append(dict(video=c['video'],frame=c['frame'],old=c['old'],new=c['new'],kind=c['kind'],clipped=any(bb[0]<=0 or bb[1]<=0 or bb[2]>=1280 or bb[3]>=720 for bb in [a['box'],b['box']]),old_cells=0,joint_cells=len(new['cells']),anchor_changed=new['anchor']['changed'],reference_raw=a['box'].tolist(),current_raw=b['box'].tolist()))
    sources += [npz,npz.with_suffix('.json'),ROOT/info['raw']]
with (OUT/'excluded_geometry.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,records=rows,no_scores_computed=True,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources}),f,ensure_ascii=False,indent=2)
print(rows)
