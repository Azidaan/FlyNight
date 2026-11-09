"""Read exact native owners of detections, without replaying the tracker."""
import study as s
from study import *
def register():
    files=[Path(__file__),OUT/'candidate_audit.json',OUT/'candidate_audit_protocol.json',OUT/'protocol.json',ROOT/'runs/direct_memory_anchor_inventory_20261109/protocol.json',ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json',ROOT/'runs/clipped_anchor_readout_20261109/study.py',ROOT/'research_bbox_confirmation_inventory_20261108/export.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py']
    write('ownership_protocol.json',dict(registered=stamp(),selection='All3 foreign events from candidate_audit, no selection by score. Map every actually matched raw identity detection to an exact public native output using saved subset-local indices and confidence/class witnesses. No new tracker replay, cost matrix, model, threshold or similarity.',scope='Candidate list is free same-class HIGH detections AFTER native first global assignment. A missing raw candidate may already belong to a public track; low confidence may fall below the track low threshold or be unavailable to a Lost track. Do not infer an unseen full cost matrix or invent why another track won.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
def run():
    op=read(OUT/'ownership_protocol.json');verify(op);p=read(OUT/'protocol.json');verify(p);parent=read(ROOT/'runs/direct_memory_anchor_inventory_20261109/protocol.json');cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config'];audit=read(OUT/'candidate_audit.json');records=[]
    ns=dict(np=np);source=ROOT/'runs/clipped_anchor_readout_20261109/study.py';node=next(n for n in ast.parse(source.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='raw_observation');exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    for r in audit['records']:
        if r['kind']!='foreign':continue
        info=parent['videos'][r['video']]
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={n:z[n] for n in ['frame_idx','conf','cls','xyxy']}
        observations=[dict(tid=int(row[5]),observation=ns['raw_observation'](native,raw,int(row[5]),r['frame'],cfg)) for row in native[native[:,0]==r['frame']]]
        targets=[]
        for det in r.get('identity_detections',[]):
            owners=[x for x in observations if x['observation']['raw_id']==det['raw_id']];assert len(owners)<=1
            targets.append(dict(**det,public_native_owners=owners,below_native_low=det['confidence']<=cfg['track_low_thresh']))
        records.append(dict(video=r['video'],frame=r['frame'],old=r['old'],foreign_new=r['new'],earlier_identity=r['earlier_identity'],own_identity_detections=targets,old_public_at_current=any(x['tid']==r['old'] for x in observations),foreign_candidate_public=next(x for x in observations if x['tid']==r['new'])))
    verify(op);verify(p);write('ownership_audit.json',dict(passed=True,records=records,zero_tracker_replay=True,full_native_cost_matrix_not_saved=True));print('Exact ownership audit complete',flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
