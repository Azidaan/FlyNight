"""Data-only explanation: was the earlier verified identity a current candidate?"""
import study as s
from study import *
def register():
    parent=ROOT/'runs/direct_memory_anchor_inventory_20261109/protocol.json';p=read(parent);verify(p)
    write('candidate_audit_protocol.json',dict(registered=stamp(),selection='All19 geometrically supported events, including ALL3 foreign events regardless of measured score. No model, frame decode or new similarity. Locate exact earlier GT identity at the exact current annotated frame; absence of its label is not proof of physical absence. Inspect raw detector overlap, confidence, original candidate list, saved forwardIoU and current eye coverage. No GT interpolation or threshold tuning.',criteria='Every supported event accounted for, actual current/earlier annotation identities preserved, missing identity label stays explicitly unknown, existing raw/candidate values reused.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),parent,OUT/'protocol.json',OUT/'results.json',ROOT/'runs/direct_memory_anchor_inventory_20261109/results.json',ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json']}))
def overlap(a,b):
    i=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();u=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-i
    return float(i/u) if u>0 else 0.
def run():
    ap=read(OUT/'candidate_audit_protocol.json');verify(ap);p=read(OUT/'protocol.json');verify(p);source=read(ROOT/'runs/direct_memory_anchor_inventory_20261109/results.json');parent=read(ROOT/'runs/direct_memory_anchor_inventory_20261109/protocol.json');cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);records=[]
    for c in p['cases']:
        if len(c['commonplan']['cells'])<3:continue
        def eq(r):return all(c[k]==r[k] for k in ['video','frame','old','new'])
        r=next(x for x in source['records'] if eq(x));info=parent['videos'][c['video']];gt=read(ROOT/info['annotation']);alignment=read(ROOT/info['alignment'])['matches'];frame_index=[int(x['video_frame_index']) for x in alignment].index(c['frame']);labels=gt[frame_index].get('labels') or [];identity=r['reference_gt_hits'][0]['identity'];oldlabels=[x for x in labels if str(x['id'])==identity]
        out={k:c[k] for k in ['video','frame','old','new','kind']};out.update(earlier_identity=identity,current_candidate_identity=r['current_gt_hits'][0]['identity'],exact_current_identity_labels=len(oldlabels),source_annotation=info['annotation'],current_label_frame_index=frame_index)
        assert len(oldlabels)<=1
        if not oldlabels:out['status']='earlier_identity_has_no_current_annotation_not_physical_absence_proof'
        else:
            label=oldlabels[0];box=np.array([label['box2d'][k] for k in ['x1','y1','x2','y2']]);out.update(identity_box=box,identity_attributes=label.get('attributes') or {})
            with np.load(ROOT/info['raw'],allow_pickle=False) as z:
                ids=np.flatnonzero((z['frame_idx']==c['frame'])&(z['cls']==2));boxes=z['xyxy'][ids];scores=z['conf'][ids];iou=np.array([overlap(box,b) for b in boxes]);order=np.argsort(-iou,kind='stable');selected=order[:5]
                out['best_raw_detections']=[dict(raw_id=int(ids[j]),box=boxes[j],confidence=float(scores[j]),gt_iou=float(iou[j]),matches_identity=bool(iou[j]>=.5)) for j in selected]
                best=out['best_raw_detections'][0] if len(selected) else None
                crop=np.array(r['track']['current_crop']);sc=(crop[2:]-crop[:2])/g['eye_size'];center=(box[:2]+box[2:])/2;q=(center-crop[:2])/sc;out['identity_center_inside_eye_hull']=bool(reader.mesh.find_simplex(q)>=0);out['best_gt_iou']=best['gt_iou'] if best else None
                admitted=[]
                for j in order:
                    if iou[j]<.5:continue
                    rid=int(ids[j]);existing=next((x for x in r['track'].get('candidates',[]) if x.get('raw_id')==rid),None)
                    # Full saved row is read rather than reconstructing its filter.
                    rows=read((ROOT/info['native']).with_suffix('.json'))['events'];event=next(e for e in rows if e['frame']==c['frame']);row=next(x for x in event['rows'] if x['old_tid']==c['old']);candidate=next((x for x in row['candidates'] if x['raw_id']==rid),None)
                    mean=np.array(r['track']['mean'][:4]);predbox=np.r_[mean[:2]-mean[2:]/2,mean[:2]+mean[2:]/2];fiou=candidate['forward_iou'] if candidate else overlap(predbox,boxes[j]);early=r['reference_raw'];pred=mean[:2]-(np.array(early['state_box'][:2])+early['state_box'][2:])/2
                    plan=reader.plan(early['state_box'],r['reference_recipe'][-1][1],boxes[j],crop,pred,c['bplan']['offsets'],reference_observation=early['box'])
                    admitted.append(dict(raw_id=rid,confidence=float(scores[j]),gt_iou=float(iou[j]),saved_candidate=candidate,forward_iou=fiou,passes_frozen_geometry=fiou>=.2,passes_native_high_confidence=bool(scores[j]>=cfg['track_high_thresh']),earlier_template_19shift_support=len(plan['cells'])))
                out.update(identity_detections=admitted,status='exact_current_identity_annotation_available')
        records.append(out)
    assert len(records)==19 and sum(r['kind']=='foreign' for r in records)==3
    verify(ap);verify(p);write('candidate_audit.json',dict(passed=True,records=records,no_model_or_similarity_calls=True,no_missing_identity_imputation=True));print('Candidate availability audit complete',flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
