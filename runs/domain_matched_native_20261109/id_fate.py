"""Trace actual public IDs, preserving unknowns and literal raw provenance."""
from evaluation_common import *
start(3600)
import gzip,collections
import numpy as np
p=read(OUT/'protocol.json');phase=read(OUT/'evaluation_protocol.json');verify(phase)
assert all(read(OUT/(a+'_complete.json'))['passed'] for a in p['arms'])
def match(box,objects):
    hits=[];box=np.asarray(box)
    for obj in objects:
        if obj['category']!='car' or (obj.get('attributes') or {}).get('crowd',False):continue
        b=np.array([obj['box2d'][k] for k in ['x1','y1','x2','y2']])
        area=np.maximum(np.minimum(box[2:],b[2:])-np.maximum(box[:2],b[:2]),0).prod()
        union=np.maximum(box[2:]-box[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-area
        if union>0 and area/union>=.5:hits.append(dict(identity=str(obj['id']),iou=float(area/union)))
    return hits
def future_observations(flat,tid,first,ann,target):
    result=[]
    for row in flat[(flat[:,0]>=first)&(flat[:,5]==tid)]:
        frame=int(row[0])
        if frame not in ann:continue
        hits=match(row[1:5],ann[frame]) if int(row[7])==2 else []
        kind='unknown' if len(hits)!=1 else ('own' if hits[0]['identity']==target else 'foreign')
        result.append(dict(frame=frame,kind=kind,hits=hits,box=row[1:5].tolist()))
    return result
all_records=[];summary={};sources={};origin_audits=[]
for video,entry in p['videos'].items():
    clock=read(LEGACY/'runtime_inputs'/(video+'.json'));gtpath=ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json';gt=read(gtpath)
    sources[gtpath.relative_to(ROOT).as_posix()]=sha(gtpath)
    by_index={int(fr.get('frameIndex',i)):fr for i,fr in enumerate(gt)};ann={}
    for frame,anchor in clock['anchors'].items():
        fr=by_index[anchor['frameIndex']];assert fr['name']==anchor['name'];ann[int(frame)]=fr.get('labels') or []
    with np.load(ROOT/entry['detections'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
    with np.load(ROOT/entry['raw_array'],allow_pickle=False) as z:baseline=z['outputs']
    rawframes={};baseline_owners={};future_cache={}
    def owners(frame):
        if frame not in baseline_owners:
            ids=rawframes.setdefault(frame,np.flatnonzero(raw['frame_idx']==frame));result={};unknown=[]
            for row in baseline[baseline[:,0]==frame]:
                hits=ids[(raw['conf'][ids].astype(np.float32)==np.float32(row[6]))&(raw['cls'][ids]==int(row[7]))]
                if len(hits)==1:result.setdefault(int(hits[0]),[]).append(int(row[5]))
                else:unknown.append(int(row[5]))
            baseline_owners[frame]=(result,unknown)
        return baseline_owners[frame]
    for arm in p['arms']:
        eventpath=OUT/'rollouts'/arm/(video+'.events.json.gz');rawpath=OUT/'rollouts'/arm/(video+'.npz');receipt=read(OUT/'rollouts'/arm/(video+'.receipt.json'))
        for path in [eventpath,rawpath]:
            name=path.relative_to(ROOT).as_posix();assert sha(path)==receipt['artifacts'][name];sources[name]=sha(path)
        with gzip.open(eventpath,'rt',encoding='utf8') as f:data=json.load(f)
        with np.load(rawpath,allow_pickle=False) as z:flat=z['outputs']
        public=collections.defaultdict(list)
        for row in flat:public[(int(row[0]),int(row[5]))].append(row)
        assignments=collections.defaultdict(list);candidate_owners=collections.defaultdict(list);audit=collections.Counter()
        for item in data['assignments']:
            item=dict(item);frame=int(item['frame']);rid=int(item['raw_id']);tid=int(item['tid']);rows=public.get((frame,tid),[])
            assert not item['emitted'] or len(rows)==1,(video,arm,frame,tid)
            if rid<0:
                item['raw_id']=None;audit['unknown_raw_origin']+=1
            else:
                assert rid<len(raw['frame_idx']) and int(raw['frame_idx'][rid])==frame
                if item['emitted']:
                    assert np.float32(raw['conf'][rid])==np.float32(rows[0][6]) and int(raw['cls'][rid])==int(rows[0][7])
                    candidate_owners[(frame,rid)].append(tid)
                audit['valid_literal_current_raw']+=1
            assignments[(frame,tid)].append(item)
        assert sum(len(v) for v in public.values())==sum(int(a['emitted']) for a in data['assignments'])
        origin_audits.append(dict(video=video,arm=arm,counts=dict(audit),public_rows=len(flat)))
        counts=summary.setdefault(arm,collections.Counter())
        for event in data['events']:
            frame=event['frame'];actual={tuple(x) for x in event['actual_first']}
            for proposal in event['proposals']:
                counts['all_logged_proposals']+=1;scored=proposal['reason'] in ['head_accept','head_refuse'];counts['scored_proposals']+=int(scored)
                old=proposal['old_id'];rid=proposal['raw_id'];rf=proposal['reference_frame'];reference=assignments.get((rf,old),[]);current=assignments.get((frame,old),[])
                pair=(proposal['row'],proposal['column']);j=proposal['column']
                winners=[dict(row=i,column=c,tid=event['rows'][i]['tid'],state=event['rows'][i]['state'],original_cost=event['C'][i][c],corrected_cost=event['corrected'][i][c]) for i,c in event['actual_first'] if c==j]
                record=dict(video=video,arm=arm,frame=frame,old_id=old,raw_id=rid,reference_frame=rf,reason=proposal['reason'],scored=scored,
                    head_accept=proposal['accepted'],original_cost=proposal['original_cost'],corrected_cost=event['corrected'][pair[0]][pair[1]],logit=proposal.get('logit'),kind='unknown',
                    global_selected=pair in actual,column_global_winners=winners,candidate_public_owners=candidate_owners.get((frame,rid),[]),
                    old_id_emitted_on_candidate=True if any(x['emitted'] and x['raw_id']==rid for x in current) else (None if any(x['emitted'] and x['raw_id'] is None for x in current) else False),
                    current_old_assignments=current,duplicates=event['duplicates'])
                if len(reference)!=1 or not reference[0]['emitted'] or reference[0]['raw_id'] is None:record['unknown_reason']='reference_not_a_unique_public_observation_with_verified_raw'
                elif rf not in ann or frame not in ann:record['unknown_reason']='reference_or_current_not_an_actual_annotation_frame'
                else:
                    rr=reference[0]['raw_id'];record['reference_raw_id']=rr
                    assert int(raw['frame_idx'][rr])==rf and int(raw['frame_idx'][rid])==frame
                    a=match(raw['xyxy'][rr],ann[rf]) if int(raw['cls'][rr])==2 else [];b=match(raw['xyxy'][rid],ann[frame]) if int(raw['cls'][rid])==2 else []
                    record.update(reference_hits=a,current_hits=b)
                    if len(a)!=1 or len(b)!=1:record['unknown_reason']='nonunique_or_missing_direct_GT'
                    else:
                        target=a[0]['identity'];record.update(kind='own' if b[0]['identity']==target else 'foreign',reference_identity=target)
                        native_ref,native_unknown_ref=owners(rf);native_cur,native_unknown_cur=owners(frame);native_ref=native_ref.get(rr,[]);native_cur=native_cur.get(rid,[])
                        record.update(native_reference_owners=native_ref,native_candidate_owners=native_cur,native_ambiguous_reference_ids=native_unknown_ref,native_ambiguous_current_ids=native_unknown_cur)
                        record['native_preserved_id_on_candidate']=None if len(native_ref)!=1 or native_ref[0] in native_unknown_cur else native_ref[0] in native_cur
                        key=(arm,old,frame,target)
                        if key not in future_cache:future_cache[key]=future_observations(flat,old,frame,ann,target)
                        future=future_cache[key];record['future_public_old_ID_at_annotated_frames']=future
                        record['first_future_own_frame']=next((x['frame'] for x in future if x['kind']=='own'),None)
                        record['first_future_foreign_frame']=next((x['frame'] for x in future if x['kind']=='foreign'),None)
                        if len(native_ref)==1:
                            key=('B0',native_ref[0],frame,target)
                            if key not in future_cache:future_cache[key]=future_observations(baseline,native_ref[0],frame,ann,target)
                            bf=future_cache[key];record['native_future_public_old_ID_at_annotated_frames']=bf
                            record['native_first_future_own_frame']=next((x['frame'] for x in bf if x['kind']=='own'),None)
                if scored:
                    label=record['kind'];decision='accepted' if record['head_accept'] else 'refused';counts[decision+'_'+label]+=1
                    counts['global_selected_'+label]+=int(record['global_selected']);counts['old_ID_emitted_'+label]+=int(record['old_id_emitted_on_candidate'] is True)
                    if record['head_accept'] and not record['global_selected']:counts['accepted_but_global_blocked_'+label]+=1
                    if record['global_selected'] and record['old_id_emitted_on_candidate'] is False:counts['global_selected_without_public_oldID_'+label]+=1
                    if label=='unknown':counts['unknown_'+record['unknown_reason']]+=1
                all_records.append(record)
        print('ID_FATE',arm,video,dict(counts),flush=True)
    status('direct_ID_fate',video=video,done=list(p['videos']).index(video)+1,total=20,records=len(all_records))
verify(phase)
write('id_fate.json',dict(passed=True,summary=summary,records=all_records,sources=sources,origin_audits=origin_audits,
    evaluation_sha256=sha(OUT/'evaluation_protocol.json'),
    limits='Direct labels require actual annotated reference and current observations, unique noncrowdcarGTIoU>=0.5 and a public reference with verified raw origin. Unknowns stay unknown. Endpoints do not prove full-history purity. Repeated proposals are not independent objects. Cross-run IDs are linked through reference raw provenance. Earlier return is separate from a rescue; no observed future return is not proof of permanent loss.'))
status('direct_ID_fate_complete',records=len(all_records));print('FATE_SUMMARY',json.dumps(cast(summary)),flush=True)
