"""Direct identities on archived PRE-assignment two-box car queries, no replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,6,35,tzinfo=MSK)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(STOP-datetime.now(MSK)).total_seconds()
if remaining<=0:raise SystemExit('Save/stop reached')
guard_entrypoint(timeout=min(900,remaining))
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [cast(x) for x in v]
    return v
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
def register():
    old=ROOT/'runs/flyvis_pair_support_overlap_20261107/recovery1';p=read(ROOT/'runs/z13_spatial_bins_gate_20261025/recovery1/protocol.json');videos=p['fit84'];assert len(videos)==84
    sources=[Path(__file__),old/'protocol.json',old/'input_sha256.json',ROOT/'runs/z13_spatial_bins_gate_20261025/recovery1/protocol.json',ROOT/'runs/interpolation_constant_audit_20261109/audit.json',ROOT/'runs/free_candidate_identity_inventory_20261109/audit.json',ROOT/'scripts/z13_data.py',ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/hex_linear_interpolation.py',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'runs/frozen_reader_transfer_20261109/external_inventory.py',ROOT/'runs/z13_local_peak_20261021/geometry.json',ROOT/'tracking/z13_region_assignment_context.py'];info={}
    original=read(old/'input_sha256.json')
    for v in videos:
        files=dict(data=f'runs/z13_data/{v}.npz',raw=f'runs/train_cache/{v}/detections/raw_video.npz',native=f'runs/fly_signatures/{v}/signatures.npz',labels=f'runs/z13_region_assignment_20261025/labels/videos/{v}.json',temporal=f'runs/z13_temporal_observation_gate_20261025/videos/{v}.json',context=f'runs/z13_box_supervision_followup_20261025/cache/videos/{v}.json',annotation=f'data/bdd100k/labels/box_track_20/train/{v}.json',alignment=f'runs/train_cache/{v}/alignment.json')
        for k in ['data','raw','labels','temporal']:assert sha(ROOT/files[k])==original[files[k]],files[k]
        sources.extend(ROOT/f for f in files.values());info[v]=files
    write('protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline='2026-11-09T06:40:00+03:00',scientific_save_stop=STOP.isoformat(),videos=info,
        population='The84 historical fit84 train videos, disjoint from prior20 reader-development and26 evaluation videos. These videos were used by older learned readers, not a new dataset split. Reuse exactly-two-real-candidate car queries from archived PRE-native-first-assignment records; no rerun of the130-video overlap count or tracker.',
        selection='Unperturbed episode, exactly2 saved real same-class high>=.25 boxes, original native row unmatched, last observed warm frame age<=6. Current frame must be an actual annotation frame. Choose latest actual annotation timestamp among four warm observations, never skip a timestamp because identity is inconvenient. Require at least2 distinct input times by reference. Map native reference row to raw detector by UNIQUE exact float32 score/class/frame, never nearest-box substitution; then require one noncrowd car GT IoU>=.5.',
        clean='Both current raw candidate boxes uniquely GT labelled, one same ID as reference and one different. Both frozen19-shift supports>=3, different reference-cell sets (same sets and same query tensor imply exact ties for the present reader). Exclude a reference with another public native detection overlapping the same target GT>=.5; retain unknown reference provenance. Store current native owners without inventing a global assignment reason.',
        math='All maps for a query can share the archived prefix: early reference snapshot and current snapshot from one state trajectory. For no image clipping, current query of a reference point is independent of candidate; identical cell sets imply identical scores. A pooled across-video AUROC does not replace an own-vs-foreign choice WITHIN each query.',
        ready='Require>=5cleanqueries on>=5videos before any feature restoration. Select at most12, exactly the first clean UID per video in the frozen84-video order. No scores consulted. All retinal views already exist in fr_hex; any subsequent model stage needs a separate frozen protocol and current deadline. GT is evaluation only, not runtime selection.',
        no_activity='Do not decompress fr_hex or run image/network/detector/tracker/score calculations in this gate. Metadata and geometric support only; reuse original cost context[choice][0] as no-FlyVis baseline. Original context flag and native output are checked explicitly.',
        cannot_claim='Identity at the saved reference/current frames does not prove every intermediate tracking ID correct. A pre-assignment choice can include a box already owned by another native track. This is a reader challenge, not an intervention or proof of preserved IDs. Earlier attempt on free residual candidates had no clean queries; this is a different observed pool, not relaxation of its gate.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources}))
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments']);cat['experiments'].append(dict(id=OUT.name,title='Два объекта до общего назначения — прямые метки для проверки считывателя',area='flyvis',state='running',started='2026-11-09',question='Есть ли в прежних сохранённых запросах до общего назначения прямо подтверждённые своя и чужая машины для одной памяти?',hypothesis='Ограничение первого публичного подтверждения теряет размеченные моменты. Прежние запросы глаза до назначения могут дать прямой внутрикандидатный контроль, но промежуточныеIDи конкуренция требуют отдельного учёта.',expected='Новый аудит прямыхGTна84историческихвидео, исходные пары и входы переиспользованы; никаких новых карт или повторения старых подсчётов.',acceptance=[dict(criterion='Прямые метки, точный raw/native/prefix provenance, все исключения и неизвестные сохранены; no inference/score/replay.',met=None),dict(criterion='Не менее5чистыхзапросов/5видео с различными допустимыми поддержками для дальнейшего сравнения.',met=None)],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Пул до первого назначения и правила до новых GTметок')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf8');now=datetime.now(MSK);entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Поиск прямых сравнений в сохранённом пуле до общего назначения.\n\n*Постановка (`{OUT.name}`).* Только84историческихfit84видео вне20/26последнихпространственныхпроверок. В отличие от свободного остатка, это прежние реальные запросы глаза до первого общего назначения, уже приходящиеся на размеченные кадры. Нужны две сырые рамки одной машине памяти: своя/чужая, единственнаяGTобеих и прошлойопоры, недублирующийся источникID,≥3ячейки и разныеподдержки. Прямая опора выбирается среди4готовыхведомыхвходов; меткиGTне входят в будущую рабочуюполитику. Доактивности минимум5запросов/5видео; отборпервыхUID, максимум12. Исторические1368пересечений/130видео не пересчитываются. Только метаданные, срок06:40/сохранение06:35. Условия `runs/{OUT.name}/protocol.json`.\n\n';path.write_text(s.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8');print('Registered historical pre-assignment direct query gate',flush=True)
def run():
    p=read(OUT/'protocol.json');verify(p)
    if (OUT/'results.json').exists():raise SystemExit('Ready inventory exists')
    ns=dict(np=np);source=ROOT/'runs/frozen_reader_transfer_20261109/external_inventory.py';node=next(n for n in ast.parse(source.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='match');exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns);match=ns['match'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2]);records=[];clean=[];counts={};selected=[]
    for video,info in p['videos'].items():
        with np.load(ROOT/info['data'],allow_pickle=False) as z:data={k:z[k] for k in ['ep_offset','ep_tid','ep_pert','fr_frame','fr_crop','lab_ep','lab_pos','lab_pred']}
        with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','conf','cls','xyxy']}
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs'];assert native.shape[1]>=8
        labels=read(ROOT/info['labels'])['rows'];temporal={r['uid']:r for r in read(ROOT/info['temporal'])['rows']};contexts=read(ROOT/info['context'])['entries'];gt=read(ROOT/info['annotation']);align=read(ROOT/info['alignment'])['matches'];ann={int(a['video_frame_index']):f.get('labels') or [] for f,a in zip(gt,align)};rawframes={}
        def observe(row):
            frame=int(row[0]);ids=rawframes.setdefault(frame,np.flatnonzero(raw['frame_idx']==frame));hits=ids[(raw['conf'][ids].astype(np.float32)==np.float32(row[6]))&(raw['cls'][ids]==int(row[7]))]
            return None if len(hits)!=1 else dict(raw_id=int(hits[0]),box=raw['xyxy'][hits[0]],state_box=row[1:5],confidence=float(row[6]),cls=int(row[7]),frame=frame)
        bylab={(int(e),int(pos)):pred for e,pos,pred in zip(data['lab_ep'],data['lab_pos'],data['lab_pred'])}
        for label in labels:
            if label['regions']['track_class']!=2 or len(label['regions']['choices'])!=4:continue
            uid=label['uid'];meta=temporal[uid];context=contexts[str(uid)];e=meta['episode_local'];off=int(data['ep_offset'][e]);current=int(meta['current_array_index']);frame=label['frame'];last=int(data['fr_frame'][off+3]);r=dict(video=video,uid=uid,frame=frame,tid=label['tid'],status='unknown',clean=False)
            assert not data['ep_pert'][e] and int(data['ep_tid'][e])==r['tid'] and data['fr_frame'][current]==frame and frame in ann
            if context['context'][0][4]!=0:r['status']='native_first_row_already_matched'
            elif frame-last>6:r['status']='last_observation_older_than6'
            elif np.any((native[:,0]==frame)&(native[:,5]==r['tid'])):r['status']='native_old_id_public_current'
            else:
                refs=[i for i in range(off,off+4) if int(data['fr_frame'][i]) in ann and data['fr_frame'][i]<frame]
                if not refs:r['status']='no_direct_warm_reference'
                else:
                    index=max(refs,key=lambda i:(int(data['fr_frame'][i]),i));rf=int(data['fr_frame'][index]);rows=native[(native[:,0]==rf)&(native[:,5]==r['tid'])];r.update(reference_frame=rf,reference_index=index)
                    if len(set(map(int,data['fr_frame'][off:index+1])))<2:r['status']='reference_has_only_one_distinct_input'
                    elif len(rows)!=1:r['status']='reference_native_row_missing'
                    else:
                        obs=observe(rows[0]);hits=match(obs['box'],ann[rf]) if obs is not None else [];r.update(reference_raw=obs,reference_hits=hits)
                        if obs is None:r['status']='reference_raw_not_unique_by_exact_score'
                        elif len(hits)!=1:r['status']='reference_gt_not_unique'
                        else:
                            identity=hits[0]['identity'];target=next(x for x in ann[rf] if str(x['id'])==identity);tb=np.array([target['box2d'][k] for k in ['x1','y1','x2','y2']]);duplicates=[]
                            for row in native[(native[:,0]==rf)&(native[:,5]!=r['tid'])&(native[:,7]==2)]:
                                other=observe(row)
                                if other is not None and any(h['identity']==identity for h in match(other['box'],ann[rf])):duplicates.append(int(row[5]))
                            r['other_native_same_gt_at_reference']=duplicates
                            if duplicates:r['status']='reference_has_another_native_owner'
                            else:
                                hi=np.flatnonzero((raw['frame_idx']==frame)&(raw['conf']>=.25));assert len(hi)==label['regions']['high_detections'];choices=label['regions']['choices'][2:];assert choices==meta['current_choices'][1:]==context['choices'][1:];rc=np.array(data['fr_crop'][index]);ct=np.array(data['fr_crop'][current]);pred=np.array(bylab[(e,current-off-4)])-(obs['state_box'][:2]+obs['state_box'][2:])/2;scale=np.maximum((rc[2:]-rc[:2])/g['eye_size'],(ct[2:]-ct[:2])/g['eye_size']);offsets=(axes@np.array([[13.,6.5],[0.,13.]]))*scale;cs=[]
                                for pos,j in enumerate(choices):
                                    rid=int(hi[j]);box=raw['xyxy'][rid];ch=match(box,ann[frame]);kind='own' if len(ch)==1 and ch[0]['identity']==identity else ('foreign' if len(ch)==1 else 'unknown');plan=reader.plan(obs['state_box'],rc,box,ct,pred,offsets,reference_observation=obs['box']);owners=[]
                                    for row in native[native[:,0]==frame]:
                                        owned=observe(row)
                                        if owned is not None and owned['raw_id']==rid:owners.append(int(row[5]))
                                    cs.append(dict(column=j,raw_id=rid,box=box,confidence=float(raw['conf'][rid]),kind=kind,hits=ch,plan=plan,original_cost=float(context['context'][pos+1][0]),current_native_owners=owners))
                                r['candidates']=cs;kinds={c['kind'] for c in cs}
                                if kinds!={'own','foreign'}:r['status']='not_unique_own_and_foreign'
                                elif min(len(c['plan']['cells']) for c in cs)<3:r['status']='fewer_than3_in_one_candidate'
                                elif np.array_equal(cs[0]['plan']['cells'],cs[1]['plan']['cells']) and np.array_equal(cs[0]['plan']['query'],cs[1]['plan']['query']):r['status']='identical_queries_force_tie'
                                else:
                                    r.update(status='clean',clean=True,episode=e,reference_relative=index-off,current_relative=current-off,sequence=[[int(data['fr_frame'][i]),data['fr_crop'][i].tolist()] for i in range(off,current+1)],prediction=bylab[(e,current-off-4)].tolist());clean.append(r)
            records.append(r);counts[r['status']]=counts.get(r['status'],0)+1
        candidates=[r for r in clean if r['video']==video]
        if candidates and len(selected)<12:selected.append(min(candidates,key=lambda r:r['uid']))
        (OUT/'status.json').write_text(json.dumps(dict(updated=datetime.now(MSK).isoformat(),video=video,videos_done=list(p['videos']).index(video)+1,total_videos=84,queries=len(records),clean=len(clean)),ensure_ascii=False,indent=2),encoding='utf8');print('INVENTORY',video,len(records),len(clean),flush=True)
    verify(p);videos=len({r['video'] for r in clean});write('results.json',dict(passed=True,records=records,counts=counts,clean_queries=len(clean),clean_videos=videos,selected=selected,readiness_gate=len(clean)>=5 and videos>=5,no_fr_hex_decompression=True,no_model_image_tracker_score_calls=True));print('READY',len(clean),videos,len(selected),flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
