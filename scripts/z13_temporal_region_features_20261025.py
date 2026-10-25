"""Frozen matched D/C/R1 features for earlier real regions; no model fitting."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_temporal_region_features_20261025'
GATE=ROOT/'runs/z13_temporal_observation_gate_20261025'
IDENT='z13_temporal_region_features_v1'


def prepare():
    assert not (RUN/'protocol.json').exists()
    from scripts.z13_localization_labels_20261023 import verify_sources
    parent=read(ROOT/'runs/z13_region_assignment_20261025/recovery2/protocol.json');verify_sources(parent)
    assert read(GATE/'report.json')['resource_admission'] and read(GATE/'independent_audit.json')['all_saved_rows_bindings_prior_frames_and_oracle_intervals_verified']
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_bounded_recovery_controller_20261022.py','scripts/z13_target_model.py','scripts/z13_train.py','tracking/z13_region_assignment.py','tracking/z13_readout.py','runs/z13_temporal_observation_gate_20261025/report.json','runs/z13_temporal_observation_gate_20261025/independent_audit.json']
    hashes={**parent['sha256'],**{n:sha(ROOT/n) for n in names},**read(GATE/'report.json')['manifest']}
    stop=min(now()+timedelta(seconds=7200),datetime.fromisoformat(read(ROOT/'runs/z13_weekend_20261024/authorization.json')['save_at']))
    p=dict(parent,id=IDENT,created=now().isoformat(),device='cuda',timezone='Europe/Moscow',budget_seconds=7200,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),sha256=hashes,
        action='Prepare previous-region68activity and observable geometry for one future learned cross-frame FlyVis readout. Existing current features/target/action labels stay immutable and are referenced byUID, not replaced. D/C/R1 have identical selected observations/regions.',
        reference='Use exactly eligible reference rows from verified temporal observation gate,22090UIDs. Need current step4+pos and previous stored-frame index relative to episode. For the single reference at step0, read actual fade-in state.nodes.activity[:,backbone.index]; do not invent an extra step or change current history. All other states use original init/step and neighbors. Receptor R1 uses original function. Selection never reads true identity.',
        features='Previous masks from unchanged regions() on sameclass high-score(.25) raw detections; verify exact raw indices against metadata. Store means68,mask,global/null indicator,6observable values(center relative crop midpoint x/y,logsize x/y,confidence,visible fraction),frame gap,UID. This center is an ROI coordinate, NOT an invented old Kalman prediction. Geometry output of regions() is discarded. Current tensors remain old caches.',
        parity='For every eligible current query, old primary peak center<.5px and score<.001. Same weights, input sequence, normalization and target coefficients. No tolerance weakening. Stored previous arrays finite and save/reload exact; all22090UIDs and previous region indices same D/C/R1. Replay from saved data before any future learner.',
        cost='OneGPUmutex;16episode inference packing,not learning batch. First2Dgroups:2*measured seconds per weighted-frame*remainingD+Cweightedframes+600 must fit remaining7200seconds. If not, preserve and stop. R1 remains deterministic retinal feature computation onCPU, not CPU learning. Independent controller hiddenIdle; no coretraining/dev/HOTA/heldout75/download.',
        next='On completion independently verify sources/UID bindings and exact replay. No automatic learning: learner specification, unchanged-readout regression, matched no-FlyVis control and measured GPU cost require a separate preregistered stage. Preparation does not claim an effective identity memory.',
        counterargument='Earlier same-class regions may be distractors or too stale, and FlyVis may not encode enough identity. Availability/oracle passed, but learned benefit remains unknown. Spatial-set module stays deferred; this is a distinct temporal input, not resuming its75models.')
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',updated=now().isoformat(),stage='Признаки прежних областей: подготовка',done=0,total=None))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Равные признаки прежних областей D/C/R1 для временного считывания',area='flyvis',state='running',started=now().strftime('%Y-%m-%d'),question='Можно ли получить равные причинные признаки прежних областей без изменения нынешнего ответа FlyVis?',hypothesis='Проверенная доступность прошлых наблюдений допускает отдельное извлечение признаков; способность различать личность пока не проверена.',expected=p['action'],acceptance=[dict(criterion=p['parity'],met=None),dict(criterion=p['cost'],met=None)],summary='Подготовка временных признаков; обучение ещё не запускалось.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],tags=['fly-problem-P8'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Правила причинных входов, воспроизведение и7200сбюджет доGPUизвлечения')]))
    write(ROOT/'experiments/project_history.json',cat)
    from scripts.z13_temporal_observation_gate_20261025 import journal
    journal('постановка отдельнойGPUподготовки временных признаков',p['action']+'\n\n*Критерии.* '+p['parity']+'\n\n*Бюджет до запуска.* '+p['cost']+' Сохранение '+p['save_at']+', остановка '+p['hard_stop']+'.\n\n*Ограничение.* '+p['counterargument']+' '+p['next'])


def worker():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_region_assignment import regions,pool_activity
    from tracking.z13_ridge_localizer import map_center
    from tracking.z13_readout import to_frame
    from tracking.gpu_lock import GpuLock
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(RUN/'protocol.json');verify_sources(p);ns=namespace();queries,_=ns['queries_and_crops']()
    os.environ['CUDA_VISIBLE_DEVICES']='0';assert torch.cuda.device_count()==1
    geom=ns['geometry']();refs={};entries={}
    for name,digest in read(GATE/'report.json')['manifest'].items():
        assert sha(ROOT/name)==digest
        rows=read(ROOT/name)['rows'];refs.update({x['uid']:x for x in rows if x['eligible']})
        v=rows[0]['video'];entries.update({int(u):r for u,r in read(ROOT/f'runs/z13_box_supervision_followup_20261025/cache/videos/{v}.json')['entries'].items()})
    sets={};weight=0;total=0
    for split,videos,key in [('fit20',p['fit20'],'fit'),('fit84',p['fit84'],'extra84'),('train26',p['evaluate26'],'evaluate')]:
        byep={}
        for v in videos:
            for q in queries[v]:
                if q['uid'] in refs:byep.setdefault(q['label']['episode'],[]).append(q)
        batches=[sorted(byep)[i:i+16] for i in range(0,len(byep),16)]
        weights=[5+max(q['label']['pos'] for e in batch for q in byep[e]) for batch in batches]
        sets[split]=(videos,key,byep,batches,weights);weight+=sum(weights);total+=len(batches)
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    manifest={};measured=[];completed=0;detcache={};began=time.monotonic();arms_uid={}
    def status(**kw):
        if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original temporal feature budget')
        write(RUN/'status.json',dict(state='running',updated=now().isoformat(),done=completed,total=total*3,**kw))
    def features(arm,bb,hx,needed):
        if arm=='R1':return old.episode_features(torch,arm,bb,hx,geom.neighbors,needed)
        result={}
        with torch.no_grad():
            st=bb.init(hx[:,0:1]);last=hx[:,0]
            if 0 in needed:result[0]=old.neighbor_aug(st.nodes.activity[:,bb.index].double().cpu().numpy(),geom.neighbors)
            for t in range(1,hx.shape[1]):
                st,f=bb.step(st,last,hx[:,t]);last=hx[:,t]
                if t in needed:result[t]=old.neighbor_aug(f.double().cpu().numpy(),geom.neighbors)
        return result
    with GpuLock('flyvis_fovea_gpu'):
        probe=torch.ones(1,device='cuda:0');assert float(probe.sum())==1.;del probe
        for arm in ('D','C','R1'):
            n=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json');mu,sd=np.asarray(n['mean']),np.asarray(n['sd'])
            if arm=='R1':bb=bank=None;dev='cpu';torch.set_default_device('cpu')
            else:bank,_,bb=old.build_arm(arm,torch,ck);dev=bb.dev;assert str(dev).startswith('cuda')
            seen=set()
            for split,(videos,key,byep,batches,weights) in sets.items():
                data=Episodes(videos);desc=read(ROOT/f'runs/z13_filter_descriptor_fixed_20261024/cache/{arm}/{split}/descriptors.json')
                base=ROOT/'runs/z13_candidate_data104_20261021/evaluate' if split=='fit84' else ROOT/'runs/z13_candidate_reserve_20261021'/key
                canonical={}
                for path in sorted((base/arm).glob('cache_*.npz')):
                    with np.load(path) as z:canonical.update({(int(e),int(t)):(c[0],float(s[0])) for e,t,c,s in zip(z['episode'],z['t'],z['centers'],z['scores'])})
                for bi,batch in enumerate(batches):
                    path=RUN/arm/split/f'chunk_{bi:05d}.npz';marker=path.with_suffix('.json')
                    if marker.exists():raise RuntimeError('No silent repeat/resume; use separately verified recovery')
                    status(stage='Замороженные признаки прежних областей',variant=arm+'/'+split)
                    before=time.monotonic();upto=[max(q['label']['pos'] for q in byep[e]) for e in batch];hx,_=data.batch(torch,dev,batch,upto);needed=set()
                    for e in batch:
                        for q in byep[e]:
                            ref=refs[q['uid']];rel=ref['previous']['frame_array_index']-(ref['current_array_index']-(4+q['label']['pos']))
                            needed.update((int(rel),4+q['label']['pos']))
                    F=features(arm,bb,hx,needed);records=[];dc=ds=0.
                    for j,e in enumerate(batch):
                        target=np.asarray(desc[str(e)],float)
                        for q in byep[e]:
                            u=q['uid'];ref=refs[u];previous=ref['previous'];rel=previous['frame_array_index']-(ref['current_array_index']-(4+q['label']['pos']))
                            x=(F[4+q['label']['pos']][j].T-mu)/sd;center,score=map_center(x@target,geom);oldcenter,oldscore=canonical[e,4+q['label']['pos']]
                            dc=max(dc,float(np.abs(np.asarray(to_frame(center,q['label']['crop'],geom.eye_size))-oldcenter).max()));ds=max(ds,abs(score-oldscore))
                            v=q['video']
                            if v not in detcache:
                                with np.load(ROOT/f'runs/train_cache/{v}/detections/raw_video.npz') as z:detcache[v]={k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
                            d=detcache[v];ids=np.flatnonzero((d['frame_idx']==previous['frame'])&(d['conf']>=.25));crop=np.asarray(previous['crop'],float)
                            choices,masks,_=regions(geom,crop,(crop[:2]+crop[2:])/2,d['xyxy'][ids],d['conf'][ids],d['cls'][ids],entries[u]['track_class'])
                            assert ids[np.asarray(choices[1:])].tolist()==previous['raw_detection_indices']
                            xp=(F[int(rel)][j].T-mu)/sd;means=pool_activity(xp,masks);geo=np.zeros((len(choices),6),np.float32)
                            extent=np.maximum(crop[2:]-crop[:2],1.)
                            for i,index in enumerate(choices[1:],1):
                                box=d['xyxy'][ids[index]];geo[i]=np.r_[((box[:2]+box[2:])/2-(crop[:2]+crop[2:])/2)/extent,np.log(np.maximum(box[2:]-box[:2],1.)/extent),d['conf'][ids[index]],masks[i].mean()]
                            records.append((u,means,geo,q['frame']-previous['frame'],previous['frame'],int(rel)));seen.add(u)
                    assert dc<.5 and ds<.001,(arm,split,bi,dc,ds)
                    width=max(len(r[1]) for r in records);count=len(records)
                    arrays=dict(uid=np.array([r[0] for r in records],np.int64),previous_x=np.zeros((count,width,68),np.float32),previous_geo=np.zeros((count,width,6),np.float32),previous_mask=np.zeros((count,width),bool),gap_frames=np.array([r[3] for r in records],np.int32),previous_frame=np.array([r[4] for r in records],np.int32),relative_step=np.array([r[5] for r in records],np.int32))
                    for i,(_,x,g,*_) in enumerate(records):w=len(x);arrays['previous_x'][i,:w]=x;arrays['previous_geo'][i,:w]=g;arrays['previous_mask'][i,:w]=True
                    assert all(np.isfinite(v).all() for v in arrays.values())
                    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
                    with np.load(path) as z:assert all(np.array_equal(z[k],v) for k,v in arrays.items())
                    write(marker,dict(sha256=sha(path),rows=count,seconds=time.monotonic()-before,max_center_px=dc,max_score=ds,previous_save_reload_exact=True,first_state_capture_rows=int((arrays['relative_step']==0).sum())))
                    manifest[marker.relative_to(ROOT).as_posix()]=sha(marker);completed+=1
                    if arm=='D' and len(measured)<2:
                        measured.append((time.monotonic()-before,weights[bi]))
                        if len(measured)==2:
                            estimate=2*sum(s for s,w in measured)/sum(w for s,w in measured)*(2*weight-sum(w for s,w in measured))+600
                            left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds();write(RUN/'price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left));assert estimate<left,'New-stage conservative resource gate; preserve completed'
                    del F,hx;gc.collect()
                del data;gc.collect()
            assert seen==set(refs);arms_uid[arm]=len(seen);del bb,bank;gc.collect();torch.cuda.empty_cache()
    verify_sources(p);write(RUN/'report.json',dict(complete=True,rows_each_arm=arms_uid,groups=completed,manifest=manifest,all_current_primary_peaks_in_original_tolerances=True,all_saved_arrays_exact=True,model_trained=False,hota_measured=False,seconds=time.monotonic()-began))
    write(RUN/'status.json',dict(state='completed',updated=now().isoformat(),done=completed,total=completed,stage='Признаки готовы; нужна независимая сверка'))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
