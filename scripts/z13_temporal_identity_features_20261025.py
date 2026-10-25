"""Matched frozen features on an observable reference clock, retaining original episode state."""
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter
import os,sys,time,gc,inspect,types,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_temporal_pair_readiness_20261025 import read,write,sha,now,journal as original_journal
BASE=ROOT/'runs/z13_temporal_identity_20261025'
RUN=BASE/'features'
REF=ROOT/'runs/z13_temporal_pair_readiness_20261025'
IDENT='z13_temporal_identity_features_v1'


def journal(title,text):
    from scripts import z13_temporal_pair_readiness_20261025 as module
    prior=module.IDENT
    try:
        module.IDENT=IDENT;module.journal(title,text)
    finally:
        module.IDENT=prior


def prepare():
    assert not RUN.exists()
    assert read(BASE/'cuda_probe/report.json')['gradient_and_host_device_transfer_verified']
    assert read(BASE/'cuda_probe/controller_result.json')['all_owned_ended']
    old=ROOT/'runs/z13_temporal_region_features_20261025'
    end=read(old/'controller_result.json');assert end['all_owned_ended'] and not end['remaining']
    saved={}
    for marker in old.glob('*/*/chunk_*.json'):
        record=read(marker);assert sha(marker.with_suffix('.npz'))==record['sha256']
        saved[marker.relative_to(ROOT).as_posix()]=sha(marker)
    write(BASE/'legacy_preservation.json',dict(at=now().isoformat(),saved_chunks=len(saved),manifest=saved,
        original_controller_result_sha256=sha(old/'controller_result.json'),original_error_preserved=True,
        reason='Original nearest-frame extraction stopped after confirmed CUDA transfer stall; new annotated-clock inputs differ, so these files are preserved, not overwritten or presented as new inputs.'))
    current=read(ROOT/'runs/z13_box_supervision_followup_20261025/protocol.json')
    files=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_readout.py','tracking/z13_region_assignment.py',
        'scripts/z13_target_model.py','scripts/z13_train.py','scripts/z13_region_assignment_recovery2_20261025.py',
        'runs/z13_temporal_pair_readiness_20261025/report.json','runs/z13_temporal_pair_readiness_20261025/label_provenance_and_cost_audit.json',
        'runs/z13_spatial_v2_20261013/D/full/model.pt','runs/z13_local_peak_20261021/geometry.json']
    files += [f'runs/z13_target_center_dev_20261018/{a}/normalization.json' for a in ('D','C','R1')]
    for split in ('fit20','fit84','train26'):
        for arm in ('D','C','R1'):
            files.append(f'runs/z13_filter_descriptor_fixed_20261024/cache/{arm}/{split}/descriptors.json')
    for video in current['fit20']+current['fit84']+current['evaluate26']:
        files += [f'runs/z13_data/{video}.npz',f'runs/train_cache/{video}/detections/raw_video.npz',
            f'runs/train_cache/{video}/alignment.json',f'runs/z13_temporal_pair_readiness_20261025/inputs/{video}.json',
            f'runs/z13_temporal_pair_readiness_20261025/supervision/{video}.json',
            f'runs/z13_box_supervision_followup_20261025/cache/videos/{video}.json']
    for folder in (ROOT/'runs/z13_candidate_reserve_20261021/fit',ROOT/'runs/z13_candidate_reserve_20261021/evaluate',ROOT/'runs/z13_candidate_data104_20261021/evaluate'):
        for arm in ('D','C','R1'):
            files += [f.relative_to(ROOT).as_posix() for f in (folder/arm).glob('cache_*.npz')]
    began=now();soft=min(began+timedelta(seconds=3600),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    p=dict(id=IDENT,created=began.isoformat(),device='cuda',timezone='Europe/Moscow',budget_seconds=3600,
        save_at=soft.isoformat(),hard_stop=(soft+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        splits={k:current[k] for k in ('fit20','fit84','evaluate26')},
        authorization='Author25.10: Давай, пробуем. Next agreed step: one reference rule, matched features and bounded readout experiment. General author window ends26.10 at07:40MSK.',
        reference_clock='Use ONLY alignment.matches[].video_frame_index as a predetermined reference-observation schedule, independent of object identity/annotations/presence. Choose latest supported stored frame on that clock strictly before current within original episode. This same clock/memory rule is used in training and application. No GT file is opened by this feature worker. Query-time absent history remains absent; all original queries will remain in learner denominators.',
        distinction='Real regional pairs, explicit offline identity supervision and an unmatched association option, unlike prior unsupervised three-peak attention. This stage extracts frozen inputs only. Prepared18824references must match observable schedule without reading annotation objects.',
        parity='Same original init, integration steps, entire episode prefix, normalization and target descriptor. For every restored query and all D/C/R1: old peak center max error<.5px,score<.001. Only cache invariant input_index on GPU once; compare cached and original step on first actual D/C batch. Current learned features remain immutable old arrays; no core training.',
        controls='Later equal readout variants D_temporal,D_blank,C_temporal,R1_temporal,geometry_temporal,D_without_identity_loss; three seeds, unchanged full-box actions/outcome labels. Current-only variant has same parameter allocation. Identity labels are offline loss targets only.',
        cost='OneGPUmutex. First2Dgroups:2*measuredseconds/weightedframes*remainingD+Cweightedframes+600 must fit remaining3600s. No budget extension on failure. Status and chunks saved after each group; independent controller preserves partials. No learning in this stage.',
        learner_plan='If features and unchanged-control regression pass: preregister90nested104video models (6variants*3seeds*5folds),1150steps,128batch,Adam.001, fixed signed outcome loss plus equally weighted known-pair BCE. D must have net>=.02 and positive video95%lower versus0,D_blank,C,R1,geometry,D_without_identity_loss. Internal failure stops before any new26evaluation. Probe cost before full training; no HOTA automatic.',
        literature='KeepTrack ICCV2021 https://arxiv.org/html/2103.16556v2 sections3.2,3.5,3.6 motivates learned candidate identities and unmatched alternatives; our internal regional readout/loss is an adaptation, not full KeepTrack.',
        cannot_claim='Data readiness does not show identity information/learnability or HOTA improvement. Schedule is a fixed sampling policy, not selection by known true identity. Reused104/26 exploratory video partitions remain non-independent.',
        sha256={n:sha(ROOT/n) for n in files})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',done=0,total=None,updated=now().isoformat()))
    catalog=read(ROOT/'experiments/project_history.json')
    oldcard=next(c for c in catalog['experiments'] if c['id']=='z13_temporal_region_features_v1')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for field,value in dict(summary='Сохранены434блока; исходная подготовка остановлена после зависания передачиCUDA. Научного результата нет.',
        obtained='runs/z13_temporal_identity_20261025/legacy_preservation.json:434готовыхблока с провереннымихешами; runs/z13_temporal_region_features_20261025/stalled_transfer_audit_20261025.json:двастекаCUDAH2D,прогресс434с22:40;controller_result.json exit4294967295/all_owned_ended после остановки проверенного дерева.',
        interpretation='Подтверждена техническая остановка, причина драйвера не установлена. ОставшиесяC/R1 не рассчитаны; метрики эффективности не получены.',
        decision='Готовые434блока сохранить. Новая выборка прошлых опорных наблюдений имеет отдельный протокол и бюджет; прежний срок не продлевается и старые источники не переписываются.').items():
        if oldcard.get(field):oldcard.setdefault('revisions',[]).append(dict(date=began.strftime('%Y-%m-%d'),commit=head,field=field,old=oldcard[field],reason='Verified technical CUDA transfer stall; preserve partial work and original error.'))
        oldcard[field]=value
    oldcard.update(state='inconclusive',done=began.strftime('%Y-%m-%d'))
    assert not any(c['id']==IDENT for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=IDENT,title='Равные временные признаки FlyVis на наблюдаемой опорной сетке',area='flyvis',state='ready',started=began.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Воспроизводятся ли исходные ответы при извлечении прошлых областей на одинаковой для обучения и применения сетке?',
        hypothesis='Подготовленные прошлые опорные наблюдения допускают равные признакиD/C/R1; личность и польза ещё не установлены.',
        expected=p['reference_clock'],acceptance=[dict(criterion=p['parity'],met=None),dict(criterion=p['cost'],met=None)],
        obtained='',interpretation='',decision='',cannot_claim=p['cannot_claim'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Единое наблюдаемое правило, бюджет и проверки до извлечения')]))
    write(ROOT/'experiments/project_history.json',catalog)
    journal('единое правило прошлого и новая проверка равных признаков','*Постановка.* '+p['reference_clock']+'\n\n*Проверки и срок.* '+p['parity']+' '+p['cost']+' Сохранение '+p['save_at']+',остановка '+p['hard_stop']+'.\n\n*Старый сбой.* '+oldcard['obtained']+'\n\n*План после допуска.* '+p['learner_plan']+'\n\n*Источник.* '+p['literature'])


def cached_step(self,state,last_hex,cur_hex):
    t=self.torch
    alpha=t.arange(1,self.steps+1,device=self.dev,dtype=t.float32).view(1,self.steps,1,1)/self.steps
    stim=last_hex.reshape(-1,1,1,last_hex.shape[-1])*(1-alpha)+cur_hex.reshape(-1,1,1,cur_hex.shape[-1])*alpha
    st=self.net.stimulus;idx=self._temporal_input_index
    buf=t.zeros((stim.shape[0],stim.shape[1],st.n_nodes),device=self.dev,dtype=t.float32)
    buf[:,:,idx]=stim.expand(-1,-1,idx.shape[0],-1)
    states=self.net.forward(buf,self.dt,state,as_states=True)
    return states[-1],states[-1].nodes.activity[:,self.index]


def worker():
    import torch
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_region_assignment import regions,pool_activity
    from tracking.z13_ridge_localizer import map_center
    from tracking.z13_readout import to_frame
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=read(RUN/'protocol.json');ns=namespace();queries,_=ns['queries_and_crops']();geom=ns['geometry']()
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    refs={};entries={};sets={};weighted=0;groups=0
    for split,videos,key in [('fit20',p['splits']['fit20'],'fit'),('fit84',p['splits']['fit84'],'extra84'),('train26',p['splits']['evaluate26'],'evaluate')]:
        byep={}
        for v in videos:
            schedule={int(r['video_frame_index']) for r in read(ROOT/f'runs/train_cache/{v}/alignment.json')['matches']}
            rows=read(REF/'inputs'/(v+'.json'))['rows']
            for r in rows:
                if r['previous']:
                    assert r['previous']['frame'] in schedule and r['previous']['frame']<r['current_frame']
                    refs[r['uid']]=r
            entries.update({int(u):e for u,e in read(ROOT/f'runs/z13_box_supervision_followup_20261025/cache/videos/{v}.json')['entries'].items()})
            for q in queries[v]:
                if q['uid'] in refs:byep.setdefault(q['label']['episode'],[]).append(q)
        batches=[sorted(byep)[i:i+16] for i in range(0,len(byep),16)]
        weights=[5+max(q['label']['pos'] for e in b for q in byep[e]) for b in batches]
        sets[split]=(videos,key,byep,batches,weights);weighted+=sum(weights);groups+=len(batches)
    assert len(refs)==18824
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    completed=0;manifest={};measured=[];before_all=time.monotonic();detcache={};parity={};uids={}
    def status(arm,split):
        if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original bounded feature deadline')
        write(RUN/'status.json',dict(state='running',stage='Равные прошлые области на опорной сетке',variant=arm+'/'+split,done=completed,total=3*groups,elapsed_seconds=time.monotonic()-before_all,updated=now().isoformat()))
    def features(arm,bb,hx,needed):
        result={}
        if arm=='R1':
            h=hx.cpu().numpy().astype(np.float64);ema=np.repeat(h[:,:1],len(old.R1_TAUS),1).astype(np.float64);a=(1./old.R1_TAUS)[None,:,None]
            for t in range(h.shape[1]):
                value=h[:,t][:,None];ema+=(value-ema)*a
                if t in needed:result[t]=old.neighbor_aug(np.concatenate([ema.copy(),value-ema],1),geom.neighbors)
        else:
            with torch.no_grad():
                state=bb.init(hx[:,0:1]);last=hx[:,0]
                if 0 in needed:result[0]=old.neighbor_aug(state.nodes.activity[:,bb.index].double().cpu().numpy(),geom.neighbors)
                for t in range(1,hx.shape[1]):
                    state,f=bb.step(state,last,hx[:,t]);last=hx[:,t]
                    if t in needed:result[t]=old.neighbor_aug(f.double().cpu().numpy(),geom.neighbors)
        return result
    with GpuLock('flyvis_fovea_gpu'):
        assert float(torch.ones(1,device='cuda:0').sum())==1.
        for arm in ('D','C','R1'):
            normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json');mu,sd=np.asarray(normal['mean']),np.asarray(normal['sd'])
            if arm=='R1':bank=bb=None;dev='cpu'
            else:
                bank,_,bb=old.build_arm(arm,torch,ck);dev=bb.dev
                bb._temporal_input_index=torch.as_tensor(np.asarray(bb.net.stimulus.input_index),device=dev)
                old_step=bb.step;bb.step=types.MethodType(cached_step,bb)
            seen=set()
            for split,(videos,key,byep,batches,weights) in sets.items():
                data=Episodes(videos);desc=read(ROOT/f'runs/z13_filter_descriptor_fixed_20261024/cache/{arm}/{split}/descriptors.json')
                cache=ROOT/'runs/z13_candidate_data104_20261021/evaluate' if split=='fit84' else ROOT/'runs/z13_candidate_reserve_20261021'/key
                canonical={}
                for f in sorted((cache/arm).glob('cache_*.npz')):
                    with np.load(f) as z:canonical.update({(int(e),int(t)):(c[0],float(s[0])) for e,t,c,s in zip(z['episode'],z['t'],z['centers'],z['scores'])})
                for bi,batch in enumerate(batches):
                    status(arm,split);dest=RUN/arm/split/f'chunk_{bi:05d}.npz';assert not dest.exists()
                    before=time.monotonic();upto=[max(q['label']['pos'] for q in byep[e]) for e in batch];hx,_=data.batch(torch,dev,batch,upto)
                    if arm in ('D','C') and arm not in parity:
                        with torch.no_grad():
                            initial=bb.init(hx[:,0:1]);a,fa=old_step(initial,hx[:,0],hx[:,1]);b,fb=bb.step(initial,hx[:,0],hx[:,1]);delta=float((fa-fb).abs().max());assert delta<1e-6
                        parity[arm]=dict(cached_input_index_max_difference=delta,original_step_checked=True)
                        write(RUN/'cached_index_parity.json',parity)
                    needed=set()
                    for e in batch:
                        for q in byep[e]:
                            r=refs[q['uid']];needed.update((r['previous']['array_index']-r['state_prefix_start'],4+q['label']['pos']))
                    F=features(arm,bb,hx,needed);records=[];dc=ds=0.
                    for j,e in enumerate(batch):
                        target=np.asarray(desc[str(e)],float)
                        for q in byep[e]:
                            uid=q['uid'];r=refs[uid];pr=r['previous'];rel=pr['array_index']-r['state_prefix_start']
                            x=(F[4+q['label']['pos']][j].T-mu)/sd;center,score=map_center(x@target,geom);oldcenter,oldscore=canonical[e,4+q['label']['pos']]
                            dc=max(dc,float(np.abs(np.asarray(to_frame(center,q['label']['crop'],geom.eye_size))-oldcenter).max()));ds=max(ds,abs(score-oldscore))
                            v=q['video']
                            if v not in detcache:
                                with np.load(ROOT/f'runs/train_cache/{v}/detections/raw_video.npz') as z:detcache[v]={k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
                            d=detcache[v];ids=np.flatnonzero((d['frame_idx']==pr['frame'])&(d['conf']>=.25));crop=np.asarray(pr['crop'],float)
                            choices,masks,_=regions(geom,crop,(crop[:2]+crop[2:])/2,d['xyxy'][ids],d['conf'][ids],d['cls'][ids],entries[uid]['track_class'])
                            assert choices[1:]==pr['choices'] and all(np.array_equal(np.flatnonzero(m),z) for m,z in zip(masks[1:],pr['masks']))
                            means=pool_activity((F[rel][j].T-mu)/sd,masks)
                            records.append((uid,means,q['frame']-pr['frame'],pr['frame']));seen.add(uid)
                    assert dc<.5 and ds<.001,(arm,split,bi,dc,ds)
                    n=len(records);w=max(len(r[1]) for r in records)
                    arrays=dict(uid=np.asarray([r[0] for r in records],np.int64),previous_x=np.zeros((n,w,68),np.float32),previous_mask=np.zeros((n,w),bool),gap_frames=np.asarray([r[2] for r in records],np.int32),previous_frame=np.asarray([r[3] for r in records],np.int32))
                    for i,r in enumerate(records):k=len(r[1]);arrays['previous_x'][i,:k]=r[1];arrays['previous_mask'][i,:k]=True
                    assert all(np.isfinite(z).all() for z in arrays.values());dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
                    with np.load(dest) as z:assert all(np.array_equal(z[k],v) for k,v in arrays.items())
                    write(dest.with_suffix('.json'),dict(rows=n,sha256=sha(dest),max_center_px=dc,max_score=ds,seconds=time.monotonic()-before,all_prior_masks_exact=True,save_reload_exact=True))
                    manifest[dest.with_suffix('.json').relative_to(ROOT).as_posix()]=sha(dest.with_suffix('.json'));completed+=1
                    if arm=='D' and len(measured)<2:
                        measured.append((time.monotonic()-before,weights[bi]))
                        if len(measured)==2:
                            estimate=2*sum(s for s,w in measured)/sum(w for s,w in measured)*(2*weighted-sum(w for s,w in measured))+600
                            left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds();write(RUN/'price.json',dict(estimate=estimate,remaining=left,proceed=bool(estimate<left)));assert estimate<left,'Feature cost exceeds unchanged3600second budget'
                    del hx,F;gc.collect()
                del data;gc.collect()
            assert seen==set(refs);uids[arm]=len(seen);del bb,bank;gc.collect();torch.cuda.empty_cache()
    for name,digest in p['sha256'].items():assert sha(ROOT/name)==digest,name
    write(RUN/'report.json',dict(complete=True,rows_each_arm=uids,groups=completed,manifest=manifest,all_current_peaks_in_original_tolerances=True,all_previous_regions_and_clock_exact=True,cached_input_index_parity=parity,models_trained=0,seconds=time.monotonic()-before_all))
    write(RUN/'status.json',dict(state='completed',done=completed,total=completed,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'failure.json',dict(at=now().isoformat(),error=repr(e)));raise
