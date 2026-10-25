"""Bounded local queue: exact labels -> frozen activity -> matched CPU readouts.

Only the per-target readout changes. No core training or tracker replacement.
"""
from pathlib import Path
from datetime import datetime,timedelta
import os
import sys
import time
import inspect
import hashlib
import subprocess
import gc
PHASE=os.environ.get('Z13_REGION_PHASE','')
if PHASE!='features':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_region_assignment_20261025'
GATE=ROOT/'runs/z13_region_choice_gate_20261025/recovery1'
OLD=ROOT/'runs/z13_assignment_effect_20261024/recovery1'
IDENT='z13_region_assignment_v1'
ARMS=('D','C','R1')
BUDGETS={'labels':900,'features':5400,'training':1800}


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as m
    old=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=old


def status(phase,stage,**kw):
    folder=RUN/phase;p=read(folder/'protocol.json')
    if (folder/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Stage budget: '+phase)
    write(folder/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))


def geometry():
    import numpy as np
    from tracking.z13_readout import HexGeometry
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    return HexGeometry(np.asarray(g['points'],np.float32),g['eye_size'])


def queries_and_crops():
    import numpy as np
    from scripts.z13_assignment_effect_20261024 import load_queries
    base=read(OLD/'protocol.json');queries,centers=load_queries(base)
    for split,videos in [('fit20',base['fit20']),('extra84',base['extra84']),('evaluate26',base['evaluate26'])]:
        offset=0
        for v in videos:
            with np.load(ROOT/f'runs/z13_data/{v}.npz') as d:off,crop,frames=d['ep_offset'],d['fr_crop'],d['fr_frame']
            for q in queries[v]:
                assert q['split']==split
                l=q['label'];e=l['episode']-offset;k=int(off[e])+4+l['pos']
                assert 0<=e<len(off)-1 and k<off[e+1] and q['frame']==int(frames[k])
                l['crop']=crop[k].tolist();q['episode_local']=e
            offset+=len(off)-1
    return queries,centers


def prepare():
    assert not RUN.exists()
    assert read(GATE/'publication.json')['push_completed'] and read(GATE/'report.json')['primary_pass']
    old=read(OLD/'protocol.json');base=read(GATE/'protocol.json')
    names=['tracking/z13_region_assignment.py','tests/test_z13_region_assignment.py','scripts/z13_region_assignment_20261025.py','scripts/z13_region_assignment_console.ps1',
        'tracking/z13_filter_descriptor.py','tracking/z13_competing_candidates.py','tracking/z13_region_choice.py','scripts/z13_region_choice_gate_20261025.py',
        'scripts/z13_template_swap_20261025.py','scripts/z13_target_model.py','tracking/gpu_lock.py',
        (GATE/'report.json').relative_to(ROOT).as_posix(),(GATE/'independent_audit.json').relative_to(ROOT).as_posix(),(GATE/'protocol.json').relative_to(ROOT).as_posix()]
    for a in ARMS:
        names += [f'runs/z13_filter_descriptor_fixed_20261024/cache/{a}/{s}/descriptors.json' for s in ('fit20','fit84','train26')]
        names += [f'runs/z13_target_center_dev_20261018/{a}/normalization.json']
    names += [(GATE/'videos'/(v+'.json')).relative_to(ROOT).as_posix() for v in old['evaluate26']]
    auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json')
    deadline=min(now()+timedelta(seconds=sum(BUDGETS.values())+120),datetime.fromisoformat(auth['save_at']))
    p=dict(id=IDENT,created=now().isoformat(),timezone='Europe/Moscow',authorization=auth['author_request'],
        pipeline_save_at=deadline.isoformat(),pipeline_hard_stop=min(deadline+timedelta(seconds=20),datetime.fromisoformat(auth['hard_stop'])).isoformat(),stage_budgets=BUDGETS,
        fit20=old['fit20'],fit84=old['extra84'],evaluate26=old['evaluate26'],seeds=[0,1,2],steps=1150,batch=128,
        action='Replace the internal FlyVis target readout: learned joint categorical region choice and explicit unmatched action, using frozen target-model coefficients and raw retinal activity pooled at real high-score same-class detection boxes. Preserve the surrounding detector, tracker and MASA. No learned core parameters in this trial.',
        target='The categorical-action oracle passed:6.358pp mean-video beneficial reserve,4.725pp more than the Dthreepeak center action. This is an action-space ceiling, not a fly result. The prior presence-only repair failed. This trial learns an actual choice, not another fixed response statistic.',
        controls='Identical action sets, labels, architecture, seeds, minibatch UIDs and optimizer forD,C,R1,geometry. Geometry uses the same D tensors but zeros both activity and target-model coefficients before embedding. C is the standard fly; D remains the retained trained core. R1 is the rich retinal-input control. New full readout vs no action, thenD vs each matched control.',
        features='Each actual region: mean68 normalized query activity in its visible rectangle. Explicit unmatched region: mean68 over the whole eye. Target input: unchanged observed-only68 ridge coefficients. Eight observable geometry values: normalized center displacementxy, log region width/height relative to crop, detector score, displacement norm, visible-column fraction, unmatched flag. No true identity, target box or query kind enters the network. Same sorted8-episode groups as existing caches; every original peak center<0.5px and height<0.001 before saving features.',
        training='Reuse the already-tested dual-embedding architecture in tracking/z13_filter_descriptor.py. Shared candidate competition and fixed unchanged logit0. Marginal cross-entropy target is all beneficial actual actions from exact one-step assignments, otherwise unchanged; positive_weight1, Adam lr0.001,1150steps batch128,three seeds0/1/2. Fit only84videos;20fit videos only for threshold calibration; train26 only for outcome comparison. All known kinds0/1 retained; unknown kind-1 excluded from gradients and reported separately.',
        policy='For each candidate group, compare action logits with fixed unchanged logit0. Select max action only if its logit>0 and its joint softmax probability>=tau. Tau from fixed[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.01] chosen on20fit mean-video(beneficial-harmful)/allknown; ties prefer fewer interventions then larger tau. 1.01 means no action. Freeze tau before evaluation26. No test coverage or threshold tuning.',
        primary='Average per-video net assignment effect across all3seeds, then paired-video95%CI. D-baseline mean>=0.02 and lower>0; D-C,D-R1,D-geometry all lower>0. Report each seed and all counts, unknowns and chosen thresholds. Passing is a gate for separately budgeted closed-loop HOTA, never automatic HOTA or final novelty.',
        technical='All130 natural-video labels: reuse completed26 gate rows exactly, replay remaining104 with float32-exact unchanged outputs. Preserve all query UID/video/frame/track/label bindings. Same supported real choices from the pure observable API. Each D/C/R1 cache reproduces the saved original centers/scores and identical ROI masks. Every phase has its own independent controller and price gate; no completed arm/chunk/model repeated. No core training/dev/HOTA/heldout75/download.',
        cost='labels900s;features5400s;training1800s, each+20s within global window. CPU first2fit replays:2*mean*102+60 must fit remaining labels budget. GPU first2D groups:2*seconds/episode-time*remaining D+C episode-time+300 must fit remaining feature budget. CPU first completed model:2*elapsed*11+60 must fit remaining training budget. Failed price gate preserves completed work and stops later phases.',
        counterargument='The oracle used truth and is equally available to non-fly representations. Low-resolution activity may not discriminate real instances; geometry may explain all utility. Some targets have no high-score detection, so this action cannot help them. One-step training omits feedback and later stages. The protocol changes the complete readout output/learning target, not an isolated causal test of a single scalar. StandardC vsD core histories differ by design. Sparse labels and repeatedly reused train26 limit generalization.',
        distinction='Unlike old three-peak rankers/filter descriptors, actions are real detector regions with actual assignment-effect labels, whose separate oracle now passed. Unlike fixed region_mean pooling, raw68 activity and observed coefficients are learned jointly with candidate context and unmatched output. Original head architecture is reused to avoid attributing differences to an untested larger network.',
        literature='SiamMOT https://arxiv.org/html/2105.11595v1 sections3,5 motivates region-conditioned matching; KeepTrack https://arxiv.org/html/2103.16556v2 motivates candidate competition and explicit unmatched. Neither supplies our action loss or expected gain.',
        next='Complete criteria pass -> separate matched closed-loop test with frozen thresholds and equal MASA/raw inputs. Fail -> preserve all controls and negative seed results, do not run expensive HOTA for this version. Technical failure -> preserve sources/logs, repair only verified service issue within original phase/global budgets.',
        sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',phase='labels',stage='Три этапа внутреннего регионального считывателя',updated=now().isoformat()))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: обучаемое считывание реальных областей по последствиям сопоставления',area='flyvis',state='ready',started=now().strftime('%Y-%m-%d'),tags=['fly-problem-P8'],question='Использует ли новый внутренний региональный узел сигнал D для полезного назначения лучше C, входа глаза и геометрии?',hypothesis='Нужен выбор реальной области с учётом соседних кандидатов и возможности не вмешаться, а не точность одного центра или бинарное присутствие.',expected=p['action']+' '+p['controls'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],summary='Три контролируемых этапа зарегистрированы до обучения.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],metrics=[],commits=[],files=names[:4],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Предварительные критерии, равные контроли и бюджеты'),dict(kind='file',ref=(GATE/'report.json').relative_to(ROOT).as_posix(),claim='Пройденная мишень действия до обучения')]))
    write(cp,cat)
    journal('постановка обучаемого регионального считывателя','*Действие.* '+p['action']+'\n\n*Критерии.* '+p['primary']+'\n\n*Равные контроли.* '+p['controls']+'\n\n*Отличие.* '+p['distinction']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Бюджет.* '+p['cost']+'\n\nИсточник: runs/z13_region_assignment_20261025/protocol.json. '+p['literature'])


def label_worker():
    import numpy as np
    from scripts import z13_assignment_effect_20261024 as original
    from scripts.z13_region_choice_gate_20261025 import replay_source
    from tracking.z13_region_assignment import regions
    from tracking.z13_region_choice import categorical_rows
    folder=RUN/'labels';p=read(folder/'protocol.json');verify_sources(p)
    queries,centers=queries_and_crops();geom=geometry();metadata={}
    def new_cost_rows(q,old,detections,track_class):
        l=q['label'];boxes=np.asarray([d.xyxy for d in detections],float).reshape(-1,4)
        confidence=np.asarray([d.score for d in detections]);classes=np.asarray([int(d.cls) for d in detections])
        choices,_,_=regions(geom,l['crop'],l['pred'],boxes,confidence,classes,track_class)
        valid=np.zeros(len(boxes),bool);valid[[x for x in choices if x>=0]]=True
        costs,indices=categorical_rows(old,confidence,valid);assert indices[1:]==choices
        metadata[q['uid']]=dict(choices=indices,supported=int(valid.sum()),high_detections=len(detections),track_class=int(track_class))
        return {'regions':costs}
    ns={**vars(original),'RUN':folder,'new_cost_rows':new_cost_rows,'T0':time.monotonic()};exec(compile(replay_source(),str(Path(__file__)),'exec'),ns)
    times=[];total=0;known=0
    allvideos=p['fit20']+p['fit84']+p['evaluate26']
    for i,v in enumerate(allvideos):
        out=folder/'videos'/(v+'.json')
        if out.exists():
            record=read(out);assert sha(out)==read(folder/'completed'/f'{v}.json')['sha256']
        elif v in p['evaluate26']:
            source=GATE/'videos'/(v+'.json');r=read(GATE/'report.json');assert sha(source)==r['video_sha256'][v]
            record=read(source)
            write(out,record);write(folder/'completed'/f'{v}.json',dict(sha256=sha(out),reused=source.relative_to(ROOT).as_posix()))
        else:
            before=time.monotonic();rows,checks=ns['replay_video'](v,queries[v],centers,p)
            for row in rows:row['regions']=metadata[row['uid']]
            record=dict(rows=rows,checks=checks,seconds=time.monotonic()-before)
            write(out,record);write(folder/'completed'/f'{v}.json',dict(sha256=sha(out),reused=False));times.append(record['seconds'])
            if len(times)==2:
                left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds();estimate=2*float(np.mean(times))*102+60
                write(folder/'price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left));assert estimate<left,'Labels price gate: preserve2 completed videos'
        assert record['checks']['raw_replay_exact']
        expected={q['uid']:q for q in queries[v]};assert len(expected)==len(record['rows'])
        for row in record['rows']:
            q=expected[row['uid']];assert all(row[k]==q[k] for k in ('video','frame','tid','split')) and row['label']==q['label']['label']
            total+=1;known+=int(row['kind']>=0)
        status('labels','Метки прямых действий; готовые26видео используются без повтора',done=i+1,total=130)
    assert total==37296 and known==26600,(total,known)
    write(folder/'report.json',dict(complete=True,rows=total,known=known,unknown=total-known,all_raw_replays_exact=True,reused_videos=26,new_videos=104,video_sha256={v:sha(folder/'videos'/(v+'.json')) for v in allvideos}))


def feature_worker():
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_region_assignment import regions,pool_activity
    from tracking.z13_ridge_localizer import map_center
    from tracking.z13_readout import to_frame
    from tracking.gpu_lock import GpuLock
    folder=RUN/'features';p=read(folder/'protocol.json');verify_sources(p)
    queries,_=queries_and_crops();byuid={q['uid']:q for qq in queries.values() for q in qq};geom=geometry();labelrows={}
    for v,h in read(RUN/'labels/report.json')['video_sha256'].items():
        path=RUN/'labels/videos'/(v+'.json');assert sha(path)==h
        labelrows.update({r['uid']:r for r in read(path)['rows']})
    splitinfo=[('fit20',p['fit20'],'fit'),('fit84',p['fit84'],'extra84'),('train26',p['evaluate26'],'evaluate')]
    sets={};reference={};detcache={};weight_total=0
    for split,videos,_ in splitinfo:
        byep={}
        for v in videos:
            for q in queries[v]:byep.setdefault(q['label']['episode'],[]).append(q)
        batches=[sorted(byep)[i:i+8] for i in range(0,len(byep),8)]
        weights=[5+max(q['label']['pos'] for e in b for q in byep[e]) for b in batches]
        sets[split]=(byep,batches,weights);weight_total+=sum(weights)
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    completed=0;measured=[];manifest={};maxdc=maxds=0.
    with GpuLock('flyvis_fovea_gpu'):
        for arm in ARMS:
            normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json');mu,sd=np.asarray(normal['mean']),np.asarray(normal['sd'])
            if arm=='R1':bank=bb=None;dev='cpu';torch.set_default_device('cpu')
            else:bank,_,bb=old.build_arm(arm,torch,ck);dev=bb.dev;assert str(dev).startswith('cuda')
            for split,videos,cachekey in splitinfo:
                data=Episodes(videos);byep,batches,weights=sets[split]
                descriptor=read(ROOT/f'runs/z13_filter_descriptor_fixed_20261024/cache/{arm}/{split}/descriptors.json')
                cachebase=ROOT/'runs/z13_candidate_reserve_20261021'/cachekey if split!='fit84' else ROOT/'runs/z13_candidate_data104_20261021/evaluate'
                canonical={}
                for path in sorted((cachebase/arm).glob('cache_*.npz')):
                    with np.load(path) as z:
                        for e,t,c,s in zip(z['episode'],z['t'],z['centers'],z['scores']):canonical[int(e),int(t)]=(c[0],float(s[0]))
                for bi,batch in enumerate(batches):
                    path=folder/arm/split/f'chunk_{bi:05d}.npz';marker=path.with_suffix('.json')
                    if marker.exists():assert sha(path)==read(marker)['sha256'];manifest[marker.relative_to(ROOT).as_posix()]=sha(marker);completed+=1;continue
                    status('features','Извлечение сырой активности в реальных областях',variant=arm+'/'+split,done=completed,total=3*sum(len(s[1]) for s in sets.values()))
                    before=time.monotonic();upto=[max(q['label']['pos'] for q in byep[e]) for e in batch]
                    hx,_=data.batch(torch,dev,batch,upto);need={4+q['label']['pos'] for e in batch for q in byep[e]}
                    F=old.episode_features(torch,arm,bb,hx,geom.neighbors,need);records=[];dc=ds=0.
                    for j,e in enumerate(batch):
                        target=np.asarray(descriptor[str(e)],float);assert target.shape==(68,)
                        for q in byep[e]:
                            l=q['label'];x=(F[4+l['pos']][j].T-mu)/sd;crop=l['crop']
                            c,height=map_center(x@target,geom);point=np.asarray(to_frame(c,crop,geom.eye_size));ref,score=canonical[e,4+l['pos']]
                            dc=max(dc,float(np.abs(point-ref).max()));ds=max(ds,abs(height-score))
                            v=q['video']
                            if v not in detcache:
                                with np.load(ROOT/f'runs/train_cache/{v}/detections/raw_video.npz') as z:detcache[v]={k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
                            d=detcache[v];ix=(d['frame_idx']==q['frame'])&(d['conf']>=.25)
                            row=labelrows[q['uid']];choices,masks,geo=regions(geom,crop,l['pred'],d['xyxy'][ix],d['conf'][ix],d['cls'][ix],row['regions']['track_class'])
                            assert choices==row['regions']['choices'][1:] and len(choices)==len(row['arms']['regions'])-1
                            records.append((q,row,pool_activity(x,masks),geo,target))
                    assert dc<.5 and ds<.001,(arm,split,bi,dc,ds)
                    maxdc=max(maxdc,dc);maxds=max(maxds,ds);n=len(records);width=max(len(r[2]) for r in records)
                    arrays=dict(x=np.zeros((n,width,68),np.float32),geo=np.zeros((n,width,8),np.float32),seed=np.zeros((n,68),np.float32),mask=np.zeros((n,width),bool),good=np.zeros((n,width),bool),harm=np.zeros((n,width),bool),uid=np.zeros(n,np.int64),kind=np.zeros(n,np.int8),video=np.zeros(n,np.int16))
                    for ri,(q,row,x,geo,target) in enumerate(records):
                        w=len(x);arrays['x'][ri,:w]=x;arrays['geo'][ri,:w]=geo;arrays['seed'][ri]=target;arrays['mask'][ri,:w]=True
                        arrays['good'][ri,:w]=[bool(o['beneficial']) for o in row['arms']['regions'][1:]];arrays['harm'][ri,:w]=[bool(o['harmful']) for o in row['arms']['regions'][1:]]
                        arrays['uid'][ri]=q['uid'];arrays['kind'][ri]=row['kind'];arrays['video'][ri]=videos.index(q['video'])
                    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
                    write(marker,dict(sha256=sha(path),rows=n,max_center_px=dc,max_score=ds,seconds=time.monotonic()-before,query_input_unchanged=True))
                    manifest[marker.relative_to(ROOT).as_posix()]=sha(marker);completed+=1
                    if arm=='D' and len(measured)<2:
                        measured.append((time.monotonic()-before,weights[bi]))
                        if len(measured)==2:
                            rate=sum(s for s,w in measured)/sum(w for s,w in measured);estimate=2*rate*(2*weight_total-sum(w for s,w in measured))+300
                            left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds();write(folder/'price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left));assert estimate<left,'Features price gate: preserve completed chunks'
                    del F,hx
                del data;gc.collect()
            del bb,bank;gc.collect();torch.cuda.empty_cache()
    write(folder/'report.json',dict(complete=True,manifest=manifest,all_masks_and_choices_equal=True,max_center_px=maxdc,max_score=maxds,groups=completed,arms=list(ARMS)))


def load_split(torch,arm,split):
    import numpy as np
    markers=sorted((RUN/'features'/arm/split).glob('chunk_*.json'));parts=[];width=0
    for path in markers:
        m=read(path);assert sha(path.with_suffix('.npz'))==m['sha256']
        with np.load(path.with_suffix('.npz')) as z:parts.append({k:z[k] for k in z.files});width=max(width,parts[-1]['x'].shape[1])
    assert parts
    arrays={}
    for key in parts[0]:
        vals=[]
        for a in parts:
            v=a[key]
            if key in ('x','geo','mask','good','harm'):
                pad=[(0,0),(0,width-v.shape[1])]+([(0,0)] if v.ndim==3 else [])
                v=np.pad(v,pad)
            vals.append(v)
        arrays[key]=np.concatenate(vals)
    order=np.argsort(arrays['uid']);arrays={k:v[order] for k,v in arrays.items()}
    assert len(set(arrays['uid']))==len(arrays['uid'])
    return {k:torch.from_numpy(v).to('cpu') for k,v in arrays.items()}


def predict(torch,model,data):
    import numpy as np
    out=[];model.eval()
    with torch.no_grad():
        for left in range(0,len(data['uid']),256):
            ss=slice(left,left+256);z=model(data['x'][ss],data['seed'][ss],data['geo'][ss],data['mask'][ss]).numpy()
            out.extend(z[i,:int(data['mask'][left+i].sum())].copy() for i in range(len(z)))
    return out


def policy_rows(data,logits,threshold,videos):
    import numpy as np
    from tracking.z13_region_assignment import apply_policy
    rows=[]
    for i,z in enumerate(logits):
        choice,prob=apply_policy(z,threshold);kind=int(data['kind'][i]);known=kind>=0
        benefit=choice>=0 and known and bool(data['good'][i,choice]);harm=choice>=0 and known and bool(data['harm'][i,choice])
        rows.append(dict(uid=int(data['uid'][i]),video=videos[int(data['video'][i])],kind=kind,action=choice,probability=prob,beneficial=benefit,harmful=harm))
    per={v:sum(int(r['beneficial'])-int(r['harmful']) for r in rows if r['video']==v)/sum(r['video']==v and r['kind']>=0 for r in rows) for v in videos}
    count=dict(rows=len(rows),selected=sum(r['action']>=0 for r in rows),unknown_selected=sum(r['action']>=0 and r['kind']<0 for r in rows),beneficial=sum(r['beneficial'] for r in rows),harmful=sum(r['harmful'] for r in rows))
    return rows,per,count


def train_worker():
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu');assert not torch.cuda.is_initialized()
    from tracking.z13_filter_descriptor import make_ranker
    from tracking.z13_competing_candidates import marginal_choice_loss
    from scripts.z13_continue import paired
    folder=RUN/'training';p=read(folder/'protocol.json');verify_sources(p);began=time.monotonic();models=0;results={};bindings=None
    for variant in ('D','C','R1','geometry'):
        arm='D' if variant=='geometry' else variant
        data={s:load_split(torch,arm,s) for s in ('fit84','fit20','train26')}
        uidsets={s:data[s]['uid'].tolist() for s in data}
        if bindings is None:bindings=uidsets
        else:assert bindings==uidsets,'Controls must have identical query UIDs'
        known=np.flatnonzero(data['fit84']['kind'].numpy()>=0);assert len(known)>128
        for seed in p['seeds']:
            dest=folder/variant/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'done.json').exists():
                done=read(dest/'done.json');assert sha(dest/'model.pt')==done['model_sha256'];results[variant,seed]=read(dest/'report.json');models+=1;continue
            before=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=make_ranker(torch,activity=variant!='geometry');opt=torch.optim.Adam(model.parameters(),lr=.001)
            start=0;checkpoint=dest/'checkpoint.pt'
            if checkpoint.exists():
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['protocol_sha256']==sha(folder/'protocol.json')
                model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng'];start=saved['step']
            for step in range(start,p['steps']):
                batch=rng.choice(known,size=p['batch'],replace=False);d=data['fit84'];idx=torch.as_tensor(batch)
                model.train();opt.zero_grad(set_to_none=True);scores=model(d['x'][idx],d['seed'][idx],d['geo'][idx],d['mask'][idx])
                loss=marginal_choice_loss(torch,scores,d['good'][idx],d['mask'][idx],1.)
                assert bool(torch.isfinite(loss));loss.backward();assert all(bool(torch.isfinite(z.grad).all()) for z in model.parameters() if z.grad is not None);opt.step()
                if (step+1)%50==0 or step+1==p['steps']:
                    temp=checkpoint.with_suffix('.next.pt');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,step=step+1,protocol_sha256=sha(folder/'protocol.json')),temp);temp.replace(checkpoint)
                    status('training','Обучение совместного выбора реальных областей',variant=variant+f'/seed{seed}',done=step+1,total=p['steps'],models_done=models,models_total=12,loss=float(loss.detach()))
            fitlogits=predict(torch,model,data['fit20']);grid=[]
            for tau in (0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.01):
                _,per,count=policy_rows(data['fit20'],fitlogits,tau,p['fit20']);grid.append(dict(tau=tau,mean=float(np.mean(list(per.values()))),counts=count,per_video=per))
            choice=max(grid,key=lambda z:(z['mean'],-z['counts']['selected'],z['tau']))
            write(dest/'calibration.json',dict(grid=grid,tau=choice['tau'],frozen_before_evaluation=True))
            logits=predict(torch,model,data['train26']);rows,per,count=policy_rows(data['train26'],logits,choice['tau'],p['evaluate26'])
            write(dest/'rows.json',rows);torch.save(model.state_dict(),dest/'model.pt')
            report=dict(variant=variant,seed=seed,tau=choice['tau'],counts=count,per_video=per,steps=p['steps'],train_known=len(known),cuda_initialized=torch.cuda.is_initialized(),rows_sha256=sha(dest/'rows.json'))
            write(dest/'report.json',report);write(dest/'done.json',dict(model_sha256=sha(dest/'model.pt'),report_sha256=sha(dest/'report.json')))
            results[variant,seed]=report;models+=1
            if models==1:
                estimate=2*(time.monotonic()-before)*11+60;left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
                write(folder/'price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left));assert estimate<left,'Training price gate: preserve completed first model'
        del data;gc.collect()
    means={a:{v:float(np.mean([results[a,s]['per_video'][v] for s in p['seeds']])) for v in p['evaluate26']} for a in ('D','C','R1','geometry')}
    comp={'D-baseline':paired(means['D'],{v:0. for v in p['evaluate26']})}
    for a in ('C','R1','geometry'):comp['D-'+a]=paired(means['D'],means[a])
    passed=comp['D-baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values())
    verify_sources(p);write(folder/'report.json',dict(complete=True,primary_pass=passed,models=models,paired=comp,mean_seed_per_video=means,individual={a:{str(s):results[a,s] for s in p['seeds']} for a in means},hota_measured=False,cuda_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-began))


def phase_protocol(phase):
    p=read(RUN/'protocol.json');stop=min(now()+timedelta(seconds=BUDGETS[phase]),datetime.fromisoformat(p['pipeline_save_at']))
    assert now()<stop
    extra={}
    for previous in list(BUDGETS)[:list(BUDGETS).index(phase)]:
        report=RUN/previous/'report.json';assert read(report)['complete'];extra[report.relative_to(ROOT).as_posix()]=sha(report)
    q=dict(p,device='cuda' if phase=='features' else 'cpu',budget_seconds=BUDGETS[phase],save_at=stop.isoformat(),hard_stop=min(stop+timedelta(seconds=20),datetime.fromisoformat(p['pipeline_hard_stop'])).isoformat(),stop_request_path=(RUN/phase/'stop.request').relative_to(ROOT).as_posix(),sha256={**p['sha256'],**extra})
    write(RUN/phase/'protocol.json',q);write(RUN/phase/'status.json',dict(state='ready',stage=phase,updated=now().isoformat()))
    return q


def pipeline():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps=WindowsProcesses();p=read(RUN/'protocol.json');verify_sources(p)
    write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat()))
    for number,phase in enumerate(BUDGETS):
        folder=RUN/phase
        if not (folder/'controller_result.json').exists():
            q=phase_protocol(phase);env={**os.environ,'Z13_REGION_PHASE':phase}
            with (folder/'controller.stdout.log').open('w',encoding='utf-8') as out,(folder/'controller.stderr.log').open('w',encoding='utf-8') as err:
                process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',str(Path(__file__))],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
                birth=ps.creation(process.pid);assert birth is not None
                write(folder/'launch.json',dict(controller_launcher=process.pid,birth=birth,at=now().isoformat()))
                while process.poll() is None:
                    try:
                        s=read(folder/'status.json');write(RUN/'status.json',dict(s,phase=phase,phase_index=number+1,phases_total=3,observer_at=now().isoformat()))
                    except (OSError,ValueError):pass
                    time.sleep(2)
                process.wait()
        e=read(folder/'controller_result.json');assert e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
        assert e['exit_code']==0,(phase,'Preserve failure and repair within original phase budget')
        assert read(folder/'report.json')['complete']
        write(folder/'verified.json',dict(at=now().isoformat(),report_sha256=sha(folder/'report.json'),all_owned_ended=True))
    record()


def record():
    from scripts.z13_continue import paired
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p=read(RUN/'protocol.json');verify_sources(p);ps=WindowsProcesses()
    for phase in BUDGETS:
        e=read(RUN/phase/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    r=read(RUN/'training/report.json');assert r['complete'] and r['models']==12 and not r['cuda_initialized']
    means={}
    for variant in ('D','C','R1','geometry'):
        means[variant]={}
        for seed in p['seeds']:
            d=RUN/'training'/variant/f'seed{seed}';m=read(d/'report.json');done=read(d/'done.json')
            assert sha(d/'model.pt')==done['model_sha256'] and sha(d/'report.json')==done['report_sha256'] and sha(d/'rows.json')==m['rows_sha256']
            rows=read(d/'rows.json');assert len(rows)==8326
            per={v:sum(int(x['beneficial'])-int(x['harmful']) for x in rows if x['video']==v)/sum(x['video']==v and x['kind']>=0 for x in rows) for v in p['evaluate26']};assert per==m['per_video']
        for v in p['evaluate26']:means[variant][v]=sum(read(RUN/'training'/variant/f'seed{s}'/'report.json')['per_video'][v] for s in p['seeds'])/len(p['seeds'])
    comp={'D-baseline':paired(means['D'],{v:0. for v in p['evaluate26']})}
    for a in ('C','R1','geometry'):comp['D-'+a]=paired(means['D'],means[a])
    assert comp==r['paired'];passed=comp['D-baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values());assert passed==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),primary_rows_models_intervals_verified=True,report_sha256=sha(RUN/'training/report.json')))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    fields=dict(summary='Полный обучаемый региональный узел: основной критерий '+('пройден.' if passed else 'не пройден.'),obtained='runs/z13_region_assignment_20261025/training/report.json → paired: '+str(comp)+';12моделей,трисидa,порог fit20; первичные строки и все контроли сохранены.',interpretation='Полный внутренний узел оценён по точным однокадровым назначениям. Это не HOTA и не изоляция одного свойства ядра. Контроль геометрии отделяет пользу внешнего применения; C/R1 отделяют вклад D.',decision=p['next']+' Основной итог: '+str(passed),cannot_claim=p['counterargument'])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=key,old=card[key],reason='Проверен полный равный пилот регионального считывателя.'))
        card[key]=value
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'))
    for i,met in enumerate((True,passed)):card['acceptance'][i].update(met=met,evidence='runs/z13_region_assignment_20261025/'+('independent_audit.json' if i==0 else 'training/report.json'))
    write(cp,cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed))
    journal('итог обучаемого регионального считывателя','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nHOTA и процент к стенду не измерены; доля оракульного потолка не переносится в HOTA. Подтверждённые сравнения и архитектурную веху проверить отдельно по первичным интервалам, не красить весь узел зелёным по частному числу.')
    write(RUN/'status.json',dict(state='completed',phase='finished',stage='Все12моделей и первичные интервалы проверены',updated=now().isoformat()))
    handoff('Полный пилот окончен. '+fields['obtained']+' '+fields['decision']+' Публикация runs/z13_region_assignment_20261025/publication.json. Архитектурная проверка нового узла требуется; не пересчитывать научные данные ради оформления.')
    publish()


def handoff(text):
    h=ROOT/'runs/z13_weekend_20261024/HANDOFF.md';h.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+text+' Авторизация26.10 07:40МСК,57чужихправок и.gitignore сохранять.\n\nНиже прежние передачи сохранены.\n\n'+h.read_text(encoding='utf-8'),encoding='utf-8')


def publish():
    from scripts import z13_presence_scope_20261025 as pub
    source=inspect.getsource(pub.publish).replace('tracking/z13_presence_scope.py','tracking/z13_region_assignment.py').replace('tests/test_z13_presence_scope.py','tests/test_z13_region_assignment.py').replace('scripts/z13_presence_scope_20261025.py','scripts/z13_region_assignment_20261025.py').replace('scripts/z13_presence_scope_console.ps1','scripts/z13_region_assignment_console.ps1').replace('Audit target-presence supervision before refusal training','Evaluate matched learned regional FlyVis assignment readout')
    source=source.replace("RUN.glob('*.json')","RUN.rglob('*.json')")
    ns={**vars(pub),'RUN':RUN};exec(compile(source,str(Path(__file__)),'exec'),ns);ns['publish']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:
        try:pipeline()
        except Exception as exc:
            write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),phase=PHASE,error=repr(exc)));handoff('Цепочка регионального считывателя остановилась: '+repr(exc)+'. Смотреть pipeline_failure.json и последний фазовый controller_result/worker.stderr. Старые источники/готовые этапы не повторять; исправить только проверенную служебную ошибку в первоначальном фазовом/global окне. Не объявлять научный отрицательный исход по техническому сбою.');raise
    elif '--worker' in sys.argv:
        try:
            {'labels':label_worker,'features':feature_worker,'training':train_worker}[PHASE]()
            write(RUN/PHASE/'status.json',dict(state='completed',stage='Этап полностью сохранён',updated=now().isoformat()))
        except Exception as exc:write(RUN/PHASE/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
