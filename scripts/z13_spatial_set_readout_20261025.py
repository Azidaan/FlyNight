"""Bounded queue: full column activity -> paired spatial-set readouts -> admission."""
from pathlib import Path
import os,sys,time,gc,subprocess,shutil
from datetime import datetime,timedelta
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_continue import paired
RUN=ROOT/'runs/z13_spatial_set_readout_20261025'
BASE=ROOT/'runs/z13_region_assignment_20261025'
OLD=BASE/'recovery2';FOLLOW=ROOT/'runs/z13_box_supervision_followup_20261025'
IDENT='z13_spatial_set_readout_v1';ARMS=('D_spatial','D_mean','C_spatial','R1_spatial','geometry')
BUDGETS={'features':4800,'training':10800}


def journal(title,body):
    from scripts import z13_candidate_reserve_20261021 as m
    previous=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,body)
    finally:m.IDENT=previous


def handoff(body):
    p=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    p.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+body+'\n\nПрежние передачи сохранены.\n\n'+p.read_text(encoding='utf8'),encoding='utf8')


def status(phase,**kw):
    p=read(RUN/phase/'protocol.json')
    if now()>=datetime.fromisoformat(p['save_at']) or (ROOT/p['stop_request_path']).exists():raise TimeoutError('Original spatial-set stage deadline')
    free=shutil.disk_usage(ROOT).free
    if free<20*1024**3 or p.get('initial_free_bytes',free)-free>80*1024**3:raise RuntimeError('Preserve disk reserve; stop new cache allocation')
    write(RUN/phase/'status.json',dict(state='running',updated=now().isoformat(),**kw))


def preflight():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from tracking.z13_spatial_set_readout import make_model
    from tracking.z13_default_class import action_loss
    torch.manual_seed(0);B,N,K=128,721,3
    full=torch.randn(B,N,68);points=torch.randn(N,2);mask=torch.ones(B,K,dtype=torch.bool)
    rm=torch.zeros(B,K,N,dtype=torch.bool);rm[:,0]=True;rm[:,1,:100]=True;rm[:,2,100:200]=True
    x=torch.stack([full[:,rm[0,k]].mean(1) for k in range(K)],1);seed=torch.randn(B,68);geo=torch.zeros(B,K,14)
    centers=torch.zeros(B,K,2);scales=torch.ones(B,K,2);good=torch.zeros(B,K,dtype=torch.bool);harm=~good
    model=make_model(torch);opt=torch.optim.Adam(model.parameters(),lr=.001)
    before=time.monotonic()
    for i in range(12):
        opt.zero_grad(set_to_none=True);z=model(x,seed,geo,mask,full,rm,centers,scales,points)
        loss=action_loss(torch,z,good,harm,mask,'cost_ce',.01);loss.backward();opt.step()
    elapsed=time.monotonic()-before;assert not torch.cuda.is_initialized()
    record=dict(at=now().isoformat(),steps=12,batch=B,columns=N,regions=K,seconds=elapsed,
        conservative_training_seconds=2*elapsed/12*1150*90+600,budget_seconds=7200,
        proceed=2*elapsed/12*1150*90+600<7200,parameters=sum(p.numel() for p in model.parameters()),cuda_initialized=False,
        limitation='Synthetic cost only; not natural data or predicted accuracy. Recheck actual first model and extraction price. All supported columns, no sampling.')
    write(RUN/'preflight.json',record);print(record,flush=True)


def prepare():
    assert not (RUN/'protocol.json').exists()
    pre=read(RUN/'preflight.json');assert pre['conservative_training_seconds']<BUDGETS['training']
    # The synthetic draft7200s allocation failed. No scientific protocol or natural
    # run existed then; register10800s BEFORE the actual experiment, keep old receipt.
    write(RUN/'preflight_allocation.json',dict(at=now().isoformat(),old_draft_budget=pre['budget_seconds'],old_draft_pass=pre['proceed'],new_registered_cpu_budget=BUDGETS['training'],conservative_seconds=pre['conservative_training_seconds'],reason='Initial allocation before protocol/natural data; fits existing author window; no running stage extended.'))
    assert shutil.disk_usage(ROOT).free>100*1024**3,'Preserve >=20GiB after conservative80GiB cache allowance'
    assert read(ROOT/'runs/z13_default_class_trial_20261025/publication.json')['push_completed']
    assert not read(ROOT/'runs/z13_default_class_trial_20261025/report.json')['internal_pass']
    old=read(OLD/'protocol.json');verify_sources(old)
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps=WindowsProcesses();prior=read(ROOT/'runs/z13_default_class_trial_20261025/controller_result.json')
    assert prior['exit_code']==0 and prior['all_owned_ended'] and not prior['remaining']
    assert all(ps.creation(int(pid))!=b for pid,b in prior['owned'].items())
    previous=read(ROOT/'runs/z13_default_class_trial_20261025/protocol.json')
    start=now();stop=min(start+timedelta(seconds=15720),datetime.fromisoformat(read(ROOT/'runs/z13_weekend_20261024/authorization.json')['save_at']))
    names=['tracking/z13_spatial_set_readout.py','tests/test_z13_spatial_set_readout.py',Path(__file__).relative_to(ROOT).as_posix(),
        'scripts/z13_bounded_recovery_controller_20261022.py','tracking/z13_default_class.py','tracking/z13_margin_rule.py',
        'scripts/z13_target_model.py','scripts/z13_region_assignment_recovery2_20261025.py','scripts/z13_region_assignment_20261025.py',
        'runs/z13_default_class_trial_20261025/internal_report.json','runs/z13_spatial_bins_gate_20261025/report.json','runs/z13_spatial_derivatives_20261025/replay/report.json',
        (RUN/'preflight.json').relative_to(ROOT).as_posix(),(RUN/'preflight_allocation.json').relative_to(ROOT).as_posix()]
    p=dict(old,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',device='cpu',budget_seconds=15720,
        save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        phase_budgets=BUDGETS,initial_free_bytes=shutil.disk_usage(ROOT).free,arms=list(ARMS),folds=previous['folds'],steps=1150,batch=128,seeds=[0,1,2],learning_rate=.001,epsilon=.01,
        action='Replace the internal regional embedding of FlyVis activity with a Deep-Sets-style mean of learned functions of every supported (68-channel activity,2relative coordinates) pair. Keep actual full-box/NULL actions and unchanged target descriptor. No quadrants, subsampling or occupancy gate. New outputs can change the chosen actual assignment action.',
        formula='Region bounds from supported observed retinal points; relative coordinate=(point-midpoint)/max(axis-span,lattice-spacing). Point MLP70->16,tanh; mean over all supported columns; target68->16; original94->32->1 competition and46->32->1 default. No LayerNorm on point activity; original train normalization retained. All arms have exactly the same parameters; costCE,epsilon=.01,Adam.001,1150,batch128,margin grid retained from previous trial.',
        controls='D_spatial versus D_mean: latter repeats the68mean computed from the EXACT SAME new fullfloat32activity at each same supported coordinate before the SAME point network. Thus all positions/support/cardinality, train splits, initialization, parameters and loss identical. C_spatial stock fly; R1_spatial equal-dimensional rich retinal history; geometry has identical support/coordinates/context but zeros activity and target. Three seeds equally averaged, no best cell selection.',
        distinction='The failed max operation pooled a scalar response; spatial derivatives changed each column descriptor by two first moments; neither learned a nonlinear joint function of raw region activity and relative position. The four-bin prototype remains inconclusive and is NOT advanced: this set encoder uses every nonempty observed region, requires no fully populated quadrants, and has a separate matched mean control. A geometric availability result is not model performance.',
        technical='Unchanged130query/episode/frame/choices/masks binding and original normalization/weights/history. Every old peak center<0.5px,score<.001, old tolerances unchanged. Mean-control features are computed with the ORIGINAL pool_activity on the SAME new fullfloat32arrays, save/reload means exact. Against previous separately extracted mean arrays record maximum difference descriptively; separateGPUpasses are not assumed bit-exact. New models exactly reproduce saved validation predictions from weights.',
        internal='Same five nested video folds onfit104 as previous trial; threshold selected only on disjoint calibration videos within each fold. Primary D_spatial mean-video(help-harm)/known>=.02 and ALL paired95%lower>0 versus zero,D_mean,C_spatial,R1_spatial,geometry. Seeds equally averaged. Conditional descriptive intervals on reused fit104; no information-absence claim on failure.',
        final='Only if internal admission passes:15new models fit84/cal20; once evaluate26, same criteria including frozen previous D_context from full-box followup. No automatic HOTA. Failed internal stops before26.',
        price='GPU first2D chunks:2*measured_seconds/weighted_frames*remainingD+Cweighted_frames+600 must fit remaining4800s. Synthetic CPU estimate8949s exceeded draft7200s; before any natural run register10800s, retained preflight_allocation. Actual first trained model:2*elapsed*89+600 must fit remaining10800s. Full disk <=80GiB and >=20GiB free. No old active budget extension or completed-chunk repetition.',
        counterargument='Relative spatial evidence may be redundant/noisy; only16aggregate dimensions and1150steps may be insufficient. Target still the prior68descriptor, so failure does not test full spatial exemplar matching. Irregular support supplies geometry, hence geometry and repeated-mean controls mandatory. Point-set theory supplies an architecture, NOT a guarantee of universal representation with this finite embedding or tracking improvement.',
        ranking=['1 Position-conditioned raw-column set encoder, current action: previous costCE D+0.01647ppCIzero, while original mean discard is explicit; equal mean control and bounded price.',
        '2 Full spatial exemplar/query correspondence: richer target and more extraction; next only if current evidence justifies.',
        '3 Learned temporal correspondence:14252/37296supported pairs, availability gap must be repaired first.',
        '4 Directional spatial filter: local derivatives already failed; no radius/scale sweep.',
        '5 Four-bin ROI features: prior gate inconclusive; not admitted.',
        '6 Scalar max/mean: failed; not repeated.',
        '7 Loss/default variants:120models failed; frozen as shared training rule only.',
        '8 Threshold-only repair: failed; no new sweep.',
        '9 Search-state reset: failed.',
        '10 Physical history clock: failed.',
        '11 More epochs with prior loss: negative continuation.',
        '12 RGB/re-ID replacement: new scope/dependencies, not current cheap step.'],
        literature='Zaheer etal,DeepSets,NeurIPS2017,section3.1 https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf: transform individual set elements then aggregate; here elements=(activity,position). Yan etal,AlphaRefine,CVPR2021,section3.3 https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Alpha-Refine_Boosting_Tracking_Performance_by_Precise_Bounding_Box_Estimation_CVPR_2021_paper.pdf: preserving spatial detail motivates a distinct representation test, but box refinement is not candidate identity/assignment.',
        limits='Features4800sGPU,training10800sCPU,each independent20s shutdown;global15720s+20s, all within26Oct07:35save/07:39:30hard. OneGPUmutex,hiddenIdle,no downloads/coretraining/dev/HOTA/heldout75. Current authorization: user Давай пробовать25Oct, ongoingweekend scope.',
        sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Полная пространственная активность → равные считыватели',updated=start.isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Считывание FlyVis по активности отдельных колонок и их положениям',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Даёт ли расположение активности внутри реальной области пользу, потерянную её средним?',hypothesis='Рабочая гипотеза: пары активность–положение различают похожие области при одинаковой средней активности. Структурная возможность проверена синтетически; причина реальных ошибок и польза не установлены.',
        expected=p['action']+' '+p['controls'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['internal'],met=None),dict(criterion=p['final'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Отличия, равные контроли, критерии, источники и бюджет до новых признаков')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка считывания отдельных колонок с положением','*Действие.* '+p['action']+'\n\n*Контрпроверка.* '+p['controls']+'\n\n*Отличие.* '+p['distinction']+'\n\n*Критерии.* '+p['technical']+' '+p['internal']+' '+p['final']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Отбор.* '+' '.join(p['ranking'])+'\n\n*Источники.* '+p['literature']+'\n\n*Бюджет.* '+p['limits']+' Save '+p['save_at']+' / hard '+p['hard_stop']+'. Пробы на синтетике не прогнозируют качество. Проверка равенства при перестановке допускает округлениеfloat32 в новом модульном тесте; старые численные допуски неизменны.')
    handoff('Новый '+IDENT+': '+p['action']+' '+p['limits']+' protocol до запуска. features4800с → training10800с только при старой peak-проверке и равных новых входах.75внутренних моделей, условно15итоговых. Все5основныхпарныхсравнений обязательны,>=2п.п. CPU/GPU price gates. Старые4bins не продолжаются. Последний итог969d93c7 опубликован;57чужихправок/.gitignore сохранить. После результата первичные строки/хеши/карточка/revisions/журнал/тесты/preserving graph/точныйpush. Никакого обещания прироста/HOTA.')


def feature_worker():
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['Z13_REGION_PHASE']='features'
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    ns=namespace();os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_region_assignment import regions,pool_activity
    from tracking.z13_spatial_set_readout import region_coordinates
    from tracking.z13_ridge_localizer import map_center
    from tracking.z13_readout import to_frame
    from tracking.gpu_lock import GpuLock
    p=read(RUN/'features/protocol.json');verify_sources(p);queries,_=ns['queries_and_crops']();geom=ns['geometry']()
    labelrows={}
    for v,h in read(BASE/'labels/report.json')['video_sha256'].items():
        path=BASE/'labels/videos'/(v+'.json');assert sha(path)==h
        labelrows.update({r['uid']:r for r in read(path)['rows']})
    splitinfo=[('fit20',p['fit20'],'fit'),('fit84',p['fit84'],'extra84'),('train26',p['evaluate26'],'evaluate')]
    sets={};total=0;detcache={};manifest={};measured=[];completed=0
    for split,videos,_ in splitinfo:
        byep={}
        for v in videos:
            for q in queries[v]:byep.setdefault(q['label']['episode'],[]).append(q)
        batches=[sorted(byep)[i:i+8] for i in range(0,len(byep),8)]
        weights=[5+max(q['label']['pos'] for e in b for q in byep[e]) for b in batches]
        sets[split]=(byep,batches,weights);total+=sum(weights)
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    before_all=time.monotonic()
    with GpuLock('flyvis_fovea_gpu'):
        for arm in ('D','C','R1'):
            normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json');mu,sd=np.asarray(normal['mean']),np.asarray(normal['sd'])
            if arm=='R1':bank=bb=None;dev='cpu';torch.set_default_device('cpu')
            else:bank,_,bb=old.build_arm(arm,torch,ck);dev=bb.dev;assert str(dev).startswith('cuda')
            for split,videos,key in splitinfo:
                data=Episodes(videos);byep,batches,weights=sets[split]
                desc=read(ROOT/f'runs/z13_filter_descriptor_fixed_20261024/cache/{arm}/{split}/descriptors.json')
                cache=ROOT/'runs/z13_candidate_reserve_20261021'/key if split!='fit84' else ROOT/'runs/z13_candidate_data104_20261021/evaluate'
                canonical={}
                for path in sorted((cache/arm).glob('cache_*.npz')):
                    with np.load(path) as z:
                        canonical.update({(int(e),int(t)):(c[0],float(s[0])) for e,t,c,s in zip(z['episode'],z['t'],z['centers'],z['scores'])})
                for bi,batch in enumerate(batches):
                    path=RUN/'features'/arm/split/f'chunk_{bi:05d}.npz';marker=path.with_suffix('.json')
                    if marker.exists():assert sha(path)==read(marker)['sha256'];manifest[marker.relative_to(ROOT).as_posix()]=sha(marker);completed+=1;continue
                    status('features',stage='Активность всех колонок с прежним состоянием',variant=arm+'/'+split,done=completed,total=3*sum(len(v[1]) for v in sets.values()))
                    before=time.monotonic();upto=[max(q['label']['pos'] for q in byep[e]) for e in batch]
                    hx,_=data.batch(torch,dev,batch,upto);need={4+q['label']['pos'] for e in batch for q in byep[e]}
                    F=old.episode_features(torch,arm,bb,hx,geom.neighbors,need);records=[];dc=ds=delta=0.
                    oldpath=OLD/'features'/arm/split/f'chunk_{bi:05d}.npz'
                    assert sha(oldpath)==read(oldpath.with_suffix('.json'))['sha256']
                    with np.load(oldpath) as z:oldrows={int(u):(z['x'][i],z['mask'][i]) for i,u in enumerate(z['uid'])}
                    for j,e in enumerate(batch):
                        target=np.asarray(desc[str(e)],float)
                        for q in byep[e]:
                            l=q['label'];x=(F[4+l['pos']][j].T-mu)/sd;crop=l['crop']
                            center,height=map_center(x@target,geom);ref,score=canonical[e,4+l['pos']]
                            dc=max(dc,float(np.abs(np.asarray(to_frame(center,crop,geom.eye_size))-ref).max()));ds=max(ds,abs(height-score))
                            v=q['video']
                            if v not in detcache:
                                with np.load(ROOT/f'runs/train_cache/{v}/detections/raw_video.npz') as z:detcache[v]={k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
                            d=detcache[v];ix=(d['frame_idx']==q['frame'])&(d['conf']>=.25);row=labelrows[q['uid']]
                            choices,masks,geo=regions(geom,crop,l['pred'],d['xyxy'][ix],d['conf'][ix],d['cls'][ix],row['regions']['track_class'])
                            assert choices==row['regions']['choices'][1:]
                            means=pool_activity(x,masks);prior,valid=oldrows[q['uid']]
                            assert means.shape==prior[valid].shape
                            delta=max(delta,float(np.max(np.abs(means-prior[valid]))))
                            full=x.astype(np.float32);assert np.isfinite(full).all()
                            means=pool_activity(full.astype(np.float64),masks)
                            c,s=region_coordinates(geom.points,masks,geom.spacing);records.append((q['uid'],full,masks,c,s,means))
                    assert dc<.5 and ds<.001,(arm,split,bi,dc,ds)
                    width=max(len(r[2]) for r in records);n=len(records)
                    arrays=dict(uid=np.array([r[0] for r in records],np.int64),full=np.stack([r[1] for r in records]),pooled=np.zeros((n,width,68),np.float32),retinal_mask=np.zeros((n,width,721),bool),centers=np.zeros((n,width,2),np.float32),scales=np.ones((n,width,2),np.float32))
                    for ri,(_,_,m,c,s,means) in enumerate(records):w=len(m);arrays['retinal_mask'][ri,:w]=m;arrays['centers'][ri,:w]=c;arrays['scales'][ri,:w]=s;arrays['pooled'][ri,:w]=means
                    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**arrays)
                    with np.load(path) as check:
                        for ri in range(n):
                            valid=check['retinal_mask'][ri].any(1)
                            assert np.array_equal(pool_activity(check['full'][ri].astype(np.float64),check['retinal_mask'][ri,valid]),check['pooled'][ri,valid])
                    write(marker,dict(sha256=sha(path),rows=n,seconds=time.monotonic()-before,max_center_px=dc,max_score=ds,same_full_mean_exact=True,previous_pooled_max_difference=delta,uid_sha256=sha(oldpath),full_dtype='float32'))
                    manifest[marker.relative_to(ROOT).as_posix()]=sha(marker);completed+=1
                    if arm=='D' and len(measured)<2:
                        measured.append((time.monotonic()-before,weights[bi]))
                        if len(measured)==2:
                            rate=sum(s for s,w in measured)/sum(w for s,w in measured);estimate=2*rate*(2*total-sum(w for s,w in measured))+600
                            left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds();write(RUN/'features/price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left));assert estimate<left,'Feature price gate, preserve completed'
                    del F,hx;gc.collect()
                del data;gc.collect()
            del bb,bank;gc.collect();torch.cuda.empty_cache()
    verify_sources(p);write(RUN/'features/report.json',dict(complete=True,manifest=manifest,groups=completed,all_same_full_mean_exact=True,all_original_peak_parity=True,seconds=time.monotonic()-before_all))
    write(RUN/'features/status.json',dict(state='completed',done=completed,updated=now().isoformat()))


def load_data(torch,arm,entries,include26=False):
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts.z13_default_class_trial_20261025 import load,merge
    ns=namespace();basearm='D' if arm in ('geometry','D_mean') else arm.split('_')[0]
    data=merge(torch,load(torch,ns,basearm,'fit84',entries),load(torch,ns,basearm,'fit20',entries))
    splits=['fit84','fit20']
    if include26:
        ev=load(torch,ns,basearm,'train26',entries);ev=dict(ev,video=ev['video']+20);data=merge(torch,data,ev);splits.append('train26')
    out=RUN/'training/consolidated'/basearm/('all130' if include26 else 'fit104');out.mkdir(parents=True,exist_ok=True)
    N=len(data['uid']);K=data['mask'].shape[1];keys=dict(full=(N,721,68),pooled=(N,K,68),retinal_mask=(N,K,721),centers=(N,K,2),scales=(N,K,2))
    if not (out/'done.json').exists():
        arrays={k:np.lib.format.open_memmap(out/(k+'.npy'),mode='w+',dtype=np.bool_ if k=='retinal_mask' else np.float32,shape=shape) for k,shape in keys.items()}
        lookup={int(u):i for i,u in enumerate(data['uid'].tolist())};seen=set()
        for split in splits:
            for marker in sorted((RUN/'features'/basearm/split).glob('chunk_*.json')):
                status('training',stage='Подготовка сохранённых пространственных входов',variant=basearm+'/'+split)
                path=marker.with_suffix('.npz');assert sha(path)==read(marker)['sha256']
                with np.load(path) as z:
                    idx=np.array([lookup[int(u)] for u in z['uid']]);assert not seen.intersection(idx.tolist());seen.update(idx.tolist())
                    arrays['full'][idx]=z['full'];w=z['retinal_mask'].shape[1]
                    for k in ('pooled','retinal_mask','centers','scales'):arrays[k][idx,:w]=z[k]
        assert len(seen)==N
        for a in arrays.values():a.flush()
        write(out/'done.json',dict(rows=N,shapes={k:list(v) for k,v in keys.items()},uid=data['uid'].tolist()))
        del arrays;gc.collect()
    assert read(out/'done.json')['uid']==data['uid'].tolist()
    for k in keys:data['x' if k=='pooled' else k]=np.load(out/(k+'.npy'),mmap_mode='r')
    return data


def take(torch,data,idx):
    return {k:torch.from_numpy(np.array(v[idx],copy=True)) if isinstance(v,np.ndarray) else v[idx] for k,v in data.items()}


def predict(torch,model,data,idx,points):
    out=[];model.eval()
    with torch.no_grad():
        for left in range(0,len(idx),64):
            b=take(torch,data,idx[left:left+64]);z=model(b['x'],b['seed'],b['geo'],b['mask'],b['full'],b['retinal_mask'],b['centers'],b['scales'],points)
            out.extend(v[:int(m.sum())].numpy().copy() for v,m in zip(z,b['mask']))
    return out


def one(torch,p,data,videos,split,arm,seed,folder,done_count,points):
    from tracking.z13_spatial_set_readout import make_model
    from tracking.z13_default_class import action_loss
    from tracking.z13_margin_rule import MARGIN_GRID
    from scripts.z13_default_class_trial_20261025 import selection
    if (folder/'done.json').exists():
        done=read(folder/'done.json');assert sha(folder/'model.pt')==done['model_sha256'] and sha(folder/'report.json')==done['report_sha256'];return read(folder/'report.json')
    folder.mkdir(parents=True,exist_ok=True);vi=data['video'].numpy();kind=data['kind'].numpy()
    train=np.flatnonzero(np.isin(vi,[videos.index(v) for v in split['train']])&(kind>=0))
    cal=np.flatnonzero(np.isin(vi,[videos.index(v) for v in split['calibration']]))
    val=np.flatnonzero(np.isin(vi,[videos.index(v) for v in split['validation']]))
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=make_model(torch,'mean' if arm=='D_mean' else 'spatial',activity=arm!='geometry');opt=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    before=time.monotonic()
    for step in range(p['steps']):
        b=take(torch,data,rng.choice(train,p['batch'],replace=False));model.train();opt.zero_grad(set_to_none=True)
        z=model(b['x'],b['seed'],b['geo'],b['mask'],b['full'],b['retinal_mask'],b['centers'],b['scales'],points)
        loss=action_loss(torch,z,b['good'],b['harm'],b['mask'],'cost_ce',p['epsilon']);assert torch.isfinite(loss)
        loss.backward();assert all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in model.parameters());opt.step()
        if (step+1)%50==0 or step+1==p['steps']:
            tmp=folder/'checkpoint.next.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,step=step+1,protocol_sha256=sha(RUN/'protocol.json')),tmp);tmp.replace(folder/'checkpoint.pt')
            status('training',stage='Считывание активности и положения',variant=arm,seed=seed,step=step+1,steps=p['steps'],models_done=done_count,models_total=75,folder=folder.relative_to(RUN).as_posix(),loss=float(loss.detach()))
    torch.save(model.state_dict(),folder/'model.pt');zcal=predict(torch,model,data,cal,points);zval=predict(torch,model,data,val,points)
    restored=make_model(torch,'mean' if arm=='D_mean' else 'spatial',activity=arm!='geometry');restored.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
    replay=predict(torch,restored,data,val,points);assert all(np.array_equal(a,b) for a,b in zip(zval,replay))
    plain=lambda idx:{k:v[idx] for k,v in data.items() if not isinstance(v,np.ndarray)}
    cd,vd=plain(cal),plain(val);grid=[]
    for t in MARGIN_GRID:
        _,per,counts=selection(cd,zcal,'margin',t,videos);assert set(per)==set(split['calibration'])
        grid.append(dict(threshold=t,mean=float(np.mean(list(per.values()))),counts=counts,per_video=per))
    best=max(range(len(grid)),key=lambda j:(grid[j]['mean'],-grid[j]['counts']['selected'],j));threshold=MARGIN_GRID[best]
    rows,per,count=selection(vd,zval,'margin',threshold,videos);assert set(per)==set(split['validation'])
    write(folder/'rows.json',rows);r=dict(threshold=threshold,per_video=per,counts=count,calibration=grid,rows_sha256=sha(folder/'rows.json'),seconds=time.monotonic()-before)
    write(folder/'report.json',r);write(folder/'done.json',dict(model_sha256=sha(folder/'model.pt'),report_sha256=sha(folder/'report.json'),predictions_reproduced_exact=True,steps=p['steps']))
    return r


def comparisons(means):
    d=means['D_spatial'];out={'D-baseline':paired(d,{v:0. for v in d})}
    for a in ARMS[1:]:out['D-'+a]=paired(d,means[a])
    return out


def train_worker():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu');p=read(RUN/'training/protocol.json');verify_sources(p)
    assert read(RUN/'features/report.json')['complete'];entries={}
    for v,h in read(FOLLOW/'cache/report.json')['manifest'].items():
        path=FOLLOW/'cache/videos'/(v+'.json');assert sha(path)==h;entries.update({int(k):x for k,x in read(path)['entries'].items()})
    points=torch.tensor(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],dtype=torch.float32)
    videos=p['fit84']+p['fit20'];results={};models=0;began=time.monotonic();bindings=None
    for arm in ARMS:
        data=load_data(torch,arm,entries)
        if bindings is None:bindings=data['uid'].tolist()
        else:assert bindings==data['uid'].tolist()
        for seed in p['seeds']:
            per={}
            for f,split in enumerate(p['folds']):
                r=one(torch,p,data,videos,split,arm,seed,RUN/'training/internal'/arm/f'seed{seed}'/f'fold{f}',models,points);models+=1;per.update(r['per_video'])
                if models==1:
                    estimate=2*r['seconds']*89+600;left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
                    write(RUN/'training/price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left));assert estimate<left,'Actual model price gate; preserve completed'
            assert set(per)==set(videos);results[arm,seed]=per
        del data;gc.collect()
    means={a:{v:float(np.mean([results[a,s][v] for s in p['seeds']])) for v in videos} for a in ARMS}
    comp=comparisons(means);passed=comp['D-baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values())
    write(RUN/'training/internal_report.json',dict(complete=True,models=models,means=means,paired=comp,admit_final=passed,conditional_intervals=True,new_train26_evaluations=0))
    if passed:
        final={};allvideos=videos+p['evaluate26'];split=dict(train=p['fit84'],calibration=p['fit20'],validation=p['evaluate26'])
        for arm in ARMS:
            data=load_data(torch,arm,entries,include26=True)
            for seed in p['seeds']:
                r=one(torch,p,data,allvideos,split,arm,seed,RUN/'training/final'/arm/f'seed{seed}',models,points);models+=1;final[arm,seed]=r['per_video']
            del data;gc.collect()
        fm={a:{v:float(np.mean([final[a,s][v] for s in p['seeds']])) for v in p['evaluate26']} for a in ARMS};fc=comparisons(fm)
        fc['D-original_D']=paired(fm['D_spatial'],read(FOLLOW/'training/report.json')['mean_seed_per_video']['D_context'])
        write(RUN/'training/final_report.json',dict(complete=True,means=fm,paired=fc,primary_pass=fc['D-baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in fc.values()),new_train26_evaluations=1))
    verify_sources(p);assert not torch.cuda.is_initialized()
    write(RUN/'training/report.json',dict(complete=True,models=models,internal_pass=passed,final_evaluated=passed,hota_measured=False,cuda_initialized=False,seconds=time.monotonic()-began))
    write(RUN/'training/status.json',dict(state='completed',models_done=models,updated=now().isoformat()))


def pipeline():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p=read(RUN/'protocol.json');verify_sources(p);ps=WindowsProcesses()
    for phase in BUDGETS:
        folder=RUN/phase
        if (folder/'controller_result.json').exists():
            e=read(folder/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended']
            assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items());assert read(folder/'report.json')['complete'];continue
        stop=min(now()+timedelta(seconds=BUDGETS[phase]),datetime.fromisoformat(p['save_at']))
        q=dict(p,device='cuda' if phase=='features' else 'cpu',budget_seconds=BUDGETS[phase],save_at=stop.isoformat(),hard_stop=min(stop+timedelta(seconds=20),datetime.fromisoformat(p['hard_stop'])).isoformat(),stop_request_path=(folder/'stop.request').relative_to(ROOT).as_posix())
        write(folder/'protocol.json',q);write(RUN/'status.json',dict(state='running',stage=phase,updated=now().isoformat(),save_at=q['save_at']))
        env={**os.environ,'Z13_SPATIAL_SET_PHASE':phase}
        code=subprocess.call([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',str(Path(__file__))],cwd=ROOT,env=env,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
        assert code==0,(phase,code)
        e=read(folder/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items()) and read(folder/'report.json')['complete']
    write(RUN/'report.json',dict(complete=True,training=read(RUN/'training/report.json'),needs_independent_finish=True,hota_measured=False))
    write(RUN/'status.json',dict(state='completed',stage='Ожидает независимой сверки и записи',updated=now().isoformat()))


if __name__=='__main__':
    if '--preflight' in sys.argv:preflight()
    elif '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        phase=os.environ.get('Z13_SPATIAL_SET_PHASE','pipeline')
        {'features':feature_worker,'training':train_worker,'pipeline':pipeline}[phase]()
    else:raise RuntimeError('Explicit mode required')
