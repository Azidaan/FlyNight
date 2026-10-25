"""Bounded matched regional identity readout with nested video controls and a final gate."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,time,gc,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_temporal_pair_readiness_20261025 import read,write,sha,now,paired
from scripts.z13_temporal_identity_features_20261025 import journal
BASE=ROOT/'runs/z13_temporal_identity_20261025'
FEATURES=BASE/'features';RUN=BASE/'training'
FOLLOW=ROOT/'runs/z13_box_supervision_followup_20261025'
REF=ROOT/'runs/z13_temporal_pair_readiness_20261025'
IDENT='z13_temporal_identity_readout_v1'
ARMS=('D','D_blank','C','R1','geometry','D_no_identity')
THRESHOLDS=(0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.01)


def note(title,text):
    from scripts import z13_temporal_identity_features_20261025 as module
    old=module.IDENT
    try:module.IDENT=IDENT;module.journal(title,text)
    finally:module.IDENT=old


def prepare():
    assert not RUN.exists()
    original=read(FOLLOW/'protocol.json');folds=read(ROOT/'runs/z13_default_class_trial_20261025/protocol.json')['folds']
    created=now();soft=min(created+timedelta(seconds=7200),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_temporal_identity_readout.py',
        'tracking/z13_filter_descriptor.py','tracking/z13_region_assignment_context.py','tracking/z13_region_value_loss.py',
        'tests/test_z13_temporal_identity_readout.py','runs/z13_default_class_trial_20261025/protocol.json',
        'runs/z13_temporal_pair_readiness_20261025/report.json','runs/z13_temporal_identity_20261025/features/protocol.json']
    p=dict(id=IDENT,created=created.isoformat(),device='cuda',timezone='Europe/Moscow',
        budget_seconds=7200,save_at=soft.isoformat(),hard_stop=(soft+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        internal_budget_seconds=1800,final_budget_seconds=600,arms=list(ARMS),seeds=[0,1,2],steps=1150,batch=128,lr=.001,
        fit84=original['fit84'],fit20=original['fit20'],evaluate26=original['evaluate26'],folds=folds,thresholds=list(THRESHOLDS),
        action='Replace internal full-box action readout by target-conditioned cross-frame real-region association; residual action scores can change the selected old full-box/NULL action. Keep old current features, labels and action semantics.',
        reference='Latest supported strictly earlier stored observation on the fixed alignment reference clock, same episode; exact same reference selection at training/application. Labels never choose reference. Missing prior history gives zero temporal residual; all original queries remain evaluated.',
        architecture='Original68/68embeddings16 and94->32->1 current head. Known-current/past candidate pair features:abs16,product16 plus5observables(global center displacement2 normalized by currentcrop extent,logsize ratio2,log1p gap/log31). Pair MLP37->16tanh->1,learned unmatched scalar,softmax over supported real past candidates+unmatched. Attended16+5+unmatched1 append to original94 then116->32->1 zero-initialized residual. NULL action is not treated as physical identity. No one-to-one or calibrated unmatched claim.',
        loss='Original mean-query signed MSE over beneficial(+1),harmful(-1),neutral(0) actions; unknown current kinds excluded as before. Add weight1mean per-query BCE ONLY on known real candidate identity pairs. Unknown endpoints never negatives. D_no_identity zeros only this additional loss. D_blank zeros history availability and thus pair loss/residual. Geometry zeros current/past activity and target descriptor, retaining same observable pair geometry/context.',
        matched='Same6architectures/parameter allocation/init per seed,UID minibatch draws,1150updates,batch128,Adam.001 and all original outcome labels. D/C/R1 share actual frames/regions/masks. No architecture/loss/threshold/seed choice after outcomes.',
        regression='All12previous contextual heads4representations*3seeds on8326queries: restored saved rows/probabilities/outcomes exact AND new zero residual exact same logits with prior weights. Pure CPU regression while GPU feature extraction runs. New trained predictions exactly reproduced from saved weights on their validation inputs.',
        calibration='Fixed old probability grid[0,.1,...,.9,1.01],separate nonoverlapping calibration videos in each nested fold; maximize mean-video(help-harm)/known,ties fewer interventions then higher threshold. Freeze before validation; average3seeds equally.',
        internal_primary='On full104out-of-fold videos: D mean-video net>=.02 AND every paired-video95%lower>0 versus0,D_blank,C,R1,geometry,D_no_identity. All required; otherwise stop before any new26evaluation. Intervals conditional on reused104/fixed folds.',
        final_primary='Only after internal pass:18newmodels fit84/cal20,one evaluation on26reusedvideos with same6contrasts and frozen previousD_context contrast; alllower>0,net>=.02. No automatic HOTA. Diagnostic pairAUROC is secondary and cannot rescue action gate.',
        price='Internal GPU budget1800s,total90models. After first full model:2*seconds*89+180 must fit remaining internal budget. Preserve completed model and stop if not. Final18models have own600s price/deadline. Feature readiness and CPU regression required before first GPU model. OneCUDAprocess,independent hard deadline and checkpoints50steps; never repeat completed models.',
        counterargument='GT-known associations may be insufficient or unrelated to beneficial intervention; neighboring objects and sparse clock can still confuse FlyVis. Improved pairAUROC alone is not benefit to the stand. The same architecture may perform equally well using geometry or retinal input.',
        cannot_claim='Not fullKeepTrack,not a retained identity bank or full spatial exemplar matching. No HOTA/core-training/independentconfirmation/calibrated absence. Finite regional mean68/old target68 and fixed clock limit mechanism. Reused104and26 exploratory.',
        literature='KeepTrack https://arxiv.org/html/2103.16556v2 §§3.2,3.5,3.6 supports learned candidate associations and unmatched alternatives. Our regional/sign-action loss and maskedBCE are adaptations; no expected improvement imported.',
        sha256={n:sha(ROOT/n) for n in names})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='waiting_features',models_done=0,models_total=90,updated=now().isoformat()))
    catalog=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=IDENT,title='З13: внутреннее межкадровое считывание с метками личности реальных областей',area='flyvis',state='planned',started=created.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Даёт ли обучаемое соответствие реальных областей пользу выбора действий сверх истории без меток, текущего чтения, штатнойC,входаR1 игеометрии?',
        hypothesis='Явное обучение известных соответствий может помочь сохранять личность и выбирать полезную старую рамку; это рабочая гипотеза,а не установленная причина неудач.',
        expected=p['action']+' '+p['reference'],acceptance=[dict(criterion=p[k],met=None) for k in ('regression','matched','internal_primary','final_primary')],
        obtained='',interpretation='',decision='',cannot_claim=p['cannot_claim']+' '+p['counterargument'],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Архитектура,одинаковое прошлое,отличия,контроли,цена икритерии дообучения')]))
    write(ROOT/'experiments/project_history.json',catalog)
    glossary=read(ROOT/'experiments/metrics_glossary.json');assert 'candidate_identity_auroc' not in glossary['metrics']
    glossary['metrics']['candidate_identity_auroc']=dict(title='AUROC соответствия известных областей',ru='Площадь под ROC-кривой бинарной оценки одного объекта по паре областей; только пары с известной личностью обоих концов.',better='higher',unit='fraction',range=[0,1],tool='sklearn.metrics.roc_auc_score',caveat='Условная диагностика пар,не сохранение личности полного трекера и не HOTA. Зависимые пары требуют повидео сравнения для научного вывода.')
    write(ROOT/'experiments/metrics_glossary.json',glossary)
    note('постановка считывания реальных межкадровых соответствий','*Действие.* '+p['action']+'\n\n*Узел.* '+p['architecture']+'\n\n*Один набор условий.* '+p['matched']+'\n\n*Проверки.* '+p['regression']+'\n\n*Критерии до результата.* '+p['internal_primary']+' '+p['final_primary']+'\n\n*Цена.* '+p['price']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Источник.* '+p['literature'])


def current_data(torch,ns,arm,split):
    from scripts.z13_default_class_trial_20261025 import load
    entries={}
    p=read(RUN/'protocol.json');videos=p['fit20'] if split=='fit20' else p['fit84'] if split=='fit84' else p['evaluate26']
    for v in videos:entries.update({int(u):e for u,e in read(FOLLOW/'cache/videos'/(v+'.json'))['entries'].items()})
    return load(torch,ns,arm,split,entries)


def regression():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from tracking.z13_region_assignment_context import make_context_ranker
    from tracking.z13_temporal_identity_readout import make_model
    torch.set_num_threads(2);torch.set_default_device('cpu');ns=namespace();p=read(RUN/'protocol.json');results={}
    for arm in ('D','C','R1','geometry'):
        data=current_data(torch,ns,arm,'train26')
        for seed in p['seeds']:
            folder=FOLLOW/'training'/(arm+'_context')/f'seed{seed}'
            state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            old=make_context_ranker(torch,activity=arm!='geometry');old.load_state_dict(state)
            model=make_model(torch,activity=arm!='geometry');model.base.load_state_dict(state);model.eval();logits=[]
            with torch.no_grad():
                for left in range(0,len(data['uid']),128):
                    sl=slice(left,left+128);x,s,g,m=[data[k][sl] for k in ('x','seed','geo','mask')]
                    px=torch.zeros((len(x),1,68));pm=torch.zeros((len(x),1),dtype=torch.bool);pg=torch.zeros((len(x),x.shape[1],1,5))
                    actual,_,_=model(x,s,g,m,px,pm,pg);expected=old(x,s,g,m);assert torch.equal(actual,expected)
                    logits.extend(z[:int(mask.sum())].numpy().copy() for z,mask in zip(actual,m))
            r=read(folder/'report.json');rows,per,counts=ns['policy_rows'](data,logits,r['tau'],p['evaluate26'])
            assert rows==read(folder/'rows.json') and per==r['per_video'] and counts==r['counts']
            results[arm+str(seed)]=dict(rows=len(rows),zero_residual_exact=True,all_saved_predictions_exact=True,source_model_sha256=sha(folder/'model.pt'))
        del data;gc.collect()
    assert len(results)==12 and not torch.cuda.is_initialized()
    write(RUN/'original_regression.json',dict(complete=True,models=results,old_models=12,all_exact=True,cuda_initialized=False,at=now().isoformat()))


def load_data(torch,arm,include26=False):
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts.z13_default_class_trial_20261025 import merge
    ns=namespace();base='D' if arm in ('D_blank','D_no_identity','geometry') else arm
    data=merge(torch,current_data(torch,ns,base,'fit84'),current_data(torch,ns,base,'fit20'))
    p=read(RUN/'protocol.json');splits=['fit84','fit20'];videos=p['fit84']+p['fit20']
    if include26:
        ev=current_data(torch,ns,base,'train26');width=max(data['mask'].shape[1],ev['mask'].shape[1]);out={}
        ev=dict(ev,video=ev['video']+104)
        for k in data:
            values=[]
            for d in (data,ev):
                z=d[k]
                if k in ('x','geo','mask','good','harm'):
                    shape=list(z.shape);shape[1]=width-z.shape[1];z=torch.cat([z,torch.zeros(shape,dtype=z.dtype)],1)
                values.append(z)
            out[k]=torch.cat(values)
        data=out;splits.append('train26');videos+=p['evaluate26']
    lookup={int(u):i for i,u in enumerate(data['uid'])};parts=[];width=1
    feature_report=read(FEATURES/'report.json')
    for split in splits:
        for marker in sorted((FEATURES/base/split).glob('chunk_*.json')):
            assert sha(marker)==feature_report['manifest'][marker.relative_to(ROOT).as_posix()]
            assert sha(marker.with_suffix('.npz'))==read(marker)['sha256']
            with np.load(marker.with_suffix('.npz')) as z:parts.append({k:z[k] for k in z.files});width=max(width,parts[-1]['previous_x'].shape[1])
    n=len(data['uid']);k=data['mask'].shape[1]
    px=np.zeros((n,width,68),np.float32);pm=np.zeros((n,width),bool);pg=np.zeros((n,k,width,5),np.float32)
    target=np.zeros((n,k,width),np.float32);known=np.zeros((n,k,width),bool);seen=set()
    for a in parts:
        for j,u in enumerate(a['uid']):
            u=int(u);assert u not in seen;seen.add(u);i=lookup[u];w=int(a['previous_mask'][j].sum())
            px[i,:w]=a['previous_x'][j,:w];pm[i,1:w]=True
    for video in videos:
        rows=read(REF/'inputs'/(video+'.json'))['rows'];labels=read(REF/'supervision'/(video+'.json'))['rows']
        with np.load(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz') as z:d={q:z[q] for q in ('xyxy','conf','frame_idx')}
        for r,l in zip(rows,labels):
            u=r['uid'];i=lookup[u]
            if not r['previous']:
                assert u not in seen;continue
            assert u in seen
            ids=np.flatnonzero((d['frame_idx']==r['current_frame'])&(d['conf']>=.25));crop=np.asarray(r['current_crop'],float);extent=np.maximum(crop[2:]-crop[:2],1.)
            boxes=d['xyxy'][ids[np.asarray(r['current_choices'],int)]];past=np.asarray(r['previous']['boxes'],float)
            curcenter=(boxes[:,:2]+boxes[:,2:])/2;pastcenter=(past[:,:2]+past[:,2:])/2
            geometry=np.concatenate([((curcenter[:,None]-pastcenter[None])/extent),
                np.log(np.maximum(boxes[:,None,2:]-boxes[:,None,:2],1.)/np.maximum(past[None,:,2:]-past[None,:,:2],1.)),
                np.full((len(boxes),len(past),1),np.log1p(r['current_frame']-r['previous']['frame'])/np.log(31.))],-1)
            pg[i,1:len(boxes)+1,1:len(past)+1]=geometry
            for ci,pi,value in l['correspondence']:target[i,ci+1,pi+1]=value;known[i,ci+1,pi+1]=True
    assert len(seen)==sum(1 for v in videos for r in read(REF/'inputs'/(v+'.json'))['rows'] if r['previous'])
    data.update(px=torch.from_numpy(px),pm=torch.from_numpy(pm),pg=torch.from_numpy(pg),id_target=torch.from_numpy(target),id_known=torch.from_numpy(known))
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    return data,videos


def forward(model,data,idx):
    return model(*[data[k][idx] for k in ('x','seed','geo','mask','px','pm','pg')])


def predict(torch,model,data,indices):
    result=[];pair_y=[];pair_z=[];model.eval()
    with torch.no_grad():
        for left in range(0,len(indices),256):
            ix=indices[left:left+256];score,logits,valid=forward(model,data,ix)
            masks=data['mask'][ix].cpu().numpy();sc=score.cpu().numpy()
            result.extend(z[:int(m.sum())].copy() for z,m in zip(sc,masks))
            use=data['id_known'][ix]&valid
            pair_y.extend(data['id_target'][ix][use].cpu().numpy().tolist());pair_z.extend(logits[use].cpu().numpy().tolist())
    if len(set(pair_y))==2:
        from sklearn.metrics import roc_auc_score
        auc=float(roc_auc_score(pair_y,pair_z))
    else:auc=None
    return result,dict(known_pairs=len(pair_y),positives=int(sum(pair_y)),candidate_identity_auroc=auc,conditional_only=True)


def selection(cpu,indices,logits,threshold,videos):
    from tracking.z13_region_assignment import apply_policy
    rows=[];sums={v:[0,0] for v in videos}
    for i,z in zip(indices,logits):
        choice,prob=apply_policy(z,threshold);kind=int(cpu['kind'][i]);known=kind>=0;v=videos[int(cpu['video'][i])]
        good=choice>=0 and known and bool(cpu['good'][i,choice]);harm=choice>=0 and known and bool(cpu['harm'][i,choice])
        sums[v][0]+=int(good)-int(harm);sums[v][1]+=int(known)
        rows.append(dict(uid=int(cpu['uid'][i]),video=v,kind=kind,action=choice,probability=prob,beneficial=good,harmful=harm))
    per={v:a/b for v,(a,b) in sums.items() if b}
    count=dict(rows=len(rows),selected=sum(r['action']>=0 for r in rows),beneficial=sum(r['beneficial'] for r in rows),harmful=sum(r['harmful'] for r in rows),unknown_selected=sum(r['action']>=0 and r['kind']<0 for r in rows))
    return rows,per,count


def one(torch,p,cpu,data,videos,split,arm,seed,folder,deadline,models_done):
    from tracking.z13_temporal_identity_readout import make_model,identity_loss
    from tracking.z13_region_value_loss import signed_action_loss
    if (folder/'done.json').exists():
        receipt=read(folder/'done.json');assert sha(folder/'model.pt')==receipt['model_sha256'];assert sha(folder/'report.json')==receipt['report_sha256'];return read(folder/'report.json')
    vi=cpu['video'].numpy();kind=cpu['kind'].numpy()
    train=np.flatnonzero(np.isin(vi,[videos.index(v) for v in split['train']])&(kind>=0))
    calibration=np.flatnonzero(np.isin(vi,[videos.index(v) for v in split['calibration']]))
    validation=np.flatnonzero(np.isin(vi,[videos.index(v) for v in split['validation']]))
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=make_model(torch,activity=arm!='geometry',history=arm!='D_blank').to('cuda:0')
    optimizer=torch.optim.Adam(model.parameters(),lr=p['lr']);batches=torch.from_numpy(np.stack([rng.choice(train,p['batch'],replace=False) for _ in range(p['steps'])])).to('cuda:0')
    folder.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    for step in range(p['steps']):
        if step%50==0:
            if now()>=deadline or (RUN/'stop.request').exists():raise TimeoutError('Original learner phase deadline; checkpoints preserved')
            write(RUN/'status.json',dict(state='running',variant=arm,seed=seed,step=step,steps=p['steps'],models_done=models_done,models_total=90,updated=now().isoformat()))
        idx=batches[step];model.train();optimizer.zero_grad(set_to_none=True);scores,association,valid=forward(model,data,idx)
        action=signed_action_loss(torch,scores,data['good'][idx],data['harm'][idx],data['mask'][idx])
        matching=identity_loss(torch,association,data['id_target'][idx],data['id_known'][idx],valid)
        loss=action+(0 if arm=='D_no_identity' else 1)*matching
        loss.backward();optimizer.step()
        if (step+1)%50==0:
            assert bool(torch.isfinite(loss))
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,protocol_sha256=sha(RUN/'protocol.json')),folder/'checkpoint.pt')
    torch.save(model.state_dict(),folder/'model.pt')
    zcal,_=predict(torch,model,data,calibration);zval,identity= predict(torch,model,data,validation)
    restored=make_model(torch,activity=arm!='geometry',history=arm!='D_blank').to('cuda:0');restored.load_state_dict(torch.load(folder/'model.pt',map_location='cuda:0',weights_only=True))
    replay,_=predict(torch,restored,data,validation);assert all(np.array_equal(a,b) for a,b in zip(zval,replay))
    grid=[]
    for t in THRESHOLDS:
        _,per,counts=selection(cpu,calibration,zcal,t,videos);assert set(per)==set(split['calibration'])
        grid.append(dict(threshold=t,mean=float(np.mean(list(per.values()))),counts=counts))
    best=max(range(len(grid)),key=lambda i:(grid[i]['mean'],-grid[i]['counts']['selected'],i));tau=THRESHOLDS[best]
    rows,per,counts=selection(cpu,validation,zval,tau,videos);assert set(per)==set(split['validation'])
    write(folder/'rows.json',rows)
    record=dict(tau=tau,per_video=per,counts=counts,calibration=grid,identity_diagnostic=identity,seconds=time.monotonic()-began,rows_sha256=sha(folder/'rows.json'))
    write(folder/'report.json',record);write(folder/'done.json',dict(model_sha256=sha(folder/'model.pt'),report_sha256=sha(folder/'report.json'),predictions_reproduced_exact=True,steps=p['steps']))
    del model,restored,optimizer,batches;torch.cuda.empty_cache();return record


def contrast(means):
    base=means['D'];zero={v:0. for v in base}
    def compare(a,b):
        keys=sorted(a);d=np.array([a[v]-b[v] for v in keys]);bs=d[np.random.default_rng(1).integers(0,len(d),(5000,len(d)))].mean(1)
        return dict(mean_per_video=float(d.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),better=int((d>0).sum()),n=len(d))
    result={'D-baseline':compare(base,zero)}
    for a in ARMS[1:]:result['D-'+a]=compare(base,means[a])
    return result


def passes(comparison):
    return comparison['D-baseline']['mean_per_video']>=.02 and all(x['ci95'][0]>0 for x in comparison.values())


def worker():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);torch.set_default_device('cpu');p=read(RUN/'protocol.json')
    assert read(RUN/'original_regression.json')['all_exact'];assert read(FEATURES/'independent_audit.json')['all_bindings_verified']
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    models=0;results={};all_reports={};bindings=None;started=time.monotonic()
    deadline=min(now()+timedelta(seconds=p['internal_budget_seconds']),datetime.fromisoformat(p['save_at']))
    with GpuLock('flyvis_fovea_gpu'):
        for arm in ARMS:
            cpu,videos=load_data(torch,arm);os.environ['CUDA_VISIBLE_DEVICES']='0'
            if bindings is None:bindings=cpu['uid'].tolist()
            else:assert bindings==cpu['uid'].tolist()
            data={k:v.to('cuda:0') for k,v in cpu.items()}
            for seed in p['seeds']:
                per={}
                for fold,split in enumerate(p['folds']):
                    folder=RUN/'internal'/arm/f'seed{seed}'/f'fold{fold}'
                    r=one(torch,p,cpu,data,videos,split,arm,seed,folder,deadline,models);models+=1;per.update(r['per_video']);all_reports[str(folder.relative_to(ROOT))]=sha(folder/'report.json')
                    if models==1:
                        estimate=2*r['seconds']*89+180;left=(deadline-now()).total_seconds();write(RUN/'price.json',dict(first_model_seconds=r['seconds'],conservative_remaining_seconds=estimate,remaining=left,proceed=estimate<left));assert estimate<left,'Internal90models exceed original1800second budget'
                assert set(per)==set(videos);results[arm,seed]=per
            del data,cpu;gc.collect();torch.cuda.empty_cache()
        means={a:{v:float(np.mean([results[a,s][v] for s in p['seeds']])) for v in videos} for a in ARMS}
        comparisons=contrast(means);passed=passes(comparisons)
        write(RUN/'internal_report.json',dict(complete=True,models=models,means=means,paired=comparisons,admit_final=passed,conditional_intervals=True,new26evaluations=0))
        final_pass=None
        if passed:
            deadline=min(now()+timedelta(seconds=p['final_budget_seconds']),datetime.fromisoformat(p['save_at']));final={}
            split=dict(train=p['fit84'],calibration=p['fit20'],validation=p['evaluate26'])
            for arm in ARMS:
                cpu,videos=load_data(torch,arm,include26=True);data={k:v.to('cuda:0') for k,v in cpu.items()}
                for seed in p['seeds']:
                    folder=RUN/'final'/arm/f'seed{seed}';r=one(torch,p,cpu,data,videos,split,arm,seed,folder,deadline,models);models+=1;final[arm,seed]=r['per_video'];all_reports[str(folder.relative_to(ROOT))]=sha(folder/'report.json')
                del data,cpu;gc.collect();torch.cuda.empty_cache()
            fm={a:{v:float(np.mean([final[a,s][v] for s in p['seeds']])) for v in p['evaluate26']} for a in ARMS};fc=contrast(fm);final_pass=passes(fc)
            write(RUN/'final_report.json',dict(complete=True,means=fm,paired=fc,primary_pass=final_pass,previous_D_context=read(FOLLOW/'training/report.json')['mean_seed_per_video']['D_context'],new26evaluations=1))
    write(RUN/'report.json',dict(complete=True,models=models,internal_pass=passed,final_pass=final_pass,final_evaluated=passed,models_report_sha256=all_reports,hota_measured=False,core_trained=False,seconds=time.monotonic()-started))
    write(RUN/'status.json',dict(state='completed',models_done=models,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--regression' in sys.argv:regression()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'failure.json',dict(at=now().isoformat(),error=repr(error)));raise
