"""Matched rankers: observed ridge coefficients versus foreground mean."""
from pathlib import Path
from datetime import timedelta,datetime
import os,sys,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from scripts import z13_localization_labels_20261023 as prior
from scripts.z13_candidate_reserve_20261021 import read,write,sha
RUN=ROOT/'runs/z13_filter_descriptor_20261023';OLD=prior.OLD;CACHE=prior.CACHE;ARMS=prior.ARMS;IDENT='z13_filter_descriptor_v1';now=prior.now;T0=time.monotonic()


def journal(title,text):
    old=prior.IDENT
    try:prior.IDENT=IDENT;prior.journal(title,text)
    finally:prior.IDENT=old


def status(stage,**kw):
    p=read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Original budget/Moscow deadline')
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),models_total=36,**kw))


def prepare():
    assert not RUN.exists();from scripts.z13_filter_descriptor_cache_20261023 import RUN as CRUN
    cache=read(CRUN/'protocol.json');prior.verify_sources(cache)
    end=read(CRUN/'controller_result.json');assert end['exit_code']==0 and end['all_owned_ended'] and end['worker_ended']
    assert read(CRUN/'report.json')['parity_pass']
    names=['scripts/z13_filter_descriptor_20261023.py','tracking/z13_filter_descriptor.py','tests/test_z13_filter_descriptor.py','scripts/z13_filter_descriptor_console.ps1','runs/z13_filter_descriptor_cache_20261023/report.json']
    names += [p.relative_to(ROOT).as_posix() for p in CRUN.glob('*/*/descriptors.json')]
    base=read(prior.RUN/'protocol.json');p={**cache,'created':now().isoformat(),'budget_seconds':1800,'stop_request_path':(RUN/'stop.request').relative_to(ROOT).as_posix(),'train':base['train'],'evaluate':base['evaluate'],'seeds':[0,1,2],'arms':list(ARMS),'sha256':{**cache['sha256'],**{n:sha(ROOT/n) for n in names}}}
    write(RUN/'protocol.json',p);journal('переход к обучению описания цели','Воспроизведение наблюдаемых входов и все коэффициенты сохранены: runs/z13_filter_descriptor_cache_20261023/report.json. Отдельный CPU-процесс,36моделей и1800с, основные критерии прежние. Протокол runs/z13_filter_descriptor_20261023/protocol.json.')


def worker():
    import gc,numpy as np,torch
    torch.set_default_device('cpu');torch.set_num_threads(2)
    from tracking.z13_filter_descriptor import make_ranker as make_dual
    from tracking.z13_candidate_ranker import make_ranker
    from scripts.z13_candidate_ranker_20261021 import pack,predict,make_rows
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    import scripts.z13_candidate_identity_20261021 as reader
    p=read(RUN/'protocol.json');prior.verify_sources(p);l20=read(CACHE/'fit/labels.json');extra=read(OLD/'evaluate/labels.json');labels=l20+extra;ev=read(CACHE/'evaluate/labels.json')
    ids=np.asarray([i for i,l in enumerate(labels) if l['kind']>=0]);nref=sum(l['kind']>=0 for l in l20);assert len(ids)==20778 and nref==2919
    data={};result={};manifest={};replay={};seedpv={}
    for ai,arm in enumerate(ARMS):
        status('Проверка кэшей и прежнего считывания',variant=arm,done=ai,total=3,models_done=0)
        seq,m=reader.load('fit',arm);manifest.update(m);a=pack(l20,seq);old=reader.CACHE
        try:reader.CACHE=OLD;seq,m=reader.load('evaluate',arm)
        finally:reader.CACHE=old
        manifest.update(m);b=pack(extra,seq);ta=tuple(np.concatenate([x,y]) for x,y in zip(a,b))
        seq,m=reader.load('evaluate',arm);manifest.update(m);ea=pack(ev,seq);data[arm]=(ta,ea)
        for mode,activity in [('activity',True),('geometry',False)]:
            predictions=[]
            for seed in p['seeds']:
                model=make_ranker(torch,activity);model.load_state_dict(torch.load(OLD/arm/mode/f'seed{seed}/model.pt',map_location='cpu',weights_only=True));predictions.append(predict(torch,model,ea))
            fresh=make_rows(ev,ea,np.mean(predictions,0));saved=read(OLD/arm/mode/'ensemble_rows.json');delta=max(abs(a['p_present']-b['p_present']) for a,b in zip(fresh,saved));assert len(fresh)==len(saved)==8326 and delta<=1e-5
            assert all(a['candidate']==b['candidate'] and a['center']==b['center'] and a.get('useful')==b.get('useful') for a,b in zip(fresh,saved))
            replay[arm+'_'+mode]=dict(rows=len(fresh),max_abs_score=delta,centers_outcomes_equal=True);result[arm+'_old_'+mode]=saved
        del seq,a,b;gc.collect()
    write(RUN/'reproduction.json',replay);write(RUN/'cache_manifest.json',manifest)
    t=[torch.as_tensor(a[ids],device='cpu') for a in data['D'][0][:5]];ref=read(OLD/'D/activity/seed0/training.json');torch.manual_seed(0);model=make_ranker(torch,True)
    loss,_=prior.train_steps(torch,model,t,nref,0,ref['pos_weight'],maxsteps=32);delta=max(abs(a-b) for a,b in zip(loss,ref['loss'][:32]));assert delta<=1e-6
    torch.manual_seed(0);model=make_dual(torch,True);start=time.monotonic();prior.train_steps(torch,model,t,nref,0,ref['pos_weight'],maxsteps=32);seconds=time.monotonic()-start
    estimate=seconds/32*1150*36*2+120;write(RUN/'timing.json',dict(old_loss_max_abs=delta,new32_seconds=seconds,conservative_seconds=estimate,budget_seconds=p['budget_seconds']))
    assert estimate<p['budget_seconds']-(time.monotonic()-T0),'Measured work exceeds budget';del t,model;gc.collect();done=0
    for arm in ARMS:
        ta0,ea0=data[arm]
        from scripts.z13_filter_descriptor_cache_20261023 import RUN as CRUN
        # Episode numbers restart for each split. Keep the two training stores separate.
        d20=read(CRUN/arm/'fit20/descriptors.json');d84=read(CRUN/arm/'fit84/descriptors.json');de=read(CRUN/arm/'train26/descriptors.json')
        for access in ('ridge','mean'):
            ta=list(ta0);ea=list(ea0)
            if access=='ridge':
                ta[1]=np.asarray([d20[str(l['episode'])] for l in l20]+[d84[str(l['episode'])] for l in extra],dtype=np.float32)
                ea[1]=np.asarray([de[str(l['episode'])] for l in ev],dtype=np.float32)
            t=[torch.as_tensor(a[ids],device='cpu') for a in ta[:5]]
            for mode,activity in [('activity',True),('geometry',False)]:
                key=arm+'_'+access+'_'+mode;seedpv[key]={};predictions=[]
                for seed in p['seeds']:
                    sd=RUN/arm/access/mode/f'seed{seed}';ref=read(OLD/arm/mode/f'seed{seed}/training.json');torch.manual_seed(seed);model=make_dual(torch,activity)
                    if (sd/'done.json').exists():
                        assert sha(sd/'model.pt')==read(sd/'training.json')['model_sha256'];model.load_state_dict(torch.load(sd/'model.pt',map_location='cpu',weights_only=True))
                    else:
                        def progress(step):status('Обучение выборщика: фильтр против среднего',variant=key+' seed'+str(seed),done=step,total=1150,models_done=done)
                        loss,n=prior.train_steps(torch,model,t,nref,seed,ref['pos_weight'],progress=progress);assert len(loss)==1150 and n==145950
                        sd.mkdir(parents=True,exist_ok=True);tmp=sd/'model.tmp.pt';torch.save(model.state_dict(),tmp);tmp.replace(sd/'model.pt')
                        write(sd/'training.json',dict(seed=seed,steps=len(loss),presented_rows=n,pos_weight=ref['pos_weight'],parameters=sum(p.numel() for p in model.parameters()),loss=loss,model_sha256=sha(sd/'model.pt')));write(sd/'done.json',dict(ended=now().isoformat()))
                    pred=predict(torch,model,ea);predictions.append(pred);rr=make_rows(ev,ea,pred);seedpv[key][str(seed)]=per_video_useful(rr);write(sd/'rows.json',rr);done+=1
                rr=make_rows(ev,ea,np.mean(predictions,0));result[key]=rr;write(RUN/arm/access/mode/'ensemble_rows.json',rr)
        del t;gc.collect()
    status('Парные итоги по26видео',models_done=done)
    for arm in ARMS:
        for b in ('top1','kalman'):result[arm+'_'+b]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{b}_rows.json')
    pv={k:per_video_useful(r) for k,r in result.items()};pairs=[]
    for arm in ARMS:
        pairs.extend((arm+'_ridge_'+m,arm+'_mean_'+m) for m in ('activity','geometry'))
        pairs.extend((arm+'_ridge_activity',arm+'_'+b) for b in ('old_activity','old_geometry','ridge_geometry','top1','kalman'))
    pairs += [('D_ridge_activity','R1_ridge_activity'),('D_ridge_activity','C_ridge_activity'),('C_ridge_activity','R1_ridge_activity')]
    comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs};c=comp['D_ridge_activity-D_mean_activity']
    criteria=dict(main=c['mean_per_video']>=.02 and c['ci95'][0]>0,controls=all(comp['D_ridge_activity-'+b]['ci95'][0]>0 for b in ('D_old_activity','D_old_geometry','R1_ridge_activity','D_ridge_geometry')))
    summary={}
    for key,rr in result.items():
        cov=coverage_table(rr)
        for q,v in cov.items():
            v['known_nonuseful']=v.pop('harmful');selected=sorted(rr,key=lambda r:-r['p_present'])[:v['selected']];v['strict_center_worse_than_prediction']=sum(r['kind']==1 and r['err']>r['err_pred'] for r in selected);v['known_unavailable']=sum(r['kind']==0 for r in selected)
        summary[key]=dict(rows=len(rr),useful_mean_video=float(np.mean(list(pv[key].values()))),coverage=cov)
    sg={a:{s:paired(v,seedpv[a+'_mean_activity'][s]) for s,v in seedpv[a+'_ridge_activity'].items()} for a in ARMS}
    prior.verify_sources(p);assert all(sha(ROOT/n)==h for n,h in manifest.items());write(RUN/'seed_metrics.json',dict(per_video=seedpv,matched_gain=sg));write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired=comp,per_video=pv,summary=summary,seeds=sg,limitations=p['counterargument']+' No HOTA/calibrated refusal; coefficients only from observed frames; exploratory reused26train.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',action='store_true');a=ap.parse_args()
    if a.prepare:prepare()
    elif a.worker:
        try:worker();write(RUN/'status.json',dict(state='completed',stage='36моделей и все сравнения сохранены',updated=now().isoformat(),models_done=36,models_total=36,elapsed_seconds=round(time.monotonic()-T0,2)))
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:ap.error('Choose prepare/worker')
