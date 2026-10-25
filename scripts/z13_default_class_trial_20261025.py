"""Nested video-separated CPU trial of explicit-default readouts, with bounded continuation."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys, time, gc, subprocess
from datetime import datetime, timedelta
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_continue import paired
from tracking.z13_margin_rule import actions,outcomes,MARGIN_GRID,PROBABILITY_GRID
from tracking.z13_default_class import make_model,action_loss
RUN=ROOT/'runs/z13_default_class_trial_20261025'
OLD=ROOT/'runs/z13_box_supervision_followup_20261025'
AUDIT=ROOT/'runs/z13_margin_rule_audit_20261025'
IDENT='z13_default_class_readout_v1'


def journal(title,body):
    from scripts import z13_candidate_reserve_20261021 as m
    old=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,body)
    finally:m.IDENT=old


def prepare():
    assert not RUN.exists();old=read(OLD/'protocol.json');verify_sources(old)
    a=read(AUDIT/'report.json');e=read(AUDIT/'controller_result.json')
    assert a['complete'] and a['models_reproduced']==24 and e['all_owned_ended'] and e['exit_code']==0
    videos=old['fit84']+old['fit20'];assert len(set(videos))==104
    outer={v:i%5 for i,v in enumerate(sorted(videos))};folds=[]
    for k in range(5):
        remain=sorted(v for v in videos if outer[v]!=k)
        cal=remain[::4];train=[v for v in remain if v not in cal];test=[v for v in videos if outer[v]==k]
        assert not(set(train)&set(cal) or set(cal)&set(test) or set(train)&set(test))
        folds.append(dict(train=train,calibration=cal,validation=test))
    names=['tracking/z13_default_class.py','tests/test_z13_default_class.py','scripts/z13_default_class_trial_20261025.py','tracking/z13_margin_rule.py','scripts/z13_bounded_recovery_controller_20261022.py',(AUDIT/'report.json').relative_to(ROOT).as_posix()]
    start=now();soft=min(start+timedelta(seconds=7200),datetime.fromisoformat(read(ROOT/'runs/z13_weekend_20261024/authorization.json')['save_at']))
    p=dict(id=IDENT,created=start.isoformat(),device='cpu',timezone='Europe/Moscow',budget_seconds=7200,save_at=soft.isoformat(),hard_stop=(soft+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        fit84=old['fit84'],fit20=old['fit20'],evaluate26=old['evaluate26'],folds=folds,seeds=[0,1,2],steps=1150,batch=128,learning_rate=.001,epsilon=.01,
        arms=['D','C','R1','geometry'],losses=['mse','cost_ce'],rules=['probability','margin'],margin_grid=MARGIN_GRID,probability_grid=PROBABILITY_GRID,
        action='One unchanged full-box or NULL action, versus leave unchanged. Only the internal readout gets a learned default score and the cost-sensitive CE loss; no core training, new feature extraction, video computation or GPU.',
        matched='Both MSE and costCE use identical augmented architecture, initialisation, training/video partitions, epsilon=.01 and optimizer. Input geometry/context identical; geometry control zeros both activity and observed target descriptor including default head. Four loss/rule combinations are reported; costCE+margin is primary, no winner selection.',
        epsilon_reason='Fixed before learning: a neutral intervention costs .01 in units where helpful=+1,harmful=-1,default=0. No sweep. Both loss arms pay the same cost; reported primary metric remains original unpenalized helpful-minus-harmful.',
        internal='5outer video folds across fit104. Each remaining pool has disjoint fixed calibration subset every4th sorted video; train only remainder. Fit score models on train; select threshold on calibration mean-video utility, ties fewer actions then larger threshold. Evaluate once on outer validation. Three seeds averaged equally. OOF interval descriptive conditional on folds; it does not include whole-pipeline retraining uncertainty. Reused development videos.',
        admission='Primary costCE+margin D OOF mean>=.02 and all paired95%lower>0 against zero, sameMSE+marginD,C,R1,geometry. This is conservative resource admission, not proof that information is absent when failed. AUC has no mandatory cutoff. If failed, preserve120models and stop this implementation; no26/HOTA.',
        final='Only after internal pass:24fresh finalheads (4arms*2losses*3seeds) trainedfit84, calibrationfit20, frozen once before26. Compare primary D against zero, originalD and matched MSE,C,R1,geometry; mean>=.02 and all lower>0. Report otherloss/rulecells without selecting them. Never auto-launch HOTA.',
        price='After first model require2*elapsed_per_model*143+300 within remaining7200s. Checkpoint every50steps. Save existing models, do not extend on failure.',
        sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Внутренняя проверка обучаемого отказа',updated=start.isoformat(),models_done=0,models_total=120))
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT)
    from scripts.z13_margin_rule_audit_20261025 import revise
    revise(card,'expected',p['matched']+' '+p['internal']+' '+p['admission']+' '+p['final']+' CPU7200с+20с;epsilon=.01;'+p['price'],'Дописано до обучения по разрешению автора продолжить планы; конкретизация С1б после отрицательной С1а.')
    card['state']='running';card['evidence'].append(dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='С1б: бюджет,стоимость вмешательства,вложенное разбиение и автоматический переход только после допуска'))
    write(ROOT/'experiments/project_history.json',cat)
    journal('подготовка С1б: обучаемый отказ с равными контролями','*Постановка.* '+p['action']+'\n\n*Контрпроверка.* '+p['matched']+'\n\n*Критерии.* '+p['internal']+' '+p['admission']+' '+p['final']+'\n\n*Параметры.* '+p['epsilon_reason']+' CPU7200с+20с,120внутренних моделей и условно24итоговых, срок '+p['hard_stop']+'. ГраницаAUC0,93 не используется. Обучение ядра отсутствует.')
    h=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    h.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\nНовый CPU-опыт '+str(RUN.relative_to(ROOT))+'; protocol.json до запуска, независимый bounded_recovery_controller,7200с+20с. С0/С1а завершены:24точных воспроизведения, прямой порог D -0,0589п.п. наfit20, допуск не пройден. Теперь120моделей вложенного разбиенияfit104: MSE/costCE, D/C/R1/геометрия,3сида,5частей. Epsilon=.01. Только если внутренний допуск пройден — автоматически24итоговых головы fit84/cal20 и единственная новая26оценка. Иначе стоп,никакогоHOTA. Скрипт z13_default_class_trial_20261025.py; результат internal_report.json, final_report.json при допуске, report.json,controller_result.json; завершённые модели не повторять. После результата независимая проверка→карточка/revisions→журнал→тесты→preserving graph→точныйкоммит/push. Все окна скрыты, чужиеправки/.gitignoreсохранены. Старые бюджеты не продлевались.\n\nПрежние передачи сохранены.\n\n'+h.read_text(encoding='utf8'),encoding='utf8')


def status(**kw):
    p=read(RUN/'protocol.json')
    if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original7200s budget')
    write(RUN/'status.json',dict(state='running',updated=now().isoformat(),**kw))


def load(torch,ns,arm,split,entries):
    d=ns['load_split'](torch,'D' if arm=='geometry' else arm,split)
    extra=torch.zeros((*d['mask'].shape,6),dtype=torch.float32)
    for i,u in enumerate(d['uid'].tolist()):
        e=entries[u];n=int(d['mask'][i].sum());assert n==len(e['choices']) and int(d['kind'][i])==e['kind']
        d['good'][i,:n]=torch.tensor(e['good'],dtype=torch.bool);d['harm'][i,:n]=torch.tensor(e['harm'],dtype=torch.bool);extra[i,:n]=torch.tensor(e['context'])
    return dict(d,geo=torch.cat([d['geo'],extra],-1))


def merge(torch,a,b):
    b=dict(b,video=b['video']+84);width=max(a['mask'].shape[1],b['mask'].shape[1]);out={}
    for k in a:
        vals=[]
        for d in (a,b):
            v=d[k]
            if k in ('x','geo','mask','good','harm'):
                shape=list(v.shape);shape[1]=width-v.shape[1]
                v=torch.cat([v,torch.zeros(shape,dtype=v.dtype)],1)
            vals.append(v)
        out[k]=torch.cat(vals,0)
    assert len(set(out['uid'].tolist()))==len(out['uid']);return out


def subset(d,idx):return {k:v[idx] for k,v in d.items()}


def selection(d,logits,rule,threshold,videos):
    chosen=actions(logits,rule,threshold);kind=d['kind'].numpy();vi=d['video'].numpy()
    net=outcomes(chosen,d['good'].numpy(),d['harm'].numpy(),kind)
    per={v:float(net[(vi==i)&(kind>=0)].mean()) for i,v in enumerate(videos) if ((vi==i)&(kind>=0)).any()}
    rows=[dict(uid=int(u),video=videos[int(vi[i])],kind=int(kind[i]),action=int(chosen[i]),beneficial=bool(net[i]>0),harmful=bool(net[i]<0)) for i,u in enumerate(d['uid'])]
    return rows,per,dict(selected=int((chosen>=0).sum()),beneficial=int((net>0).sum()),harmful=int((net<0).sum()),unknown_selected=int(((chosen>=0)&(kind<0)).sum()))


def one(torch,ns,p,data,videos,split,arm,loss_name,seed,folder,models_done):
    assert not (folder/'done.json').exists();folder.mkdir(parents=True,exist_ok=True)
    vi=data['video'].numpy();kind=data['kind'].numpy()
    train=np.flatnonzero(np.isin(vi,[videos.index(v) for v in split['train']])&(kind>=0))
    cal=subset(data,np.flatnonzero(np.isin(vi,[videos.index(v) for v in split['calibration']])))
    val=subset(data,np.flatnonzero(np.isin(vi,[videos.index(v) for v in split['validation']])))
    assert not(set(cal['uid'].tolist())&set(val['uid'].tolist()))
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=make_model(torch,activity=arm!='geometry');opt=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    for step in range(p['steps']):
        idx=torch.as_tensor(rng.choice(train,p['batch'],replace=False));d=subset(data,idx)
        model.train();opt.zero_grad(set_to_none=True);z=model(d['x'],d['seed'],d['geo'],d['mask'])
        loss=action_loss(torch,z,d['good'],d['harm'],d['mask'],loss_name,p['epsilon'])
        assert bool(torch.isfinite(loss));loss.backward();assert all(bool(torch.isfinite(q.grad).all()) for q in model.parameters() if q.grad is not None);opt.step()
        if (step+1)%50==0 or step+1==p['steps']:
            temp=folder/'checkpoint.next.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,step=step+1,protocol_sha256=sha(RUN/'protocol.json')),temp);temp.replace(folder/'checkpoint.pt')
            status(stage='Обучение внутреннего считывания',variant=arm+'/'+loss_name,seed=seed,step=step+1,steps=p['steps'],models_done=models_done,models_total=120,folder=folder.relative_to(RUN).as_posix(),loss=float(loss.detach()))
    torch.save(model.state_dict(),folder/'model.pt')
    zcal=ns['predict'](torch,model,cal);zval=ns['predict'](torch,model,val)
    restored=make_model(torch,activity=arm!='geometry');restored.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
    replay=ns['predict'](torch,restored,val);assert all(np.array_equal(a,b) for a,b in zip(zval,replay))
    result={}
    for rule,grid in (('probability',PROBABILITY_GRID),('margin',MARGIN_GRID)):
        calibration=[]
        for t in grid:
            _,per,count=selection(cal,zcal,rule,t,videos);assert set(per)==set(split['calibration'])
            calibration.append(dict(threshold=t,mean=float(np.mean(list(per.values()))),counts=count,per_video=per))
        best=max(range(len(grid)),key=lambda j:(calibration[j]['mean'],-calibration[j]['counts']['selected'],j));threshold=grid[best]
        rows,per,count=selection(val,zval,rule,threshold,videos);assert set(per)==set(split['validation'])
        write(folder/rule/'rows.json',rows);result[rule]=dict(threshold=threshold,per_video=per,counts=count,calibration=calibration,rows_sha256=sha(folder/rule/'rows.json'))
    write(folder/'report.json',result);write(folder/'done.json',dict(model_sha256=sha(folder/'model.pt'),report_sha256=sha(folder/'report.json'),predictions_reproduced_exact=True,steps=p['steps']))
    return result


def comparisons(means):
    d=means['cost_ce']['margin']['D'];out={'D-baseline':paired(d,{v:0. for v in d}),'D-MSE':paired(d,means['mse']['margin']['D'])}
    for a in ('C','R1','geometry'):out['D-'+a]=paired(d,means['cost_ce']['margin'][a])
    return out


def worker():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu');p=read(RUN/'protocol.json');verify_sources(p)
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    ns=namespace();entries={};began=time.monotonic();models=0
    for v,h in read(OLD/'cache/report.json')['manifest'].items():
        f=OLD/'cache/videos'/(v+'.json');assert sha(f)==h;entries.update({int(k):x for k,x in read(f)['entries'].items()})
    videos=p['fit84']+p['fit20'];results={};bindings=None
    for arm in p['arms']:
        data=merge(torch,load(torch,ns,arm,'fit84',entries),load(torch,ns,arm,'fit20',entries))
        if bindings is None:bindings=data['uid'].tolist()
        else:assert bindings==data['uid'].tolist()
        for loss_name in p['losses']:
            for seed in p['seeds']:
                per={rule:{} for rule in p['rules']}
                for f,split in enumerate(p['folds']):
                    before=time.monotonic();folder=RUN/'internal'/arm/loss_name/f'seed{seed}'/f'fold{f}'
                    r=one(torch,ns,p,data,videos,split,arm,loss_name,seed,folder,models);models+=1
                    for rule in per:
                        assert not(set(per[rule])&set(r[rule]['per_video']));per[rule].update(r[rule]['per_video'])
                    if models==1:
                        estimate=2*(time.monotonic()-before)*143+300;left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
                        write(RUN/'price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left));assert estimate<left,'Preserve first model; price gate failed'
                for rule in per:assert set(per[rule])==set(videos)
                results[arm,loss_name,seed]=per
        del data;gc.collect()
    means={loss:{rule:{arm:{v:float(np.mean([results[arm,loss,s][rule][v] for s in p['seeds']])) for v in videos} for arm in p['arms']} for rule in p['rules']} for loss in p['losses']}
    comp=comparisons(means);passed=comp['D-baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values())
    write(RUN/'internal_report.json',dict(complete=True,models=models,paired=comp,admit_final=passed,means=means,conditional_intervals=True,new_train26_evaluations=0))
    if passed:
        # A separate final fit/calibration/evaluation with fixed protocol, never train on26.
        final={};allvideos=videos+p['evaluate26'];split=dict(train=p['fit84'],calibration=p['fit20'],validation=p['evaluate26'])
        for arm in p['arms']:
            data=merge(torch,load(torch,ns,arm,'fit84',entries),load(torch,ns,arm,'fit20',entries))
            evaldata=load(torch,ns,arm,'train26',entries)
            # merge offsets second input by84; add20 to obtain104 for evaluation.
            evaldata=dict(evaldata,video=evaldata['video']+20);data=merge(torch,data,evaldata)
            for loss_name in p['losses']:
                for seed in p['seeds']:
                    folder=RUN/'final'/arm/loss_name/f'seed{seed}'
                    final[arm,loss_name,seed]=one(torch,ns,p,data,allvideos,split,arm,loss_name,seed,folder,models);models+=1
            del data;gc.collect()
        fm={loss:{rule:{arm:{v:float(np.mean([final[arm,loss,s][rule]['per_video'][v] for s in p['seeds']])) for v in p['evaluate26']} for arm in p['arms']} for rule in p['rules']} for loss in p['losses']}
        fc=comparisons(fm);fc['D-original_D']=paired(fm['cost_ce']['margin']['D'],read(OLD/'training/report.json')['mean_seed_per_video']['D_context'])
        fp=fc['D-baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in fc.values())
        write(RUN/'final_report.json',dict(complete=True,paired=fc,primary_pass=fp,means=fm,new_train26_evaluations=1))
    verify_sources(p);assert not torch.cuda.is_initialized()
    write(RUN/'report.json',dict(complete=True,models=models,internal_pass=passed,final_evaluated=passed,primary_pass=read(RUN/'final_report.json')['primary_pass'] if passed else None,internal_report_sha256=sha(RUN/'internal_report.json'),seconds=time.monotonic()-began,cuda_initialized=False,hota_measured=False))
    write(RUN/'status.json',dict(state='completed',stage='Результаты сохранены; требуется независимая запись',models_done=models,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
    else:raise RuntimeError('Explicit mode required')
