"""Bounded CPU pilot: same candidate readout, joint action/null likelihood loss."""
from pathlib import Path
from datetime import datetime
import os,sys,time,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from scripts import z13_localization_labels_20261023 as P
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_competing_candidates_20261024';OLD=P.OLD;CACHE=P.CACHE;IDENT='z13_competing_candidates_v1';T0=time.monotonic();ARMS=('D','C','R1')


def journal(title,text):
    before=P.IDENT
    try:P.IDENT=IDENT;P.journal(title,text)
    finally:P.IDENT=before


def status(stage,**kw):
    p=read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Independent CPU budget/deadline')
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),models_total=18,**kw))


def prepare():
    assert not RUN.exists();base=read(ROOT/'runs/z13_localization_labels_20261023/protocol.json');P.verify_sources(base)
    auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json');assert now()<datetime.fromisoformat(auth['save_at'])
    files=['scripts/z13_competing_candidates_20261024.py','tracking/z13_competing_candidates.py','tests/test_z13_competing_candidates.py','tracking/z13_candidate_ranker.py','tracking/z13_matched_data_budget.py','scripts/z13_candidate_ranker_20261021.py','scripts/z13_localization_labels_20261023.py','scripts/z13_candidate_identity_20261021.py']
    ids=['z13_filter_descriptor_fixed_v2','z13_search_conditioned_v1','z13_temporal_candidates_v1','z13_learned_association_v1','z13_semantic_sampling_v1','z13_localization_labels_v1','z13_observed_memory_v1','z13_query_memory_v1','z13_balanced_ridge_v1','z13_aligned_core_v1']
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);cards={c['id']:c for c in cat['experiments']}
    selection=[dict(id=i,state=cards[i]['state'],source='experiments/project_history.json',decision=cards[i].get('decision','')) for i in ids]
    p=dict(id=IDENT,created=now().isoformat(),device='cpu',budget_seconds=1800,save_at=auth['save_at'],hard_stop=auth['hard_stop'],stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),authorization='Explicit autonomous necessary work until26.10.2026 07:40MSK; user requests automatic next stage without another message. This is a new objective-only CPU pilot, not extension of stock control budget.',arms=list(ARMS),seeds=[0,1,2],train=base['train'],evaluate=base['evaluate'],
      action='Replace independent weighted candidate BCE by a joint categorical action likelihood over three candidate logits and a fixed zero null logit. Null means no useful candidate, not necessarily physical target absence. Acceptable set includes every candidate already labelled useful. Loss=-log(sum probability of acceptable actions); null is the sole acceptable action if the original useful labels are all zero. Same original positive weight multiplies useful-row loss; fixed optimizer/batches/seeds/1150steps/145950presentations. Objective and its normalization are the intervention; no new parameters.',
      matched_control='Same original mean-exemplar ranker and initialization,68-16 embedding/88-32-1 head, same cached104train and26evaluation, core and candidates fixed. Reuse18 original BCE models only after full ensemble replay and32-step original loss reproduction. Both BCE and new likelihood use the identical four-way normalized maximum candidate probability for confidence; additionally report original BCE native-logit selection. Geometry and R1 use exactly the same intervention. Unknown training labels excluded.',
      primary='D joint-common minus D BCE-common precision among known selected cases at fixed per-video5% of ALL rows >=0.05 and paired-video95%lower>0; all26videos must have known selected outcomes for all required comparisons. D joint precision also exceeds D joint geometry and R1 joint activity: both lower95>0. D joint useful-center mean must not fall below D BCE common. All required; C cannot rescue.',
      secondary='All18seeds and ensembles; center fractions, C/R1 contrasts, original BCE logit quota; selected/known/unknown/nonuseful/strictly worse/known unavailable per video. Quota ceil(.05*all rows) selected before examining outcomes; unknown excluded only from precision denominator and reported, never counted as an observed error. No fitted rejection threshold, no calibrated guarantee.',
      counterargument='Independent BCE can already learn negatives. Joint competition may add no target identity information, suppress several useful candidates or change gradient scaling. A gain in confidence could be score normalization alone; common-normalization BCE separates that. Unequal knownness of selected cases limits interpretation. It does not establish the cause of FlyVis failures or replace a closed-loop test.',
      distinction='Previous learned association tested cross-frame graph matching/Sinkhorn, and temporal context tested past-vs-current features. Here no graph, memory, new feature or capacity is added: change only the same frame candidate/null learning objective. Failed coefficient descriptor is not advanced.',
      sources=[dict(url='https://arxiv.org/html/2103.16556v2',section='3.3,3.5,3.6',support='Explicit missing-match option and assignment negative log likelihood; our same-frame useful-action marginal is an adaptation, not KeepTrack reproduction.')],
      gates='Unit loss/padding/permutation/null gradients; full18controls replay; original32loss<=1e-6; conservative2x first32-step projected18models plus120s fits1800s; sources/caches/weights final hashes; independent controller and visible console.',
      selection=dict(chosen='Joint candidate/null loss: existing arrays and unchanged model; one cheap loss intervention before any additional feature extraction.',alternatives=selection,reason='Most listed concrete implementations did not meet criteria. Another expensive core/descriptor run has weaker immediate decision value; replayed CPU loss change is cheap and directly tests competition/calibration without claiming it will work.'),
      prohibitions='No GPU/core training/downloads/dev/HOTA/heldout75, no modification of frozen sources, no best seed or threshold chosen on26train.',sha256={**base['sha256'],**{n:sha(ROOT/n) for n in files}})
    write(RUN/'protocol.json',p)
    glossarypath=ROOT/'experiments/metrics_glossary.json';g=read(glossarypath)
    assert 'known_useful_selected_fraction' not in g['metrics']
    g['metrics']['known_useful_selected_fraction']=dict(title='Доля полезных среди выбранных с известным исходом',ru='В каждом видео выбираются ceil(0,05*N) строк из всех N по баллу считывателя без использования исхода. Доля полезных считается среди выбранных строк с известной разметкой; затем среднее по видео.',better='higher',unit='доля',range=[0,1],tool='tracking/z13_competing_candidates.py:quota_precision',caveat='Не HOTA и не гарантия отказа. Неизвестные исходы отдельно; изменение их доли может влиять на сравнение. Квота по каждому видео отличается от прежнего общего5%отбора.',source='tracking/z13_competing_candidates.py:quota_precision');write(glossarypath,g)
    assert IDENT not in cards
    cat['experiments'].append(dict(id=IDENT,title='З13: совместный выбор кандидата или отсутствия полезного ответа',area='flyvis',state='running',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','engineering-future'],question='Улучшит ли одна замена функции ошибки выбор полезных вмешательств при той же архитектуре и информации?',hypothesis='Конкуренция кандидатов с вариантом отсутствия полезного ответа может улучшить ранжирование уверенности; преимущества над BCE заранее нет.',expected=p['action']+' '+p['matched_control'],acceptance=[dict(criterion=p['gates'],met=None),dict(criterion='Основной Dприрост точности среди известных при одинаковом покадровом охвате5%по видео>=5п.п. и нижняя95%ДИ>0;26видео имеют определённую метрику.',met=None),dict(criterion='Dсовместный превосходит соответствующие геометрию и R1 по той же метрике с нижней95%ДИ>0; средняя доля полезных центровDне ниже BCE.',met=None)],summary='Критерии и сравнения зафиксированы до результатов;дешёвая CPUступень до1800с.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument']+' Повторно использованные26train;неполнаяKeepTrack,неHOTA.',metrics=[],commits=[],files=files[:3],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Метод,критерии,сравнения и хеши до прогона')]))
    stock=cards['z13_matched_stock_controls_v1'];oldsummary=stock.get('summary','');head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    if oldsummary:stock.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field='summary',old=oldsummary,reason='Прежний бюджет сохранён;неполный запуск остановлен технической ценой.'))
    stock.update(state='inconclusive',done=f'{now():%Y-%m-%d}',summary='Частичный результат:5из35заданий,цена полного остатка не помещается в исходный бюджет;вывода об эффективности нет.',obtained='runs/z13_matched_trackers_20261024/stock_recovery1/price.json: conservative_remaining_seconds=10509,18;available_seconds=8938,789432;proceed=false. controller_result.json:exit1,all_owned_ended=true;готовы5маркеров/jobs,исходыC/B/R1по7видео не измерены.',interpretation='АдаптерR1восстановлен и полный первый ролик сохранён;технический критерий цены не пройден. Это не отрицательный научный итог контролей.',decision='Сохранить5готовыхзаданий и остаток30,не продлевать исходный бюджет10800с. Не строить неполную таблицу10систем. Новая дешёваяCPUгипотеза регистрируется отдельной карточкой,не как продолжение бюджетаstock.')
    stock['acceptance'][0].update(met=False,evidence='runs/z13_matched_trackers_20261024/stock_recovery1/price.json;controller_result.json');write(catpath,cat)
    journal('новая ступень: совместная ошибка кандидатов и отказа','*Постановка.* '+p['action']+'\n\n*Критерии до результата.* '+p['primary']+'\n\n*Контроль и ограничения.* '+p['matched_control']+' '+p['counterargument']+'\n\nПервоисточник KeepTrack§3.3/3.5/3.6,https://arxiv.org/html/2103.16556v2;заимствован только принцип отсутствующей пары/NLL,это не полнаяKeepTrack. Цена32шагов до18новыхCPUмоделей,не более1800с;старые18контролейвоспроизвести. Каталог и словарь метрики заполнены до старта. Stockпослеполного первого ролика остановлен своимкритериемцены:10509,18с против8938,789432с,5/35заданий сохранены,полного научного вывода нет;исходныйбюджетнепродлён.')


def train(torch,model,tensors,nref,seed,weight,maxsteps=None,progress=None):
    from tracking.z13_matched_data_budget import matched_batches
    from tracking.z13_competing_candidates import marginal_choice_loss
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);losses=[];presented=0
    for step,indices in enumerate(matched_batches(len(tensors[0]),nref,50,128,seed)):
        if maxsteps is not None and step>=maxsteps:break
        if progress and step%23==0:progress(step)
        ix=torch.as_tensor(indices,device='cpu');x,s,g,mask,y=[t[ix] for t in tensors]
        loss=marginal_choice_loss(torch,model(x,s,g,mask),y,mask,weight);assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();losses.append(float(loss.detach()));presented+=len(indices)
    return losses,presented


def worker():
    import gc,numpy as np,torch
    torch.cuda.is_available=lambda:False;torch.set_default_device('cpu');torch.set_num_threads(2)
    from tracking.z13_candidate_ranker import make_ranker
    from tracking.z13_competing_candidates import choice_rows,quota_precision
    from scripts.z13_candidate_ranker_20261021 import pack,predict,make_rows
    from scripts.z13_continue import paired,per_video_useful
    import scripts.z13_candidate_identity_20261021 as reader
    p=read(RUN/'protocol.json');P.verify_sources(p);l20=read(CACHE/'fit/labels.json');extra=read(OLD/'evaluate/labels.json');labels=l20+extra;ev=read(CACHE/'evaluate/labels.json');ids=np.asarray([i for i,l in enumerate(labels) if l['kind']>=0]);nref=sum(l['kind']>=0 for l in l20);assert len(ids)==20778 and nref==2919
    data={};results={};manifest={};replay={};done=0
    for arm in ARMS:
        status('Воспроизведение прежнего считывания',variant=arm,models_done=0)
        seq,m=reader.load('fit',arm);manifest.update(m);a=pack(l20,seq);saved=reader.CACHE
        try:reader.CACHE=OLD;seq,m=reader.load('evaluate',arm)
        finally:reader.CACHE=saved
        manifest.update(m);b=pack(extra,seq);ta=tuple(np.concatenate([x,y]) for x,y in zip(a,b));seq,m=reader.load('evaluate',arm);manifest.update(m);ea=pack(ev,seq);data[arm]=(ta,ea)
        for mode,activity in (('activity',True),('geometry',False)):
            predictions=[]
            for seed in (0,1,2):
                model=make_ranker(torch,activity);model.load_state_dict(torch.load(OLD/arm/mode/f'seed{seed}/model.pt',map_location='cpu',weights_only=True));predictions.append(predict(torch,model,ea))
            logits=np.mean(predictions,0);rr=make_rows(ev,ea,logits);old=read(OLD/arm/mode/'ensemble_rows.json');delta=max(abs(a['p_present']-b['p_present']) for a,b in zip(rr,old));assert len(rr)==len(old)==8326 and delta<=1e-5
            assert all(a['center']==b['center'] and a['candidate']==b['candidate'] and a.get('useful')==b.get('useful') for a,b in zip(rr,old))
            replay[arm+'_'+mode]=dict(rows=len(rr),max_abs_score=delta,centers_equal=True);results[arm+'_bce_native_'+mode]=old;results[arm+'_bce_common_'+mode]=choice_rows(ev,ea,logits)
        del seq,a,b;gc.collect()
    write(RUN/'reproduction.json',replay);write(RUN/'cache_manifest.json',manifest)
    tensors=[torch.as_tensor(a[ids],device='cpu') for a in data['D'][0][:5]];ref=read(OLD/'D/activity/seed0/training.json');torch.manual_seed(0);model=make_ranker(torch,True);loss,_=P.train_steps(torch,model,tensors,nref,0,ref['pos_weight'],maxsteps=32);delta=max(abs(a-b) for a,b in zip(loss,ref['loss'][:32]));assert delta<=1e-6
    torch.manual_seed(0);model=make_ranker(torch,True);start=time.monotonic();train(torch,model,tensors,nref,0,ref['pos_weight'],maxsteps=32);seconds=time.monotonic()-start;estimate=2*seconds/32*1150*18+120
    write(RUN/'timing.json',dict(old32_loss_max_abs=delta,new32_seconds=seconds,conservative_seconds=estimate));assert estimate<p['budget_seconds']-(time.monotonic()-T0)
    seedmetrics={}
    for arm in ARMS:
        ta,ea=data[arm];tensors=[torch.as_tensor(a[ids],device='cpu') for a in ta[:5]]
        for mode,activity in (('activity',True),('geometry',False)):
            predictions=[];seedmetrics[arm+'_'+mode]={}
            for seed in (0,1,2):
                folder=RUN/arm/mode/f'seed{seed}';ref=read(OLD/arm/mode/f'seed{seed}/training.json');torch.manual_seed(seed);model=make_ranker(torch,activity)
                if (folder/'done.json').exists():assert sha(folder/'model.pt')==read(folder/'training.json')['model_sha256'];model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
                else:
                    loss,n=train(torch,model,tensors,nref,seed,ref['pos_weight'],progress=lambda step:status('Обучение совместного выбора и отсутствия полезного ответа',variant=arm+' '+mode+' seed'+str(seed),models_done=done,done=step,total=1150));assert len(loss)==1150 and n==145950
                    folder.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),folder/'model.pt');write(folder/'training.json',dict(seed=seed,steps=1150,presented_rows=n,pos_weight=ref['pos_weight'],loss=loss,parameters=sum(p.numel() for p in model.parameters()),model_sha256=sha(folder/'model.pt')));write(folder/'done.json',dict(at=now().isoformat()))
                logits=predict(torch,model,ea);predictions.append(logits);rr=choice_rows(ev,ea,logits);write(folder/'rows.json',rr);seedmetrics[arm+'_'+mode][str(seed)]=dict(precision=quota_precision(rr)[0],centers=per_video_useful(rr));done+=1
            key=arm+'_joint_'+mode;results[key]=choice_rows(ev,ea,np.mean(predictions,0));write(RUN/arm/mode/'ensemble_rows.json',results[key])
        gc.collect()
    precision={k:quota_precision(v)[0] for k,v in results.items()};details={k:quota_precision(v)[1] for k,v in results.items()};centers={k:per_video_useful(v) for k,v in results.items()};comparisons={};centercomp={}
    for arm in ARMS:
        for baseline in (arm+'_bce_common_activity',arm+'_bce_native_activity',arm+'_joint_geometry'):
            key=arm+'_joint_activity-'+baseline;comparisons[key]=paired(precision[arm+'_joint_activity'],precision[baseline]);centercomp[key]=paired(centers[arm+'_joint_activity'],centers[baseline])
    for arm in ('D','C'):
        key=arm+'_joint_activity-R1_joint_activity';comparisons[key]=paired(precision[arm+'_joint_activity'],precision['R1_joint_activity']);centercomp[key]=paired(centers[arm+'_joint_activity'],centers['R1_joint_activity'])
    main=comparisons['D_joint_activity-D_bce_common_activity'];controls=['D_joint_activity-D_joint_geometry','D_joint_activity-R1_joint_activity'];complete=all(comparisons[k]['n']==26 for k in ['D_joint_activity-D_bce_common_activity',*controls]);criteria=dict(complete_metrics=complete,precision_main=main['mean_per_video']>=.05 and main['ci95'][0]>0,fly_controls=all(comparisons[k]['ci95'][0]>0 for k in controls),centers_not_lower=centercomp['D_joint_activity-D_bce_common_activity']['mean_per_video']>=0)
    P.verify_sources(p);assert all(sha(ROOT/n)==h for n,h in manifest.items())
    summary={k:dict(known_precision_mean_video=float(np.mean(list(precision[k].values()))),known_precision_videos=len(precision[k]),useful_centers_mean_video=float(np.mean(list(centers[k].values())))) for k in results}
    write(RUN/'seed_metrics.json',seedmetrics);write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired_precision=comparisons,paired_centers=centercomp,summary=summary,precision_per_video=precision,centers_per_video=centers,quota_details=details,limitations=p['counterargument']+' Exploratory reused26train;no HOTA or calibrated rejection.'))
    status('18моделей и парные итоги сохранены',models_done=18)
    write(RUN/'status.json',dict(state='completed',stage='Совместная ошибка: расчёт завершён, требуется проверка',updated=now().isoformat(),models_done=18,models_total=18))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
