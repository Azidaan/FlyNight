"""Registered CPU comparison of cached readouts; frozen core is never imported."""
import os,sys,json,time,csv,gzip,subprocess,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
OUT=Path('runs/flyvis_readout_comparison_20261107')
BASE=Path('runs/z13_region_assignment_20261025/recovery2/features')
PRIOR=Path('runs/flyvis_exclusive_t4t5_20261107')
META=Path('runs/z13_temporal_pair_readiness_20261025/inputs')
GUARD=ROOT/'runs/flyvis_repairs_20261106/unification_20261107'
IDENT='flyvis_cached_readout_comparison_v1'
KINDS=('linear','mlp','products','sigma_pi')

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.pending');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(temp,p)
def stamp():return datetime.now(timezone(timedelta(hours=3))).isoformat()
def status(stage,**values):write(OUT/'status.json',dict(stage=stage,updated=stamp(),**values))

def prepare():
    assert not OUT.exists()
    old=read(PRIOR/'protocol.json');videos=old['videos']
    assert all(not(set(videos[a])&set(videos[b])) for a,b in [('fit84','fit20'),('fit84','evaluate26'),('fit20','evaluate26')])
    sources=['tracking/flyvis_cached_readouts.py','tests/test_flyvis_cached_readouts.py',Path(__file__).relative_to(ROOT).as_posix(),
        'scripts/audit_flyvis_readout_comparison_20261107.py','scripts/chain_flyvis_readout_comparison_20261107.py','scripts/check_flyvis_readout_comparison_20261107.py',str(GUARD/'system_memory_guard.py'),
        str(PRIOR/'choices.csv.gz'),str(PRIOR/'protocol.json'),str(PRIOR/'measurement.json'),str(PRIOR/'inputs_used.json'),str(BASE/'report.json'),
        'runs/flyvis/results/flow/0000/000/best_chkpt','tracking/z13_readout.py','tracking/flyvis_motion.py',
        'runs/flyvis_two_frame_input_20261107/Portsev_pu_426.pdf']
    p=dict(id=IDENT,registered=stamp(),git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        authorization='Давай проверим считыватели на сохранённых признаках, ядро FlyVis не меняем. Если будет подтверждённая прибавка в выборе машины — проверяем лучший вариант в полном трекере.',
        scope='Readout only, original frozen C flow/0000/000 best. Reuse original common-map caches, no core/detector/retina/video/GPU inference. Historical research video splits; not a fresh independent final test and not HOTA.',
        hypothesis='Learned combination of the existing eight observable T4/T5 channel contributions improves target choice relative to their frozen sum. Sigma-Pi is one competing head, not a presupposed winner.',
        selection='All prior exactly2==1,label_reason==ok rows. Do NOT require nonempty exclusive support: whole-box readout does not delete common cells. Exclude unknown identities and ambiguous/both/neither target, report counts. Original172 subset only reproduces old result, not model selection.',
        inputs='g[c]=saved pooled standardized activity of candidate in T4/T5 channel c TIMES saved observed-exemplar coefficient c. Eight g values per candidate. Frozen old score=sum(g). Each candidate scored independently by same head: no candidate index/GT/video/frame/identity/geometry as model input. Original full candidate masks; no exclusive support or new crops.',
        heads=dict(baseline='sum(g)',linear='sum(g)+linear(z),9 parameters',mlp='sum(g)+linear(tanh(linear8to2(z))),21 parameters',
            products='sum(g)+linear([z,28 fixed distinct pair-products]),37 parameters',
            sigma_pi='sum(g)+(1-sigmoid(gamma))*(w.z+b)+sigmoid(gamma)*m*product(1-sigmoid(alpha_i)+sigmoid(alpha_i)*z_i),19 parameters; paper equations6-8, residual adapter'),
        training=dict(seeds=[42,43,44],regularization=[0.0,.001,.01],updates=600,batch=128,optimizer='Adam',lr=.01,gradient_clip_norm=10,
            dtype='float64 CPU for all heads',normalization='mean/sd of fit84 candidate g only,sd floor1e-8; no clipping',
            loss='pair BCE(scoreA-scoreB,target_side==1), mean minibatch + .5*lambda*sum(all parameter squares)',
            init='All residual outputs exactly0. MLP input weights N(0,.1), final weights0; sigma alpha N(0,.05), all other parameters0. Same sampled minibatch indices within seed across heads/regularization. No early stopping, no quality-based architecture changes.'),
        videos=videos,fit='fit84 only for gradients;fit20 only for regularizer/family selection;evaluate26 only after selection.json frozen',
        choice='For each family/reg, average scores of3 seeds. Select reg by dev correct fraction, then lower pair BCE, then larger regularization. Select ONE family by same dev rules, then fewer parameters, then fixed family order. Save selection.json BEFORE loading evaluation inputs or labels. No replacement by a better test family if selected fails.',
        primary='Four ensemble heads versus old sum, pooled paired correct-choice difference. Original26-video cluster bootstrap20000/seed42, ordinary95% and Bonferroni98.75%. Operational gate: DEV-selected head lower98.75%>0 and all3 seed deltas>=0. No test retuning or fallback selection.',
        sigma_secondary='Sigma-Pi minus linear/mlp/products, separate predeclared family3 with Bonferroni98.333333%; only positive lower on all3 supports a Sigma-specific benefit. These do not change operational gate.',
        technical='All caches/manifests/UID/video/candidate bindings exact; original172 baseline104/ties0/rank exact, margin abs error<=2e-5; zero residual reproduces baseline; candidate permutation equivariant; no CUDA. Independent NumPy prediction repeat from saved parameters<=1e-9 and every discrete choice exact; independent bootstrap and labels recomputed from saved CSV.',
        limitation='Only eligible two-candidate known-target historical examples, not missing/unknown objects or all tracker decisions. Eight contributions retain fixed spatial averaging. Better head cannot recover absent information; capacity/ordinary interactions can explain benefit. All repeated research videos remain exploratory.',
        conditional_tracker='Only if operational gate passes, prepare and run a separate prospectively frozen matched full tracker experiment with the already dev-selected ensemble; original detector/feature preparation/query timing and controls. Compare native, original adapter, selected adapter and a matched no-core comparator. First full-video resource/price check, exact independent replay and paired HOTA/AssA/IDF1/FP/FN/IDSw. No full tracker run after failed gate; exact implementation protocol must precede new outputs.',
        resource=dict(device='CPU only;CUDA_VISIBLE_DEVICES empty before Torch',threads=2,guard='external reserve max(6GiB,20% physical),+2GiB at start;no arbitrary per-process cap',disk_reserve_gib=5,watchdog_s=2400,nearest_weekday_stop='2026-11-09T07:40:00+03:00'),
        literature=dict(url='http://pu.mtas.ru/archive/Portsev_pu_426.pdf',pages='18 equations6-8;21 experimental selection;25 limitations',interpretation='Arithmetic factor multiplication is not object separation; published evidence is not tracking benefit'),
        source_sha256={s:sha(s) for s in sources})
    OUT.mkdir();(OUT/'models').mkdir();(OUT/'predictions').mkdir()
    (OUT/'.gitignore').write_text('*.npz\nmodels/\npredictions/\nvault_check_copy/\n',encoding='utf8')
    write(OUT/'protocol.json',p)
    cat=read('experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Считыватели T4/T5 — сумма, обычная сеть и ΣΠ при неизменном ядре',area='flyvis',state='ready',started=p['registered'][:10],
        question='Улучшает ли новый считыватель выбор продолжения объекта на прежних признаках без изменения ядра?',hypothesis=p['hypothesis'],expected='Выбранный на fit20 вариант проходит основной критерий; тогда отдельная проверка полного трекера.',
        acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None),dict(criterion=p['sigma_secondary'],met=None)],
        summary='',obtained='',interpretation='',decision='',cannot_claim=p['limitation'],confirmed_results=[],
        evidence=[dict(kind='file',ref=(OUT/'protocol.json').as_posix(),claim='Критерии, неизменные входы и переход к полному трекеру записаны до обучения')]))
    write('experiments/project_history.json',cat)
    plan=Path('dissertation/ЕДИНЫЙ_ПЛАН.md');text=plan.read_text(encoding='utf8')
    entry=f'\n**{datetime.now(timezone(timedelta(hours=3))):%d.%m.%Y, %H:%M} MSK** — `{IDENT}` зарегистрирован по прямому поручению автора. *Постановка.* Четыре обучаемых остаточных считывателя против прежней суммы восьми вкладов T4/T5; ядро и кэш неизменны. Fit84/fit20/evaluate26 по разным видео; один победитель фиксируется по dev до оценки. *Критерии.* {p["primary"]} Только после pass разрешён полный трекер; ΣΠ не назначается победителем заранее. Протокол:`{OUT}/protocol.json`.\n'
    plan.write_text(text.replace('## Журнал изменений','## Журнал изменений\n'+entry,1),encoding='utf8')
    status('registered')

def verify(p):
    for s,h in p['source_sha256'].items():assert sha(s)==h,s

def data(obs):
    import numpy as np
    p=read(OUT/'protocol.json');verify(p)
    assert read(PRIOR/'measurement.json')['rows_sha256']==sha(PRIOR/'choices.csv.gz')
    source_report=read(BASE/'report.json');assert source_report['complete'] and source_report['all_masks_and_choices_equal']
    old_bindings=read(PRIOR/'inputs_used.json'); truth={};excluded={}
    with gzip.open(PRIOR/'choices.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            if row['exactly2']!='1':continue
            key=row['split']
            if row['label_reason']!='ok':
                excluded[(key,row['label_reason'])]=excluded.get((key,row['label_reason']),0)+1;continue
            uid=int(row['uid']);assert uid not in truth
            truth[uid]=dict(row,uid=uid)
    binds={};counts={};regression=[];began=time.perf_counter()
    for split,folder in [('fit84','fit84'),('fit20','fit20'),('evaluate26','train26')]:
        desired={u:r for u,r in truth.items() if r['split']==split}; meta={}
        for video in p['videos'][split]:
            file=META/(video+'.json');assert sha(file)==old_bindings[file.as_posix()];binds[file.as_posix()]=sha(file)
            for row in read(file)['rows']:
                if row['uid'] in desired:meta[row['uid']]=(video,row['current_choices'])
        assert set(meta)==set(desired)
        arrays={k:[] for k in ['g','activation','target','uid','video','y','legacy_margin','regression172']};seen=set()
        files=sorted((BASE/'C'/folder).glob('chunk*.npz'))
        for number,file in enumerate(files):
            marker=file.with_suffix('.json');key=marker.as_posix()
            assert sha(marker)==source_report['manifest'][key]
            assert sha(file)==read(marker)['sha256'];binds[key]=sha(marker);binds[file.as_posix()]=sha(file)
            with np.load(file,allow_pickle=False) as z:
                uid=z['uid'];slots=[(i,int(u)) for i,u in enumerate(uid) if int(u) in desired]
                if slots:
                    x=z['x'];w=z['seed'];m=z['mask'];v=z['video']
                    for i,u in slots:
                        row=desired[u];video,choices=meta[u];assert u not in seen;seen.add(u)
                        assert choices==[int(row['j']),int(row['k'])] and int(m[i].sum())==3
                        assert video==row['video']==p['videos'][split][int(v[i])]
                        activation=x[i,1:3,4:12].astype('float64');target=w[i,4:12].astype('float64');g=activation*target
                        assert np.isfinite(g).all()
                        old_margin=float(row['whole_margin']);margin=g[0].sum()-g[1].sum()
                        reg=split=='evaluate26' and bool(row['exclusive_choice'])
                        if reg:
                            assert abs(margin-old_margin)<=2e-5 and int(margin>1e-10)-int(margin< -1e-10)==int(row['whole_choice'])
                            regression.append((margin,int(row['target_side'])))
                        for k,value in dict(g=g,activation=activation,target=target,uid=u,video=p['videos'][split].index(video),y=int(row['target_side'])==1,legacy_margin=old_margin,regression172=reg).items():arrays[k].append(value)
            if number%50==0:status('prepare_cached_pairs',split=split,done=number+1,total=len(files),guard=obs.read())
        assert seen==set(desired)
        order=np.argsort(arrays['uid']);arrays={k:np.asarray(v)[order] for k,v in arrays.items()}
        with (OUT/('data_'+split+'.npz')).open('xb') as stream:np.savez_compressed(stream,**arrays)
        counts[split]=dict(pairs=len(seen),videos=len(np.unique(arrays['video'])))
    assert len(regression)==172 and sum(m*s>1e-10 for m,s in regression)==104
    verify(p)
    write(OUT/'data_prepared.json',dict(passed=True,counts=counts,excluded={'/'.join(k):v for k,v in excluded.items()},source_sha256=binds,
        data_sha256={s:sha(OUT/('data_'+s+'.npz')) for s in counts},regression172_correct=104,seconds=time.perf_counter()-began,guard=obs.read(),protocol_sha256=sha(OUT/'protocol.json')))
    print('Data ready',counts,flush=True)

def train(obs):
    import numpy as np
    import torch
    from tracking.flyvis_cached_readouts import make_model,export_parameters
    p=read(OUT/'protocol.json');verify(p);pack=read(OUT/'data_prepared.json');t=p['training']
    for s,h in pack['data_sha256'].items():assert sha(OUT/('data_'+s+'.npz'))==h
    with np.load(OUT/'data_fit84.npz') as z:fit=z['g'];labels=z['y'].astype('float64')
    with np.load(OUT/'data_fit20.npz') as z:dev=z['g'];devy=z['y'].astype('float64');devuid=z['uid']
    mu=fit.reshape(-1,8).mean(0);sd=np.maximum(fit.reshape(-1,8).std(0),1e-8)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_default_device('cpu')
    tensors=[torch.tensor(v,dtype=torch.float64,device='cpu') for v in [fit,labels,dev,mu,sd]]
    fg,fy,dg,tm,ts=tensors
    manifest={};configs=[];before=time.perf_counter();first=None
    for kind in KINDS:
        for reg in t['regularization']:
            predictions=[]
            for seed in t['seeds']:
                key=f'{kind}_l{reg:g}_s{seed}';path=OUT/'models'/(key+'.npz');done=path.with_suffix('.json')
                if done.exists():
                    receipt=read(done);assert sha(path)==receipt['sha256']
                    with np.load(OUT/'predictions'/(key+'_dev.npz')) as z:prediction=z['scores']
                else:
                    assert not path.exists()
                    model=make_model(torch,kind,seed);initial=model(fg,tm,ts)
                    assert torch.equal(initial,fg.sum(-1))
                    opt=torch.optim.Adam(model.parameters(),lr=t['lr']);rng=np.random.default_rng(seed)
                    started=time.perf_counter();losses=[]
                    for step in range(t['updates']):
                        indices=rng.integers(0,len(fit),size=t['batch']);x=fg[indices];y=fy[indices]
                        opt.zero_grad(set_to_none=True);scores=model(x,tm,ts)
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(scores[:,0]-scores[:,1],y)
                        loss=loss+.5*reg*sum((v*v).sum() for v in model.parameters())
                        assert torch.isfinite(loss)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip_norm'],error_if_nonfinite=True);opt.step()
                        if step%100==0:losses.append(float(loss.detach()))
                    with torch.no_grad():prediction=model(dg,tm,ts).numpy()
                    assert np.isfinite(prediction).all() and not torch.cuda.is_initialized()
                    with path.open('xb') as stream:np.savez_compressed(stream,**export_parameters(model),mu=mu,sd=sd)
                    with (OUT/'predictions'/(key+'_dev.npz')).open('xb') as stream:np.savez_compressed(stream,scores=prediction,uid=devuid)
                    receipt=dict(sha256=sha(path),kind=kind,seed=seed,regularization=reg,parameters=sum(v.numel() for v in model.parameters()),seconds=time.perf_counter()-started,loss_trace=losses,initial_identity_exact=True,cuda_initialized=False)
                    write(done,receipt)
                    if first is None:
                        first=dict(seconds=receipt['seconds'],conservative_remaining_s=2*receipt['seconds']*35,guard=obs.read());write(OUT/'first_model.json',first)
                manifest[key]=receipt;predictions.append(prediction)
                status('train_cached_readouts',done=len(manifest),total=36,variant=key,seconds=time.perf_counter()-before,guard=obs.read())
            mean=np.mean(predictions,axis=0);margin=mean[:,0]-mean[:,1];sign=np.where(abs(margin)<=1e-10,.5,(margin>0).astype(float))
            correct=np.where(sign==.5,.5,(sign==devy).astype(float));loss=float(np.mean(np.logaddexp(0,margin)-devy*margin))
            configs.append(dict(kind=kind,regularization=reg,dev_accuracy=float(correct.mean()),dev_bce=loss,parameters=receipt['parameters'],keys=[f'{kind}_l{reg:g}_s{s}' for s in t['seeds']]))
    best={kind:sorted([c for c in configs if c['kind']==kind],key=lambda c:(-c['dev_accuracy'],c['dev_bce'],-c['regularization']))[0] for kind in KINDS}
    winner=sorted(best.values(),key=lambda c:(-c['dev_accuracy'],c['dev_bce'],c['parameters'],KINDS.index(c['kind'])))[0]
    write(OUT/'selection.json',dict(selected=winner,best_per_family=best,all_configs=configs,manifest=manifest,protocol_sha256=sha(OUT/'protocol.json'),selected_before_evaluation=True))
    # Evaluation inputs are first opened ONLY after the immutable dev choice above.
    with np.load(OUT/'data_evaluate26.npz') as z:eg=z['g'];eu=z['uid']
    et=torch.tensor(eg,dtype=torch.float64,device='cpu');frozen=sha(OUT/'selection.json');predicted={}
    for kind,config in best.items():
        for key in config['keys']:
            receipt=manifest[key];file=OUT/'models'/(key+'.npz');assert sha(file)==receipt['sha256']
            model=make_model(torch,kind,receipt['seed'])
            with np.load(file) as z:
                model.load_state_dict({k:torch.tensor(z[k],dtype=torch.float64) for k,_ in model.named_parameters()})
            with torch.no_grad():scores=model(et,tm,ts).numpy()
            dest=OUT/'predictions'/(key+'_eval.npz')
            with dest.open('xb') as stream:np.savez_compressed(stream,scores=scores,uid=eu)
            predicted[key]=sha(dest)
    assert sha(OUT/'selection.json')==frozen and not torch.cuda.is_initialized()
    verify(p)
    write(OUT/'training_completed.json',dict(passed=True,models=len(manifest),selection_sha256=frozen,evaluation_predictions_sha256=predicted,seconds=time.perf_counter()-before,guard=obs.read(),cuda_initialized=False))
    print('Training complete; selection frozen before evaluation',winner,flush=True)

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
        sys.path.insert(0,str(GUARD));from system_memory_guard import guard_entrypoint
        observation=guard_entrypoint(timeout=2400)
        if '--data' in sys.argv:data(observation)
        elif '--train' in sys.argv:train(observation)
        else:raise ValueError('Explicit stage required')
