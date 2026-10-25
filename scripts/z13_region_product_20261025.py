"""Bounded paired CPU experiment: preserve the original 68-channel target interaction."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import hashlib
import subprocess
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_context_20261025 as context
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_region_product import product_scale,product_features,make_product_ranker
RUN=ROOT/'runs/z13_region_product_20261025'
PREVIOUS=context.RUN
IDENT='z13_region_product_v1'
VARIANTS=tuple(a+'_'+m for a in ('D','C','R1','geometry') for m in ('product','blank'))


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)


def training_source():
    code=context.training_source()
    old='from tracking.z13_region_assignment_context import make_context_ranker as make_ranker'
    assert code.count(old)==1
    return code.replace(old,'from tracking.z13_region_product import make_product_ranker as make_ranker').replace('load_context_split','load_product_split')


def comparisons(means,videos):
    from scripts.z13_continue import paired
    out={'Dproduct-baseline':paired(means['D_product'],{v:0. for v in videos})}
    for variant in ('D_blank','C_product','R1_product','geometry_product'):
        out['Dproduct-'+variant]=paired(means['D_product'],means[variant])
    out['Dproduct-Dprevious']=paired(means['D_product'],read(PREVIOUS/'training/report.json')['mean_seed_per_video']['D_context'])
    return out


def primary(comp):
    return comp['Dproduct-baseline']['mean_per_video']>=.02 and comp['Dproduct-D_blank']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values())


def status(phase,stage,**kw):
    p=read(RUN/'protocol.json')
    if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original product stage budget ended')
    write(RUN/'training/status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=time.monotonic()-START,**kw))


def build_loader(torch,normalizations=None):
    ns=context.original.namespace();base=ns['load_split'];entries={};scales={}
    report=read(PREVIOUS/'cache/report.json');assert report['complete'] and report['all_raw_replays_exact']
    for video,h in report['manifest'].items():
        path=PREVIOUS/'cache/videos'/(video+'.json');assert sha(path)==h
        for uid,entry in read(path)['entries'].items():
            assert int(uid) not in entries;entries[int(uid)]=entry
    def load(torch,arm,split,mode):
        d=base(torch,arm,split)
        if arm not in scales:
            fit=d if split=='fit84' else base(torch,arm,'fit84')
            scales[arm]=product_scale(fit['x'].numpy(),fit['seed'].numpy(),fit['mask'].numpy())
            path=RUN/'normalization'/(arm+'.json')
            value=dict(source='fit84 observable original products only; no label/evaluation fitting',scale=scales[arm].tolist(),channels=68)
            if normalizations is None:
                if path.exists():assert read(path)==value
                else:write(path,value)
            else:assert value==normalizations[arm]
        n,w=d['mask'].shape;extra=torch.zeros((n,w,74),dtype=torch.float32)
        for i,uid in enumerate(d['uid'].tolist()):
            count=int(d['mask'][i].sum());value=entries[uid]['context'];assert len(value)==count
            extra[i,:count,:6]=torch.as_tensor(value,dtype=torch.float32)
        if mode=='product':extra[:,:,6:]=torch.from_numpy(product_features(d['x'].numpy(),d['seed'].numpy(),scales[arm]))
        else:assert mode=='blank'
        d['geo']=torch.cat([d['geo'],extra],-1)
        return d
    return ns,load


def worker():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=read(RUN/'protocol.json');verify_sources(p)
    ns,loader=build_loader(torch)
    def load_product_split(torch,arm,split,mode):return loader(torch,arm,split,mode)
    # Geometry uses the identical tensors, but all target/activity-derived inputs are zero.
    code=training_source();old="data={s:load_product_split(torch,arm,s,mode) for s in ('fit84','fit20','train26')}"
    assert code.count(old)==1
    code=code.replace(old,old+"\n        if base_variant=='geometry':\n            for one in data.values():one['geo'][...,14:]=0.")
    assert hashlib.sha256(code.encode()).hexdigest()==p['executed_training_sha256']
    ns.update(RUN=RUN,VARIANTS=VARIANTS,load_product_split=load_product_split,signed_action_loss=context.signed_action_loss,status=status,comparisons=comparisons,primary=primary)
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['train_worker']()


def executed_source():
    code=training_source();old="data={s:load_product_split(torch,arm,s,mode) for s in ('fit84','fit20','train26')}"
    assert code.count(old)==1
    return code.replace(old,old+"\n        if base_variant=='geometry':\n            for one in data.values():one['geo'][...,14:]=0.")


def prepare():
    import torch
    assert not RUN.exists() and read(PREVIOUS/'publication.json')['push_completed']
    assert not read(PREVIOUS/'training/report.json')['primary_pass']
    ps=WindowsProcesses()
    for phase in ('cache','training'):
        c=read(PREVIOUS/phase/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining']
        assert all(ps.creation(int(pid))!=birth for pid,birth in c['owned'].items())
    torch.set_num_threads(2);torch.set_default_device('cpu');torch.manual_seed(0)
    model=make_product_ranker(torch);x=torch.ones(2,2,68);seed=torch.ones(2,68);geo=torch.zeros(2,2,82);mask=torch.ones(2,2,dtype=torch.bool)
    score=model(x,seed,geo,mask);loss=context.signed_action_loss(torch,score,torch.zeros_like(mask),torch.zeros_like(mask),mask);loss.backward()
    assert bool(loss.isfinite()) and all(bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None) and not torch.cuda.is_initialized()
    start=now();stop=min(start+timedelta(seconds=1800),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    preflight=ROOT/'runs/z13_region_product_execution_check_20261025/receipt.json'
    assert read(preflight)['all24_synthetic_variant_seed_paths_passed'] and read(preflight)['code_sha256']==hashlib.sha256(executed_source().encode()).hexdigest()
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_product_console.ps1','tracking/z13_region_product.py','tests/test_z13_region_product.py',
        'scripts/z13_region_product_execution_check_20261025.py',preflight.relative_to(ROOT).as_posix(),
        'scripts/z13_region_context_20261025.py','tracking/z13_region_assignment_context.py','tracking/z13_filter_descriptor.py',
        'runs/z13_region_context_20261025/training/report.json','runs/z13_region_context_20261025/cache/report.json']
    names += [(PREVIOUS/'cache/videos'/(v+'.json')).relative_to(ROOT).as_posix() for v in read(PREVIOUS/'cache/report.json')['manifest']]
    p=dict(read(PREVIOUS/'protocol.json'),id=IDENT,created=start.isoformat(),device='cpu',budget_seconds=1800,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),
        stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),variants=list(VARIANTS),
        action='Add direct68 channelwise products between original normalized pooled query activity and its unchanged observed-only ridge coefficients, before the two independent68->16 embeddings. Same prior6 assignment inputs remain for all arms. Equal blank controls add exactly68 zeros to the same162->32->1 network.',
        scaling='Per-original-channel RMS over all valid observed fit84 regions, floor1e-6; no centering or labels. Freeze before fit20/train26. Multiplying by saved scale and summing reconstructs the original linear filter response. Geometry zeros activity,target and added68 products, retaining identical geometry/context. All seeds/UID batches,1150steps,128batch,Adam0.001,signedMSE,threshold grid unchanged.',
        distinction='Not another loss or threshold: preserves original channel correspondence before independent learned compression. Earlier fixed region_mean used one scalar mean ridge response; this exposes68 separate contributions to a learned joint action model. No claim that previous embeddings provably destroyed information.',
        primary='Dproduct-baseline and Dproduct-Dblank means both>=2pp; all6paired-video95%CI lower>0 against baseline,blank,Cproduct,R1product,geometryproduct,previousDcontext.24models,3seeds.',
        technical='Reuse every frozen original activity/context/label cache; fit84-only RMS. No extraction or completed-model repetition. First full model2*elapsed*23+60 must fit original1800seconds. Independently reproduce every8326-row evaluation of all24models and paired intervals before catalog acceptance.',
        literature='Bertinetto et al.,ECCV2016,Fully-Convolutional Siamese Networks for Object Tracking,section2.1 equation2: https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto16fully.pdf . Direct channel-aligned correlation motivates retaining the analytic interaction; our query activations and ridge coefficients are a primal/dual pair, not two SiamFC image embeddings. No published improvement is transferred.',
        counterargument='Mean pooling may already discard the needed spatial information; adding products cannot restore it. A wider network can overfit rare assignment events, so equal-capacity blank and C/R1/geometry contrasts remain mandatory. Repeatedtrain26 and one-row labels are exploratory, not HOTA or causal proof.',
        next='Pass -> separately budgeted frozen closed-loop parity/price/HOTA with standardC and no-fly controls. Fail -> preserve this interaction implementation; do not enumerate further scalar losses or thresholds. Next action must examine the spatial pooling node using one preregistered cheap check before any new extraction.',
        executed_training_sha256=hashlib.sha256(executed_source().encode()).hexdigest(),sha256={**read(PREVIOUS/'protocol.json')['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'training/protocol.json',p)
    write(RUN/'smoke.json',dict(at=now().isoformat(),real162_input_forward_backward_finite=True,cuda_initialized=False))
    write(RUN/'status.json',dict(state='ready',stage='24 matched CPU readouts; cached inputs',updated=now().isoformat(),models_done=0,models_total=24))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Прямое поканальное сравнение модели цели и активности FlyVis',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Поможет ли сохранение исходного поканального взаимодействия перед независимым сжатием в считывателе?',
        hypothesis='Рабочая гипотеза: прямое взаимодействие68 исходных каналов полезнее, чем только произведение независимо обученных16мерных вложений. Это не установленная причина отказов.',
        expected=p['action'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Неизменные входы, парные контроли, критерии и бюджет до расчёта')],obtained='',interpretation='',decision='',cannot_claim=p['counterargument']))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка прямого поканального сравнения','*Постановка.* '+p['action']+' '+p['scaling']+'\n\n*Критерии.* '+p['primary']+'\n\n*Отличие.* '+p['distinction']+'\n\n*Сильнейшее возражение.* '+p['counterargument']+'\n\n*Бюджет.* CPU1800с+20с; save '+p['save_at']+', hard '+p['hard_stop']+'. НикакихGPU/dev/HOTA/heldout75/загрузок. Источник runs/z13_region_product_20261025/protocol.json. '+p['literature'])


def audit():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=read(RUN/'protocol.json');verify_sources(p);r=read(RUN/'training/report.json');assert r['complete'] and r['models']==24 and not r['cuda_initialized']
    norm={a:read(RUN/'normalization'/(a+'.json')) for a in ('D','C','R1')};ns,load=build_loader(torch,norm);means={};models={}
    for variant in VARIANTS:
        arm,mode=variant.rsplit('_',1);base='D' if arm=='geometry' else arm
        d=load(torch,base,'train26',mode)
        if arm=='geometry':d['geo'][...,14:]=0.
        per=[]
        for seed in p['seeds']:
            dest=RUN/'training'/variant/f'seed{seed}';done=read(dest/'done.json');report=read(dest/'report.json')
            assert sha(dest/'model.pt')==done['model_sha256'] and sha(dest/'report.json')==done['report_sha256'] and sha(dest/'rows.json')==report['rows_sha256']
            model=make_product_ranker(torch,activity=arm!='geometry').cpu();model.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
            logits=ns['predict'](torch,model,d);rows,one,counts=ns['policy_rows'](d,logits,report['tau'],p['evaluate26'])
            assert rows==read(dest/'rows.json') and one==report['per_video'] and counts==report['counts']
            assert len(rows)==8326 and sum(x['kind']>=0 for x in rows)==5822
            per.append(one);models[variant+'/'+str(seed)]=done
        means[variant]={v:float(np.mean([x[v] for x in per])) for v in p['evaluate26']}
    comp=comparisons(means,p['evaluate26']);assert comp==r['paired'] and primary(comp)==r['primary_pass'] and not torch.cuda.is_initialized()
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all24_models_reinferred_exact=True,rows_per_model=8326,known_per_model=5822,all_intervals_verified=True,report_sha256=sha(RUN/'training/report.json'),model_hashes=models))
    return r


def finish():
    ps=WindowsProcesses();c=read(RUN/'training/controller_result.json')
    assert c['exit_code']==0 and c['worker_ended'] and c['all_owned_ended'] and not c['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in c['owned'].items())
    r=audit();p=read(RUN/'protocol.json');passed=r['primary_pass'];cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT)
    fields=dict(summary='Прямое поканальное сравнение: критерий '+('пройден.' if passed else 'не пройден.'),obtained=(RUN/'training/report.json').relative_to(ROOT).as_posix()+' → paired: '+str(r['paired'])+';24модели,3сида,равный162входный нулевой контроль.',
        interpretation='Изменён внутренний узел сопоставления активности с моделью цели. Все первичные выходы24моделей воспроизведены независимо; это последствия первых назначений, не HOTA и не доказанная причина всех отказов.',decision=p['next']+' Основной критерий: '+str(passed),cannot_claim=p['counterargument'])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for k,v in fields.items():
        if card.get(k) and card[k]!=v:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=k,old=card[k],reason='Полный парный опыт проверен по первичным выходам.'))
        card[k]=v
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'))
    for i,value in enumerate((True,passed)):card['acceptance'][i].update(met=value,evidence=(RUN/('independent_audit.json' if i==0 else 'training/report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed))
    journal('итог прямого поканального сравнения','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision'])
    handoff('Полный z13_region_product_v1 окончен. '+fields['obtained']+' '+fields['decision']+' Публикация runs/z13_region_product_20261025/publication.json. Архитектурные вехи context/product и confirmed_results проверить по первичным источникам; проигравшее кHOTA не продвигать.')
    publish('final')


def handoff(text):
    context.original.namespace()['handoff'](text)


def publish(stage):
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged()
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_region_product.py','-q']]):
        with (RUN/f'{stage}_checks_{i}.log').open('w',encoding='utf-8') as out:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/(stage+'_owned_graph.json');write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/('vault_'+stage+'.json')),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_product.py','tests/test_z13_region_product.py','scripts/z13_region_product_console.ps1','scripts/z13_region_product_execution_check_20261025.py','runs/z13_region_product_execution_check_20261025/receipt.json','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [p.relative_to(ROOT).as_posix() for p in RUN.rglob('*.json') if p.name not in ('publication.json','start_publication.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m',('Register' if stage=='start' else 'Evaluate')+' direct channel-aligned FlyVis target interaction','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/('start_publication.json' if stage=='start' else 'publication.json'),dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        folder=RUN/'training'
        with (folder/'controller.stdout.log').open('w',encoding='utf-8') as out,(folder/'controller.stderr.log').open('w',encoding='utf-8') as err:
            child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',str(Path(__file__))],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
            while child.poll() is None:
                if (folder/'status.json').exists():
                    try:write(RUN/'status.json',read(folder/'status.json'))
                    except (OSError,ValueError):pass
                time.sleep(2)
            child.wait()
        finish()
    except Exception as error:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(error)));raise


START=time.monotonic()
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    elif '--publish-start' in sys.argv:publish('start')
    else:raise RuntimeError('Explicit mode required')
