"""Paired target-only intervention: correct decisions versus rare beneficial changes."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import ast
import inspect
import hashlib
import subprocess
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_assignment_recovery2_20261025 as original
from scripts import z13_region_balanced_20261025 as flow
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_region_semantic_loss import semantic_choice_loss
BASE=original.RUN
RUN=ROOT/'runs/z13_region_semantic_20261025'
IDENT='z13_region_semantic_v1'
START=time.monotonic()


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)


def primary(comp):
    return comp['D-baseline']['mean_per_video']>=.02 and comp['Dsemantic-Dutility']['mean_per_video']>=.02 and all(v['ci95'][0]>0 for v in comp.values())


def training_source():
    text=original.source();tree=ast.parse(text)
    code=ast.get_source_segment(text,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_worker'))
    changes=[('loss=marginal_choice_loss(torch,scores,d[\'good\'][idx],d[\'mask\'][idx],1.)',
              "loss=semantic_choice_loss(torch,scores,d['semantic_good'][idx],d['baseline_good'][idx],d['mask'][idx])"),
        ("    passed=comp['D-baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values())",
         "    comp['Dsemantic-Dutility']=paired(means['D'],read(BASE/'training/report.json')['mean_seed_per_video']['D'])\n    passed=primary(comp)")]
    for old,new in changes:assert code.count(old)==1,old;code=code.replace(old,new)
    return code


def status(phase,stage,**kw):
    p=read(RUN/'protocol.json')
    if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original semantic stage budget ended')
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=time.monotonic()-START,**kw))


def label_metadata():
    root=ROOT/'runs/z13_region_assignment_20261025/labels';report=read(root/'report.json');result={}
    for video,h in report['video_sha256'].items():
        path=root/'videos'/(video+'.json');assert sha(path)==h
        for row in read(path)['rows']:
            uid=row['uid'];assert uid not in result
            if row['kind']>=0:
                acts=row['arms']['regions'];good=[bool(x['own_after']) and not bool(x['harmful']) for x in acts[1:]];baseline=bool(acts[0]['own_before'])
            else:good=[False]*(len(row['arms']['regions'])-1);baseline=False
            result[uid]=(good,baseline)
    assert len(result)==37296
    return result


def worker():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=read(RUN/'protocol.json');verify_sources(p);metadata=label_metadata();original_ns=original.namespace();loader=original_ns['load_split']
    def load_split(torch,arm,split):
        d=loader(torch,arm,split);good=torch.zeros_like(d['mask']);baseline=torch.zeros(len(d['uid']),dtype=torch.bool)
        for i,uid in enumerate(d['uid'].tolist()):
            values,base=metadata[uid];count=int(d['mask'][i].sum());assert len(values)==count
            good[i,:count]=torch.tensor(values,dtype=torch.bool);baseline[i]=base
        d['semantic_good']=good;d['baseline_good']=baseline;return d
    ns={**original_ns,'RUN':RUN,'BASE':BASE,'load_split':load_split,'semantic_choice_loss':semantic_choice_loss,'primary':primary,'status':status}
    code=training_source();assert hashlib.sha256(code.encode()).hexdigest()==p['executed_training_sha256']
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['train_worker']()


def prepare():
    import torch
    from tracking.z13_filter_descriptor import make_ranker
    assert not RUN.exists()
    assert read(ROOT/'runs/z13_region_supervision_audit_20261025/report.json')['availability_gate']
    assert read(ROOT/'runs/z13_region_pooling_20261025/publication.json')['push_completed']
    old=read(BASE/'training/report.json');assert old['complete'] and not old['primary_pass']
    torch.set_num_threads(2);torch.set_default_device('cpu');ns=original.namespace();controls={}
    for arm in ('D','C','R1','geometry'):
        d=ns['load_split'](torch,'D' if arm=='geometry' else arm,'train26')
        for seed in (0,1,2):
            dest=BASE/'training'/arm/f'seed{seed}';done=read(dest/'done.json');report=read(dest/'report.json')
            assert sha(dest/'model.pt')==done['model_sha256'] and sha(dest/'report.json')==done['report_sha256']
            model=make_ranker(torch,activity=arm!='geometry').cpu();model.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
            rows,per,count=ns['policy_rows'](d,ns['predict'](torch,model,d),report['tau'],read(BASE/'protocol.json')['evaluate26'])
            assert rows==read(dest/'rows.json') and per==report['per_video'] and count==report['counts'];controls[arm+'/'+str(seed)]=done
    metadata=label_metadata();d=ns['load_split'](torch,'D','fit84');idx=torch.arange(2);good=torch.zeros_like(d['mask'][idx]);baseline=torch.zeros(2,dtype=torch.bool)
    for i in (0,1):
        v,b=metadata[int(d['uid'][i])];good[i,:len(v)]=torch.tensor(v);baseline[i]=b
    model=make_ranker(torch).cpu();z=model(d['x'][idx],d['seed'][idx],d['geo'][idx],d['mask'][idx]);loss=semantic_choice_loss(torch,z,good,baseline,d['mask'][idx]);loss.backward()
    assert bool(loss.isfinite()) and all(bool(x.grad.isfinite().all()) for x in model.parameters() if x.grad is not None) and not torch.cuda.is_initialized()
    start=now();stop=min(start+timedelta(seconds=1800),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_semantic_loss.py','tests/test_z13_region_semantic_loss.py','scripts/z13_region_supervision_audit_20261025.py',
        'runs/z13_region_supervision_audit_20261025/protocol.json','runs/z13_region_supervision_audit_20261025/report.json',
        'runs/z13_region_pooling_20261025/report.json','runs/z13_region_assignment_20261025/labels/report.json',
        (BASE/'training/report.json').relative_to(ROOT).as_posix(),(BASE/'features/report.json').relative_to(ROOT).as_posix()]
    features=read(BASE/'features/report.json')
    for name,h in features['manifest'].items():assert sha(ROOT/name)==h and sha((ROOT/name).with_suffix('.npz'))==read(ROOT/name)['sha256']
    p=dict(read(BASE/'training/protocol.json'),id=IDENT,created=start.isoformat(),device='cpu',budget_seconds=1800,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        action='Change the acceptable action target, not the likelihood family or input: include every own-correct, nonharmful action instead of only strictly beneficial changes. Include unchanged in the positive marginal whenever the original own assignment was correct. If neither an action nor baseline is correct, unchanged is the conservative expressible fallback.',
        controls='Same original88->32->1 region model, frozen68activity/68target/geometric8 inputs, seeds0/1/2,UID minibatches,1150updates,batch128,Adam0.001 and fit20 threshold grid. Utility controls are the12existing original equal-weight marginal likelihood models, all8326 outputs re-inferred exactly before launch; do not retrain.12newsemantic D/C/R1/geometry models. Failed context/product inputs are not carried forward.',
        target_availability='fit84:17859known queries,14646original own decisions correct,16737safe-own-correct actions vs1356strictly beneficial actions. Source runs/z13_region_supervision_audit_20261025/report.json. These are existing semantic outcome booleans, not new labels for unknown identity. Unknown queries remain excluded from gradients.',
        primary='Dsemantic-baseline and Dsemantic-Dutility means both>=2pp; all5paired-video95%lower>0: baseline,utilityD,semanticC,semanticR1,semanticgeometry. All26videos and3seeds. Same net-beneficial assignment metric; semantic label accuracy cannot rescue a failed primary.',
        technical='All12utility models exactly replayed; actual fit84 two-query CPU forward/backward finite. Mask/NULL/fallback gradients and utility special-case value+gradient equivalence tested. Same full caches/sources/UID batches, no completed training repeated. First full model2*seconds*11+60 fits original1800s; independent controller only own tree. Independently re-infer12new models/rows/calibration and paired intervals before final acceptance.',
        distinction='Earlier semantic_sampling changed hard-row frequency while preserving center-useful labels. Balancing changed rare query weights; signed MSE changed regression objective; none made correct neutral decisions acceptable while retaining correct baseline. This trial changes the supervised decision target under the same marginal likelihood and architecture as its reused utility control.',
        literature='Bhat et al.,DiMP,ICCV2019 https://arxiv.org/html/1904.07220v1 sections3.1,3.5: discriminative target/background labels and prediction error, not only occasions when an external tracker can be improved. This supports comparing supervision semantics; our acceptable-set loss is not the DiMP optimizer or an inherited published gain.',
        counterargument='Correct neutral actions need not improve the stand; expanding the acceptable set may dilute rare useful corrections or leave an easy unchanged solution. Fixed response/mean-pooled representation may remain insufficient. One-row outcomes omit feedback; repeatedtrain26 is exploratory, not HOTA or proven causal diagnosis.',
        next='Pass -> separately budgeted frozen-policy closed-loop parity/price/HOTA with original/no-fly controls. Fail -> preserve this supervised-target implementation. No further scalar loss/threshold variants under this action; choose a different internal representation/state mechanism only with a measured cheap prerequisite.',
        executed_training_sha256=hashlib.sha256(training_source().encode()).hexdigest(),sha256={**read(BASE/'training/protocol.json')['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'training/protocol.json',p);write(RUN/'preflight.json',dict(at=now().isoformat(),all12_utility_controls_reinferred_exact=True,controls=controls,real_fit84_forward_backward_finite=True,cuda_initialized=False))
    write(RUN/'status.json',dict(state='ready',stage='Semantic acceptable targets;12newCPU models,12controls reused',updated=now().isoformat(),models_done=0,models_total=12))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Правильный ответ как цель обучения регионального считывателя',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],question='Поможет ли обучение всем правильным невредным решениям вместо только редких улучшений, с допустимым правильным бездействием?',hypothesis='Рабочая гипотеза: обучение редкому улучшению может не научить различать сам правильный ответ. Наличие больше положительных меток не доказывает эту причину.',expected=p['action'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Предварительные критерии, неизменные контроли/пакеты и семантика цели')],obtained='',interpretation='',decision='',cannot_claim=p['counterargument']))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка правильного ответа как цели обучения','*Постановка.* '+p['action']+' '+p['controls']+'\n\n*Мишень.* '+p['target_availability']+'\n\n*Критерии.* '+p['primary']+' '+p['technical']+'\n\n*Отличие.* '+p['distinction']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Бюджет.* CPU1800с+20с, скрыто, save '+p['save_at']+' / hard '+p['hard_stop']+'. НикакихGPU/ядра/dev/HOTA/75/загрузок. Источник runs/z13_region_semantic_20261025/protocol.json. '+p['literature'])


def finish():
    code=inspect.getsource(flow.finish)
    changes=[(" and m['positive_weight']==p['positive_weight']",''),('D-balanced-D-uniform','Dsemantic-Dutility'),
        ("passed=comp['D-baseline']['mean_per_video']>=.02 and all(x['ci95'][0]>0 for x in comp.values())",'passed=primary(comp)'),
        ('Равный вес запросов: основной критерий','Правильный ответ как цель: основной критерий'),
        ('Изменён только вес классов запроса. Результат не доказывает причину всех отказов; качество принудительного выбора не равно качеству допуска. Однокадровые последствия, не замкнутый трекер.',
         'Изменено множество допустимых ответов в той же функции обучения. Правильное бездействие остаётся допустимым. Результат не доказывает общую причину отказов; это первые назначения, не HOTA.'),
        ('Проверен полный пилот взвешивания классов.','Проверен полный пилот семантики допустимых ответов.'),
        ('итог равного веса запросов вмешательства','итог обучения правильным невредным решениям'),('Балансировка классов регионального узла закончена.','Обучение семантическим региональным ответам закончено.')]
    for old,new in changes:assert code.count(old)==1,old;code=code.replace(old,new)
    ns={**vars(flow),'RUN':RUN,'IDENT':IDENT,'__file__':str(Path(__file__)),'journal':journal,'publish':lambda:publish('final'),'primary':primary}
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['finish']()


def publish(stage):
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged()
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_region_semantic_loss.py','-q']]):
        with (RUN/f'{stage}_checks_{i}.log').open('w',encoding='utf-8') as out:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/(stage+'_owned_graph.json');write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/('vault_'+stage+'.json')),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_semantic_loss.py','tests/test_z13_region_semantic_loss.py','scripts/z13_region_supervision_audit_20261025.py','runs/z13_region_supervision_audit_20261025/protocol.json','runs/z13_region_supervision_audit_20261025/report.json','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    paths=list(RUN.rglob('*.json')) if stage=='final' else [RUN/n for n in ('protocol.json','preflight.json','launch.json','pipeline.json','controller.json') if (RUN/n).exists()]
    names += [p.relative_to(ROOT).as_posix() for p in paths if p.name not in ('publication.json','start_publication.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists() and subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):
        for attempt in range(3):
            result=subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+25]],cwd=ROOT,capture_output=True)
            if result.returncode==0:break
            write(RUN/f'git_add_failure_{i}_{attempt}.json',dict(at=now().isoformat(),returncode=result.returncode,stderr=result.stderr.decode('utf-8',errors='replace'),paths=names[i:i+25]));time.sleep(.5)
        result.check_returncode()
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m',('Register' if stage=='start' else 'Evaluate')+' semantic correct-action supervision within FlyVis readout','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/('start_publication.json' if stage=='start' else 'publication.json'),dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        with (RUN/'controller.stdout.log').open('w',encoding='utf-8') as out,(RUN/'controller.stderr.log').open('w',encoding='utf-8') as err:
            child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(RUN),'--script',str(Path(__file__))],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW);child.wait()
        finish()
    except Exception as error:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(error)));raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--publish-start' in sys.argv:publish('start')
    else:raise RuntimeError('Explicit mode required')
