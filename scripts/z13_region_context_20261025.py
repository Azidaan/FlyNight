"""Bounded CPU queue: observable assignment context, then matched context/blank heads."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import ast
import hashlib
import time
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_assignment_recovery2_20261025 as original
from scripts.z13_region_choice_gate_20261025 import replay_source as old_replay
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_region_assignment_context import assignment_state_context,make_context_ranker
from tracking.z13_region_value_loss import signed_action_loss
RUN=ROOT/'runs/z13_region_context_20261025'
BASE=original.RUN
PREVIOUS=ROOT/'runs/z13_region_value_20261025'
IDENT='z13_region_context_v1'
PHASE=os.environ.get('Z13_CONTEXT_PHASE','')
VARIANTS=tuple(a+'_'+mode for a in ('D','C','R1','geometry') for mode in ('context','blank'))
BUDGETS={'cache':1200,'training':1800}


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)


def replay_source():
    text=old_replay();old='new_cost_rows(q,costs[ri],detections,t.cls)'
    assert text.count(old)==1
    return text.replace(old,'new_cost_rows(q,costs,ri,base,detections,t.cls)')


def comparisons(means,videos):
    from scripts.z13_continue import paired
    answer={'Dcontext-baseline':paired(means['D_context'],{v:0. for v in videos})}
    for name in ('D_blank','C_context','R1_context','geometry_context'):
        answer['Dcontext-'+name]=paired(means['D_context'],means[name])
    answer['Dcontext-Dprevious']=paired(means['D_context'],read(PREVIOUS/'training/report.json')['mean_seed_per_video']['D'])
    return answer


def primary(comp):
    return comp['Dcontext-baseline']['mean_per_video']>=.02 and comp['Dcontext-D_blank']['mean_per_video']>=.02 and all(x['ci95'][0]>0 for x in comp.values())


def training_source():
    text=original.source();tree=ast.parse(text)
    code=ast.get_source_segment(text,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_worker'))
    changes=[("from tracking.z13_filter_descriptor import make_ranker","from tracking.z13_region_assignment_context import make_context_ranker as make_ranker"),
      ("for variant in ('D','C','R1','geometry'):","for variant in VARIANTS:"),
      ("arm='D' if variant=='geometry' else variant","base_variant,mode=variant.rsplit('_',1);arm='D' if base_variant=='geometry' else base_variant"),
      ("data={s:load_split(torch,arm,s) for s in ('fit84','fit20','train26')}","data={s:load_context_split(torch,arm,s,mode) for s in ('fit84','fit20','train26')}"),
      ("activity=variant!='geometry'","activity=base_variant!='geometry'"),
      ("loss=marginal_choice_loss(torch,scores,d['good'][idx],d['mask'][idx],1.)","loss=signed_action_loss(torch,scores,d['good'][idx],d['harm'][idx],d['mask'][idx])"),
      ("models_total=12","models_total=24"),
      ("*11+60","*23+60"),
      ("for a in ('D','C','R1','geometry')}","for a in VARIANTS}")]
    for old,new in changes:assert code.count(old)==1,old;code=code.replace(old,new)
    first=code.index("    comp={'D-baseline'")
    last=code.index("    verify_sources(p);write(folder/'report.json'")
    code=code[:first]+"    comp=comparisons(means,p['evaluate26']);passed=primary(comp)\n"+code[last:]
    return code


def status(phase,stage,**kw):
    p=read(RUN/phase/'protocol.json')
    if (RUN/phase/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Original context phase budget ended')
    write(RUN/phase/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))


def prepare():
    assert not RUN.exists()
    assert read(PREVIOUS/'publication.json')['push_completed'] and not read(PREVIOUS/'training/report.json')['primary_pass']
    p=read(BASE/'protocol.json');start=now();stop=min(start+timedelta(seconds=3120),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    model=make_context_ranker(torch);x=torch.zeros(2,2,68);seed=torch.zeros(2,68);geo=torch.zeros(2,2,14);mask=torch.ones(2,2,dtype=torch.bool)
    score=model(x,seed,geo,mask);loss=signed_action_loss(torch,score,torch.zeros_like(mask),torch.zeros_like(mask),mask);loss.backward()
    assert bool(torch.isfinite(loss)) and all(bool(t.grad.isfinite().all()) for t in model.parameters() if t.grad is not None) and not torch.cuda.is_initialized()
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_context_console.ps1','tracking/z13_region_assignment_context.py',
        'tests/test_z13_region_assignment_context.py','tracking/z13_region_value_loss.py','scripts/z13_region_value_20261025.py',
        'runs/z13_region_value_20261025/training/report.json','runs/z13_region_value_20261025/independent_audit.json',
        'runs/z13_region_assignment_20261025/labels/report.json','runs/z13_region_assignment_20261025/recovery2/features/report.json']
    protocol=dict(p,id=IDENT,created=start.isoformat(),device='cpu',pipeline_save_at=stop.isoformat(),pipeline_hard_stop=(stop+timedelta(seconds=20)).isoformat(),
        phase_budgets=BUDGETS,variants=list(VARIANTS),
        action='Add6 observable original first-assignment features per action: own cost, lowest rival-track cost, best alternative cost, own-minus-rival, original row matched flag, action equals original assignment flag. No annotations or identities enter these features.',
        conventions='Missing rival/alternative cost1. NULL: own-row minimum,1,own-row minimum,0,original-matched,original-unmatched. Original matching uses unchanged native threshold and full matrix. No cost/state change during collection.',
        architecture='Same68/68 embeddings16 and readout, geo8+context6 gives94->32->1. Context and blank controls have identical initialization/capacity; blank zeros exactly the6new inputs. D/C/R1/geometry times2modes times3seeds=24models. Geometry zeros activity/target only. Same signed assignment loss,1150updates,batch128,Adam0.001,UID minibatches and frozen fit20 threshold grid.',
        primary='Dcontext-baseline and Dcontext-Dblank both means>=2pp, and all6paired-video lower95% bounds>0: baseline,Dblank,Ccontext,R1context,geometrycontext,previous signedD.',
        technical='Cache130 baseline replays float32-exact, all37296query UIDs/causal IDs/predictions/labels and region order equal saved labels; first2videos2*mean_seconds*128+60 fits1200remaining. No completed cache video repeated. Training first complete model2*seconds*23+60 fits remaining1800s.',
        counterargument='Assignment context may help geometric processing alone, without FlyVis. This is a small readout adaptation, not a graph neural tracker. One-row outcomes omit joint corrections and closed-loop feedback; repeatedtrain26 is exploratory.',
        literature='Braso and Leal-Taixe,CVPR2020,Learning a Neural Solver for Multiple Object Tracking,sections3.3-4: https://openaccess.thecvf.com/content_CVPR_2020/papers/Braso_Learning_a_Neural_Solver_for_Multiple_Object_Tracking_CVPR_2020_paper.pdf . Global association context motivates this cheap input check; our6statistics are not their message-passing architecture or published result.',
        next='Pass -> separately budgeted frozen closed-loop parity/price/HOTA, native/C/R1/no-fly controls. Fail -> preserve this context implementation; no HOTA or core training by default, select a different measured internal node.',
        executed_replay_sha256=hashlib.sha256(replay_source().encode()).hexdigest(),executed_training_sha256=hashlib.sha256(training_source().encode()).hexdigest(),
        sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',protocol);write(RUN/'smoke.json',dict(at=now().isoformat(),real94_input_forward_backward_finite=True,cuda_initialized=False))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Контекст исходного назначения внутри считывателя FlyVis',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
       question='Улучшат ли исходные назначения и конкуренция других треков решение считывателя о вмешательстве?',
       hypothesis='Рабочая гипотеза: часть различий полезного/нейтрального/вредного действия определяется отсутствующим в прежних входах контекстом назначения. Проверяется добавлением наблюдаемых6признаков против их нулевого контроля той же ёмкости.',
       expected='Улучшить D минимум на2п.п. относительно равного пустого контроля и без правки; подтвердить преимущество над C/R1/геометрией.',
       acceptance=[dict(criterion=protocol['technical'],met=None),dict(criterion=protocol['primary'],met=None)],
       evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Критерии, равная ёмкость, наблюдаемые входы и бюджеты до расчёта')],
       obtained='',interpretation='',decision='',cannot_claim=protocol['counterargument']))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка наблюдаемого контекста назначения',
       '*Постановка.* '+protocol['action']+' '+protocol['architecture']+' '+protocol['primary']+' '
       'Все130 CPU-повторов нужны для ранее не сохранённой матрицы; прежние признаки и метки не пересчитываются. '
       'Cache1200с -> CPU24модели1800с, независимые контроллеры, общий конец '+protocol['pipeline_hard_stop']+'. '
       'До запуска пройден реальный94входный прямой/обратный проход CPU без CUDA. '+protocol['counterargument']+' '
       'Источник runs/z13_region_context_20261025/protocol.json. '+protocol['literature'])


def cache_worker():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from scripts import z13_assignment_effect_20261024 as producer
    from tracking.z13_region_assignment import regions
    folder=RUN/'cache';p=read(folder/'protocol.json');verify_sources(p)
    query_source=original.namespace();queries,centers=query_source['queries_and_crops']();geom=query_source['geometry']();metadata={}
    def new_cost_rows(q,costs,row,base,detections,track_class):
        boxes=np.asarray([d.xyxy for d in detections],float).reshape(-1,4);confidence=np.asarray([d.score for d in detections]);classes=np.asarray([int(d.cls) for d in detections])
        choices,_,_=regions(geom,q['label']['crop'],q['label']['pred'],boxes,confidence,classes,track_class)
        metadata[q['uid']]=dict(choices=choices,context=assignment_state_context(costs,row,choices,base).tolist(),track_class=int(track_class))
        return {'regions':[costs[row].copy()]}
    ns={**vars(producer),'RUN':folder,'new_cost_rows':new_cost_rows,'T0':time.monotonic()};code=replay_source()
    assert hashlib.sha256(code.encode()).hexdigest()==p['executed_replay_sha256'];exec(compile(code,str(Path(__file__)),'exec'),ns)
    videos=p['fit20']+p['fit84']+p['evaluate26'];times=[];manifest={};rows_total=0
    labels=read(ROOT/'runs/z13_region_assignment_20261025/labels/report.json')
    for i,video in enumerate(videos):
        out=folder/'videos'/(video+'.json')
        if out.exists():
            record=read(out);assert sha(out)==read(folder/'completed'/(video+'.json'))['sha256']
        else:
            before=time.monotonic();rows,checks=ns['replay_video'](video,queries[video],centers,p);assert checks['raw_replay_exact']
            labelpath=ROOT/'runs/z13_region_assignment_20261025/labels/videos'/(video+'.json');assert sha(labelpath)==labels['video_sha256'][video]
            old={r['uid']:r for r in read(labelpath)['rows']};assert len(old)==len(rows)
            entries={}
            for row in rows:
                ref=old[row['uid']];m=metadata[row['uid']]
                assert all(row[k]==ref[k] for k in ('uid','video','frame','tid','split','label','kind'))
                assert m['choices']==ref['regions']['choices'][1:] and m['track_class']==ref['regions']['track_class']
                entries[str(row['uid'])]=m
            record=dict(complete=True,raw_replay_exact=True,entries=entries,seconds=time.monotonic()-before)
            write(out,record);write(folder/'completed'/(video+'.json'),dict(sha256=sha(out)));times.append(record['seconds'])
            if len(times)==2:
                estimate=2*float(np.mean(times))*128+60;remaining=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
                write(folder/'price.json',dict(estimate=estimate,remaining=remaining,proceed=estimate<remaining));assert estimate<remaining,'Cache price gate; preserve completed'
        rows_total+=len(record['entries']);manifest[video]=sha(out)
        status('cache','Observable original assignment context',done=i+1,total=130,variant=video)
    assert rows_total==37296 and not torch.cuda.is_initialized()
    write(folder/'report.json',dict(complete=True,rows=rows_total,videos=130,manifest=manifest,all_raw_replays_exact=True,cuda_initialized=False))


def train_worker():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    ns=original.namespace();loader=ns['load_split'];contexts={}
    for video,h in read(RUN/'cache/report.json')['manifest'].items():
        path=RUN/'cache/videos'/(video+'.json');assert sha(path)==h
        for uid,entry in read(path)['entries'].items():assert int(uid) not in contexts;contexts[int(uid)]=entry
    def load_context_split(torch,arm,split,mode):
        data=loader(torch,arm,split);extra=torch.zeros((*data['mask'].shape,6),dtype=torch.float32)
        for i,uid in enumerate(data['uid'].tolist()):
            n=int(data['mask'][i].sum());value=contexts[uid]['context'];assert len(value)==n
            if mode=='context':extra[i,:n]=torch.as_tensor(value,dtype=torch.float32)
        data['geo']=torch.cat([data['geo'],extra],-1);return data
    ns={**ns,'RUN':RUN,'VARIANTS':VARIANTS,'load_context_split':load_context_split,'signed_action_loss':signed_action_loss,'status':status,'comparisons':comparisons,'primary':primary}
    assert hashlib.sha256(training_source().encode()).hexdigest()==read(RUN/'protocol.json')['executed_training_sha256']
    exec(compile(training_source(),str(Path(__file__)),'exec'),ns);ns['train_worker']()


def phase_protocol(phase):
    p=read(RUN/'protocol.json');stop=min(now()+timedelta(seconds=BUDGETS[phase]),datetime.fromisoformat(p['pipeline_save_at']))
    extra={}
    if phase=='training':extra[(RUN/'cache/report.json').relative_to(ROOT).as_posix()]=sha(RUN/'cache/report.json')
    q=dict(p,budget_seconds=BUDGETS[phase],save_at=stop.isoformat(),hard_stop=min(stop+timedelta(seconds=20),datetime.fromisoformat(p['pipeline_hard_stop'])).isoformat(),
        stop_request_path=(RUN/phase/'stop.request').relative_to(ROOT).as_posix(),sha256={**p['sha256'],**extra})
    write(RUN/phase/'protocol.json',q);return q


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        for phase in BUDGETS:
            folder=RUN/phase
            if not (folder/'controller_result.json').exists():
                phase_protocol(phase);env={**os.environ,'Z13_CONTEXT_PHASE':phase}
                with (folder/'controller.stdout.log').open('w',encoding='utf-8') as out,(folder/'controller.stderr.log').open('w',encoding='utf-8') as err:
                    child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',str(Path(__file__))],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
                    while child.poll() is None:
                        path=folder/'status.json'
                        if path.exists():
                            try:write(RUN/'status.json',dict(read(path),phase=phase,observer_at=now().isoformat()))
                            except (OSError,ValueError):pass
                        time.sleep(2)
                    child.wait()
            result=read(folder/'controller_result.json');assert result['exit_code']==0 and result['all_owned_ended'] and not result['remaining']
            assert all(ps.creation(int(pid))!=birth for pid,birth in result['owned'].items()) and read(folder/'report.json')['complete']
        finish()
    except Exception as error:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),phase=phase,error=repr(error)));raise


def finish():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=read(RUN/'protocol.json');verify_sources(p);report=read(RUN/'training/report.json');assert report['complete'] and report['models']==24 and not report['cuda_initialized']
    means={};records={}
    for variant in VARIANTS:
        records[variant]={}
        for seed in p['seeds']:
            dest=RUN/'training'/variant/f'seed{seed}';done=read(dest/'done.json');r=read(dest/'report.json');rows=read(dest/'rows.json')
            assert sha(dest/'model.pt')==done['model_sha256'] and sha(dest/'report.json')==done['report_sha256'] and sha(dest/'rows.json')==r['rows_sha256']
            assert len(rows)==8326 and sum(x['kind']>=0 for x in rows)==5822 and r['steps']==1150
            per={v:sum(int(x['beneficial'])-int(x['harmful']) for x in rows if x['video']==v)/sum(x['video']==v and x['kind']>=0 for x in rows) for v in p['evaluate26']};assert per==r['per_video']
            records[variant][seed]=r
        means[variant]={v:float(np.mean([records[variant][s]['per_video'][v] for s in p['seeds']])) for v in p['evaluate26']}
    comp=comparisons(means,p['evaluate26']);assert comp==report['paired'] and primary(comp)==report['primary_pass']
    passed=primary(comp);write(RUN/'independent_audit.json',dict(at=now().isoformat(),all24_models_rows_counts_intervals_verified=True,report_sha256=sha(RUN/'training/report.json')))
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT)
    fields=dict(summary='Наблюдаемый контекст назначения: критерий '+('пройден.' if passed else 'не пройден.'),
       obtained=(RUN/'training/report.json').relative_to(ROOT).as_posix()+' → paired: '+str(comp)+';24модели,3сида,8равных вариантов.',
       interpretation='Контекст сопоставления проверен против его нулевого контроля той же ёмкости; это внутренний считыватель с внешними наблюдаемыми входами. Не HOTA и не полноценный графовый трекер.',
       decision=p['next']+' Измеренный основной критерий: '+str(passed),cannot_claim=p['counterargument'])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=key,old=card[key],reason='Проверен полный контекстный пилот.'))
        card[key]=value
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'))
    for i,met in enumerate((True,passed)):card['acceptance'][i].update(met=met,evidence=(RUN/('independent_audit.json' if i==0 else 'training/report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed))
    journal('итог наблюдаемого контекста назначения','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision'])
    original.namespace()['handoff']('Полный z13_region_context_v1 окончен. '+fields['obtained']+' '+fields['decision']+' Публикация runs/z13_region_context_20261025/publication.json; архитектурная веха и отдельные confirmed_results требуют проверки по первичным источникам.')
    publish()


def publish():
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged()
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_region_assignment_context.py','-q']]):
        with (RUN/f'checks_{i}.log').open('w',encoding='utf-8') as out:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/'owned_graph_before.json';write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_final.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_assignment_context.py','tests/test_z13_region_assignment_context.py','scripts/z13_region_context_console.ps1',
       'experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [p.relative_to(ROOT).as_posix() for p in RUN.rglob('*.json') if p.name!='publication.json']
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Evaluate observable assignment context within the FlyVis readout','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:
        try:{'cache':cache_worker,'training':train_worker}[PHASE]();write(RUN/PHASE/'status.json',dict(state='completed',stage='Phase saved',updated=now().isoformat()))
        except Exception as error:write(RUN/PHASE/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
