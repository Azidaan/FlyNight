"""Learn the existing internal readout against actual full-box action outcomes."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import time
import inspect
import hashlib
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_context_20261025 as context
from scripts import z13_box_proposal_20261025 as box
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=ROOT/'runs/z13_full_box_supervision_20261025'
IDENT='z13_full_box_supervision_v1'
PHASE=os.environ.get('Z13_FULL_BOX_PHASE','')
BUDGETS={'cache':1200,'training':1800}
VARIANTS=context.VARIANTS


def journal(title,text):
    from scripts import z13_candidate_reserve_20261021 as m
    old=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=old


def handoff(text):
    h=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    h.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+text+'\n\nНиже прежние передачи сохранены.\n\n'+h.read_text(encoding='utf8'),encoding='utf8')


def comparisons(means,videos):
    from scripts.z13_continue import paired
    out={'Dbox-baseline':paired(means['D_context'],{v:0. for v in videos})}
    for a in ('D_blank','C_context','R1_context','geometry_context'):
        out['Dbox-'+a]=paired(means['D_context'],means[a])
    prior=read(context.RUN/'training/report.json')['mean_seed_per_video']['D_context']
    out['Dbox-Dcategorical']=paired(means['D_context'],prior)
    return out


def primary(c):
    return c['Dbox-baseline']['mean_per_video']>=.02 and all(v['ci95'][0]>0 for v in c.values())


def training_source():
    return context.training_source()


def loader_source():
    s=inspect.getsource(context.train_worker)
    a="n=int(data['mask'][i].sum());value=contexts[uid]['context'];assert len(value)==n"
    b=a+"\n            data['good'][i,:n]=torch.as_tensor(contexts[uid]['good'],dtype=torch.bool)\n            data['harm'][i,:n]=torch.as_tensor(contexts[uid]['harm'],dtype=torch.bool)"
    assert s.count(a)==1
    return s.replace(a,b)


def prepare():
    assert not RUN.exists()
    previous=ROOT/'runs/z13_unmatched_box_20261025'
    assert read(previous/'publication.json')['push_completed'] and not read(previous/'report.json')['primary_pass']
    p=read(context.RUN/'protocol.json');verify_sources(p)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_box_action_labels.py','tests/test_z13_box_action_labels.py',
        'scripts/z13_region_context_20261025.py','tracking/z13_region_assignment_context.py','tracking/z13_region_value_loss.py',
        'scripts/z13_box_proposal_20261025.py','tracking/z13_box_proposal.py',
        'runs/z13_region_context_20261025/cache/report.json','runs/z13_region_context_20261025/training/report.json',
        'runs/z13_unmatched_box_20261025/report.json','runs/z13_box_proposal_20261025/final_summary.json']
    # Hash every reused feature chunk/context row, not just its manifest.
    names += [n for n in read(context.BASE/'features/report.json')['manifest']]
    names += [(context.RUN/'cache/videos'/(v+'.json')).relative_to(ROOT).as_posix() for v in p['fit20']+p['fit84']+p['evaluate26']]
    start=now();stop=min(start+timedelta(seconds=3120),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    q=dict(p,id=IDENT,created=start.isoformat(),device='cpu',phase_budgets=BUDGETS,
        pipeline_save_at=stop.isoformat(),pipeline_hard_stop=(stop+timedelta(seconds=20)).isoformat(),
        action='Replace the internal regional readout action semantics and corresponding supervision: selected real region now transfers complete XYWH and uses unchanged native get_dists, instead of a categorical cost row. Explicit unmatched still sets this row costs to1; fixed unchanged logit still means no action. No row or target identity is supplied at runtime.',
        difference='Earlier signed/context heads learned outcomes of categorical replacement costs. The frozen-map full-box prototype did not learn safety for the actual geometric action. This trains identical old readout architectures/features against new full-box outcomes. It is a coupled output-interface/target change, not proof that one scalar loss caused previous failures. Regional support is the exact prior trainable support, including nearest-column check, not the wider fixed-map prototype support.',
        fixed='Reuse all D/C/R1 raw68 regional activity, observed target coefficients, original assignment context6, eight geometry inputs and query UID order. Same24heads (4representations x context/blank x3seeds),1150steps,batch128,Adam0.001,signed good-harm loss, fixed fit20 threshold grid. Train84; calibrate20; evaluate26. Only outcome labels and applied candidate-box action change together. No new feature/core computation or GPU.',
        primary='Preregister D_context as primary, no choosing blank after evaluation: net baseline effect mean>=2pp and all6paired95%video lower>0 versus baseline,D_blank,C_context,R1_context,geometry_context,previous Dcontext categorical action. Report all24models and frozen fit20 thresholds, including zero-action and unfavorable outcomes.',
        technical='All130 baseline raw outputs exactly reproduce original. All37296query bindings/region orders match original labels and saved context; same observed features reused. Label arrays contain actual full assignment outcomes for each full-box/null action, not IoU or identity surrogate. Complete means restored; unchanged action exact. All input hashes; every model/hash/row/interval independently audited. First2label replays2*mean*128+60 must fit1200s. Firsttraining model2*time*23+60 must fit1800s. No completed video/model repeated.',
        counterargument='The old representations may not identify absent targets even with correct action labels. Severe imbalance may yield no action; the same geometry controls may suffice. Action reserve does not imply learnability. Both20fit and26evaluation repeatedly used, so this is exploratory, not independent confirmation. No manual guards or threshold edits after evaluation.',
        literature='SiamMOT https://arxiv.org/html/2105.11595 sections3.1-3.3 equations1,3,6 learns visibility and full-box output before association. Supports aligning a learned output with its geometric action, not our signed loss or an expected improvement. This is an adaptation inside FlyVis readout, not reproduction of SiamMOT.',
        next='Pass -> separately preregister matched runtime and closed-loop HOTA with native/C/R1/no-FlyVis controls. Fail -> preserve this exact action-supervision implementation and stop this categorical-to-box family; no new loss/guard/threshold sweep on26. A different internal mechanism needs its own measured rationale.',
        executed_replay_sha256=hashlib.sha256(box.source().encode()).hexdigest(),executed_training_sha256=hashlib.sha256(training_source().encode()).hexdigest(),executed_loader_sha256=hashlib.sha256(loader_source().encode()).hexdigest(),
        sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',q);write(RUN/'status.json',dict(state='ready',stage='Полная рамка и согласованные метки действия',updated=start.isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Внутреннее считывание FlyVis: обучение последствиям полной рамки',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8','fly-problem-P9'],
        question='Сможет ли внутренний выбор/отказ использовать запас полной рамки при обучении именно этому действию?',hypothesis='Рабочая гипотеза: метки прежнего категориального изменения стоимости не переносятся автоматически на другую геометрию действия. Обучение полной рамке при равных признаках может изменить полезность.',
        expected=q['action']+' '+q['fixed'],acceptance=[dict(criterion=q['technical'],met=None),dict(criterion=q['primary'],met=None)],obtained='',interpretation='',decision='',cannot_claim=q['counterargument'],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Связанный выход/метки, равные признаки и критерии до обучения')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка обучения последствиям полной рамки','*Действие.* '+q['action']+'\n\n*Отличие от испытанного.* '+q['difference']+'\n\n*Фиксировано.* '+q['fixed']+'\n\n*Критерии до прогона.* '+q['primary']+' '+q['technical']+'\n\n*Возражение.* '+q['counterargument']+'\n\n*Источник.* '+q['literature']+'\n\nCPU1200с метки → CPU1800с24модели, каждый+20с, независимые скрытые контроллеры. '+q['next'])


def status(phase,stage,**kw):
    p=read(RUN/phase/'protocol.json')
    if (RUN/phase/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Original full-box phase deadline')
    write(RUN/phase/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))


def cache_worker():
    from scripts import z13_assignment_effect_20261024 as producer
    from tracking.z13_region_assignment import regions
    from tracking.z13_box_action_labels import full_box_rows
    p=read(RUN/'cache/protocol.json');verify_sources(p)
    original=context.original.namespace();queries,centers=original['queries_and_crops']();geom=original['geometry']();meta={}
    def new_cost_rows(q,old,detections,track,get_dists):
        choices,_,_=regions(geom,q['label']['crop'],q['label']['pred'],[d.xyxy for d in detections],[d.score for d in detections],[int(d.cls) for d in detections],int(track.cls))
        meta[q['uid']]=choices
        return {'regions':full_box_rows(track,detections,old,get_dists,choices)}
    code=box.source();assert hashlib.sha256(code.encode()).hexdigest()==p['executed_replay_sha256']
    ns={**vars(producer),'RUN':RUN/'cache','new_cost_rows':new_cost_rows,'T0':time.monotonic()};exec(compile(code,__file__,'exec'),ns)
    videos=p['fit20']+p['fit84']+p['evaluate26'];manifest={};times=[];total=known=0
    prior_manifest=read(context.RUN/'cache/report.json')['manifest']
    for i,v in enumerate(videos):
        dest=RUN/'cache/videos'/(v+'.json');assert not dest.exists()
        before=time.monotonic();rows,checks=ns['replay_video'](v,queries[v],centers,p)
        ref=context.RUN/'cache/videos'/(v+'.json');assert sha(ref)==prior_manifest[v]
        old=read(ref)['entries'];assert len(old)==len(rows);entries={}
        for row in rows:
            uid=row['uid'];entry=old[str(uid)];assert entry['choices']==meta[uid]
            outcome=row['arms']['regions'][1:];assert len(outcome)==len(entry['context'])
            entry=dict(entry,kind=row['kind'],good=[bool(x['beneficial']) for x in outcome],harm=[bool(x['harmful']) for x in outcome])
            assert not any(a and b for a,b in zip(entry['good'],entry['harm']))
            entries[str(uid)]=entry;total+=1;known+=row['kind']>=0
        write(dest,dict(complete=True,entries=entries,checks=checks,seconds=time.monotonic()-before));manifest[v]=sha(dest);times.append(time.monotonic()-before)
        status('cache','Точные последствия полной рамки; исходный трекер неизменен',done=i+1,total=130,variant=v)
        if i==1:
            estimate=2*float(np.mean(times))*128+60;remaining=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
            write(RUN/'cache/price.json',dict(estimate=estimate,remaining=remaining,proceed=estimate<remaining));assert estimate<remaining
    assert total==37296 and known==26600
    verify_sources(p);write(RUN/'cache/report.json',dict(complete=True,rows=total,known=known,videos=130,manifest=manifest,all_raw_replays_exact=True))


def train_worker():
    p=read(RUN/'training/protocol.json');s=loader_source()
    assert hashlib.sha256(s.encode()).hexdigest()==p['executed_loader_sha256']
    ns={**vars(context),**globals(),'original':context.original,'signed_action_loss':context.signed_action_loss}
    exec(compile(s,__file__,'exec'),ns);ns['train_worker']()


def phase_protocol(phase):
    p=read(RUN/'protocol.json');stop=min(now()+timedelta(seconds=BUDGETS[phase]),datetime.fromisoformat(p['pipeline_save_at']))
    assert now()<stop
    extra={}
    if phase=='training':extra[(RUN/'cache/report.json').relative_to(ROOT).as_posix()]=sha(RUN/'cache/report.json')
    q=dict(p,budget_seconds=BUDGETS[phase],save_at=stop.isoformat(),hard_stop=min(stop+timedelta(seconds=20),datetime.fromisoformat(p['pipeline_hard_stop'])).isoformat(),stop_request_path=(RUN/phase/'stop.request').relative_to(ROOT).as_posix(),sha256={**p['sha256'],**extra})
    write(RUN/phase/'protocol.json',q);write(RUN/phase/'status.json',dict(state='ready',stage=phase,updated=now().isoformat()))


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat()))
    try:
        for phase in BUDGETS:
            phase_protocol(phase);folder=RUN/phase
            with (folder/'controller.stdout.log').open('w',encoding='utf8') as out,(folder/'controller.stderr.log').open('w',encoding='utf8') as err:
                proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',str(Path(__file__))],cwd=ROOT,env={**os.environ,'Z13_FULL_BOX_PHASE':phase},stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
                write(folder/'launch.json',dict(pid=proc.pid,birth=ps.creation(proc.pid),at=now().isoformat()))
                while proc.poll() is None:
                    try:write(RUN/'status.json',dict(read(folder/'status.json'),phase=phase,observer_at=now().isoformat()))
                    except (OSError,ValueError):pass
                    time.sleep(2)
                proc.wait()
            e=read(folder/'controller_result.json')
            assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
            assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items()) and read(folder/'report.json')['complete']
        write(RUN/'compute_complete.json',dict(at=now().isoformat(),both_phases_complete=True,recording_pending=True))
    except Exception as error:
        write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),phase=phase,error=repr(error)));raise


def finish():
    p=read(RUN/'protocol.json');verify_sources(p);ps=WindowsProcesses()
    for phase in BUDGETS:
        e=read(RUN/phase/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    cache=read(RUN/'cache/report.json');labels={}
    for v,h in cache['manifest'].items():
        f=RUN/'cache/videos'/(v+'.json');assert sha(f)==h;labels.update({int(u):x for u,x in read(f)['entries'].items()})
    r=read(RUN/'training/report.json');assert r['models']==24 and r['complete'] and not r['cuda_initialized'];means={}
    for a in VARIANTS:
        records=[]
        for seed in p['seeds']:
            dest=RUN/'training'/a/f'seed{seed}';done=read(dest/'done.json');rec=read(dest/'report.json');rows=read(dest/'rows.json')
            assert sha(dest/'model.pt')==done['model_sha256'] and sha(dest/'report.json')==done['report_sha256'] and sha(dest/'rows.json')==rec['rows_sha256']
            assert len(rows)==8326 and sum(x['kind']>=0 for x in rows)==5822 and rec['steps']==1150
            for x in rows:
                label=labels[x['uid']];i=x['action'];known=x['kind']>=0
                assert x['kind']==label['kind'] and x['beneficial']==bool(i>=0 and known and label['good'][i]) and x['harmful']==bool(i>=0 and known and label['harm'][i])
            per={v:sum(int(x['beneficial'])-int(x['harmful']) for x in rows if x['video']==v)/sum(x['video']==v and x['kind']>=0 for x in rows) for v in p['evaluate26']}
            assert per==rec['per_video'];records.append(per)
        means[a]={v:float(np.mean([x[v] for x in records])) for v in p['evaluate26']}
    c=comparisons(means,p['evaluate26']);assert means==r['mean_seed_per_video'] and c==r['paired'] and primary(c)==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all24_models_rows_action_labels_and_intervals_verified=True,report_sha256=sha(RUN/'training/report.json')))
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT)
    card.update(state='completed_positive' if r['primary_pass'] else 'completed_negative',done=now().strftime('%Y-%m-%d'),summary='Обучение последствиям полной рамки: критерий '+('пройден.' if r['primary_pass'] else 'не пройден.'),obtained=(RUN/'training/report.json').relative_to(ROOT).as_posix()+' → paired: '+str(c),interpretation='Одинаковое внутреннее считывание обучено фактическим действиям полной рамки с равными контролями. Изменены сопряжённые выход и метки, не доказана одна причина прежнего провала. Повторные train26, первые назначения, не HOTA.',decision=p['next']+' Основной критерий: '+str(r['primary_pass']))
    for i,met in enumerate((True,r['primary_pass'])):card['acceptance'][i].update(met=met,evidence=(RUN/('independent_audit.json' if i==0 else 'training/report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=r['primary_pass']))
    journal('итог обучения действию полной рамки','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+'\n\nHOTA и относительный процент к стенду не измерены; доля потолка не установлена.')
    handoff(IDENT+' завершён. '+card['obtained']+' '+card['decision']+' Все свои деревья проверены поbirth. Итог требует каталога/тестов/графа/коммита. Скрытые процессы;57чужихправок/.gitignoreсохранить. Общее окно26.10до07:40МСК.')


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:
        try:{'cache':cache_worker,'training':train_worker}[PHASE]();write(RUN/PHASE/'status.json',dict(state='completed',stage='Фаза сохранена',updated=now().isoformat()))
        except Exception as error:write(RUN/PHASE/'failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--finish' in sys.argv:finish()
    else:raise RuntimeError('Explicit mode required')
