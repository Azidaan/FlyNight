"""CPU paired action: make the NULL branch explicitly preserve the original row."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,inspect,hashlib,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_readout_20261025 as decoder
from scripts import z13_region_readout_recovery1_20261025 as recovery
from scripts import z13_region_pooling_20261025 as template
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=ROOT/'runs/z13_null_mixture_20261025'
IDENT='z13_null_mixture_v1'
ARMS=tuple(a+'_'+mode for a in ('D','C','R1','geometry') for mode in ('old','null_mix'))
ALPHAS=(0.,.25)

def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)

def comparisons(values):
    from scripts.z13_continue import paired
    videos=list(values['D_null_mix']);out={'Dnull-baseline':paired(values['D_null_mix'],{v:0. for v in videos}),'Dnull-Dold':paired(values['D_null_mix'],values['D_old'])}
    for a in ('C','R1','geometry'):out['Dnull-'+a+'null']=paired(values['D_null_mix'],values[a+'_null_mix'])
    out['null_effect_D-geometry']=paired({v:values['D_null_mix'][v]-values['D_old'][v] for v in videos},{v:values['geometry_null_mix'][v]-values['geometry_old'][v] for v in videos})
    return out

def primary(c):
    return all(c[k]['mean_per_video']>=.02 for k in ('Dnull-baseline','Dnull-Dold')) and all(v['ci95'][0]>0 for v in c.values())

def worker_source():
    s=recovery.worker_source()
    changes=[
        ('from tracking.z13_region_readout import region_probabilities,cost_row','from tracking.z13_region_readout import region_probabilities\n    from tracking.z13_null_mixture import cost_row'),
        ("for split in ('fit','evaluate'):","for split in ('evaluate',):"),
        ("field=geometry if arm=='geometry' else maps[split,arm][l['label']]","name,mode=arm.split('_',1)\n            field=geometry if name=='geometry' else maps[split,name][l['label']]"),
        ('cost_row(old,probabilities,confidence,valid,a) for a in ALPHAS','cost_row(old,probabilities,confidence,valid,a,mode) for a in ALPHAS'),
        ("choices={a:max(range(len(ALPHAS)),key=lambda j:np.mean(list(fit[a][j].values()))) for a in ARMS}","choices={a:1 for a in ARMS} # Same fixedalpha0.25, no fitting"),
        ("contrasts={'D-baseline':paired(chosen['D'],{v:0. for v in p['evaluate26']})}\n    for arm in ('C','R1','geometry'):contrasts['D-'+arm]=paired(chosen['D'],chosen[arm])\n    passed=contrasts['D-baseline']['mean_per_video']>=.02 and all(v['ci95'][0]>0 for v in contrasts.values())",'contrasts=comparisons(chosen)\n    passed=primary(contrasts)'),
        ('exact_baseline_videos=46','exact_baseline_videos=26')]
    for a,b in changes:assert s.count(a)==1,a;s=s.replace(a,b)
    return s.replace('total=46','total=26')

def worker():
    p=read(RUN/'protocol.json');s=worker_source();assert hashlib.sha256(s.encode()).hexdigest()==p['executed_worker_sha256']
    ns={**vars(decoder),'RUN':RUN,'ARMS':ARMS,'ALPHAS':ALPHAS,'comparisons':comparisons,'primary':primary};exec(compile(s,__file__,'exec'),ns);ns['worker']()

def prepare():
    from tracking.z13_null_mixture import cost_row
    assert not RUN.exists();parent=ROOT/'runs/z13_ridge_uncertainty_20261025';gate=ROOT/'runs/z13_observed_negatives_gate_20261025'
    assert read(parent/'publication.json')['push_completed'] and not read(parent/'replay/report.json')['primary_pass']
    assert read(gate/'publication.json')['push_completed'] and not read(gate/'report.json')['admission_pass']
    ps=WindowsProcesses();c=read(gate/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    old=read(template.RUN/'protocol.json');verify_sources(old)
    x=np.array([.123,.6]);p0=np.zeros(2);confidence=np.array([.9,.8]);support=np.ones(2,bool)
    prior=cost_row(x,p0,confidence,support,.25,'old');new=cost_row(x,p0,confidence,support,.25,'null_mix');assert np.array_equal(x,new) and prior[0]>x[0]
    write(RUN/'cost_contract_check.json',dict(at=now().isoformat(),original=x.tolist(),probabilities=p0.tolist(),old_cost=prior.tolist(),new_cost=new.tolist(),new_full_null_exact_noop=True,
        claim='Deterministic contract example, not an attribution of any observed tracking error. Previous NULL meant no candidate; treating it as epistemic abstention in the uncertainty interpretation was inaccurate.'))
    start=now();stop=min(start+timedelta(seconds=900),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_null_mixture.py','tests/test_z13_null_mixture.py','runs/z13_ridge_uncertainty_20261025/replay/report.json','runs/z13_observed_negatives_gate_20261025/report.json',(RUN/'cost_contract_check.json').relative_to(ROOT).as_posix()]
    q=dict(old,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',device='cpu',budget_seconds=900,pipeline_save_at=stop.isoformat(),pipeline_hard_stop=(stop+timedelta(seconds=20)).isoformat(),save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),videos=old['evaluate26'],fit20=[],
        action='New readout application contract: NULL probability leaves the original association row unchanged instead of raising every supported cost. Keep original mean maps, original unit-clock state, weights, normalization, null logit0, support and alpha0.25. No uncertainty penalty from the failed parent is combined.8paired conditions C/D/R1/geometry timesold/null_mix.',
        formula='q=sum(p_regions); old_cost=(1-alpha)*c+alpha*(1-conf*p); new_cost=c+alpha*(q*(1-c)-conf*p). This equals (1-alpha)*c+alpha*(p_NULL*c+sum_j p_j*action_j), where action_j costs1 except its chosen detection costs1-conf. Unsupported entries unchanged. q0exactnoop; q1oldformula; alpha0exactbaseline.',
        distinction='This is a scientifically new interpretation of the readout NULL, not service repair of the previous declared no-candidate cost. The previous contract could intentionally reject all candidates when target absent. The new one treats NULL as no trustworthy intervention; it can preserve incorrect baseline matches. It tests an interface mechanism, not an intrinsic improvement of FlyVis core.',
        primary='Dnull-baseline and Dnull-Dold means>=2pp; all6paired-video95%lower>0 includingCnull,R1null,geometrynull and (Dnull-Dold)-(geometrynull-geometryold). A shared improvement without a D-specific gain is an application change, not a FlyVis result. No favorable subset/alpha/null-logit selection.',
        technical='Three unit checks: fullNULLnoop, explicit expected-cost mixture, q1oldformula/alpha0exact. All26baseline outputs float32exact; everyold branch action and geometry exactly reproduce the prior mean-region report; old center<0.5px/score<0.001 unchanged. First2CPUreplays conservative old2*mean_seconds*44+120 estimate fits remaining900seconds or preservepartial/stop. Primary rows/hashes/paired intervals independently verified.',
        counterargument='NULL conflates absence and uncertainty; suppressing its rejection effect can leave a false or wrong baseline match untouched and reduce beneficial rejection. Generic reduced intervention may explain everything without FlyVis, so geometry/R1/C and interaction controls are mandatory. One-step naturaltrain26 is repeatedly used; no HOTA or claim of causal explanation of all previous failures.',
        literature='Rasmussen and Williams,GPML2006,chapter2 section2.4 https://gaussianprocess.org/gpml/chapters/RW2.pdf : decision loss averaged over predictive alternatives. The explicit NULL-as-unchanged cost and its8arm tracking test are our engineering proposal; the book supplies no tracking improvement.',
        next='Pass -> distinct runtime contract parity/price and frozen matched closed-loop stage; still separate application from intrinsic FlyVis improvement. Fail -> preserve this exact NULL expectation contract and do not retune alpha/probability thresholds. Observed-negative and four-bin gates remain inconclusive and do not advance.',
        limits='NewCPU900s+20 only, hidden independent owned-tree deadline. No GPU/features/training/dev/HOTA/75/download. Author saveMonday26Oct07:35/hard07:39:30; smaller900s not extended.',
        executed_worker_sha256=hashlib.sha256(worker_source().encode()).hexdigest(),sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',q);write(RUN/'status.json',dict(state='ready',stage='NULL-as-unchanged; eight paired CPU conditions',updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(x['id']==IDENT for x in cat['experiments'])
    previous=next(x for x in cat['experiments'] if x['id']=='z13_ridge_uncertainty_v1');field='interpretation';oldtext=previous[field]
    newtext='Измерены первые назначения10парных условий при прежних средних откликах/весах/времени. Постоянный штраф отделяет общий сдвиг вероятностей кандидатов и стоимости сопоставления от пространственной неопределённости. Это не ужесточение отказа от вмешательства: в прежнем cost_row уменьшение вероятности кандидата повышает его стоимость, а полный NULL не возвращает исходную строку. Это не калиброванная вероятность полезности, не общая причина ошибок FlyVis и не HOTA.'
    if oldtext!=newtext:
        previous.setdefault('revisions',[]).append(dict(date=start.strftime('%Y-%m-%d'),commit=read(parent/'publication.json')['commit'],field=field,old=oldtext,reason='Термин «ужесточение отказа» неточен: проверена формула cost_row и контрпример полного NULL, прежняя версия сохранена.'))
        previous[field]=newtext;previous.setdefault('evidence',[]).append(dict(kind='file',ref=(RUN/'cost_contract_check.json').relative_to(ROOT).as_posix(),claim='NULL прежнего контракта не означает отсутствие вмешательства'))
    cat['experiments'].append(dict(id=IDENT,title='Применение считывателя: NULL как сохранение решения трекера',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],question='Поможет ли явное сохранение исходной строки в ветви NULL при той же карте мухи и без нового штрафа неопределённости?',hypothesis='Рабочая гипотеза: неопределённый ответ не должен автоматически означать повышение стоимости всех кандидатов. Это новый контракт применения, не доказанная ошибка прежнего смысла «цели нет».',expected=q['action'],acceptance=[dict(criterion=q['technical'],met=None),dict(criterion=q['primary'],met=None)],obtained='',interpretation='',decision='',cannot_claim=q['counterargument']+' '+q['distinction'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Одна замена контракта, контроли и критерии до расчёта')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('поправка собственного вывода об отказе и постановка явного NULL',
        '*Поправка.* В итоговом объяснении z13_ridge_uncertainty_v1 я неточно назвал постоянный штраф ужесточением отказа. Старый текст сохранён в revisions. При p=0 прежняя стоимость0,123 становится0,34225, а не возвращается к0,123; это пример формулы, не измеренная причина конкретного ухудшения. Источник runs/z13_null_mixture_20261025/cost_contract_check.json.\n\n'
        '*Новое действие.* '+q['action']+' '+q['formula']+'\n\n*Граница.* '+q['distinction']+'\n\n*Критерии до расчёта.* '+q['primary']+' '+q['technical']+'\n\n*Возражение.* '+q['counterargument']+'\n\n*Бюджет.* '+q['limits']+' Save '+q['save_at']+' / hard '+q['hard_stop']+'. Источник runs/z13_null_mixture_20261025/protocol.json. '+q['literature'])

def audit():
    p=read(RUN/'protocol.json');verify_sources(p);r=read(RUN/'report.json');assert r['complete'] and r['exact_baseline_videos']==26;values={a:{} for a in ARMS}
    for v in p['videos']:
        path=RUN/'videos'/(v+'.json');assert sha(path)==r['video_sha256'][v];record=read(path);assert record['checks']['raw_replay_exact']
        old={x['uid']:x for x in read(recovery.RUN/'videos'/(v+'.json'))['rows']}
        for row in record['rows']:
            previous=old[row['uid']];assert all(row[k]==previous[k] for k in ('uid','video','frame','tid','kind','label','split'))
            for a in ('D','C','R1','geometry'):assert row['arms'][a+'_old'][1]==previous['arms'][a][1]
        known=[x for x in record['rows'] if x['kind']>=0]
        for a in ARMS:values[a][v]=float(np.mean([int(x['arms'][a][1]['beneficial'])-int(x['arms'][a][1]['harmful']) for x in known]))
    c=comparisons(values);assert values==r['per_video'] and c==r['paired'] and primary(c)==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all26_old_branches_and_baselines_exact=True,all8_conditions_and_intervals_verified=True,report_sha256=sha(RUN/'report.json')));return r

def finish():
    ps=WindowsProcesses();c=read(RUN/'controller_result.json');assert c['exit_code']==0 and c['worker_ended'] and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    r=audit();p=read(RUN/'protocol.json');cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT)
    card.update(state='completed_positive' if r['primary_pass'] else 'completed_negative',done=now().strftime('%Y-%m-%d'),summary='NULL как сохранение строки: критерий '+('пройден.' if r['primary_pass'] else 'не пройден.'),obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → paired: '+str(r['paired']),interpretation='Изменён только контракт применения NULL при той же исходной карте: неиспользованная масса вероятности сохраняет исходную строку. Это8парных условий первых назначений, не HOTA и не самостоятельное доказательство модернизации ядра FlyVis.',decision=p['next']+' Primary: '+str(r['primary_pass']))
    for i,met in enumerate((True,r['primary_pass'])):card['acceptance'][i].update(met=met,evidence=(RUN/('independent_audit.json' if i==0 else 'report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=r['primary_pass']));journal('итог явного сохранения строки при NULL','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision'])
    template.handoff('Полный z13_null_mixture_v1 завершён. '+card['obtained']+' '+card['decision']+' Не путать исправленную формулировку отказа с доказательством причины всех ошибок. Обе availabilityгипотезы осталисьнеопределёнными; их пороги не ослаблять. Проверить архитектурнуювехуuncertainty/NULL и подтверждённыечастныеэффекты. Всё скрыто,57чужихправок/.gitignore сохранять.')
    publish('final')

def publish(stage):
    s=inspect.getsource(template.publish).replace('tracking/z13_region_pooling.py','tracking/z13_null_mixture.py').replace('tests/test_z13_region_pooling.py','tests/test_z13_null_mixture.py').replace('paired spatial pooling of FlyVis target responses','explicit NULL mixture contract in FlyVis readout')
    ns=dict(globals());exec(compile(s,__file__,'exec'),ns);ns['publish'](stage)

def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    template.handoff('ТЕКУЩИЙ z13_null_mixture_v1: runs/z13_null_mixture_20261025; scripts/z13_null_mixture_20261025.py --pipeline. CPU900с+20с, всё скрыто, controller.json/launch.json/status.json. Это новый научный контракт NULL как сохранение строки, не служебная правка прошлого результата. Старые карты/веса/время/alpha0.25 неизменны;8условийC/D/R1/geometry old/null_mix. Не комбинирует проваленную неопределённость; все26oldназначенияобязаныточносовпасть. После первого2видеоценапрежнегоокна. Полныйитог→первичныехеши/ДИ/каталог/61тест/сохраняющийграф/scopedpush. Старыйuncertaintyотрицательныйa30b200f, observednegativesgateнеопределённый; четырёхчастныйgateтоже. Готовое не повторять, критерии не ослаблять, меньший900снепродлевать; никакихGPU/обучения/dev/HOTA/75/загрузок.57чужихправок/.gitignore сохранять. Архитектурнуювехуuncertainty/NULL проверить после блока.')
    try:
        with (RUN/'controller.stdout.log').open('w',encoding='utf-8') as out,(RUN/'controller.stderr.log').open('w',encoding='utf-8') as err:
            child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(RUN),'--script',__file__],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW);child.wait()
        finish()
    except Exception as e:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(e)));raise

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    else:raise RuntimeError('Explicit mode required')
