"""New science, not service recovery: two observable guards, fixed full-strength box."""
from pathlib import Path
from datetime import datetime,timedelta
import os
import sys
import inspect
import hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import z13_box_proposal_20261025 as base
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_unmatched_box_20261025'
IDENT='z13_unmatched_box_v1'
ARMS=tuple(a+'_'+mode for a in ('D','C','R1','geometry') for mode in ('full','unmatched','free'))
ALPHAS=(0.,1.)


def journal(title,text):
    from scripts import z13_candidate_reserve_20261021 as m
    old=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=old


def source():
    s=base.source();old='new_cost_rows(q,costs[ri],detections,t,self.get_dists)'
    assert s.count(old)==1
    return s.replace(old,'new_cost_rows(q,costs[ri],detections,t,self.get_dists,base,ri)')


def worker_source():
    s=inspect.getsource(base.worker)
    replacements=[
        ('from tracking.z13_box_proposal import choose_region,proposal_costs','from tracking.z13_box_proposal import choose_region,proposal_costs\n    from tracking.z13_unmatched_box import admit'),
        ('def new_cost_rows(q,original_row,detections,track,get_dists):','def new_cost_rows(q,original_row,detections,track,get_dists,assignment,row_index):'),
        ("for kind in ('center','box'):\n                answer[a+'_'+kind]=proposal_costs(track,detections,original_row,get_dists,index,ALPHAS,kind=='box')",
         "for kind in ('full','unmatched','free'):\n                allowed=admit(kind,assignment,row_index,index)\n                selected[a][kind+'_admitted']=bool(allowed)\n                answer[a+'_'+kind]=proposal_costs(track,detections,original_row,get_dists,index if allowed else -1,ALPHAS,True)"),
        ("choices={a:max(range(len(ALPHAS)),key=lambda j:np.mean(list(fit[a][j].values()))) for a in ARMS}","choices={a:1 for a in ARMS}"),
        ("stage='Одна область: перенос центра против полной рамки'","stage='Полная рамка: без защиты / только неназначенный / свободная детекция'"),
        ("stage='Полный парный прототип сохранён; не HOTA'","stage='Обе защиты применения сохранены; не HOTA'")]
    for old,new in replacements:
        assert s.count(old)==1,old
        s=s.replace(old,new)
    return s


def paired975(a,b):
    videos=sorted(set(a)&set(b));d=np.array([a[v]-b[v] for v in videos])
    samples=d[np.random.default_rng(1).integers(0,len(d),(5000,len(d)))].mean(1)
    return dict(mean_per_video=float(d.mean()),ci975=[float(np.quantile(samples,.0125)),float(np.quantile(samples,.9875))],better=int((d>0).sum()),n=len(videos))


def contrasts(v):
    zero={x:0. for x in v['D_full']};out={}
    for mode in ('unmatched','free'):
        stem='D_'+mode
        for other in ('baseline','D_full','Dold','C_'+mode,'R1_'+mode,'geometry_'+mode):
            out[stem+'-'+other]=paired975(v[stem],zero if other=='baseline' else v[other])
        for a in ('C','R1'):
            out[mode+'_effect_D-'+a]=paired975({x:v[stem][x]-v['D_full'][x] for x in zero},
                {x:v[a+'_'+mode][x]-v[a+'_full'][x] for x in zero})
    return out


def passed_modes(c):
    return {mode:c['D_'+mode+'-baseline']['mean_per_video']>=.02 and all(x['ci975'][0]>0 for k,x in c.items() if k.startswith('D_'+mode+'-') or k.startswith(mode+'_effect_')) for mode in ('unmatched','free')}


def primary(c):return any(passed_modes(c).values())


def prepare():
    assert not RUN.exists();old=read(base.RUN/'protocol.json');result=read(base.RUN/'report.json')
    assert read(base.RUN/'publication.json')['push_completed'] and not result['primary_pass']
    assert read(base.RUN/'final_summary.json')['D_selection_among_these']['helpful_full_box_chosen_but_alpha025_not_helpful']==246
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_unmatched_box.py','tests/test_z13_unmatched_box.py',
        'runs/z13_box_proposal_20261025/report.json','runs/z13_box_proposal_20261025/final_summary.json',
        'scripts/z13_record_box_proposal_20261025.py']
    start=now();stop=min(start+timedelta(seconds=1200),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    p=dict(old,id=IDENT,created=start.isoformat(),budget_seconds=1200,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),arms=list(ARMS),alphas=list(ALPHAS),
        action='Test two observable application contracts for the existing target-model full-box output. Full-strength action is permitted only if the original first-stage assignment left this row unmatched; stricter free rule also requires proposed detection not assigned to any other row. This is an application-boundary experiment, not an intrinsic modernization of FlyVis or a service fix.',
        method='Same original maps, region selection, NULL, support and full-XYWH proposal as prior prototype. Three fixed contracts perD/C/R1/geometry: unconditional full strength, unmatched-only, unmatched-and-free. Both new contracts use original baseline assignment computed before any intervention. No GT or labels in admission. Same complete assignment after one-row replacement. No alpha sweep: active alpha1, alpha0 only exact regression control.',
        calibration='No calibration/selection: every contract uses alpha1 fixed before run. Fit20 replay is reported as development audit; indices1 are fixed and written before evaluate26. Original Dold retains original saved tau. All three contracts/all controls reported. No conditional threshold estimated on26.',
        criteria='For either of2preregistered guards: Dguard-baseline mean>=2pp; every matched lower97.5%video interval>0 against baseline, unconditional Dfull, original Dold, same C/R1/geometry guard, and differential guard effects D-C and D-R1. Bonferroni for two guard alternatives: bootstrap5000seed1 quantiles0.0125/0.9875, field ci975. A better guard than a catastrophic unconditional policy is insufficient without absolute benefit and FlyVis-specific controls.',
        technical=old['technical']+' All alpha1guard decisions observable and alpha0exact. Prior unconditional-full outcomes are reproduced exactly on all26rows as part of the final audit, to isolate only the new admission contract. Two helper tests cover NULL, already-assigned row, and claimed detection. No scientific tolerance changed.',
        counterargument='The original tracker may be correctly unmatched when no compatible detection exists. A new box can create a false match or steal from a valid neighbor. The free rule may remove useful recovery of a detection incorrectly held by another track. Both guards may give the same effect without FlyVis, which would be an application improvement only. Reused26 was seen to formulate this new hypothesis; no independent validation claim.',
        difference='Prior fixedalpha0.25 missed246useful full-box actions after choosing a potentially helpful box, but unconditional alpha1 had severe harm. This changes only admission of the same full-strength output based on pre-existing assignment, not weights/pooling/NULL threshold, and compares against that exact unconditional alpha1. New explicit author request25Oct: continue testing after these findings.',
        next='If a guard passes all corrected criteria, preregister a separate frozen matched closed-loop gate; it still counts as application of FlyVis, not an intrinsic new model. If both fail, keep both negative and stop this rule family; do not tune confidence/displacement/admission thresholds on26. Report causes and full working results distinctly.',
        executed_source_sha256=hashlib.sha256(source().encode()).hexdigest(),executed_worker_sha256=hashlib.sha256(worker_source().encode()).hexdigest(),sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Два фиксированных правила допуска полной рамки',done=0,total=46,updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Применение полного ответа FlyVis: неназначенный трек и свободная детекция',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P9'],question='Защищает ли наблюдаемое исходное назначение от вреда полной поправки при пользе сверх равных контролей?',hypothesis='Рабочая гипотеза: часть вреда создаёт полная поправка там, где исходное назначение уже существует; доступное состояние сопоставления может позволить безопаснее применить полный ответ.',expected=p['action']+' '+p['method'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['criteria'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['counterargument']+' Внешний контракт применения не объявляется модернизацией ядра или считывания.',evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Два заранее определённых правила, исправление множественности и фиксированная сила поправки')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка двух наблюдаемых защит полной поправки','*Действие.* '+p['action']+'\n\n*Метод и фиксированные условия.* '+p['method']+' '+p['calibration']+'\n\n*Критерии до прогона.* '+p['criteria']+'\n\n*Причина.* '+p['difference']+'\n\n*Возражение.* '+p['counterargument']+'\n\nДо1200сCPU+20с, скрытый независимый контроллер. Это способ применения ответа, не заявленная модернизация мухи. '+p['next'])


def namespace():
    ns={**vars(base),**globals()}
    exec(compile(inspect.getsource(base.aggregate),__file__,'exec'),ns)
    return ns


def worker():
    p=read(RUN/'protocol.json');s=worker_source();assert hashlib.sha256(s.encode()).hexdigest()==p['executed_worker_sha256']
    ns=namespace();exec(compile(s,__file__,'exec'),ns);ns['worker']()


def finish():
    ns=namespace();s=inspect.getsource(base.finish)
    a="assert values==r['all_alpha_evaluate']";assert s.count(a)==1
    s=s.replace(a,"assert __import__('json').loads(__import__('json').dumps(values))==r['all_alpha_evaluate']")
    s=s.replace('Выход полной рамки: ','Защиты применения полной рамки: ').replace('итог рабочего выхода модели цели в полную рамку','итог двух защит применения полной рамки')
    s=s.replace('Проверен рабочий выбор по отклику обученной модели цели без разметки на входе. Центр и полная рамка одной выбранной области — парные варианты. Это первые назначения на повторно используемом train26, не HOTA и не предел полезности FlyVis.','Проверены два фиксированных контракта применения одного полного ответа. Они читают исходное назначение, а не разметку. Интервалы97,5% учитывают две альтернативы; парные C/R1/геометрия отделяют вклад мухи. Это применение сигнала, не внутренняя модернизация; повторныйtrain26 и первые назначения, не HOTA.')
    s=s.replace('Источник runs/z13_box_proposal_20261025/report.json; выбор alpha только20fit.','Источник runs/z13_unmatched_box_20261025/report.json; alpha1 зафиксирована до расчёта, без подбора.')
    # Verify the already completed unconditional arm, not a newly selected baseline.
    prior=read(base.RUN/'report.json');p=read(RUN/'protocol.json');checked=0
    for v in p['evaluate26']:
        old={r['uid']:r for r in read(base.RUN/'videos'/(v+'.json'))['rows']}
        for row in read(RUN/'videos'/(v+'.json'))['rows']:
            for a in ('D','C','R1','geometry'):
                assert row['arms'][a+'_full'][1]==old[row['uid']]['arms'][a+'_box'][3]
            checked+=1
    assert checked==8326
    write(RUN/'unconditional_reproduction.json',dict(at=now().isoformat(),rows=checked,all4unconditional_full_arms_exact=True))
    exec(compile(s,__file__,'exec'),ns);ns['finish']()
    r=read(RUN/'report.json')
    write(RUN/'guard_results.json',dict(passed_modes=passed_modes(r['paired']),interval='97.5% video bootstrap; Bonferroni for2guards',primary_unchanged=True))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--finish' in sys.argv:finish()
    else:raise RuntimeError('Explicit mode required')
