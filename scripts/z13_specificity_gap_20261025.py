"""Cheap fixed-confidence replacement on saved maps and canonical assignments."""
from pathlib import Path
from datetime import datetime,timedelta
import inspect
import os
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_specificity_gap_20261025'
SOURCE=ROOT/'runs/z13_template_swap_20261025/recovery1'
IDENT='z13_specificity_gap_v1'
ARMS=('D','C','R1');T0=time.monotonic()


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as old
    before=old.IDENT
    try:old.IDENT=IDENT;old.journal(title,text)
    finally:old.IDENT=before


def prepare():
    assert not RUN.exists()
    assert read(SOURCE/'publication.json')['push_completed']
    base=read(SOURCE/'protocol.json');verify_sources(base)
    assert read(SOURCE/'finish_recorded.json')['complete']
    deadline=now()+timedelta(seconds=300)
    names=['tracking/z13_specificity_gap.py','tests/test_z13_specificity_gap.py',
        'scripts/z13_specificity_gap_20261025.py','scripts/z13_specificity_gap_console.ps1',
        'scripts/z13_finish_template_swap_20261025.py','runs/z13_template_swap_20261025/recovery1/report.json',
        'runs/z13_template_swap_20261025/recovery1/independent_audit.json','runs/z13_template_swap_20261025/recovery1/controller_result.json',
        'runs/z13_refusal_causes_20261025/rows.json']
    p=dict(id=IDENT,created=now().isoformat(),timezone='Europe/Moscow',device='cpu',budget_seconds=300,save_at=deadline.isoformat(),hard_stop=(deadline+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        action='Replace the internal confidence formula, keeping the canonical proposed center unchanged: normalized own response at its argmax minus mean normalized foreign-filter response at that SAME column. No learned coefficients/grid or threshold optimization.',
        source='Saved complete maps from the isolated template experiment; same4844 exact-geometry eligible queries forD/C/R1. Unknown observed identities and unmatched geometry remain outside this diagnostic; they are not silently refused as part of an inference policy.',
        outcome='Canonical one-step assignment effect for the original top1 center, read from z13_refusal_causes rows. Helpful/harmful refer to actual first-stage assignment; neutral is neither. All8326 labels/arm and scores bound before restricting the cohort.',
        primary='Frozen topceil(1%*4844)=49 per arm/score with ascending label tie-break; compare gap vs original height on the IDENTICAL cohort. Per-video net=(beneficial-harmful)/all known eligible queries in that video. Require Dgap-baseline0, Dgap-Dheight, Dgap-Cgap,Dgap-R1gap all lower95%paired-videoCI>0. A pass is only a target for a separate deployable full-cohort calibration; never automatic HOTA.',
        counterargument='The strict cohort uses known observed identity to construct a diagnostic intervention, and is not an inference-ready admission rule. Counterfactual maps may amplify background; three donors may be unrepresentative. Selection is retrospective fixed-coverage, not a threshold frozen on independent calibration data. Reused train26; single-step canonical states omit feedback. A positive outcome is not a calibrated refusal or whole-tracker improvement.',
        literature='Goroshin et al.2001.02593 investigates sensitivity to target-template substitution. Reading a same-location foreign-filter margin is our fixed prototype, not their published auxiliary detector.',
        next='If the complete primary gate fails, do not train/calibrate/promote this fixed formula. If it passes, prepare an annotation-free donor-bank/coverage implementation and fit-only calibration before any deployment.',
        sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Контраст уверенности в прежней точке',done=0,total=822,updated=now().isoformat()))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: контраст с чужими моделями в выбранной точке',area='flyvis',state='ready',started=now().strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Помогает ли оценка специфичности в выбранной точке отличить полезное для сопоставления вмешательство от высокого неспецифического пика?',
        hypothesis='Правильный образец влияет на центр, но высота максимума чужой модели может оставаться высокой в другом месте. Контраст в одной точке проверяет иной сигнал отказа, сохраняя центр.',
        expected=p['action']+' CPU до300с, никаких новых признаков или обучения.',acceptance=[dict(criterion=p['outcome'],met=None),dict(criterion=p['primary'],met=None)],summary='Одна фиксированная замена формулы уверенности зарегистрирована до исходов.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],metrics=[],commits=[],files=names[:4],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Формула, контроль и критерии до расчёта')]))
    write(cp,cat)
    journal('постановка контраста уверенности в прежней точке','*Действие.* '+p['action']+'\n\n*Критерии.* '+p['primary']+'\n\n*Граница вывода.* '+p['counterargument']+'\n\nCPU300с+20с, независимый контроллер, сохранённые карты. Другая формула, не повтор удаления SD или фиксированного косинуса наблюдавшегося фона.')


def worker():
    import numpy as np
    from tracking.z13_specificity_gap import specificity_gap,select
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');verify_sources(p)
    src=read(SOURCE/'report.json')
    original=read(ROOT/'runs/z13_refusal_causes_20261025/rows.json')
    reference={a:{r['label']:r for r in original if r['arm']==a and r['candidate']==0} for a in ARMS}
    assert all(len(x)==8326 for x in reference.values())
    allrows=[];pv={};counts={};done=0;videos=read(SOURCE/'protocol.json')['evaluate']
    for arm in ARMS:
        rows=[]
        for path in sorted((SOURCE/arm).glob('chunk_*.json')):
            assert time.monotonic()-T0<p['budget_seconds'] and now()<datetime.fromisoformat(p['save_at']) and not (RUN/'stop.request').exists()
            assert sha(path)==src['chunk_sha256'][path.relative_to(ROOT).as_posix()]
            job=read(path);assert sha(path.with_suffix('.npz'))==job['maps_sha256']
            with np.load(path.with_suffix('.npz')) as z:maps=z['maps'];ids=z['labels']
            for i,r in enumerate(job['rows']):
                old=reference[arm][r['label']]
                assert int(ids[i])==r['label'] and old['video']==r['video'] and old['kind']==r['kind']
                assert old['kind']<0 or old['outcome'] in ('beneficial','harmful','neutral')
                assert abs(old['score']-r['outputs'][0]['p_present'])<.001
                if not r['eligible']:continue
                rows.append(dict(arm=arm,label=r['label'],video=r['video'],kind=r['kind'],height=r['outputs'][0]['p_present'],gap=specificity_gap(maps[i]),outcome=old['outcome']))
            done+=1
            write(RUN/'status.json',dict(state='running',stage='Контраст в прежней точке, исход сопоставления фиксирован',variant=arm,done=done,total=822,updated=now().isoformat()))
        assert len(rows)==4844
        allrows.extend(rows)
        for mode in ('height','gap'):
            selected=select(rows,mode);assert len(selected)==49
            counts[arm+'_'+mode]={k:sum(r['outcome']==k for r in selected if r['kind']>=0) for k in ('beneficial','harmful','neutral')}
            counts[arm+'_'+mode].update(selected=49,unknown=sum(r['kind']<0 for r in selected),labels=[r['label'] for r in selected])
            pv[arm+'_'+mode]={v:sum((int(r['outcome']=='beneficial')-int(r['outcome']=='harmful')) for r in selected if r['video']==v and r['kind']>=0)/sum(r['video']==v and r['kind']>=0 for r in rows) for v in videos}
    comp={'Dgap-baseline':paired(pv['D_gap'],{v:0. for v in videos})}
    for name in ('D_height','C_gap','R1_gap'):comp['Dgap-'+name]=paired(pv['D_gap'],pv[name])
    write(RUN/'rows.json',allrows)
    verify_sources(p)
    write(RUN/'report.json',dict(complete=True,counts=counts,per_video=pv,paired=comp,primary_pass=all(x['ci95'][0]>0 for x in comp.values()),rows_sha256=sha(RUN/'rows.json'),limits=p['counterargument']))
    write(RUN/'status.json',dict(state='completed',stage='Фиксированная формула и одинаковые контроли проверены',done=822,total=822,updated=now().isoformat()))


def finish():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p=read(RUN/'protocol.json');ps=WindowsProcesses()
    while not (RUN/'controller_result.json').exists():
        if now()>datetime.fromisoformat(p['hard_stop'])+timedelta(seconds=60):raise TimeoutError('Missing controller result')
        time.sleep(2)
    e=read(RUN/'controller_result.json');assert e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    assert e['exit_code']==0,'Preserve technical failure; repair in original budget before recording efficacy'
    verify_sources(p);r=read(RUN/'report.json');assert r['complete'] and sha(RUN/'rows.json')==r['rows_sha256']
    # Recompute selection, counts and video intervals independently from raw scalar rows.
    import numpy as np
    from scripts.z13_continue import paired
    rows=read(RUN/'rows.json');pv={};counts={};videos=list(next(iter(r['per_video'].values())))
    for arm in ARMS:
        rr=[x for x in rows if x['arm']==arm];assert len(rr)==4844
        for mode in ('height','gap'):
            chosen=sorted(rr,key=lambda x:(-x[mode],x['label']))[:49]
            counts[arm+'_'+mode]={k:sum(x['outcome']==k for x in chosen if x['kind']>=0) for k in ('beneficial','harmful','neutral')}
            counts[arm+'_'+mode].update(selected=49,unknown=sum(x['kind']<0 for x in chosen),labels=[x['label'] for x in chosen])
            pv[arm+'_'+mode]={v:sum((int(x['outcome']=='beneficial')-int(x['outcome']=='harmful')) for x in chosen if x['video']==v and x['kind']>=0)/sum(x['video']==v and x['kind']>=0 for x in rr) for v in videos}
    assert counts==r['counts'] and pv==r['per_video']
    comparisons={'Dgap-baseline':paired(pv['D_gap'],{v:0. for v in videos})}
    for name in ('D_height','C_gap','R1_gap'):comparisons['Dgap-'+name]=paired(pv['D_gap'],pv[name])
    assert comparisons==r['paired'];passed=all(x['ci95'][0]>0 for x in comparisons.values());assert passed==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),counts_and_intervals_verified=True,report_sha256=sha(RUN/'report.json')))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    source=(RUN/'report.json').relative_to(ROOT).as_posix()
    fields=dict(summary='Проверена одна фиксированная формула специфичности; основной критерий '+('пройден.' if passed else 'не пройден.'),
        obtained=source+' → counts: '+str(counts)+'; paired: '+str(comparisons)+'. Неизменные центры и исходы сопоставления; по4844строки/49выбранных каждого варианта.',
        interpretation='Это ограниченная проверка на строгом диагностическом подмножестве; не рабочее правило допуска без разметки и не HOTA.',
        decision=('Подготовить отдельную реализацию банка без разметки и калибровку только на fit, не переносить этот отбор напрямую в стенд.' if passed else 'Не продвигать эту фиксированную формулу к обучению или HOTA. Сохранить отрицательный исход; он не закрывает другие внутренние способы оценки достоверности.'),cannot_claim=p['counterargument'])
    head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for k,v in fields.items():
        if card.get(k) and card[k]!=v:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=k,old=card[k],reason='Проверен итог фиксированного контраста.'))
        card[k]=v
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'))
    card['acceptance'][0].update(met=True,evidence=(RUN/'independent_audit.json').relative_to(ROOT).as_posix());card['acceptance'][1].update(met=passed,evidence=source)
    card['evidence'].append(dict(kind='file',ref=source,claim='Все исходы и одинаковые контроли'))
    write(cp,cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed,report_sha256=sha(RUN/'report.json'),architecture_review_due=True))
    journal('итог контраста уверенности в прежней точке','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nHOTA и процент к стенду не измерены; доля потолка не определена. Проверка не принята как подтверждённое улучшение стенда.')
    path=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    path.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+IDENT+' окончен. '+fields['obtained']+' '+fields['decision']+' Публикация: runs/z13_specificity_gap_20261025/publication.json. Прежняя изолированная подмена завершена и опубликована5e39f3a; её D+16,58п.п. означает вклад образца, не новую пользу стенду.\n\nНиже прежние передачи сохранены.\n\n'+path.read_text(encoding='utf-8'),encoding='utf-8')
    from scripts import z13_finish_template_swap_20261025 as publisher
    code=inspect.getsource(publisher.publish)
    code=code.replace("names=['tracking/z13_template_swap.py'","names=EXTRA+['tracking/z13_template_swap.py'")
    code=code.replace("'tests/test_z13_template_swap.py','-q'","'tests/test_z13_template_swap.py','tests/test_z13_specificity_gap.py','-q'")
    code=code.replace('Verify isolated template dependence with frozen FlyVis query states','Test same-location specificity confidence on frozen assignment outcomes')
    extra=['tracking/z13_specificity_gap.py','tests/test_z13_specificity_gap.py','scripts/z13_specificity_gap_20261025.py','scripts/z13_specificity_gap_console.ps1']
    ns={**vars(publisher),'RUN':RUN,'EXTRA':extra}
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['publish']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as exc:write(RUN/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    elif '--finish' in sys.argv:
        try:finish()
        except Exception as exc:write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(exc)));raise
    else:raise RuntimeError('Explicit mode required')
