import ast,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_error_audit_20261108.common import *


def main():
    import psutil
    assert not (AGE/'comparison_phase.json').exists(),'Avoid changing the catalog during queued tests.'
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    p=dict(id='singleton_error_source_audit_v1_20261108',registered=datetime.now(MSK).isoformat(),active_problem=1,deferred=[2,3,4],
        question='When sole-free recovery fails, is the chosen detection a different object, is its correctly identified box displaced by the filter, or is the new pair removed by native deduplication?',
        action='Read only the completed and independently audited full outputs, current raw detections, sparse labels and recorded assignments. No tracker/network/evaluator rerun, parameter tuning or update intervention.',
        arms='Same SUPPORT, IMAGE and FLYVIS full trajectories; add AGEonly if its separately registered full study finishes and passes its audit.',
        definitions=dict(added_first_pairs='actual_first minus native_first within each variant, never by numeric ID comparison across variants.',
            raw_identity='Current high-confidence detections matched jointly to same-classGT at IoU>=0.5 on actually annotated frames only.',
            target='The emitted trajectory last-known identity if its return event exists; otherwise one consistent identity from actually labeled recent stored observations. Sources recorded; conflicts and unknowns explicit.',
            box_degradation='The matched raw detection has IoU>=0.5 to its known current object, but emitted filtered box has IoU<0.5 to that same object. Count separately when prior and current identity agree.',
            assigned_but_not_emitted='First-stage added pair has no same-variant current output. Native duplicate-removal log checked separately; absence alone is not a cause.'),
        literature=dict(url='https://arxiv.org/html/2203.14360',section='4.1Observation-centric Re-Update',
            role='OC-SORT separates a re-association from correcting the accumulated Kalman state afterwards. This motivates distinguishing two failure stages; it does not prove ORUis needed or solves identity choice here.'),
        limitations='Historical20val and sparseGT. Local direct-edge diagnosis does not account for all later closed-loop effects. Unknowns are not negative labels. A different identity cannot be repaired merely by a better post-match Kalman update. No claim about problems2–4or a causal test of field coverage.',
        acceptance='All registered input hashes unchanged, native pairs retained, all direct added pairs counted once, identity and box-error populations separated. No tracker-quality acceptance inferred from this diagnostic.')
    write(RUN/'protocol.json',p,new=True)
    path=ROOT/'experiments/project_history.json';catalog=read(path);assert not any(c['id']==p['id'] for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=p['id'],title='Возврат потерянного трека — чужой объект, смещение рамки или удаление дубликата',area='flyvis',state='planned',started='2026-11-08',
        question='На каком этапе теряется польза возврата единственной свободной рамки?',hypothesis='Ошибку выбора личности и ошибку рамки после обновления нужно проверять раздельно; какая преобладает, ещё не установлено.',
        expected='Выбрать следующую правку первой проблемы по сохранённым данным, не повторяя полный трекер.',acceptance=[dict(criterion=p['acceptance'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['limitations'],
        evidence=[dict(kind='file',ref='runs/singleton_error_audit_20261108/protocol.json',claim='Границы и определения до чтения итогов полного пилота')]))
    write(path,catalog);entry=(f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — До итогов текущего полного пилота зарегистрирован разбор источников ошибки первой проблемы `singleton_error_source_audit_v1_20261108` (`runs/singleton_error_audit_20261108/protocol.json`). После текущего полного расчёта, его независимого аудита и контроля давности будут прочитаны только готовые назначения и траектории. Раздельно считаются выбор другого размеченного объекта, ухудшение положения правильно выбранной рамки после обновления и удаление назначенной пары штатной проверкой дубликатов. Разные варианты не сравниваются по числовому ID, поскольку после изменения рождения треков их нумерация может расходиться. Неразмеченные случаи остаются неизвестными. Основание различения этапов — OC-SORT, раздел 4.1 (https://arxiv.org/html/2203.14360): повторное обновление Калмана следует после сопоставления и само по себе не решает выбор личности. Это диагностика готового, без нового трекера, ядра, параметров или изменения критериев; проблемы 2–4 отложены.\n\n")
    write(RUN/'registration_journal.json',dict(entry=entry),new=True);plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1;plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    status('waiting_for_full_and_recency_audits',current_problem=1)
    with (RUN/'controller.log').open('x',encoding='utf-8') as out,(RUN/'controller.err').open('x',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u','research_singleton_error_audit_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),script='research_singleton_error_audit_20261108/controller.py');write(RUN/'started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(RUN/'deadline_registered.json',dict(passed=True,root=root,deadline='2026-11-08T14:00:00+03:00'),new=True)


if __name__=='__main__':main()
