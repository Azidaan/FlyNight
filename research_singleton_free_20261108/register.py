"""Prospective registration for the first problem only; no new map inference."""
import sys,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    assert read(BASE/'focus_order_20261108.json')['active_problem']==1
    assert read(BASE/'singleton_inventory_fit20_recovery1.json')['passed']
    comparisons=[['U_GEO','B0'],['U_GEO','U_THRESHOLD'],['U_THRESHOLD','B0']]
    comparisons += [['U_FLYVIS',a] for a in ARMS if a!='U_FLYVIS']
    assert len(comparisons)==11
    protocol=dict(id='singleton_free_cost_v1_20261108',registered=datetime.now(MSK).isoformat(),
        authorization='Current direct instruction: solve problem1 first, then the others. Continue authorized matched implementation and comparison until14:00MSK.',
        active_problem=1,deferred=[2,3,4],deadline='2026-11-08T14:00:00+03:00',
        hypothesis='Locking the original first-stage matches can preserve stable associations while permitting a sole geometrically admissible high-confidence box to rescue an unmatched Lost track.',
        intervention='First solve the native complete matrix. Keep its matched pairs fixed. On remaining rows/columns, allow a new edge only for a same-class Lost row with exactly one valid choice among its two historical causal search boxes and positive native IoU (C<1). Change only cost, not the Kalman mean/covariance, memory, identity owner or duplicate cleanup. Resolve all newly allowed edges together and then run the unchanged full native lifecycle.',
        visual='Reuse the frozen eight-map centered relative correction from the parent; apply the corrected cost only to the same admissible free edges. The original reference/current eye, history, rival mean and undefined-score behavior are unchanged. A claimed target is not taken from another native match.',
        strongest_alternative='A simple relaxation of the matching threshold on the free remainder may explain the benefit; geometric eligibility or FlyVis may add nothing.',
        controls=dict(B0='Native first-stage matching and full lifecycle, unchanged.',
            U_THRESHOLD='Same locked native pairs. Reduce cost by alpha for all free same-class Lost-to-high edges with C<1, without the exactly-two/unique-geometric rule. Fit its alpha with the same5-value budget.',
            U_GEO='Sole admissible free edge; no image or FlyVis.',
            U_SHIFT='Same geometry, uniform beta cost reduction on identical allowed edges.',
            U_IMAGE='Same geometry and direct processed brightness/derivative map.',U_R1='Same geometry and direct causal receptor temporal features.',
            U_DIS='Same geometry and cached ordinary motion.',U_FLYVIS='Same geometry and fixed T4/T5 spatial maps.',U_PERM='Same geometry and32fixed spatial permutations of the same FlyVis maps.'),
        arms=ARMS,native_config=read(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json')['native_config'],
        calibration=dict(cohort='fit20 only, all323known queries;300shadow queries explicitly retained.',grid=[0,.1,.2,.4,.8],
            criterion='Maximize J=correct-wrong; unassigned0; unknown0 and counted.',tie='Fewer changed final emitted track-to-raw bindings, then smaller parameter.',
            alpha='U_GEO selects shared alpha; U_THRESHOLD selects its own alpha with equal budget.',beta='Each image/motion/FlyVis/shift/permutation control selects beta at the common geometry alpha.',
            freeze='Save selection before instantiating evaluate26 evaluators or labeling new outcomes.'),
        evaluation='All641known historical queries from26videos plus727shadow queries. Same rows and no favorable subgroup selection. Reuse archived full lifecycle outcomes whenever the complete first match set and sources are identical.',
        prior_experiment=dict(id='z13_unmatched_box_v1',source='runs/z13_unmatched_box_20261025/report.json',
            outcome='Negative: D_free90beneficial/215harmful among5822known; mean-video change−0.0256007,97.5%CI[−0.0394417,−0.0139097].',
            distinction='The admission principle was already tested and is not new. The old intervention replaced the predicted fullXYWH by a selected map/box proposal at full strength. This intervention retains the prediction and all native first-stage matched pairs and changes only costs of uniquely admissible free residual edges.'),
        primary_comparisons=comparisons,full_tracker_gate='Proceed to a separately frozen full causal20val pilot only if paired-video lower95%deltaJ vsB0>0 for U_GEO,U_THRESHOLD orU_FLYVIS. Conditional screening is not tracker acceptance.',
        full_acceptance=dict(family=11,primary_ci_percent=100*(1-.05/11),bootstrap_samples=20000,seed=42,unit='paired_video',
            geometry='U_GEO-B0 corrected lower pooled-HOTA bound>0 and no harm. U_GEO-U_THRESHOLD explains whether geometry adds value; no claim of that contribution without its positive interval.',
            threshold='U_THRESHOLD-B0 corrected lower pooled-HOTA bound>0 and no harm is a useful ordinary-tracker result, not FlyVis or novel geometry.',
            flyvis='U_FLYVIS must beatB0and every7matched comparator with corrected lower pooled-HOTA bounds>0 and no harm.',
            no_harm='Nonnegative changes in mean-videoAssA/IDF1 and no increase in totalIDSw,FP,FN for each acceptance comparator.',
            foreign_returns='Report correct/wrong/unknown returns after full lifecycle, not first-stage wins. A one-frame gain cannot close problem1.'),
        literature=[dict(url='https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09351.pdf',
            role='BUSCA establishes an existing after-association unmatched-track stage. Our cost/geometry rule is a separate hypothesis, not BUSCA implementation or evidence of benefit.'),
            dict(url='https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py',role='Original assignment and same-ID reactivation; those are baseline mechanisms.')],
        stop='Preserve failures; never retune on26or20val; if screening or full acceptance fails, do not deploy this version or mark problem1solved. Other problems stay deferred.')
    write(RUN/'protocol.json',protocol,new=True)
    sources={};names=['tracking/free_singleton_assignment.py','tracking/spatial_identity_assignment.py','tests/test_free_singleton_assignment.py',
        'research_identity_assignment_run_20261108/replay.py','research_identity_assignment_run_20261108/simulate_recovery1.py',
        'research_identity_assignment_run_20261108/compare_recovery1.py','research_identity_assignment_run_20261108/evaluator.py',
        'scripts/association_log.py','scripts/output_filter_oracle.py']
    for path in (ROOT/'research_singleton_free_20261108').glob('*.py'):names.append(path.relative_to(ROOT).as_posix())
    for name in names:
        p=ROOT/name;ast.parse(p.read_text(encoding='utf-8-sig'));sources[name]=sha(p)
    for name in ['protocol.json']:
        path=RUN/name;sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [BASE/'comparison_report.json',BASE/'selection.json',BASE/'independent_audit.json',BASE/'singleton_inventory_fit20_recovery1.json',
        BASE/'export_complete.json',BASE/'scores_complete.json',BASE/'zero_native_outputs.npz',ROOT/'runs/z13_unmatched_box_20261025/report.json',
        ROOT/'research_identity_assignment_protocol_20261108/queries.json',ROOT/'research_identity_assignment_protocol_20261108/evaluation_strata.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,scientific_protocol_sha256=sha(RUN/'protocol.json')),new=True)
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);assert not any(e['id']==protocol['id'] for e in cat['experiments'])
    cat['experiments'].append(dict(id=protocol['id'],title='Единственная свободная рамка — возврат без изменения штатных назначений',area='flyvis',state='planned',
        started=datetime.now(MSK).date().isoformat(),question='Можно ли устранить отказ при одной допустимой свободной рамке без чужих назначений и потери качества полного трекера?',
        hypothesis=protocol['hypothesis'],expected='Сохранение исходных назначений и полезный возврат ID; источник эффекта отделён от порога и FlyVis.',
        acceptance=[dict(criterion='Нулевая правка точно воспроизводит полный исходный выход; штатные первые назначения фиксированы.',met=None),
            dict(criterion='Параметры выбраны только наfit20, сохранены до26; все первичные и теневые запросы учитываются.',met=None),
            dict(criterion='Полный пилот только по заранее заданномуJ-гейту; принятие поHOTAи ограничениям вреда.',met=None)],
        obtained='',interpretation='',decision='',cannot_claim='Свободная рамка не гарантирует правильную личность. Принципunmatched/free уже проверялся с отрицательным исходом в другой операции. Условная проверка не равна улучшению полного трекера.',
        evidence=[dict(kind='file',ref=str((RUN/'protocol.json').relative_to(ROOT)),claim='Фокус, прежний отрицательный опыт, новые действия, контроли и критерии до исполнения')]))
    write(catpath,cat)
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Для первой проблемы зарегистрирован `singleton_free_cost_v1_20261108`: сохранить штатные первые назначения и разрешить стоимостью возврат только на свободных остатках Lost/детекций с единственной геометрически допустимой рамкой иненулевымIoU. СостояниеКалмана,владениеID,очисткадубликатовне меняются. Контроли: обычное ослабление порога, геометрия, общий сдвиг, изображение,R1,DIS,FlyVisи перестановки. Параметры толькоfit20; полный20val только послеJ-гейта. Прежний отрицательный `z13_unmatched_box_v1` учтён: там подменялся полный прогнозXYWH, принципсвободной рамкисам по себе не новый. Основание — толькоfit20-инвентарь:8размеченных свободных одиночных кандидатов,7правильных возвратов прежней геометрией,1чужаянепринята (`runs/identity_assignment_20261108/singleton_inventory_fit20_recovery1.json`). В первой служебной сводке ошибочно истолкованtarget_side; метки исправлены изcurrent_identity/target_identity, геометрия и стоимости не повторялись,128проверок привязки прошли. Новых научных исходов пока нет. Запуск после завершения текущего полного пилота; проблемы2–4отложены.\n\n"
    write(RUN/'registration_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')


if __name__=='__main__':main()
