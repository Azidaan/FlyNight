"""Register a no-image recency control without changing the running study."""
from research_singleton_age_20261108.common import *


def main():
    assert read(OWN/'independent_audit.json')['passed'];assert not (RUN/'protocol.json').exists()
    protocol=dict(id='singleton_recency_control_v1_20261108',registered=datetime.now(MSK).isoformat(),active_problem=1,deferred=[2,3,4],deadline='2026-11-08T14:00:00+03:00',
        observation='The4new known correctFlyVis returns in the conditional26cohort all have gap2frames: UID30818,33601,33988,36902. This motivated a recency control; it does not prove the cause.',
        question='Does own-template spatial similarity add identity evidence beyond a simple restriction on time since the last observation?',
        hypothesis='A fixed lost-age gate may explain some or all apparent content benefit without image orFlyVis.',
        intervention='Exactly the same locked native pairs, free sole-geometric edges, C<1, same-classLost restriction, fixedalpha0.2 and common>=3reference-cell/DIS-domain support asSUPPORT. Allow cost discount only when frame-last_history_frame<=selected_cap. No image values orFlyVis calls.',
        calibration=dict(cohort='Exactly the previousfit20calibration rows:73known positives/negatives, unknowns retained separately; no evaluate26labels for selection.',
            caps=[0,1,2,3,4,6,8,12,20,30],budget=10,
            objective='SameTPR-FPR objective as the spatial gates; tie fewer false positives, more true positives, then smaller cap. Cap0rejects allLost recovery.',
            freeze='Write selection before calculating AGEoutcomes on26or full20val. No retuning of the running spatial study.'),
        conditional='Same641known and727shadow queries. Reuse all complete prior first-match outcomes under exact full first-match-set identity.',
        full_gate='After the running full study and its independent audit: perform fixedAGE20val if AGEhas positive lower95%deltaJ vsB0with no extra known wrong assignments, or if the running selected system is accepted and needs the recency explanation checked.',
        full_comparisons=[['AGE','B0'],['FLYVIS','AGE'],['IMAGE','AGE']],
        joint_analysis=dict(family=14,samples=20000,seed=42,unit='paired_video',confidence_percent=100*(1-.05/14),
            role='Separate explanatory follow-up: conservative joint family of the running study11planned comparisons plus3new ones. Original11-comparison results remain unchanged.',
            no_harm='Mean-videoAssA/IDF1 nonnegative; totalIDSw/FP/FN and known wrong-target returns do not increase.',
            identity_closure='A useful AGEresult also needs more known correct returns and more sustained-success returns thanB0; short-gap success does not establish reliable long-occlusion recovery.',
            signal_claim='No incrementalFlyVis claim without outperformingAGEand all previously matched controls under the stronger joint bound.'),
        limitation='This follow-up was motivated after inspecting the historical26result and uses previously studied data. It is not an independent final test; gains confined to short gaps cannot be generalized to long occlusions.',
        literature=dict(url='https://arxiv.org/pdf/1703.07402',section='2.3MatchingCascade',
            role='DeepSORT prioritizes recently seen tracks when uncertainty grows. The hard cap tested here is a simple separate control, not its full matching cascade.'),
        stop='Do not repeat finished maps, inference or original full trajectories. Preserve all original criteria/results. Problems2–4remain deferred.')
    write(RUN/'protocol.json',protocol,new=True)
    catalog_path=ROOT/'experiments/project_history.json';catalog=read(catalog_path)
    assert not any(c['id']==protocol['id'] for c in catalog['experiments'])
    old=next(c for c in catalog['experiments'] if c['id']=='singleton_own_template_v1_20261108');assert old['state']=='planned';old['state']='running'
    catalog['experiments'].append(dict(id=protocol['id'],title='Давность пропуска — контроль возврата без изображения и FlyVis',area='flyvis',state='planned',started='2026-11-08',
        question='Может ли одна давность пропуска объяснить прибавку проверки по прошлой пространственной карте?',
        hypothesis='Разрешение только недавних потерянных треков может воспроизвести часть эффекта без зрительного содержания.',
        expected='Отделить полезность короткого прогноза от содержимого карты; полезный вариант безFlyVisсчитать результатом всей системы.',
        acceptance=[dict(criterion='Одинаковые штатные пары, геометрия, стоимость0,2и общая пространственная поддержка; меняется только допуск по давности.',met=None),
            dict(criterion='Граница выбрана на прежних73известныхfit20примерах доAGEисходов26и20val.',met=None),
            dict(criterion='Полный пилот по заранее заданному условию; отдельные парные сравнения, вред и устойчивость правильногоID.',met=None)],
        obtained='',interpretation='',decision='',cannot_claim=protocol['limitation'],
        evidence=[dict(kind='file',ref='runs/singleton_age_20261108/protocol.json',claim='Контроль и условия до его настройки и запуска'),
            dict(kind='file',ref='runs/singleton_verify_20261108/comparison_report.json',claim='Исходный пространственный результат не переписывается')]))
    write(catalog_path,catalog)
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Сильнейшая дополнительная альтернатива пространственному объяснению первой проблемы: все четыре новых известных правильных возвратаFlyVisимеютдавность2кадра(UID30818,33601,33988,36902; `runs/singleton_verify_20261108/outcomes/evaluate26_FLYVIS.json`, `runs/singleton_free_20261108/free_verification_diagnosis.json`). Это наблюдение, не причина. Зарегистрированконтроль `singleton_recency_control_v1_20261108`: теже свободные пары, геометрия, общаяподдержка искидка0,2, но допуск только по давности без изображений/FlyVis. Максимальный пропуск будет выбран только на прежних73известныхfit20примерах с темже бюджетом10границ и темжеTPR−FPRкритерием. Работа начнётся последовательно после текущего полного пилота и его аудита; его источники, параметры, критерии и результаты не меняются. Совместная поясняющая оценка новых трёх сравнений используетконсервативную семью14, отдельно от исходной семьи11. Основание — DeepSORT§2.3(https://arxiv.org/pdf/1703.07402), не утверждение о реализации полногоDeepSORT. Сопоставление мотивировано уже увиденным историческим материалом; успех только коротких пропусков не доказывает восстановление после длинного перекрытия. Условия `runs/singleton_age_20261108/protocol.json`; остальные проблемы отложены.\n\n"
    write(RUN/'registration_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    sources={n:sha(ROOT/n) for n in ['research_singleton_age_20261108/common.py','research_singleton_age_20261108/register.py',
        'runs/singleton_age_20261108/protocol.json','runs/singleton_verify_20261108/calibration_rows.json',
        'runs/singleton_verify_20261108/selection.json','runs/singleton_verify_20261108/comparison_report.json','runs/singleton_verify_20261108/independent_audit.json']}
    write(RUN/'registration_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,scientific_running_study_unchanged=True),new=True)
    status('recency_control_registered_waiting_for_full_audit',deadline=protocol['deadline'])


if __name__=='__main__':main()
