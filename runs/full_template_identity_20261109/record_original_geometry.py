"""Record the final signal diagnostic and replace only the unrun pilot draft."""
from original_geometry import *
from record import revise

def main():
    p=read(OUT/'original_geometry_protocol.json');verify(p);a=read(OUT/'original_geometry_audit.json');s=read(OUT/'original_geometry_results.json');ch=read(OUT/'original_geometry_changes.json')
    assert a['passed'] and s['selected_development_arm']=='C34_full7_channels'
    lines=['# Полный шаблон при прежней геометрии: прямое сравнение',
        '1. Причина дополнительного контроля',
        'Предыдущая расширенная геометрия сама снизила число принятых своих с 13 до 10. Сравнение старой и полной опор внутри неё корректно, но прямое сравнение при исходных семи геометрических признаках отсутствовало. Поэтому до новых шести обучений зафиксирован объяснительный контроль. Он проведён после просмотра предыдущего результата и не является независимым подтверждением. Это третья и последняя сигнальная диагностика данной правки. Источник: original_geometry_protocol.json.',
        '2. Что изменено',
        'Старые модели, их оценки и обучающие нормировки геометрии переиспользованы. Шесть новых моделей получили только прежние семь геометрических признаков и уже готовые признаки полного шаблона. Новые карты, корреляции, разметка, пороги и штрафы не вычислялись и не подбирались. Полная опора вместо усечённой — единственное различие в прямой паре старой и новой моделей. 91 пара, 15 своих/76 чужих, 32 исключения целого видео; штраф 1 и порог 0.',
        '3. Основные результаты',
        '| Вариант | Своих принято из 15 | Чужих принято из 76 | Сбалансированная точность |',
        '| --- | ---: | ---: | ---: |']
    titles={'G_ready':'Прежняя геометрическая модель','G_support':'Дополнительный расширенный геометрический контроль','C34_old7_channels':'FlyVis34, прежняя геометрия и усечённая опора','C34_full7_channels':'FlyVis34, прежняя геометрия и полный шаблон','R134_spatial_full7_channels':'Пространственный вход34, прежняя геометрия и полный шаблон','R134_temporal_full7_channels':'Временной вход34, прежняя геометрия и полный шаблон'}
    for key,title in titles.items():
        m=s['metrics'][key];lines.append(f'| {title} | {m["own_accept"]} | {m["foreign_accept"]} | {m["balanced_accuracy"]:.6f} |'.replace('.',','))
    lines += ['Все числа: original_geometry_results.json → metrics. Это решения классификаторов отдельных связей, не реальные назначения трекера.',
        '4. Обмен ошибками',
        'Против прежнего усечённого FlyVis при той же исходной геометрии полный шаблон вернул 1 свою связь без потерь, отверг 5 прежних чужих и добавил 3 новые. Против геометрической модели он вернул 2 свои и потерял 2 другие: число 13 осталось тем же, но это не те же самые связи. Против геометрии отвергнуты 14 чужих и добавлены 4 новые. Источник: original_geometry_changes.json.',
        'Против полного пространственного входа у FlyVis на одну чужую принятую связь меньше, но свои/чужие случаи не совпадают: 1 своя возвращена и 1 потеряна, 4 чужие отвергнуты и 3 новые пропущены. Одного различия итоговых чисел недостаточно для вывода о пользе ядра.',
        '5. Допуск и неопределённость',
        'Только C34_full7_channels прошёл зарегистрированный допуск этой прямой проверки: не меньше принятых своих и строго меньше чужих, чем у исходной геометрии, расширенной геометрии и соответствующего старого7-признакового считывателя. Это допуск к реальному пилоту, не доказательство улучшения системы.',
        'Разница сбалансированной точности FlyVis с исходной геометрией +0,065789; 95% интервал [−0,038892; 0,165281]. С прежним усечённым FlyVis +0,046491; интервал [−0,020270; 0,127457]. С полным пространственным входом +0,006579; интервал [−0,102580; 0,094875]. Интервалы с поправкой на пять первичных сравнений также не подтверждают отдельную пользу ядра. Источник: original_geometry_results.json → paired_video_intervals.',
        '6. Следующий этап',
        'Новых вариантов каналов, геометрии и порога на этих парах не проверять. Следующий этап — причинный исполнитель перед общим решателем, две проверки нулевого вмешательства на 20 val-видео, затем реальные назначения, дальнейшая судьба ID и HOTA с согласованными геометрическим и входными контролями. Разметка не участвует в назначении; неизвестное в оценке остаётся неизвестным. Проект этапа: research_full_template_native_pilot_20261109/proposal_v2.json. Он ещё не выполнен.',
        '7. Проверки и проценты',
        'Независимое применение сохранённых коэффициентов воспроизвело решения: ошибка не больше 2,23·10⁻¹⁶; остаток градиента меньше 10⁻⁷. Старые нормировки, модели и оценки не переобучались. Источник: original_geometry_audit.json.',
        'Относительно прежней геометрической модели: принятые свои 0%, пропущенные чужие −38,462%, сбалансированная точность +8,631%. Это описательные проценты при интервале разницы, включающем ноль. Доля потолка не определена: потолок не измерен. Фактические ID и HOTA пока не измерены.']
    paragraphs=[]
    for line in lines:
        if line.startswith('|') and paragraphs and paragraphs[-1].startswith('|'):paragraphs[-1]+='\n'+line
        else:paragraphs.append(line)
    (OUT/'ПРЕЖНЯЯ_ГЕОМЕТРИЯ.md').write_bytes(('\n\n'.join(paragraphs)+'\n').encode('utf8'))
    proposal_dir=ROOT/'research_full_template_native_pilot_20261109';oldproposal=read(proposal_dir/'proposal.json');proposal=dict(oldproposal)
    proposal.update(version=2,prepared=datetime.now(MSK).isoformat(),state='draft_execution_not_ready_not_run',parent='runs/full_template_identity_20261109/original_geometry_audit.json',
        previous_version='proposal.json',previous_version_sha256=sha(proposal_dir/'proposal.json'),
        revision_reason='Third registered diagnostic isolated the same support change at original7 geometry; C34_full7_channels alone passed. Earlier9-geometry input-only proposal remains unchanged and unrun.',
        scope='Primary candidate is now FlyVis34 full-template/original7 geometry. Measure the same pipeline with geometry-only and two matched34-channel input controls. No core training.',
        selected_model='C34_full7_channels',
        deployment_training='Exactly91 ready records from32 videos. Original7 geometry normalization fitted on those training records, fixedlambda1,zero threshold; final G7,C34full7,R134spatialfull7,R134temporalfull7 models fitted once on the same data. No val label or score for selection.',
        arms=['native_ready_baseline_reused','G_geometry7_verification','G_C34_full_template7_verification','G_R134_spatial_full_template7_control','G_R134_temporal_full_template7_control'],
        control_roles='C34 is the admitted candidate. Input models are required matched scientific controls, not promoted candidate solutions; their failed discovery admissions are not rewritten.',
        primary='Four fixed paired HOTA contrasts of C34 versus native,G7,input-spatial34,input-temporal34, plus actual own/foreign/refusal counts and subsequent ID fate. Proposals only filter fixedalpha0.2 cost corrections BEFORE native global first assignment; beta0 and match_thresh0.8 unchanged. No alias or ID overwrite after matching.',
        inference='Same TrackEval settings and20000 paired video draws,seed42; four primary comparisons require98.75% family intervals. Secondary results are descriptive, no selected favorable video subset.',
        acceptance='Allfour adjusted lower HOTA bounds of C34 versus native,G7,spatial-input34,temporal-input34>0; nonnegative mean AssA/IDF1 changes and no increase of totalIDSw,FP,FN relative to native and matched controls. Otherwise additional FlyVis benefit is not established. Useful input/geometry behavior is reported separately without crediting core.',
        input_snapshot_verified_at=oldproposal['prepared'],input_snapshot_source='proposal.json',
        core='Same frozen flow/0000/000 checkpoint and34 decoder input channels; same2 integration steps,dt1/60,1-second initialization. Reuse exact ready maps and prefix states; restore missing34-channel maps only, using existing8-channel witnesses where available. One GPU mutex; identical core weights before/after.',
        resources='CPU arms set CUDA_VISIBLE_DEVICES empty before imports. GPU arm uses a separate process and explicit CUDA before loading the core; do not import CPU-only parent geometry.py into the GPU worker. All stages sequential,Idle,memory reserve,independent save/stop23:54MSK.',
        unready=['Causal runtime adapter and source passport','Final deployment models','Sparse missing-view recovery with cached-view parity','Both complete native zero gates on20 videos','Bounded missing34-channel recovery with parity where possible','New rollouts,TrackEval and direct ID-fate audit'])
    atomic_json(proposal_dir/'proposal_v2.json',proposal)
    (proposal_dir/'АКТУАЛЬНОЕ.md').write_bytes('# Актуальный проект проверки реальных ID\n\n[Версия2 — FlyVis с прежней геометрией и полным шаблоном](proposal_v2.json).\n\n[Исходный проект временного входа с расширенной геометрией](proposal.json) сохранён как прежняя версия. Ни один пилот пока не запускался.\n'.encode('utf8'))
    (proposal_dir/'ПРОТОКОЛ_v2.md').write_bytes('''# Реальные назначения: полный шаблон FlyVis при прежней геометрии

Это обновлённый проект следующего этапа. Исполнитель, восстановление недостающих видов и реальный пилот ещё не выполнены.

Третий заранее зафиксированный контроль показал: при прежних семи геометрических признаках полный шаблон FlyVis даёт 13 своих/16 чужих против 12/18 у старого усечения и 13/26 у геометрии. Только он прошёл свой допуск; интервалы отдельного преимущества пока включают ноль. Утверждение о прежнем9-признаковом варианте не меняется.

Для всех20 фиксированныхval-видео сравниваются нативный готовый результат, такая же геометрическая проверка и три полных34-канальных проверки: FlyVis, обычный пространственный и обычный временной вход. Входные модели нужны как согласованные контроли, а не как автоматически принятые решения. Пересечения с32 обучающими видео нет; это прежняя разработческая валидация, не новая финальная выборка.

Перед вмешательством обязательны два точных совпадения полного вывода со штатным трекером: отключённая поправка и активная ветвь с пустыми принятыми предложениями. Готовые базовые HOTA не пересчитываются. Коррекция0,2 и общий порог0,8 прежние; проверяющая модель имеет штраф1 и порог0. Воздействие только до общего решателя, без последующего переписыванияID. Нативные рождение,re_activate и удаление дубликатов остаются.

Большие обработанные кадры отсутствуют. Сначала использовать готовые проверенные представления; недостающие небольшие виды восстанавливать из исходных видео и ближайших прошлых состояний обработки. Точность доказать по сохранённым видам. Для FlyVis отдельно восстановить только недостающие34-канальные карты, проверяя совпадение с доступными8-канальными свидетелями; не повторять готовое. CPU и GPU этапы разнести, GPU один.

После технических ворот — все20 новых замкнутых прогонов, исходные/исправленные матрицы, предложения и отказы, общий выбор, публичныеID, удаление дубликатов и дальнейшая судьба номера. Прямые метки своих/чужих и ошибок отказа ставить только при имеющейся разметке; иначе неизвестно. TrackEval и парные интервалы оценивают полный результат, а не сходство карты.

Четыре первичных сравнения FlyVis с нативным трекером,геометрией и обоими входами требуют98,75%-х интервалов; не допускается рост суммарныхIDSw,FP,FN. Полные неизменные условия и входные паспорта — proposal_v2.json. Никакого результата по реальнымID из этого проекта ещё нет.
'''.encode('utf8'))
    path=ROOT/'experiments/project_history.json';catalog=read(path);card=next(c for c in catalog['experiments'] if c['id']==ID)
    card.update(state='inconclusive',done='2026-11-09',
        obtained='При прежних7 геометрических признаках и полном шаблоне C34 принял13 своих/16 чужих против12/18 у соответствующего старого усечения и13/26 у готовой геометрии. Полные входные контроли: пространственный34 — 13/17, временной34 — 13/23. Источник: runs/full_template_identity_20261109/original_geometry_results.json → metrics. Только C34_full7_channels прошёл допуск. Против геометрии вернулись2 свои и потерялись2 другие; отвергнуты14 прежних чужих и приняты4 новые. Против старогоC34 вернулась1 своя без потерь, отвергнуты5 чужих, приняты3 новые. Источник: original_geometry_changes.json. Разница сбалансированной точности с геометрией+0,065789,95% ДИ[−0,038892;0,165281]; со пространственным входом+0,006579,ДИ[−0,102580;0,094875], original_geometry_results.json → paired_video_intervals. Все коэффициенты/решения независимо проверены, original_geometry_audit.json. Старые оценки/нормировка переиспользованы; новых карт, корреляций и реальныхID нет.',
        interpretation='Добавленные признаки геометрической доступности оказались существенным отдельным изменением решения. Прямой контроль с исходной геометрией лучше отделяет эффект полного шаблона: C34 прошёл точечный допуск. Совпадение числа своих13 не означает сохранение тех же связей: есть обмен2 на2. Неподтверждённые интервалы и почти равный пространственный вход не позволяют объявлять ядро или полный трекер улучшенными.',
        decision='Завершить сигнальные варианты этой правки. Следующий этап — фактические назначения кандидата C34 с неизменными7 геометрическими признаками и согласованными входными/геометрическими контролями. Сначала две полные проверки нулевого вмешательства на всех20 val-видео и точность восстановления недостающих видов. Затем реальныеID/HOTA либо остановка версии. Подготовленная версия2 проекта — research_full_template_native_pilot_20261109/proposal_v2.json; пока не выполнена.',
        cannot_claim='Это третья объяснительная диагностика после просмотра результата расширенной геометрии, а не независимое подтверждение. Выборка91 ребро/32 видео уже использовалась в предыдущих прототипах. Порог0,штраф1 и состав случаев не подбирались, но выбор метода на этих данных остаётся разработческим. Интервалы сравнения с геометрией и обычными входами включают ноль. Реальная судьбаID и HOTA не измерены;2 потерянные свои и4 новые чужие связи сохранены в полном перечне.')
    card['acceptance'][0].update(met=True,evidence='runs/full_template_identity_20261109/original_geometry_audit.json')
    card['acceptance'][1].update(met=True,evidence='runs/full_template_identity_20261109/original_geometry_results.json → pilot_admission.C34_full7_channels')
    card['metrics']=[dict(key='candidate_identity_balanced_accuracy',value=s['metrics']['C34_full7_channels']['balanced_accuracy'],baseline=s['metrics']['G_ready']['balanced_accuracy'],subset='91 пара;32 видео;полныйC34 при прежнейгеометрии',source='runs/full_template_identity_20261109/original_geometry_results.json',locator='metrics.C34_full7_channels.balanced_accuracy')]
    card['evidence'] += [dict(kind='file',ref='runs/full_template_identity_20261109/'+name,claim='Прямая неизменная геометрия и независимая проверка') for name in ['original_geometry_results.json','original_geometry_audit.json','original_geometry_changes.json','ПРЕЖНЯЯ_ГЕОМЕТРИЯ.md']]+[dict(kind='file',ref='research_full_template_native_pilot_20261109/proposal_v2.json',claim='Новый кандидат в ещё не выполненном проекте реальныхID')]
    parent=next(c for c in catalog['experiments'] if c['id']==OUT.name)
    revise(parent,'decision',parent['decision']+' После третьей прямой проверки при исходной геометрии дальнейший кандидат изменён на C34_full7_channels; это другой зафиксированный вариант. Проект реальныхID обновлён до research_full_template_native_pilot_20261109/proposal_v2.json. Результаты9-признаковой версии не меняются.','Новый причинный контроль завершён после этого опыта; прошлое решение сохранено.')
    atomic_json(path,catalog)
    tpath=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(tpath);topics['topics'][ID]=['Сохранение личности','Выбор продолжения среди рамок','Соответствие участков карты'];atomic_json(tpath,topics)
    synth=ROOT/'research_identity_synthesis_20261109';text=(synth/'СВОДКА_v6.md').read_text(encoding='utf8').split('\n4. Текущее решение',1)[0]
    text+='''
4. Прямой контроль исходной геометрии

Добавление двух геометрических признаков само снизило приём своих13→10. Поэтому третья и последняя зарегистрированная сигнальная диагностика оставила исходные7 признаков и заменила только поддержку шаблона. Старые оценки и нормировка переиспользованы; карты/корреляции не повторялись.

Получено: C34 старый12 своих/18 чужих → полный13/16; геометрия13/26; полный пространственный вход13/17, временной13/23. Только C34_full7_channels прошёл свой допуск. Против геометрии есть2 возвращённые свои и2 потерянные,14 отвергнутых чужих и4 новых. Число13 не означает, что сохранены те же связи. Источники: runs/full_template_identity_20261109/original_geometry_results.json, original_geometry_changes.json, original_geometry_audit.json.

Интервалы преимущества над геометрией и обоими входами включают ноль. Это объяснительный выбор после просмотра предыдущего опыта, не независимое доказательство. Допуск к реальнымID подтверждён, улучшение ядра/полного трекера не подтверждено.

5. Текущее решение

Больше сигнальных вариантов на этих парах не делать. Следующий кандидат — полныйC34 при исходнойгеометрии. Следующий этап — две точные проверки нулевого вмешательства на20 val-видео, затем причинные назначения до общего решателя со штатными рождением,re_activate и удалением дубликатов. Согласованные контроли: геометрия и оба34-канальных обычных входа. Штраф1,порогмодели0,историческаягеометрическаяпоправка0,2 и общийпорог0,8 неизменны.

Версия2 проекта: research_full_template_native_pilot_20261109/proposal_v2.json. Исполнитель и реальныйпилот ещё не выполнены. Большие обработанные кадры отсутствуют; доступны проверенные исходныевидео, сохранённые состояния и готовыеR1-представления. Восстанавливать только недостающие виды с проверкойпокэшу, дляFlyVis — только недостающие карты. Проверять итоговуюсудьбуID, не подменять её локальнымпобедителем.

Прежние отрицательные и неопределённые результаты, условия и решения сохранены в версиях1–6. Цель сохранения личности остаётся открытой; дополнительной пользы ядра и улучшенияHOTA пока нет.
'''
    (synth/'СВОДКА_v7.md').write_bytes(text.encode('utf8'));(synth/'АКТУАЛЬНОЕ.md').write_bytes('# Актуальное накопленное заключение\n\n[Версия7 — прямой контроль с прежней геометрией](СВОДКА_v7.md).\n\n[Первичный отчёт](../runs/full_template_identity_20261109/ПРЕЖНЯЯ_ГЕОМЕТРИЯ.md).\n\n[Проект реальныхID, версия2](../research_full_template_native_pilot_20261109/ПРОТОКОЛ_v2.md). РеальныеID и HOTA ещё не измерены; прежние версии сохранены.\n'.encode('utf8'))
    path=ROOT/'experiments/system_problems.json';data=read(path);problem=next(c for c in data['problems'] if c['id']=='С2')
    revise(problem,'fact','Локализовано усечение опоры всеми19 сдвигами:90/91 расширились, медианы23→49, старые точки совпали точно. При исходной геометрии полныйC34 дал13 своих/16 чужих против12/18 старогоC34 и13/26 геометрии; только он прошёл свой допуск. Против геометрии2 свои возвращены,2 потеряны,14 чужих отвергнуты,4 новые приняты. Пространственный вход13/17 почти равенC34; интервалы добавки включают ноль. Это локализация потери измерений и разработческий допуск, не сохранённыеID. Источники: runs/full_template_identity_20261109/original_geometry_results.json, original_geometry_changes.json, original_geometry_audit.json; research_identity_synthesis_20261109/СВОДКА_v7.md.','Третий прямой контроль отделил исходную геометрию от двух добавленных признаков; прежние версии сохранены.')
    revise(problem,'next_check','Сигнальные варианты этой правки завершены. Реальный кандидат C34_full7_channels: исходные7 признаков,полнаяопора,штраф1,порог0. Подготовить причинный исполнитель с поправкой0,2 до общего первого назначения; две точные проверки без вмешательства на всех20 val-видео. Затем реальныеID/HOTA с той же геометрией и обоими34-канальными входами, сохраняя отказы/стоимости/назначения/дубликаты/дальнейшую судьбуID. Недостающие виды восстановить из проверенных исходныхвидео/состояний, не создавать большие массивы кадров. Проект proposal_v2.json ещё не запускался.','Допуск C34 при исходной геометрии требует следующего реального этапа, а не ещё одного диагностического балла.')
    problem['updated']=datetime.now(MSK).isoformat();problem['cases'].append(dict(date='2026-11-09',component='Прямой контроль полной опоры при исходной геометрии',fact=card['obtained'],hypothesis=card['interpretation'],discriminating_check=card['decision'],limitation=card['cannot_claim'],evidence=['runs/full_template_identity_20261109/original_geometry_audit.json','runs/full_template_identity_20261109/original_geometry_results.json']))
    if ID not in problem['cards']:problem['cards'].append(ID)
    for file in [OUT/'original_geometry_results.json',OUT/'original_geometry_audit.json',OUT/'original_geometry_changes.json',synth/'СВОДКА_v7.md',proposal_dir/'proposal_v2.json']:
        problem['evidence'].append(dict(source=file.relative_to(ROOT).as_posix(),sha256=sha(file),claim='Третья прямая диагностика и ещё не выполненный проект реальныхID.'))
    atomic_json(path,data)
    journal_atomic('Прямой контроль целого шаблона допущен к реальным назначениям',f'*Получено (`{ID}`).* {card["obtained"]}\n\n*Вывод.* {card["interpretation"]}\n\n*Решение.* {card["decision"]} {card["cannot_claim"]} Против исходной геометрии свои0%, чужие−38,462%, сбалансированная точность+8,631%; интервал включает ноль, потолок не измерен. Вторая версия проекта реальныхID сохраняет старуюнеисполненнуюверсию; никаких вал-исходов нового метода ещё нет. Все три научные диагностики поддержки завершены; далее исполнитель,техническаяпроверка,реальныеID/HOTA либоостановка.')
    print('Third diagnostic and revised native-ID proposal recorded',flush=True)

if __name__=='__main__':main()
