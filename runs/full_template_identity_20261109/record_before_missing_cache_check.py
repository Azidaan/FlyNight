"""Record support mechanism, matched outcomes and actual-ID pilot requirements."""
from geometry import *

def atomic_json(path,obj):
    temporary=path.with_suffix(path.suffix+'.full_template.pending')
    with temporary.open('w',encoding='utf8',newline='\n') as f:json.dump(cast(obj),f,ensure_ascii=False,indent=2);f.write('\n')
    temporary.replace(path)

def revise(obj,field,value,reason):
    if obj.get(field)!=value:
        obj.setdefault('revisions',[]).append(dict(date=datetime.now(MSK).isoformat(),field=field,old=obj.get(field),reason=reason))
        obj[field]=value

def journal_atomic(title,body):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n'
    marker='## Журнал изменений'.encode('utf8')+nl;assert before.count(marker)==1
    stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')
    entry=('**'+stamp+'** — '+title+'\n\n'+body+'\n\n').replace('\n',nl.decode()).encode('utf8')
    result=before.replace(marker,marker+nl+entry,1);temporary=path.with_name('ЕДИНЫЙ_ПЛАН.full_template_result.pending')
    assert not temporary.exists();temporary.write_bytes(result)
    assert path.read_bytes()==before,'Concurrent journal edit detected'
    temporary.replace(path);assert path.read_bytes()==result

def main():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'comparison_protocol.json'));a=read(OUT/'audit.json');s=read(OUT/'summary.json');g=read(OUT/'geometry_results.json');ch=read(OUT/'decision_changes.json')
    assert a['passed'] and s['selected_development_arm']=='R134_temporal_full_channels'
    lines=['# Целый пространственный шаблон: геометрия и решение о принадлежности',
        '1. Проверенный механизм',
        'Требование попадать внутрь рамки при каждом поисковом сдвиге в точности оставляет её внутреннее пересечение: [B_lo − min(s), B_hi − max(s)]. Это доказано алгебраически в МАТЕМАТИКА.md и проверено шестью тестами. Меняется только эта проверка; все ячейки нового шаблона доступны в поле глаза при всех 19 сдвигах.',
        'В 90 из 91 пары опора расширилась. Медианы числа ячеек: старая 23, полная 49, внутри прошлой рамки 77. Медианная сохранённая доля 0,302632 → 0,704762, с учётом общих ограничений прошлой рамкой и полем глаза. Это не отношение медиан. Старые ячейки и координаты сохранены точно. Источник: geometry_results.json.',
        '2. Условия сигнального сравнения',
        'Те же 91 пара: 15 своих, 76 чужих, 32 исторических видео. Старые признаки, исходные геометрические оценки и 114 готовых карт переиспользованы; нового вызова сети нет. Полная и старая опоры сравниваются при одинаковой расширенной геометрии: прежние семь признаков плюс размер новой доступной опоры и её доля от прошлой рамки. Этот контроль отделяет геометрическую доступность от сигнала. Штраф 1, порог 0, выбор сдвига прежним средним и исключение целого видео неизменны. Источник: protocol.json, comparison_protocol.json.',
        '3. Результаты',
        'В таблице указаны решения обучаемых моделей, не реальные назначения трекера. Исходная геометрическая модель G_ready переиспользуется; все остальные модели получают одну и ту же расширенную геометрию.',
        '| Вариант | Своих принято из 15 | Чужих принято из 76 | Сбалансированная точность |',
        '| --- | ---: | ---: | ---: |']
    titles={'G_ready':'Исходная геометрическая модель — готовый результат','G_support':'Геометрия с доступностью полного шаблона'}
    representation={'C34':'FlyVis34','R134_spatial':'Пространственный вход34','R134_temporal':'Временной вход34'}
    for key,m in s['metrics'].items():
        if key not in titles:
            name,mode,kind=key.rsplit('_',2);title=representation[name]+', '+('старая' if mode=='old' else 'полная')+' опора, '+('среднее' if kind=='mean' else 'каналы')
        else:title=titles[key]
        lines.append(f'| {title} | {m["own_accept"]} | {m["foreign_accept"]} | {m["balanced_accuracy"]:.6f} |'.replace('.',','))
    lines += ['Все числа: summary.json → metrics.',
        '4. Вывод по FlyVis',
        'При одной и той же расширенной геометрии поканальный FlyVis изменился с 11 своих/20 чужих до 12/15. Сбалансированная точность выше на 0,066228; 95% интервал [0,007246; 0,142549], но интервал с поправкой на четыре основных сравнения [0; 0,169555]. Отдельная польза ядра относительно равных входных контролей не подтверждена: интервалы включают ноль. Исходная геометрическая модель сохраняла 13 своих, поэтому FlyVis не прошёл допуск. Источник: summary.json → paired_video_intervals и pilot_admission.',
        'Против исходной геометрии полный FlyVis вернул 1 свою, потерял 2, отверг 15 прежних чужих и пропустил 4 новые. Против прежнего усечения при той же расширенной геометрии он вернул 1 свою без потерь, отверг 6 чужих и пропустил 1 новую. Источник: decision_changes.json → comparisons.',
        '5. Допущенный контроль без FlyVis',
        'Только полный шаблон с 34 временными признаками обычного входа прошёл заранее записанный допуск: 13 своих/22 чужих против 13/26 у исходной геометрии, 10/23 у расширенной геометрии и 11/23 у собственного старого шаблона. Против исходной геометрии те же самые 13 своих сохранены: возвращённых и потерянных своих нет. Отвергнуты 6 прежних чужих, пропущены 2 новые. Источник: summary.json, decision_changes.json.',
        'Допуск означает право на ограниченную проверку фактических ID. Разница сбалансированной точности с исходной геометрией 0,026316; 95% интервал [−0,008068; 0,068182]. Это не подтверждённое улучшение полного трекера. Другой входный контроль имеет более высокое точечное значение, но не прошёл условие относительно своего старого шаблона; выбирать его после результата нельзя.',
        '6. Следующий обязательный шаг',
        'Перед изменением реальных назначений нужны обученные итоговые модели на 32 видео, новый причинный исполнитель, точное совпадение без вмешательства со штатным трекером и парный запуск геометрического и входного вариантов. Проверка данных для такого этапа оформлена в research_full_template_native_pilot_20261109/proposal.json. Там 20 прежних val-видео без пересечения с обучением этой головы; новые условия ещё не запускались. Старые HOTA не пересчитываются. Возможный выигрыш входного контроля относится ко всей авторской системе, не к ядру FlyVis.',
        '7. Проверки и границы',
        'Независимое применение всех сохранённых коэффициентов воспроизвело решения: максимальная ошибка 2,22·10⁻¹⁶, градиент меньше 10⁻⁷. Утечки видео не найдено; допуск пересчитан независимо. Источник: audit.json. Неизвестные пары не добавлялись. Полные ID, HOTA и последующий вред конкурентам пока не измерены.',
        'Предыдущий строгий допуск обучаемой версии не менялся. У нового протокола до геометрического расчёта зарегистрировано сохранение числа принятых своих вместо обязательного его роста при одновременном строгом сокращении чужих; протокол объясняет причину. Это не исправление прежнего отрицательного результата задним числом.',
        '8. Проценты относительно исходной геометрии',
        'Допущенный временной вход: принятые свои 0%, пропущенные чужие −15,385%, сбалансированная точность +3,452%. Интервал разницы включает ноль. Доля измеренного потолка не определена: потолок не измерялся. Все величины вычисляются из summary.json → metrics/G_ready и metrics/R134_temporal_full_channels.']
    paragraphs=[]
    for line in lines:
        if line.startswith('|') and paragraphs and paragraphs[-1].startswith('|'):paragraphs[-1]+='\n'+line
        else:paragraphs.append(line)
    (OUT/'РЕЗУЛЬТАТ.md').write_bytes(('\n\n'.join(paragraphs)+'\n').encode('utf8'))
    # Concrete readiness, not an unrun scientific verdict. Only small sources
    # are verified here; large processed-image checks remain an execution gate.
    proposal_dir=ROOT/'research_full_template_native_pilot_20261109';proposal_dir.mkdir(exist_ok=True)
    full=ROOT/'runs/identity_assignment_20261108/full_precision';native=read(full/'protocol.json');train=set(read(BASE/'protocol.json')['videos']);assert not train&set(native['videos'])
    inputs={};entries={}
    for video,e in native['videos'].items():
        for key in ['detections','raw_array','baseline']:
            assert sha(ROOT/e[key])==e['sources'][e[key]];inputs[e[key]]=e['sources'][e[key]]
        clock=full/'runtime_inputs'/(video+'.json');inputs[clock.relative_to(ROOT).as_posix()]=sha(clock)
        marker=full/'processed'/(video+'.json');m=read(marker);image=full/'processed'/(video+'.float64')
        assert image.stat().st_size==m['bytes'];inputs[marker.relative_to(ROOT).as_posix()]=sha(marker)
        entries[video]=dict(detections=e['detections'],baseline_array=e['raw_array'],baseline_json=e['baseline'],clock=clock.relative_to(ROOT).as_posix(),frames=e['frames'],processed_image=image.relative_to(ROOT).as_posix(),processed_bytes=m['bytes'],processed_expected_sha256=m['sha256'])
    proposal=dict(prepared=datetime.now(MSK).isoformat(),state='draft_execution_not_ready_not_run',parent='runs/'+OUT.name+'/audit.json',
        authorization='research_identity_synthesis_20261109/authorization.json',scope='One selected input-image verification policy; FlyVis arm failed admission and cannot be claimed improved by this pilot.',
        source_protocol=full.relative_to(ROOT).as_posix()+'/protocol.json',source_protocol_sha256=sha(full/'protocol.json'),
        videos=entries,training_videos_disjoint=True,selection='All20 existing full_precision val videos in their fixed order; no new video selection by outcome. Previously used development validation, not a new untouched final test.',
        selected_model='R134_temporal_full_channels',deployment_training='Exactly91 existing feature records from32 videos, same fixedlambda1,zero threshold. Fit geometry-only9-feature model and geometry+34 R1 full-channel model once on all32 videos. Test video labels not inputs.',
        intervention='Filter the existing G geometric cost proposals by the verification decision before the native GLOBAL first-stage assignment; alpha0.2,beta0 fixed from the old G protocol, match_thresh0.8 unchanged. No posthoc alias, no forced ID, native re_activate,births and duplicate removal retained.',
        arms=['native_ready_baseline_reused','G_geometry_verification','G_R1_full_template_verification'],
        feature_contract='Same affine relative plan,19 offsets from frozen hex axes @[[13,6.5],[0,13]] times max(reference,current crop scale), same last4 actual observations plus causal missing-frame search history. Require at least2 distinct past observations and old support>=3. Insufficient geometry/history adds no cost correction in either new arm. Undefined R1 patch falls back to the same geometry model. No other detection or future annotation chooses features.',
        zero_gate='Before any measured intervention, reproduce all20 complete raw output arrays and annotated JSON twice: alpha0 disabled and active alpha0.2 with empty accepted votes. Both must exactly match archived native outputs; no HOTA recalculation for the ready baseline.',
        primary='Paired actual HOTA and ID fate versus archived native baseline and the same geometry-only verifier. Preserve full proposal refusals, original/corrected costs, global matches, public IDs and duplicate-removal events. Direct own/foreign/future-ID labels only when annotation actually supports them; otherwise unknown.',
        inference='Reuse the existing TrackEval configuration and 20000 paired video draws,seed42; two primary comparisons with97.5% family intervals. No threshold tuning or per-video outcome selection.',
        acceptance='Corrected lower HOTA bound of R1-native>0 and R1-geometry>0, nonnegative mean AssA/IDF1 changes and no increase of total IDSw,FP,FN relative to native. Otherwise no claim that the complete system is improved.',
        inputs_verified_now=inputs,unready=['Causal runtime adapter and its source passport','Final deployment models','Full checksums of actual large image inputs before use','Both complete native zero gates','New rollouts,TrackEval and direct ID-fate audit'],
        limits='No new rollout, ID result or HOTA from this proposal. Source markers and file lengths do not establish current full hashes of large image files. Reserved final test videos are not used. One CPU process,empty CUDA visibility,Idle,memory reserve and independent stop23:54MSK.')
    atomic_json(proposal_dir/'proposal.json',proposal)
    (proposal_dir/'ПРОТОКОЛ.md').write_bytes(('''# Фактическое сохранение ID: подготовка следующего этапа

Это конкретный проект следующей проверки, не готовый исполнитель и не выполненный опыт.

Основание — один заранее допущенный вариант: полный шаблон и 34 временных признака входа глаза. На 91 паре он сохранил те же 13 своих связи и сократил чужие с 26 до 22. Интервал включает ноль; нужна проверка реальных назначений.

Выбраны все 20 уже используемых val-видео. Пересечения с 32 обучающими видео головы нет; это прежняя разработческая валидация, не новая финальная выборка. Проверены текущие суммы детекций, исходных массивов трекера и базовых JSON; доступны готовые причинно обработанные изображения. Их большие полные суммы перед применением ещё предстоит проверить.

Изменение: допускать прежнюю геометрическую поправку G к стоимости только после решения проверяющей модели. Поправка 0,2, порог общего решателя 0,8; новые значения не подбираются. Решение принимает штатный общий решатель до обновления треков; рождение, восстановление и удаление дубликатов сохраняются. Переназначение номеров после решения не используется.

Перед вмешательством новый исполнитель обязан воспроизвести исходные массивы и JSON на всех 20 видео: с отключённой поправкой и с активной ветвью при пустом наборе принятых предложений. Готовые базовые HOTA повторно не считаются.

Парные новые варианты — та же геометрическая проверка и та же проверка с обычным временным входом. Нужно сохранить все отказы, изменённые стоимости, общие назначения, публичные ID и удаление дубликатов; проверять правильные/чужие продолжения, ошибочные отказы и последующую судьбу номера. Неизвестные не достраивать.

После двух проверок нулевого вмешательства — полный фиксированный пилот на 20 видео, TrackEval и парные интервалы по прежним настройкам. Положительный вывод требует улучшения относительно штатного трекера и геометрического контроля без роста IDSw, FP и FN. Полные критерии, входы и недостающие части перечислены в proposal.json. Польза этой ветви относится к авторской системе; отдельного вклада ядра FlyVis она не доказывает.
''').encode('utf8'))
    path=ROOT/'experiments/project_history.json';catalog=read(path);card=next(c for c in catalog['experiments'] if c['id']==OUT.name)
    card.update(state='inconclusive',done='2026-11-09',
        obtained='В 90/91 пар полная опора больше; медианы 23→49 ячеек при77 ячейках прошлой рамки, медианные доли0,302632→0,704762. Старые координаты совпали точно,6 тестов пройдены. Источник: runs/full_template_identity_20261109/geometry_results.json. При одинаковой расширенной геометрии C34 old_channels 11 своих/20 чужих → full_channels12/15; временной вход34 11/23→13/22; пространственный вход34 11/19→13/20. Исходная геометрическая модель13/26, расширенная10/23. Источник: runs/full_template_identity_20261109/summary.json → metrics. Только R134_temporal_full_channels прошёл допуск. Против исходной геометрии он сохранил те же13 своих, отверг6 старых чужих и принял2 новые. Источник: decision_changes.json. Его разность сбалансированной точности+0,026316,95% ДИ[−0,008068;0,068182], summary.json → paired_video_intervals. Новых вызовов сети и назначенийID нет; audit.json.',
        interpretation='Механизм потери пространственной поддержки локализован до проверки принадлежности всем сдвинутым рамкам и подтверждён точной заменой этой проверки. Сохранение измерений дало точечную пользу считывателю, но не подтвердило отдельную пользу ядра FlyVis. Только один входной контроль допущен к фактическим ID; это ещё не улучшение полного трекера. Оба технических критерия допуска выполнены, научный итог по личности остаётся неопределённым до полного пилота.',
        decision='Сохранить новую опору и все неблагоприятные случаи. Не объявлять FlyVis улучшенной: её вариант потерял одну свою связь относительно исходной геометрии по итоговому счёту. Подготовить фактический пилот только для зарегистрированного временного входного варианта, с тем же геометрическим контролем и обязательными двумя проверками нулевого вмешательства. Подготовлены реальные20 val-входов без пересечения с обучением: research_full_template_native_pilot_20261109/proposal.json; исполнитель и запуски ещё не готовы.',
        cannot_claim='Увеличение опоры не доказывает, что все добавленные ячейки принадлежат машине. 91 ребро из32 исторических видео не является91 независимым сюжетом; выборка уже использовалась в предшествующем прототипе. Первичный C34 выигрыш над старым усечением не подтверждён интервалом с поправкой на4 сравнения; отдельного выигрыша над входом нет. Решения классификаторов не равны реальным назначениям, сохранённымID или HOTA. Пилотный допуск изменён и зарегистрирован до новой геометрии, прежний отрицательный критерий не пересмотрен.')
    card['acceptance'][0].update(met=True,evidence='runs/'+OUT.name+'/geometry_results.json')
    card['acceptance'][1].update(met=True,evidence='runs/'+OUT.name+'/audit.json → admitted_arm=R134_temporal_full_channels; admission only')
    card['metrics']=[dict(key='candidate_identity_balanced_accuracy',value=s['metrics']['R134_temporal_full_channels']['balanced_accuracy'],baseline=s['metrics']['G_ready']['balanced_accuracy'],subset='91 строгая последняя опора;32 видео;полный временной вход34',source='runs/'+OUT.name+'/summary.json',locator='metrics.R134_temporal_full_channels.balanced_accuracy')]
    card['evidence'] += [dict(kind='file',ref='runs/'+OUT.name+'/'+name,claim='Фиксированные геометрия, сигналы и независимая проверка') for name in ['geometry_results.json','comparison_protocol.json','summary.json','audit.json','decision_changes.json','РЕЗУЛЬТАТ.md']]+[dict(kind='file',ref='research_full_template_native_pilot_20261109/proposal.json',claim='Подготовка следующего реального этапа, не выполненный результат')]
    atomic_json(path,catalog)
    topics_path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(topics_path);topics['topics'][OUT.name]=['Сохранение личности','Выбор продолжения среди рамок','Соответствие участков карты'];atomic_json(topics_path,topics)
    synthesis=ROOT/'research_identity_synthesis_20261109'
    text=(synthesis/'СВОДКА_v5.md').read_text(encoding='utf8')
    # Replace stale current hypotheses/action, keeping every old version intact.
    text=text.split('\n3. Ранжированные рабочие объяснения',1)[0]
    text+='''
3. Обновлённая локализация

Подтверждено кодом, алгеброй и данными: пересечение всех 19 сдвигов внутри рамки кандидата убирало измерения прошлого шаблона. После замены только этой проверки старые точки и значения координат сохранены точно, в 90 из 91 пары доступно больше ячеек: медианы 23→49. Источник: runs/full_template_identity_20261109/geometry_results.json. Это доказательство механизма усечения, не принадлежности добавленных ячеек объекту.

При одинаковой расширенной геометрии поканальный FlyVis дал 11 своих/20 чужих на старой опоре и 12/15 на полной. Однако исходная геометрическая модель сохраняла 13 своих, а отдельное преимущество ядра над двумя34-канальными входами не подтверждено. Источник: runs/full_template_identity_20261109/summary.json, audit.json.

Полный временной вход34 прошёл заранее зафиксированный допуск: сохранил те же13 своих и снизил чужие26→22 относительно исходной геометрии. Отвергнуты6 прежних чужих, добавлены2 новые, собственных потерь/возвратов против этой геометрии нет. Интервал общей разницы включает ноль. Источник: runs/full_template_identity_20261109/decision_changes.json, summary.json. Это основание для проверки реальныхID, не готовое доказательство улучшения системы.

Сильнейшая альтернатива — пользу дают геометрия и обычные пространственно-временные структуры, а не специфическая динамика FlyVis. Все модели имеют согласованные геометрические и входные контроли; фактические назначения ещё не измерены. Конкуренция треков и отсутствие настоящего кандидата остаются отдельными подтверждёнными ограничениями.

4. Текущее решение

Предыдущая обучаемая версия сохраняет отрицательный статус. Новая полная опора не даёт права автоматически передавать ID по FlyVis. Следующий конкретный этап — отфильтровать прежние геометрические предложения перед общим нативным решателем проверкой временного входа, сравнить с такой же геометрической моделью, сначала доказав точное нулевое вмешательство на всех20 доступныхval-видео. Эти видео не пересекаются с32 обучающими; это прежняя разработческая валидация, не новая финальная выборка.

Параметры геометрической поправки остаются историческими:0,2 и порог0,8; классификатор имеет штраф1 и порог0. Никаких псевдо-ID после решателя. Подготовка входов и границы запуска: research_full_template_native_pilot_20261109/proposal.json. Исполнитель, фактическая проверкаID и HOTA ещё не выполнены.

Ранний шаблон сам по себе, нулевой обратный возврат, прежние фоновые фильтры и настройка порога на просмотренных эпизодах не повторяются без нового механизма. Исходные результаты, ограничения и прежние заключения находятся в версиях1–5 и первичных отчётах. Главная цель сохранения личности остаётся открытой.
'''
    (synthesis/'СВОДКА_v6.md').write_bytes(text.encode('utf8'))
    (synthesis/'АКТУАЛЬНОЕ.md').write_bytes(('# Актуальная накопленная постановка\n\n[Текущая сводка — версия6](СВОДКА_v6.md).\n\n[Целый шаблон: механизм и сравнение](../runs/full_template_identity_20261109/РЕЗУЛЬТАТ.md).\n\n[Подготовка проверки реальныхID](../research_full_template_native_pilot_20261109/ПРОТОКОЛ.md). Исполнитель ещё не готов; прежние версии сохранены.\n').encode('utf8'))
    spath=ROOT/'experiments/system_problems.json';problems=read(spath);problem=next(r for r in problems['problems'] if r['id']=='С2')
    revise(problem,'fact','Локализовано усечение поддержки всеми19 сдвигами внутри рамки кандидата: новая опора сохраняет прежние точки точно и расширяется в90/91 пар, медианы23→49 ячеек. Полный поканальный FlyVis дал12 своих/15 чужих против11/20 на старой опоре при одинаковой расширенной геометрии, но не сохранил13 своих исходной геометрии. Полный временной вход34 сохранил те же13 своих и снизил чужие26→22; только он прошёл допуск. Интервал общего выигрыша включает ноль, реальныеID пока не измерены. Источники: runs/full_template_identity_20261109/geometry_results.json, summary.json, decision_changes.json, audit.json; накопленное заключение research_identity_synthesis_20261109/СВОДКА_v6.md.','Новая качественно отличная проверка опоры завершена; прежний отрицательный вариант не переписан.')
    revise(problem,'next_check','Подготовить и выполнить причинный пилот с выбранным временным входом: готовые20 val-видео без пересечения с32 обучающими; прежние поправка0,2 и общий порог0,8. Сначала полное совпадение нового исполнителя с нативными массивами/JSON при alpha0 и при активной ветви с пустыми принятыми предложениями. Затем одинаковые геометрическая и входная проверки до глобального назначения, с нативными рождением, re_activate и удалением дубликатов. Сохранять отказы и судьбуID; неизвестное не додумывать. Подготовленный проект research_full_template_native_pilot_20261109/proposal.json пока не является выполненным опытом. FlyVis-ветвь не прошла свой допуск.','Первая и вторая диагностики опоры пройдены; следующий шаг должен измерить реальную цепочку назначения, а не ещё один балл сходства.')
    problem['updated']=datetime.now(MSK).isoformat();problem['cases'].append(dict(date='2026-11-09',component='Полный пространственный шаблон и общая поддержка19 сдвигов',fact=card['obtained'],hypothesis=card['interpretation'],discriminating_check=card['decision'],limitation=card['cannot_claim'],evidence=['runs/'+OUT.name+'/audit.json','runs/'+OUT.name+'/summary.json']))
    if OUT.name not in problem['cards']:problem['cards'].append(OUT.name)
    for path in [OUT/'geometry_results.json',OUT/'summary.json',OUT/'audit.json',OUT/'decision_changes.json',synthesis/'СВОДКА_v6.md',proposal_dir/'proposal.json']:
        problem['evidence'].append(dict(source=path.relative_to(ROOT).as_posix(),sha256=sha(path),claim='Механизм опоры, согласованное сравнение и ещё не выполненная подготовка проверкиID.'))
    atomic_json(spath,problems)
    litpath=ROOT/'experiments/useful_literature.json';lit=read(litpath);assert not any(v['id']=='bertinetto2016_full_template' for v in lit['sources'])
    lit['sources'].append(dict(id='bertinetto2016_full_template',title='Целый пространственный шаблон в поисковом поле — Bertinetto и соавторы',problem_group='Сохранение личности',method_group='Соответствие пространственных шаблонов',publication='Luca Bertinetto, Jack Valmadre, Joao F. Henriques, Andrea Vedaldi, Philip H. S. Torr. Fully-Convolutional Siamese Networks for Object Tracking,2016; прочитана arXiv-версияv3. DOI ниже относится к записи arXiv.',doi='10.48550/arXiv.1606.09549',url='https://arxiv.org/pdf/1606.09549',sections='Раздел2.1, формула2: пространственный шаблон и взаимная корреляция в большем доступном поисковом поле.',established='Взаимная корреляция пространственных представлений позволяет оценить переведённые окна поискового изображения одним фиксированным шаблоном. Авторы обучали собственное представление; это не доказательство для FlyVis.',problem='Проверить, не теряет ли наш считыватель полезную опору из-за требования оставаться внутри рамки при всех сдвигах.',application='Принцип помог выбрать одно изменение: сохранить полный прошлый шаблон в поле глаза и удерживать одинаковые ячейки на всех сдвигах. Алгебраически и по данным установлено прежнее усечение; выполнено согласованное сравнение с входными контролями.',limitation='Полный трекер из статьи не переносился. Прибавка реальныхID и отдельная польза ядра FlyVis не доказаны; прошёл только диагностический допуск входного варианта.',evidence=['runs/'+OUT.name+'/МАТЕМАТИКА.md','runs/'+OUT.name+'/geometry_results.json','runs/'+OUT.name+'/summary.json','runs/'+OUT.name+'/audit.json'],related=['sys:С2']))
    required={'id','title','problem_group','method_group','publication','doi','url','sections','established','problem','application','limitation','evidence','related'}
    assert all(required<=v.keys() for v in lit['sources']);atomic_json(litpath,lit)
    journal_atomic('Целый шаблон сохранил измерения; допущен один входной вариант для реальныхID',f'*Получено (`{OUT.name}`).* {card["obtained"]}\n\n*Вывод.* {card["interpretation"]}\n\n*Решение.* {card["decision"]} {card["cannot_claim"]} Подготовка20 val-видео проверена по малым входам: сырые детекции, исходные треки, JSON и часы; большие полные суммы обработанных изображений ещё не проверены. Научный пилот не запускался. Относительно исходной геометрии выбранный вход: свои0%, чужие−15,385%, сбалансированная точность+3,452%; интервал включает ноль, потолок не измерен. Литература SiamFC сохранена как основание фиксированного шаблона, не как решение проблемы. Полный отчёт: `runs/{OUT.name}/РЕЗУЛЬТАТ.md`.')
    (OUT/'status.json').write_bytes(json.dumps(dict(stage='recorded_inconclusive',admitted='R134_temporal_full_channels',actual_ID_pilot_not_run=True,updated=datetime.now(MSK).isoformat())).encode('utf8'))
    print('Mechanism, matched signals and concrete native-ID proposal recorded',flush=True)

if __name__=='__main__':main()
