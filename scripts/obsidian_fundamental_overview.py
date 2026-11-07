"""Обзор задач: замысел → попытка → итог → следующее решение. Журнал не копируется."""
from pathlib import Path
import argparse,json,re,copy

ROOT=Path(__file__).resolve().parents[1]
TASK_HUB='Опорные задачи/Задачи и основные результаты'
TASKS=[
dict(title='Задача 1 — Получить сопоставимый базис для проверки новых модулей',
 ids=['z13_val20_native_masa_v1','z13_val20_raw_v1','tracker_zoo'],
 question='Сравнить трекеры в одинаковых условиях, чтобы следующую прибавку можно было отнести к конкретному изменению.',
 attempt='Проверили четыре исходных трекера на одних видео и детекциях, отдельно без сшивки траекторий и с одинаковой MASA (сопоставление по внешнему виду).',
 result='Сопоставимый базис получен. Это результат проверки окружения; вклад ядра FlyVis в этом сравнении не измерялся.',
 next='Новые модули сравнивать с тем же базисом, на тех же входах и с парным разбросом по видео. Не смешивать цифры разных выборок.',
 limit='Использованные видео уже участвовали в разработке; это не независимый итоговый тест.',
 outcome='Сопоставимый базис получен',
 primary=['runs/z13_val20_matched_20261024/native_masa_recovery2/report.json','runs/z13_val20_matched_20261024/raw/report.json']),
dict(title='Задача 2 — Помочь трекеру за счёт зрительного движения',
 ids=['motion_scheme_search_v1','fly_input_norm_hota_v1','fly_motion_trust_learned_v1'],
 question='Найти применение движения изображения, которое действительно помогает сопровождению объектов.',
 attempt='Проверили несколько схем поправки прогноза трекера и сравнили сигнал FlyVis с классическим оптическим потоком DIS (оценка движения по изменениям изображения).',
 result='Для классического движения полезная схема нашлась. Испытанный сигнал FlyVis в той же схеме вредил. Это отрицательный результат конкретного источника движения и способа его применения.',
 next='Возвращаться к этой задаче при обоснованной замене качества сигнала или способа применения. Отрицательная реализация не закрывает все возможные варианты движения.',
 limit='Предварительные схемы подбирались на данных разработки; их результат нельзя переносить на любой набор видео.',
 outcome='Испытанное движение FlyVis не дало полезной прибавки',
 primary=['runs/motion_scheme_search/dev.json']),
dict(title='Задача 3 — Узнавать нужный объект и выбирать полезное исправление',
 ids=['z13_box_supervision_followup_v1','z13_spatial_set_readout_v1','z13_information_ceiling_v1','z13_temporal_identity_readout_v1','z13_hybrid_fly_confirm_v1'],
 question='Научить систему выбирать полезное действие среди областей, которые она действительно может использовать.',
 attempt='Проверяли считывание активности, пространственные и временные признаки, метки личности объекта и обучение последствиям выбора полной рамки.',
 result='В наборе доступных действий полезные исправления есть, но испытанные реализации не подтвердили нужную пользу по своим критериям. Хорошее различение пар само по себе не обеспечило надёжного выбора действия.',
 next='Б1 проверяет временную поправку к сильной ветви внешнего вида; Б3 — новую память конкретных объектов. Нужна проверка полного трекера, а не только оценки сохранённых пар.',
 limit='Это вывод о проверенных реализациях. Он не доказывает отсутствия полезной информации во всей FlyVis и не является новым измерением HOTA.',
 outcome='Нужная польза испытанных считывателей не подтверждена',
 primary=['runs/z13_box_supervision_followup_20261025/strict_action_gate.json','runs/z13_information_ceiling_20261026/report.json']),
dict(title='Задача 4 — Восстановить потерянный объект без ложных продолжений',
 ids=['recovery_motion_output_control_v1','busca_output_recovery_pilot_v1','sam2_query_mask_output_pilot_v1','detection_supported_recovery_v1'],
 question='Вернуть пропущенный объект и при этом не выдавать ложные продолжения, которые ухудшают сопровождение.',
 attempt='Безусловное продолжение, выходная BUSCA и собственная масочная выдача ухудшили HOTA. Затем для Калмана и маски ввели одно правило: кандидат должен получить поддержку текущего реального обнаружения. Назначения и состояние исходного трекера сохранялись.',
 result='Поддержка текущим обнаружением дала небольшую положительную прибавку HOTA на использованных видео и для Калмана, и для маски. Дополнительная польза масочного пути над Калманом не подтверждена; это не доказательство их эквивалентности. FP и смены идентификатора всё ещё выросли.',
 next='Проверить точный перенос более дешёвого варианта Калман+обнаружение в исполняемый путь, выключенный по умолчанию. Затем требуется независимая проверка фиксированного правила. Основной baseline и пороги по этим видео не менять.',
 limit='Это предварительная польза всего проекта на повторно использованных видео. Отдельный вклад ядра или полного модуля FlyVis не измерялся. Полная BUSCA, точная LTR и плотный масочный перенос не опровергнуты. Добавка маски остаётся неопределённой.',
 outcome='Текущее обнаружение дало небольшую прибавку; дополнительная польза маски не подтверждена',
 primary=['runs/b123_20261027/recovery_motion_control/report.json','runs/b123_20261027/busca_output_pilot/report.json','runs/b123_20261027/sam2_query_output_pilot/report.json','runs/b123_20261027/detection_supported_recovery/report.json','runs/b123_20261027/detection_supported_recovery/independent_audit.json']),
dict(title='Задача 5 — Отделить пользу структуры связей мухи от расстройки обученной сети',
 ids=['fly_connectome_null_v1','fly_matched_controls_v1'],
 question='Понять, даёт ли рисунок связей FlyVis преимущество при одинаковых остальных условиях.',
 attempt='Сравнивали настоящую сеть с сетью, в которой получатели связей переставлены с сохранением числа связей и знаков.',
 result='Представление сети менялось, но такая перестановка без одинакового переобучения одновременно расстраивает уже обученную модель. Она не доказывает отдельную пользу анатомии для трекера.',
 next='Б3 должна сравнить одинаковое обучение памяти при исходных, перестроенных и обычных связях. Пользу всей системы и вклад ядра считать раздельно.',
 limit='Прежний результат нельзя объявлять доказательством превосходства структуры связей FlyVis.',
 outcome='Вклад структуры связей отдельно не установлен',
 primary=['runs/fly_reid_null/compare.json','runs/fly_reid_null/report.json'])]


def task_path(task):return 'Опорные задачи/'+task['title']


def resolved_tasks(history):
    """Один содержательный итог переноса добавляется в прежнюю задачу, без нового узла."""
    tasks=copy.deepcopy(TASKS)
    transfer=next((c for c in history['experiments'] if c['id']=='detection_support_transfer_v1'),None)
    if not transfer or transfer['state'] not in ('completed_positive','completed_negative','inconclusive') or not transfer.get('obtained'):return tasks
    task=next(t for t in tasks if t['title'].startswith('Задача 4'))
    task['ids'].append(transfer['id'])
    task['primary'] += ['runs/b123_20261027/supported_transfer/recovery2/report.json','runs/b123_20261027/supported_transfer/recovery2/independent_audit.json']
    if transfer['state']=='completed_positive':
        task['outcome']='Фиксированное правило поддержки дало прибавку на двух частях прежнего val; добавка маски не подтверждена'
        task['result']+=' Фиксированное более дешёвое правило Калман+обнаружение также дало положительный парный интервал на другой части прежнего val без пересечения с видео разработки.'
        task['next']='Сохранить опциональный модуль выключенным по умолчанию. Для итогового вывода нужны независимые данные и отдельное решение о применении; основной baseline не меняется.'
    elif transfer['state']=='completed_negative':
        task['outcome']='Частная прибавка на разработке не перенеслась: на другой части val подтверждён вред'
        task['result']+=' На другой части прежнего val фиксированное правило ухудшило HOTA. Положительный частный результат разработки сохраняется, но он не обеспечил переносимость.'
        task['next']='В основной baseline модуль не принимать. Новую замену правила проверять на заранее разделённых данных; пороги по готовым результатам не подбирать.'
    else:
        task['outcome']='Частная прибавка на разработке есть; переносимость фиксированного правила пока не подтверждена'
        task['result']+=' На другой части прежнего val польза фиксированного правила не подтверждена: парный интервал включает ноль. Это не доказательство отсутствия эффекта.'
        task['next']='В основной baseline модуль не принимать по этому результату. Сохранить опциональную реализацию и оба разных итога; дальнейшие проверки требуют заранее разделённых данных.'
    task['attempt']+=' Фиксированный дешёвый вариант перенесён в исполняемый путь и проверен на другой части прежнего val.'
    task['limit']+=' Все локальные val-видео имеют историю прежних работ; перенос на другую часть не является новым независимым итоговым тестом проекта.'
    oracle=next((c for c in history['experiments'] if c['id']=='recovery_admission_oracle_v1' and c['state']=='infrastructure' and c.get('obtained')),None)
    oracle_path=ROOT/'runs/b123_20261027/admission_oracle/report.json'
    if oracle and oracle_path.is_file():
        measured=json.loads(oracle_path.read_text(encoding='utf-8'))
        if measured['complete'] and measured['paired']['oracle_vs_baseline']['ci95'][0]>0:
            task['result']+=' Диагностический допуск с разметкой нашёл полезное сочетание тех же добавлений. Это показывает доступный пример отбора, но не качество реальной модели и не максимум HOTA.'
            task['next']='Ближайшая проверка — обучаемый допуск продолжений: подготовить отдельные обучающую и разработочную выборки и проверить, научится ли модель выбирать полезные продолжения без разметки во время работы. Неизвестную прошлую личность не превращать автоматически в отрицательную метку. Пользу проверять по HOTA полного трекера относительно того же правила без обучения; большая прибавка пока не обоснована. Для выбора следующего крупного изменения сверить уже выполненную лестницу вмешательств в детекции и уточнить резерв геометрии рамок, пропусков и сопоставления на актуальном одинаковом выходе. Отдельная проверка восстановления после перекрытия требует своей постановки. Обе работы записаны в разделе «Запланировано»; готовые расчёты не повторяются, основной базис не меняется.'
            task['limit']+=' Допуск с разметкой недоступен модели, а его неполная история истинной личности не измеряет память FlyVis. Непреодолимый предел всей системы не установлен. Проверенные выходные ветви давали лишние продолжения при неизменных исходных назначениях и состоянии трекера. Это ограничение конкретных опытов, а не доказательство бесполезности внешних методов. Следующий крупный шаг выбирается по измеренному резерву конкретных ошибок, а не по числу подключённых методов.'
            task['primary'] += ['runs/b123_20261027/admission_oracle/report.json','runs/oracle_ladder/report.json']
            coverage=ROOT/'runs/b123_20261027/admission_oracle/coverage.json'
            if coverage.is_file():task['primary'].append('runs/b123_20261027/admission_oracle/coverage.json')
    learned=next((c for c in history['experiments'] if c['id']=='recovery_admission_learning_v1' and c['state']=='inconclusive' and c.get('obtained')),None)
    learned_path=ROOT/'runs/recovery_research_20261027/learned_validation/report.json'
    if learned and learned_path.is_file():
        report=json.loads(learned_path.read_text(encoding='utf-8'))
        audit=learned_path.with_name('independent_audit.json')
        if report['complete'] and audit.is_file() and json.loads(audit.read_text(encoding='utf-8'))['passed']:
            task['ids'].append(learned['id'])
            task['attempt']+=' Затем обучили одинаковые головы допуска по числовым признакам и по реальным областям, по три сида. Веса и пороги выбраны только на обучающей и разработочной частях до полной проверки сорока прежних val-видео.'
            task['result']+=' Полная проверка обучаемого допуска завершена: вариант с областями дал около +0,20–0,23 п.п. по средней парной разности к исходному трекеру на совокупных сорока видео во всех трёх сидах. Преимущество над простым правилом поддержки обнаружением не подтверждено. Добавка внешнего вида к числовому контролю не подтверждена в одном из трёх сидов; это не доказательство эквивалентности.'
            task['outcome']='Обучаемый допуск немного лучше исходного трекера; превосходство над простым правилом не подтверждено'
            task['next']='Эту попытку допуска завершили, пороги по val не меняем. Следующее конкретное действие — ограниченная поправка рамок по текущему изображению и его прошлому наблюдению. Для геометрии и личности на актуальном выходе уже измерен значительный диагностический резерв; он не обещает достижимой прибавки. Основной baseline не меняется.'
            task['primary'] += ['runs/recovery_research_20261027/learned_validation/report.json','runs/recovery_research_20261027/learned_validation/independent_audit.json','runs/recovery_research_20261027/error_budget/recovery1/report.json']
    geometry=next((c for c in history['experiments'] if c['id']=='pixel_box_refinement_v1' and c['state']=='inconclusive' and c.get('obtained')),None)
    if geometry:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(geometry['id'])
        target['attempt']+=' Отдельно обучили поправку координат по реальным пикселям текущей области и по паре прошлой/текущей областей, при неизменных идентификаторах и состоянии трекера.'
        target['result']+=' Пиксельная поправка вне модуля FlyVis дала небольшую подтверждённую прибавку HOTA к исходному трекеру во всех трёх сидах: около +0,16–0,20 п.п. по средней парной разности на сорока прежних val. Устойчивый дополнительный вклад прошлого изображения не подтверждён. Польза пикселей сверх дешёвой калибровки рамок ещё не отделена.'
        target['outcome']='Нужная польза считывателей не подтверждена; отдельная пиксельная поправка дала малую прибавку'
        target['next']='Проверить одно более дешёвое объяснение малого эффекта — систематическую калибровку рамок без изображений. Не менять веса и пороги по val. Следующий крупный узел — основное сопоставление объектов по уже измеренному резерву; полные научные Б1–Б3 остаются отдельными незавершёнными проверками.'
        target['primary'] += ['runs/recovery_research_20261027/pixel_box_validation/report.json','runs/recovery_research_20261027/pixel_box_validation/independent_audit.json']
    numeric=next((c for c in history['experiments'] if c['id']=='pixel_box_numeric_control_v1' and c['state']=='inconclusive' and c.get('obtained')),None)
    if geometry and numeric:
        target['ids'].append(numeric['id'])
        target['result']+=' Последующий контроль без изображений завершён: линейная калибровка дала +0,170 п.п. по средней парной HOTA на тех же сорока видео. Превосходство ни одного из шести пиксельных вариантов над ней не подтверждено. Поэтому отдельную пользу пикселей эта серия не установила; эквивалентность не доказана.'
        target['next']='Геометрическую серию завершили. Следующий опыт меняет реальные спорные назначения внутри трекера с помощью внешнего вида, а затем обновляет его состояние. Сравнение включает исходный трекер и бессмысленные признаки с таким же штрафом. Готовые поправки рамок и пороги по val не перенастраиваются.'
        target['primary'] += ['runs/recovery_research_20261027/pixel_box_numeric_control/report.json','runs/recovery_research_20261027/pixel_box_numeric_control/independent_audit.json']
        task['next']='Обучаемый допуск и отдельная серия геометрии завершены. Теперь проверяется изменение реального первого сопоставления внутри трекера; исходный базис и готовые выходные ветви не перенастраиваются по val.'
    association=next((c for c in history['experiments'] if c['id']=='lazy_appearance_association_v1' and c['state']=='completed_positive' and c.get('obtained')),None)
    if association:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(association['id'])
        target['attempt']+=' Затем внешний вид реальной прошлой области и текущей детекции включили в спорное первое назначение; выбранная пара действительно обновляла состояние трекера.'
        target['result']+=' На совокупных сорока прежних val эта фиксированная ветвь дала +0,089 п.п. средней парной HOTA к исходной базе, интервал [0,004; 0,190], и превзошла три контроля с бессмысленными признаками. Каждая отдельная половина из двадцати видео остаётся неопределённой к базе. Это малая прибавка всего проекта, без ядра FlyVis.'
        target['outcome']='Реальные назначения по внешнему виду дали малую прибавку; большого прироста пока нет'
        target['next']='Проверить одну фиксированную композицию назначений и уже выбранной калибровки координат, отдельно сравнить сильную старую MASA на общих двадцати видео. Параметры по val не выбирать; основной базис сохранять.'
        target['primary'] += ['runs/recovery_research_20261027/lazy_appearance_validation/report.json','runs/recovery_research_20261027/lazy_appearance_validation/independent_audit.json']
    composition=next((c for c in history['experiments'] if c['id']=='fixed_association_geometry_composition_v1' and c['state']=='completed_positive' and c.get('obtained')),None)
    if composition:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(composition['id'])
        target['result']+=' Совместная фиксированная проверка тоже завершена: назначения с калибровкой дали +0,260 п.п. средней парной HOTA к базе на сорока видео, интервал [0,172; 0,355], с положительными интервалами к обеим отдельным поправкам. На общих первых двадцати готовая офлайн-MASA остаётся сильнее новой композиции. Та же калибровка поверх MASA дала ещё +0,175 п.п. [0,130; 0,216]; это отдельный результат на двадцати, его нельзя смешивать с показателем на сорока.'
        target['outcome']='Совместные поправки дали малую прибавку; сильная MASA на общей части остаётся впереди'
        target['next']='Заменить извлекатель людей на уже имеющийся классонезависимый MASA внутри того же правила текущего назначения. Сначала короткий причинный профиль и цена, затем отдельные полные dev/val и неизменные контроли. Это новая интеграция, старую офлайн-сшивку не повторяем.'
        target['primary'] += ['runs/recovery_research_20261027/fixed_composition/report.json','runs/recovery_research_20261027/fixed_composition/independent_audit.json']
    masa_cost=next((c for c in history['experiments'] if c['id']=='frame_masa_association_v1' and c['state']=='inconclusive' and c.get('obtained')),None)
    if masa_cost:
        target['result']+=' Полный кадровый MASA в новых назначениях на обычной числовой точности прошёл техническую пробу, но после первого полного видео не уложился в консервативную оценку стоимости. На остальных видео эта версия не проверялась; научного отрицательного вывода о качестве MASA нет.'
        target['next']='Отдельно проверить предусмотренную авторами смешанную FP16 (пониженная числовая точность): сравнить признаки и измерить цену полного видео. Только после собственного допуска проводить полную серию; старое окно и готовые FP32-выходы сохранить.'
        target['primary'].append('runs/recovery_research_20261027/frame_masa_dev/price.json')
    masa_mechanism=next((c for c in history['experiments'] if c['id']=='masa_feature_contribution_v1' and c['state']=='completed_positive' and c.get('obtained')),None)
    if masa_mechanism:
        target=next(t for t in tasks if t['title'].startswith('Задача 1'))
        target['ids'].append(masa_mechanism['id'])
        target['result']+=' Отдельная проверка нынешней замороженной головы показала:средний нейтральный MASA-вход выключил все связи и вернул исходнуюHOTA;три перестановки этого входа тоже хуже правильного. Зависимость нынешней головы от правильных признаков подтверждена. Однако в прежнем сравнении отдельно обученная сшивка безMASA также помогала,а превосходство обученнойMASA-головы надней не было подтверждено. Поэтому это не единственный доказанный способ получить прибавку.'
        target['next']='Разбор этой фиксированной связки завершён;сохранить её сильным контролем и перейти кБ1,затемБ2/Б3. Не продолжать новые покадровыеMASA.'
        target['primary'] += ['runs/recovery_research_20261027/masa_feature_contribution/report.json','runs/recovery_research_20261027/masa_feature_contribution/independent_audit.json','runs/stitch_v2/gap150_dist5/report.json']
    b1=next((c for c in history['experiments'] if c['id']=='b1_full_tracker_pilot_v1' and c['state']=='inconclusive' and c.get('obtained')),None)
    if b1:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(b1['id'])
        partial=any(e.get('ref','').endswith('b1_partial_audit/partial_source.json') for e in b1.get('evidence',[]))
        full=any(e.get('ref','').endswith('continuation_20261028/b1_tracker/independent_audit.json') for e in b1.get('evidence',[]))
        if full:
            target['result']+=' Полный пилот Б1 на двадцати видео и независимый аудит завершены. Весь вариант D выше исходного трекера, но дополнительное преимущество над той же ветвью внешнего вида, штатным ядром, входом глаза и обычной временной ветвью не подтверждено. Частная польза всей связки не изолирует вклад ядра; один начальный случай не подтверждает всю гипотезу.'
            target['primary']+=['runs/continuation_20261028/b1_tracker/report.json','runs/continuation_20261028/b1_tracker/independent_audit.json']
        elif partial:
            checked=any(e.get('ref','').endswith('b1_partial_audit/report.json') for e in b1.get('evidence',[]))
            audit_text='Готовые девятнадцать видео строго повторены на CPU, включая решения и метрики.' if checked else 'Идёт отдельная строгая проверка готовых файлов.'
            if any(e.get('ref','').endswith('b1_final_timeout_save/outcome.json') for e in b1.get('evidence',[])):
                audit_text='Дополнительный аудит девятнадцати тоже не уложился в прежний срок; финального акта нет. Готовые файлы и промежуточный прогресс сохранены, но называть все девятнадцать полностью аудированными нельзя.'
            target['result']+=' Б1 доведена до настоящих назначений и обновления состояния. После прямого поручения продолжить сохранены девятнадцать полных видео из двадцати; последнее не завершено к прежнему абсолютному сроку GPU. '+audit_text+' Это неполная серия, не научное подтверждение или опровержение Б1.'
            target['primary']+=['runs/recovery_research_20261027/b1_partial_audit/partial_source.json']
        else:
            target['result']+=' Б1 доведена до настоящего назначения и обновления состояния. Один полный видеопроход всех семи вариантов и независимый повтор завершены, но серия остановлена по заранее записанной стоимости. Этого недостаточно для вывода о пользе Б1; полных двадцати видео и парного интервала нет.'
        target['next']='Для полной Б1 требуется более дешёвое сопоставимое исполнение без подбора по первому видео. По порученному порядку следующая подготовка — научная проверка Б2, затем Б3; готовые технические пробы не повторять.'
        target['primary'] += ['runs/recovery_research_20261027/b1_tracker/recovery2/partial_result.json','runs/recovery_research_20261027/b1_tracker/recovery2/independent_audit.json']
    b1_active=next((c for c in history['experiments'] if c['id']=='b1_full_tracker_pilot_v1' and c['state']=='running' and any(e.get('ref','').endswith('continuation1/author_override.json') for e in c.get('evidence',[]))),None)
    if b1_active:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(b1_active['id'])
        target['result']+=' Б1 подключена к реальному первому назначению. Один полный видеопроход всех контролей и независимый аудит прошли; по одному видео научный вывод не делается. После первоначального ценового стопа автор прямо поручил продолжить оставшиеся девятнадцать видео с прежними весами и контролями.'
        target['next']='Завершить разрешённую серию Б1 и независимый аудит до общего срока; первое готовое видео не повторять. После результата Б1 перейти к Б2 и Б3, если остаётся разрешённое время.'
        target['primary'] += ['runs/recovery_research_20261027/b1_tracker/recovery2/partial_result.json','runs/recovery_research_20261027/b1_tracker/continuation1/author_override.json']
    history_program=next((c for c in history['experiments'] if c['id']=='history_representations_factorial_v1' and c['state'] in ('planned','ready','running')),None)
    if history_program:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['next']='Завершить текущую Б1 и независимый аудит. По новому поручению автора затем проверить изображение истории движения, долговременную память внешности и историю реальных наблюдений: три отдельно, три пары и все вместе, с общей базой и простыми контролями памяти. Сейчас подготовлена программа, а не обучение; сначала причинные входы, интерфейс и стоимость. Б2/Б3 остаются отложенными. Общий срок не продлён.'
        target['primary']+=['dissertation/Проверка трёх представлений истории — отдельно и совместно.md','runs/recovery_research_20261027/history_representations_program/program.json']
    b1_resume=next((c for c in history['experiments'] if c['id']=='b1_full_tracker_pilot_v1' and c['state']=='running' and any(e.get('ref','').endswith('continuation_20261028/authorization.json') for e in c.get('evidence',[]))),None)
    if b1_resume:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['result']+=' По новому разрешению автора 28 октября продолжается та же серия: девятнадцать готовых видео сохранены по контрольным суммам, досчитывается только последнее. Прежний незавершённый аудит не объявляется успешным; новый полный аудит сохраняет подтверждения по видео.'
        target['primary']+=['runs/continuation_20261028/authorization.json','runs/continuation_20261028/preparation.json']
    b1_full=next((c for c in history['experiments'] if c['id']=='b1_full_tracker_pilot_v1' and c['state'] in ('inconclusive','completed_positive','completed_negative') and any(e.get('ref','').endswith('continuation_20261028/b1_tracker/independent_audit.json') for e in c.get('evidence',[]))),None)
    if b1_full:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['next']='Полный пилот Б1 сохранён; готовые двадцать видео не повторять. Следующая порученная проверка — изображение истории движения, долговременная память внешности и реальные наблюдения, по отдельности и во всех сочетаниях. Нужны реальные входы, общий обучаемый интерфейс, согласованные простые контроли и фактическая стоимость, затем полная HOTA.'
    history_pilot=next((c for c in history['experiments'] if c['id']=='history_factorial_tracker_two_epoch_v1' and c['state'] in ('completed_positive','inconclusive','completed_negative') and any('history_tracker_' in e.get('ref','') and e.get('ref','').endswith('/independent_audit.json') for e in c.get('evidence',[]))),None)
    if history_pilot:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(history_pilot['id'])
        target['attempt']+=' Затем отдельно обучили все восемь сочетаний трёх представлений истории и простые контроли последнего наблюдения и усреднения. Полный трекер проверен с тремя начальными значениями, двумя эпохами и общей схемой назначений.'
        target['result']+=' '+history_pilot['interpretation']
        target['limit']+=' Новый результат относится к ограниченному двумя эпохами пилоту на сорока исторических проверочных видео. Ядро и полный модуль FlyVis в этих представлениях не используются; временное окно ограничено. Шестиэпоховая серия остановлена по стоимости.'
        target['outcome']='Матрица истории проверена в двухэпоховом пилоте; '+('главный критерий пользы пройден' if history_pilot['state']=='completed_positive' else 'главный критерий дополнительной пользы не пройден')
        target['next']='Разобрать одиночные и совместные эффекты, отделить их от простых контролей и сохранить ограничения пилота. После этой порученной матрицы вернуться к Б2, затем Б3, если позволяет разрешённое окно; готовые прогоны не повторять.'
        audit_ref=next(e['ref'] for e in history_pilot['evidence'] if 'history_tracker_' in e.get('ref','') and e.get('ref','').endswith('/independent_audit.json'))
        target['primary']+=[str(Path(audit_ref).with_name('report.json')).replace('\\','/'),audit_ref]
    elif any(c['id']=='history_factorial_tracker_two_epoch_v1' and c['state'] in ('inconclusive','running') and c.get('obtained') for c in history['experiments']):
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append('history_factorial_tracker_two_epoch_v1')
        target['attempt']+=' Для новой матрицы истории обучены тридцать моделей: все сочетания и простые контроли, три начальных значения, по две эпохи.'
        target['result']+=' Полная проверка этой матрицы на сорока видео не завершена: выполнен один полный проход тридцати двух трекеров, остальные видео не допущены по заранее заданной стоимости. Это не научный отрицательный результат качества истории.'
        target['next']='Сохранить готовые веса и первое видео. Для продолжения сверх прежнего бюджета требуется отдельное решение автора; независимая подготовка Б2 возможна без повторения её технических проб.'
        active=next(c for c in history['experiments'] if c['id']=='history_factorial_tracker_two_epoch_v1')
        if active['state']=='running':
            target['result']+=' После отдельного явного разрешения автора та же фиксированная матрица продолжается в новом окне. Готовый первый видеопроход не повторяется; исходный ценовой стоп сохранён.'
            target['next']='Завершить разрешённое продолжение и независимый аудит до нового срока, затем записать полную парную HOTA и ограничения. Не повторять готовые входы и веса; далее Б2/Б3.'
        target['primary']+=['runs/continuation_20261028/history_tracker_pilot/report.json','runs/continuation_20261028/history_tracker_pilot/price.json']
    memory_write=next((c for c in history['experiments'] if c['id']=='query_memory_write_v1' and c['state'] in ('completed_positive','completed_negative','inconclusive') and c.get('obtained')),None)
    if memory_write:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(memory_write['id'])
        target['attempt']+=' Отдельно сравнили последнее наблюдение, обычное экспоненциальное усреднение и обновление памяти по уверенности детектора при одинаковых детекциях, признаках и правиле назначения.'
        target['result']+=' '+memory_write['interpretation']
        target['limit']+=' Это проверка одного правила записи памяти, а не полного Deep OC-SORT и не ядра FlyVis; исторические двадцать видео не являются новым итоговым тестом.'
        target['primary'].append('runs/continuation_20261029/memory_write_val_audit/report.json')
        target['next']=memory_write['decision']
    admitted=next((c for c in history['experiments'] if c['id']=='query_assignment_write_v1' and c['state'] in ('completed_positive','completed_negative','inconclusive') and c.get('obtained')),None)
    if admitted:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(admitted['id'])
        target['attempt']+=' Затем отдельно проверили допуск записи памяти только после взаимно однозначной пары первого назначения, без усреднения и изменения фактического назначения.'
        target['result']+=' '+admitted['interpretation']
        target['limit']+=' Это собственная адаптация допуска записи, а не полный KeepTrack; однозначная геометрия не подтверждает истинную личность.'
        target['primary'].append('runs/continuation_20261029/assignment_write_val_audit/report.json')
        target['next']=admitted['decision']
    bounded=next((c for c in history['experiments'] if c['id']=='query_bounded_write_v1' and c['state'] in ('completed_positive','completed_negative','inconclusive') and c.get('obtained')),None)
    if bounded:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(bounded['id'])
        target['attempt']+=' Отдельно ограничили задержку записи исходным сроком жизни трека и сравнили с периодическим обновлением того же срока.'
        target['result']+=' '+bounded['interpretation']
        target['limit']+=' Обновление по сроку не подтверждает личность и применяется только после реального назначения; это не полный KeepTrack и не ядро FlyVis.'
        target['primary'].append('runs/continuation_20261029/bounded_write_val_audit/report.json')
        target['next']=bounded['decision']
    cost_use=next((c for c in history['experiments'] if c['id']=='query_cost_use_v1' and c['state'] in ('completed_positive','completed_negative','inconclusive') and c.get('obtained')),None)
    if cost_use:
        target=next(t for t in tasks if t['title'].startswith('Задача 3'))
        target['ids'].append(cost_use['id'])
        target['attempt']+=' Отдельно сравнили применение признаков: прежнюю положительную добавку, центрированную поправку и общий сдвиг без содержания внешности при прежней памяти последнего наблюдения.'
        target['result']+=' '+cost_use['interpretation']
        target['limit']+=' Выигрыш у прежней добавки сам по себе не отделяет содержание внешности от смягчения порога; это авторская адаптация сопоставления, не полный Deep SORT/BoT-SORT и не ядро FlyVis.'
        target['primary'].append('runs/continuation_20261029/cost_use_val_audit/report.json')
        target['next']=cost_use['decision']
    separated=next((c for c in history['experiments'] if c['id']=='flyvis_separate_inputs_v1'),None)
    separated_dir=ROOT/'runs/flyvis_separate_inputs_20261107'
    if separated and separated.get('done') and (separated_dir/'controller_verified.json').is_file():
        checked=json.loads((separated_dir/'controller_verified.json').read_bytes())
        if checked.get('passed') and checked.get('all_owned_ended'):
            measured=json.loads((separated_dir/'independent_audit.json').read_bytes())
            exposure=json.loads((separated_dir/'activation_exposure.json').read_bytes())
            target=next(t for t in tasks if t['title'].startswith('Задача 3'))
            values={x['name']:x['correct'] for x in measured['conditions']}
            target['ids'].append(separated['id'])
            target['attempt']+=' Проверили раздельные входные окна двух кандидатов до ядра FlyVis при прежнем масштабе и тех же кадрах, с контролем двух окон вместе.'
            target['result']+=f" На {measured['pairs']} прежних парах общая карта дала {values['common']:g} верных выборов, раздельные входы — {values['separate']:g}; главный критерий прибавки не пройден. При изменении только текущего входа с общей предысторией T4/T5 побитно не изменились во всех {exposure['pairs']} случаях: вмешательство ещё не достигло этих выходов за два внутренних обновления."
            target['limit']+=' Это ограниченные окна в прежнем поле глаза, не центрированные полные виды автомобилей; нулевой контроль одного кадра не проверяет потенциальную пользу нового текущего сигнала.'
            target['primary'] += ['runs/flyvis_separate_inputs_20261107/independent_audit.json','runs/flyvis_separate_inputs_20261107/matched_history/independent_audit.json','runs/flyvis_separate_inputs_20261107/activation_exposure.json']
    return tasks


def build(vault_path,preserve=(),authorize_masa=False,refresh_masa=False,authorize_tracking=False,authorize_core_dynamics=False):
    from scripts import build_obsidian_vault as G
    from scripts.obsidian_visible_sections import restore_saved_files
    restore_saved_files(vault_path,ROOT)
    from scripts.obsidian_checked_hypotheses import is_branch,HUB_NOTE,NOTE
    from scripts.obsidian_b123_literature import build as build_literature,INDEX,path as literature_path
    from scripts.obsidian_hypothesis_tree import build as build_tree,section,child_paths,START as TREE_START,END as TREE_END
    from scripts.obsidian_graph_policy import OVERVIEW_PATHS
    history=json.loads((ROOT/'experiments/project_history.json').read_text(encoding='utf-8'))
    cards={c['id']:c for c in history['experiments']}
    tasks=resolved_tasks(history)
    for task in tasks:
        assert all(ident in cards for ident in task['ids'])
        assert all((ROOT/source).is_file() for source in task['primary']),task['title']
    old=json.loads((vault_path/'.generated.json').read_text(encoding='utf-8')) if (vault_path/'.generated.json').exists() else []
    V=G.Vault(vault_path,preserve)
    # Обновление обычного обзора не переписывает отдельное дерево гипотез.
    from scripts.obsidian_hypothesis_tree import child_paths as protected_children
    protected=['Гипотезы.md',*[p+'.md' for p in protected_children()]]
    from scripts.obsidian_tracking_hypotheses import paths as existing_tracking_paths
    protected += [p+'.md' for p in existing_tracking_paths()]
    from scripts.obsidian_core_dynamics_hypothesis import paths as core_paths,root_section as core_root_section,build as build_core
    protected += [p+'.md' for p in core_paths()]
    protected += ['09 Запланировано/'+G.safe(c['title'])+'.md' for c in history['experiments'] if is_branch(c)]
    if authorize_masa:
        protected.remove('09 Запланировано/'+G.safe(cards['flyvis_temporal_residual_association_v1']['title'])+'.md')
    if refresh_masa:
        from scripts.obsidian_masa_hypothesis import PATH as MASA_PATH
        protected.remove(MASA_PATH+'.md')
    if authorize_tracking:
        protected.remove('Гипотезы.md')
        protected=[p for p in protected if p not in {x+'.md' for x in existing_tracking_paths()}]
    if authorize_core_dynamics:
        if 'Гипотезы.md' in protected:protected.remove('Гипотезы.md')
        protected=[p for p in protected if p not in {x+'.md' for x in core_paths()}]
    V.preserve.update(p for p in protected if (vault_path/p).exists())
    branch_cards=[c for c in history['experiments'] if is_branch(c)]
    assert len(branch_cards)==3
    card_paths={c['id']:'09 Запланировано/'+G.safe(c['title']) for c in branch_cards}
    for task in tasks:
        card_paths.update({ident:task_path(task) for ident in task['ids']})
    card_paths['b1_temporal_residual_training_v1']=card_paths['flyvis_temporal_residual_association_v1']
    G.HYPOTHESIS_PATHS.clear();G.HYPOTHESIS_PATHS.update(child_paths())
    from scripts.obsidian_tracking_hypotheses import paths as tracking_paths,root_section,build as build_tracking
    G.HYPOTHESIS_PATHS.update(tracking_paths())
    G.HYPOTHESIS_PATHS.update(core_paths())
    G.HYPOTHESIS_PATHS.update(card_paths[c['id']] for c in branch_cards)
    OVERVIEW_PATHS.clear()
    OVERVIEW_PATHS.update([TASK_HUB,INDEX,'07 Исследования/Исследования'])
    OVERVIEW_PATHS.update(task_path(t) for t in tasks)
    for task in tasks:
        path=task_path(task)
        body=f'# {task["title"]}\n\n↑ [[Обзор проекта/Эксперименты|Эксперименты]]\n\n'
        body+='**Итог проверенной попытки:** '+task['outcome']+'.\n\n'
        for key,title in [('question','Задача'),('attempt','Что попробовали'),('result','Чем закончилось'),('next','Что дальше'),('limit','Граница вывода')]:
            body+='## '+title+'\n\n'+task[key]+'\n\n'
        body+='## Первичные источники\n\n'
        for source in task['primary']:
            body+=f'[{Path(source).name}]({(ROOT/source).as_uri()}) — `{source}`.\n\n'
        body+='Полная история попыток и прежних формулировок остаётся в `dissertation/ЕДИНЫЙ_ПЛАН.md` и `experiments/project_history.json`; она не копируется в отдельные узлы.\n\n'
        if task['title'].startswith('Задача 4'):
            body+=G.ref('',literature_path('busca'),'Следующий подход — BUSCA')+'\n\n'+G.ref('',literature_path('ltr'),'Альтернатива — восстановление по маскам')+'\n\n'
        if task['title'].startswith('Задача 1'):
            from scripts.obsidian_masa_hypothesis import PATH as MASA_PATH
            body+='## Что означает прибавка связки с MASA\n\n'
            body+='Замороженный сшиватель использовал не только MASA: в оценку входили геометрия, миниатюра, гистограмма и вход глаза. Все сохранённые выходы сохранили рамки/классы/баллы/число и изменили толькоID. Для нынешней головы проверена зависимость от правильногоMASA-признака;она не равна превосходству над отдельно обученной сшивкой безMASA. Полный вывод и источники — в дочерней записи.\n\n'
            body+=G.ref('',MASA_PATH,'Проверенная связка с MASA — роль для Б1 и границы объяснения')+'\n\n'
        body+=G.ref('',HUB_NOTE,'Текущие гипотезы Б1–Б3 и способы их проверки')+'\n'
        V.write(path+'.md',G.fm({'tags':['опорная-задача','проверка-гипотезы/завершена'],
            'aliases':task['ids'],'исход':task['outcome'],'уровень':'итог задачи; не журнал исполнения'})+body)
    body='# Задачи и основные результаты\n\nОдна запись — задача, содержательная попытка, её итог и следующее решение. Промежуточные шаги не создают новых кружков.\n\n'
    for task in tasks:body+=f'[[{task_path(task)}|{task["title"]}]]\n\n{task["outcome"]}.\n\n'
    body+=G.ref('',HUB_NOTE,'Отдельное дерево текущих гипотез')+'\n'
    V.write(TASK_HUB+'.md',G.fm({'tags':['центр','обзор-графа']})+body)
    for card in branch_cards:
        path=card_paths[card['id']];existing=vault_path/(path+'.md')
        if existing.exists():text=existing.read_text(encoding='utf-8')
        else:
            text=G.fm({'tags':['проверка-гипотезы/не-завершена'],'aliases':[card['id']]})+f'# {card["title"]}\n\n'+card['hypothesis']+'\n\n'
        text=re.sub(re.escape(TREE_START)+r'.*?'+re.escape(TREE_END)+r'\s*','',text,flags=re.S).rstrip()+'\n\n'+section(card)
        V.write(path+'.md',text)
    # Прямое поручение29.10 добавляет четвёртый луч Трекинг; Б1–Б3 сохранены.
    hub='# Гипотезы\n\nТри основные гипотезы и дочерние подходы к их проверке. Связи с другими группами — только стелс-переходы.\n\n'
    for card in branch_cards:hub+=f'[[{card_paths[card["id"]]}|{card["title"]}]]\n\n'
    hub+=root_section()
    hub=G.fm({'tags':['гипотезы-узел']})+hub
    if authorize_core_dynamics and (vault_path/(HUB_NOTE+'.md')).is_file():
        hub=(vault_path/(HUB_NOTE+'.md')).read_text(encoding='utf-8')
    if core_root_section() and core_root_section().strip() not in hub:
        hub+=core_root_section()
    V.write(HUB_NOTE+'.md',hub)
    build_tree(V,history,card_paths,G.ref,G.fm)
    build_tracking(V,G)
    build_core(V,G)
    research_hub='07 Исследования/Исследования'
    build_literature(V,history,card_paths,G.ref,G.fm,research_hub)
    V.write(research_hub+'.md',G.fm({'tags':['центр','обзор-графа']})+
        '# Исследования для текущих задач\n\n'+f'[[{INDEX}|Восстановление потерянных объектов — BUSCA и маски]]\n\n'
        'Остальные прочитанные работы сохранены в исследовательских папках проекта. Подготовка и технические шаги не становятся литературными узлами.\n')
    checked='# Проверенные гипотезы — результаты и ограничения\n\nЗдесь итоги опорных задач, а не журнал всех прогонов. Зелёный означает завершённую проверку конкретной попытки, включая отрицательную; это не знак успеха всего направления.\n\n'
    for task in tasks:checked+=G.ref('',task_path(task),task['title'])+'\n\n'+task['result']+'\n\n'
    V.write(NOTE+'.md',G.fm({'tags':['проверенные-гипотезы','проверка-гипотезы/завершена']})+checked)
    home='# Проект мухи — задачи, результаты и гипотезы\n\n'
    home+='Цель — устойчивое сопровождение нескольких объектов. Обзор показывает, какую задачу решали, что получилось и что делаем дальше. Хронология и технические подробности остаются в отдельных документах проекта.\n\n'
    home+='## Открыть\n\n'
    for path,label in [(TASK_HUB,'Основные задачи и их итоги'),(HUB_NOTE,'Гипотезы → Б1–Б3 → подходы'),(research_hub,'BUSCA и восстановление по маскам'),(NOTE,'Проверенные попытки и ограничения выводов')]:
        home+=G.ref('',path,label)+'\n\n'
    home+='## Подробная история\n\n'
    home+=f'[Единый план и журнал изменений]({(ROOT/"dissertation/ЕДИНЫЙ_ПЛАН.md").as_uri()}).\n\n'
    home+=f'[Каталог всех экспериментов]({(ROOT/"experiments/project_history.json").as_uri()}).\n\n'
    home+='Эти документы не разбиваются на кружки и не копируются в хранилище. Отдельная старая доска «Карта мухи» не используется: она дублировала каталог.\n'
    V.write('00 Главная.md',G.fm({'tags':['навигация']})+home)
    from scripts.obsidian_visible_sections import build as build_sections,set_visibility,SECTIONS
    visible=build_sections(V,G,history,card_paths,tasks,task_path,research_hub)
    home=(vault_path/'00 Главная.md').read_text(encoding='utf-8')
    home+='\n## Разделы обзора\n\n'
    for title,path in SECTIONS.items():home+=G.ref('',path,title)+'\n\n'
    home+=G.ref('',research_hub,'Исследования')+'\n\n'
    home+='Все прежние файлы сохраняются в хранилище. Фильтр графа скрывает подробности, а не удаляет их.\n'
    V.write('00 Главная.md',home)
    G.graph_settings(vault_path)
    set_visibility(vault_path,visible)
    # Устаревшие производные файлы сохраняются на месте. Никакого удаления
    # или перемещения ради оформления; видимость задаёт отдельный фильтр.
    (vault_path/'.generated.json').write_text(json.dumps(sorted(set(V.written)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'[OK] Видимых заметок: {len(visible)}; прочие файлы сохранены и скрыты фильтром.')
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vault',type=Path,default=ROOT/'ObsidianFlyVis')
    parser.add_argument('--preserve-note',action='append',default=[])
    parser.add_argument('--authorize-core-dynamics',action='store_true',help='Применить прямо порученное добавление гипотезы Я1')
    args=parser.parse_args()
    return build(args.vault,args.preserve_note,authorize_core_dynamics=args.authorize_core_dynamics)
