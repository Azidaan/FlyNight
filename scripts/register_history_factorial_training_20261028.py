"""Протокол до обучения; готовую матрицу не выбирать поval."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
from scripts.b123_safe_journal_20261027 import journal
read,write,sha,now=service.read,service.write,service.sha,service.now


def main():
    from tracking.history_pair_features import VARIANTS
    assert read(RUN/'history_pair_model_real_interface.json')['passed']
    assert '4 passed' in (RUN/'history_pair_model_unit.log').read_text(encoding='utf-8')
    folder=RUN/'history_factorial_training';assert not (folder/'protocol.json').exists()
    files=['tracking/history_pair_features.py','tracking/history_pair_model.py','scripts/train_history_factorial_20261028.py',
        'scripts/audit_history_factorial_training_20261028.py','tests/test_history_pair_model.py',
        'runs/continuation_20261028/history_pair_dataset/report.json','runs/continuation_20261028/history_pair_dataset/protocol.json',
        'runs/continuation_20261028/history_visual_inputs/protocol.json',
        'runs/continuation_20261028/history_pair_model_real_interface.json']
    p=dict(created=now(),parent_experiment='history_representations_factorial_v1',budget_seconds=5400,
        seeds=[42,43,44],variants=list(VARIANTS),epochs=6,batch_size=256,learning_rate=.0005,weight_decay=.0001,
        precision='FP32; noTF32; deterministic torch/CUBLAS; frozen common visual extractor stays unchanged',
        architecture='Одна архитектура всем:Conv3→8→16→16,64x64,avg4x4/Linear32;GRUCell260→32 внешность;GRUCell8→16 наблюдения;Linear14→32 геометрия;112→32→1. БезDropout/BatchNorm. Одинаковая инициализация внутри каждого начального значения, разные обучаемые веса вариантов.',
        information='1 добавляет только канал давности изменений к общим текущему серому/маске;3 — все реальные признаки/возраст/качество;4 — последовательность рамок с переносом камеры/возрастом/качеством. До6наблюдений/60кадров≈2с, не произвольно долгая память. У0 уже есть Калман и последнее наблюдение; это не отсутствие истории. Число параметров одинаково, но у отключённого канала давности нет информационной нагрузки.',
        controls='0/1/3/4/13/14/34/134:отключённая история заменена последним доступным наблюдением,повторённым6раз для той же архитектуры. mean/1mean отдельно обучены:простое арифметическое усреднение доступных признаков/геометрии,повторённое6раз;1mean сохраняет MHI для сильного контроля134. Это собственная адаптация,не полныеMeMOTR/OCSORT.',
        camera='Только перенос320x180. Реальная прошлая рамка переносится current_camera-past_camera и нормируется текущей рамкой. Разныеcamera_epoch→маска геометрической истории. Внешность в реальном прошлом кадре сохраняется независимо от сброса камеры.',
        labels='Прежняя фиксированная цель:та же личность,что в последней реально назначенной прошлой детекции. GTтолькоtargets. После захвата чужой личности это может не исходныйID: не обучение восстановления исходногоID. БудущаяHOTA проверит итог; различие не объявляется причиной старогоБ1.',
        loss='BCEwithlogits,unknown=-1 полностью исключён; весаN/(2*Nclass) считаются ТОЛЬКОfit84 и одинаковы всем; score после балансировки не называется калиброванной вероятностью истинной личности.',
        splits='84fit84train/20fit20dev из готовых104; ни одного val для обучения/выбора. Детектор виделtrain:не доказательство его обобщения.',
        selection='Все6эпох всех30моделей;checkpoint с минимальной взвешенной dev-потерей,приравенстве ранний. Ни пороги,ни сочетания поval не выбирать.',
        future_action='Только первое назначение на прежних query-временах и прежних геометрически допустимых парах того же класса:cost+.2*(.5-sigmoid(score)),предел±.1. Alpha=.2 фиксируетсяСЕЙЧАСдообучения/val,не подбирается. Новых рёбер вне исходного порога нет;delta0 точно сохраняет baseline;второе/неподтверждённое назначение неизменны.',
        price_rule='Первый ПОЛНЫЙ train+dev проход одной эпохи0seed42 сохраняется:2*time*(6*30-1)+300<остаток5400с. Приfalse сохранитьepoch/optimizer/цену,не продолжать179,непродлевать. Приtrue первый не повторять.',
        audit_forward_tolerance=.00003,audit_loss_tolerance=.000002,
        acceptance_scope='Толькоготовностьсопоставимыхвесов. HOTAнеизмеряетсяобучением.',
        primary_scientific_criterion='После отдельной интеграции/цены:134 лучше native/0/mean/1mean по полнойпарнойHOTA наоднихвидео во всех3начальныхзначениях;обычные95%ДИ и поправка12первичных сравненийBonferroni. Отдельно все8сочетаний/взаимодействия/AssA/IDF1/FP/FN/IDSw,безсложенияэффектов. Неполный расчёт не отрицательная наука.',
        literature=[dict(url='https://www.cs.bu.edu/faculty/betke/cs591/papers/bobick-davis.pdf',role='Образдавностидвижения,статьяо распознаваниидвижений,не нашаHOTA'),
            dict(url='https://arxiv.org/abs/2307.15700',role='Памятьвнешности; собственнаяGRUнеполныйавторскийTransformer'),
            dict(url='https://arxiv.org/abs/2203.14360',role='Использованиереальныхнаблюдений;мы не воспроизводим всюOCSORT')],
        sha256={name:sha(ROOT/name) for name in files})
    write(folder/'protocol.json',p)
    history=read(ROOT/'experiments/project_history.json');ident='history_factorial_training_v1';assert not any(c['id']==ident for c in history['experiments'])
    history['experiments'].append(dict(id=ident,title='Одинаково обучить представления истории и простые контроли',area='flyvis',state='ready',started='2026-10-28',
        question='Можно ли подготовить сопоставимые обученные варианты0/1/3/4/13/14/34/134 и среднее без подбора поval?',
        hypothesis='Гипотезаготовности:в разрешённой стоимости возможны30голов одной архитектуры. Дополнительнаяпользаисторииполномутрекеру этимэтапом не проверяется.',
        expected='Все10вариантов×3начальныхзначения,по6эпох,фиксированнаяBCE/fit84/fit20/unknownмаска/отдельнообученныепоследнийисреднее;фактическаяцена после первойполнойэпохи.',
        acceptance=[dict(criterion=s,met=None) for s in ['Полные30голов/6эпохсодинаковымиархитектурой/инициализациейвнутриначальногозначения;цена прошла',
            'GTтолькоtargets,unknownисключён,весаклассовтолькоfit84;источники/разбиения/хеши неизменны',
            'Независимоповторенывсеdevпотери и первыйполныйdevвидеопроходвсех30наCPUсзаранеезаданнойточностью']],
        evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='Архитектура/контроли/цель/обучение/фиксированное действие/бюджетдообучения')],metrics=[],confirmed_results=[]))
    write(ROOT/'experiments/project_history.json',history)
    for name,sub,script,budget in [('070_history_model_checks','history_model_checks','scripts/check_history_pair_model_20261028.py',600),
        ('075_history_factorial_training','history_factorial_training','scripts/train_history_factorial_20261028.py',5400),
        ('080_history_factorial_training_audit','history_factorial_training_audit','scripts/audit_history_factorial_training_20261028.py',900)]:
        f=RUN/sub
        if sub!='history_factorial_training':write(f/'protocol.json',dict(parent_experiment=ident,budget_seconds=budget,sha256={script:sha(ROOT/script)}))
        write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=f.relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(f/'protocol.json')))
    write(folder/'prelaunch.json',dict(created=now(),unit_log='runs/continuation_20261028/history_pair_model_unit.log',initial_prelaunch_error='ВпервойпроверкесреднееNumPyповысилосьдоFloat64;явныйFloat32исправленДОзаморозки/обучения. Наука/GPU060неизменны.',training_started=False,hota_measured=False))
    journal(ident,'Зарегистрировано одинаковое обучение представлений истории и простых контролей',
        'Источник runs/continuation_20261028/history_factorial_training/protocol.json:дообучения зафиксированы8сочетаний/последний/простое среднее/1+среднее,3начальныхзначения42/43/44,6эпох/общаяархитектура/fit84fit20/взвешеннаяBCE/unknownмаска. '
        'Послеpassed065и070готова075GPU5400с,полнаяперваяэпохаопределяетцену179оставшихся;готоваяэпохане повторяется. Затем080CPU900саудитвсехdevпотерь/первогополноговидеовсех30голов. '
        'Фиксировано будущеедействие первойассоциации±0,1безновыхрёбер;этоещёнеинтеграция/HOTA. Цельостаётсяличностьюпоследнегореальногопрошлогонаблюдения,послеошибкиможетнеисходной;ограничениезаписано. '
        'ПервоначальнаяпроверкатипаусредненияобнаружилаFloat64;дообучения исправленоFloat32,четыретестапрошли. Бюджетыстарыхэтаповнепродлевались.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as out:
        out.write('\n\nСледующаяреализацияпосле065:070history_model_checks600сCPU→075history_factorial_training5400сGPU→080audit900сCPU. '
            'Протокол history_factorial_training/protocol.json ДОобучения:общаяCNN/двеGRU/геометрия/30голов/3сида42/43/44/6эпох/fit84fit20. '
            'Восемьсочетаний плюсmean/1mean отдельнообучены;0содержитКалман/последнее,не трекербезпамяти. Unknownисключён,весаклассовтолькоfit84,выборcheckpointтолькопоdevloss. '
            'ПерваяПОЛНАЯtrain+devэпоха0seed42:2*time*179+300<остаток5400,иначестопбезпродления;готовуюэпохунеповторять. '
            'Alpha.2/поправка±.1первогоcostзафиксированаДОобучения/val,новыхрёбернет,неподбиратьпоval. 080повторяетвсеdevпотериипервыйполныйdevвидеопроходкаждойголовынаCPU. '
            'ОбучениепарНЕHOTA;после080нуженАГЕНТдляреальнойинтеграциивполныйтрекер/точногонуля/причинногобанка/цены/полныхметрик. '
            'Цельявнозаданакакличностьпоследнегореальногопрошлогонаблюдения,неисходныйIDпослезахвата;невыдаватьзаобучениевосстановленияисходноготрека. '
            'Окно6наблюдений/60кадров≈2сограничено. Основноесравнение134сnative/0/mean/1meanв3сидах;95%ДИипоправка12первичныхсравнений,историческийvalненовыйитоговыйтест. '
            'CPUunit4passed;Float64усредненияисправлендозаморозкивFloat32. ГотовыеGPU060иБ1неповторять.\n')
    print('070/075/080 registered; no training started by registration')


if __name__=='__main__':main()
