"""Проверенные первичные файлы и точный остаток разрешённой Б2."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    proof={}
    for name in ('b2_causal960_checks_recovery1','b2_causal960_profile','b2_causal960_record','b2_sparse_training_preparation'):
        folder=RUN/name;c=service.read(folder/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
        proof[name]=dict(controller_sha256=service.sha(folder/'controller_result.json'),actual_all_owned_ended=True)
    folder=RUN/'b2_sparse_training_pilot';launch=service.read(folder/'launch.json')
    assert service.matching(launch['pid'],launch['created'])
    p=service.read(folder/'protocol.json');price=service.read(folder/'price.json');assert price['proceed']
    for f,h in p['sha256'].items(): assert service.sha(ROOT/f)==h
    assert service.matching(9664,1793203064.7696502) and service.matching(20528,1793203064.802624)
    proof.update(updated=service.now(),training_launch=launch,status=service.read(folder/'status.json'),
                 execution_deadline=service.read(folder/'execution_deadline.json'),price=price,
                 queue=service.read(RUN/'queue_status.json'),guard_updated=service.read(RUN/'deadline_status.json')['updated'],
                 matrix_author_override_exists=(RUN/'history_tracker_author_continuation1/author_override.json').exists())
    service.write(RUN/'b2_sparse_launch_verified.json',proof)
    text='''

АКТУАЛЬНО29.10около00:08МСК: Б2продвинутадоНОВОГОреальногообучения. 112первоначальныеCPUchecksотказалиДОGPUнаформатеevidence(строкавместоkind/ref/claim);исходныйcontrollerexit1/logсохранён. 112aотдельнаяchecks_recovery1востаткеисходных600с:каталогOK/62tests136.31с/code0/allended. 113_b2_causal960_profileПОЛНОСТЬЮготова/code0/allended:первоеtrainвидео1204raw/202query/198clips,fps30.027433701887105,nativeboxes/scores/classes960побитныraw_videoкэшу,zeroheadrawточен;кольцо5P3/P4/P5максимум36556800байт,57.625ссзагрузкой;commonadapter4609/ordinary9602,history/last/meanlossbackwardfinite,YOLOunchanged/optimizer_steps0. 114recordcode0/allended/catalogOK/62passed135.46с. Источник b2_causal960_profile/report.json. ЭтоНЕобучение/неaccuracy/HOTAиНЕпроверка734ядра. Новыйtracking/b2_causal_detector.pyсохраняетреальныеинтервалы,GTтолькотаргеты,crowdclassbackgroundignore(positiveприоритет),last/meanсодинаковойобычнойGRU,ограниченноекольцо. 4CPUunitpassed.

115_b2_sparse_training_preparation600сCPUГОТОВА/code0/allended/catalogOK/62passed138.04с. СЕЙЧАС116_b2_sparse_training_pilot3600сGPUРЕАЛЬНООБУЧАЕТ;не113профиль,нестарые3scratchsteps. Папкаb2_sparse_training_pilot,script scripts/train_b2_sparse_pilot_20261029.py,карточкаb2_sparse_training_pilot_v1,protocol.jsonДОобучения. РОВНО1эпохаseed26/6ветвейC/D/R1/B/last/mean. Разреженныйрежим—по10равномернымцелевымqueryиндексамнакаждом104историческихfit84/fit20видео(840fit/200dev),НЕполные20441клип. ВыбранпоценеДОкачества/val,неперетюнивать. ИсториякаждойцелиПОЛНЫЕ5queryкадровизисходноговыравнивания,несоседние10выбранныхцелей. ПредикторGTнечитает;меткатолькоследнегокадравloss. Commonadapter4609,одинаковаяинициализациянулевогоoutput;C/Dstockядроизоднихсуществующихвесов,норазныепараметры;Cзаморожено/Dтолько734;R1фиксированный;B/last/mean9602,одинаковыеначальныевеса,отдельноеобучение. AlphaкоррекцииP3.1,Adam.001адаптер/ordinary,CORE_LR/penalty/clampдляD,clipnorm1. Реальныевремена/FPS,FlyVisdt<=.02,TimedR1распадпореальномуинтервалусопоройна.2с. Точныйnativecacheнакаждомвычисленномqueryкаждоговидео;нулевойrawвсех6допервогоoptimizerstep. YOLObackboneодинразнакаждыйнеобходимыйquery,общиеimmutableP3всем6;отработанныеP3освобождаются,80GBневыгружается. В протоколеSHAкаждоговидео/cache/GT/alignment;104старыхGT/alignmentSHAотдельнопровереныдообучения.

ПЕРВЫЙГЕЙТ116УЖЕПРОЙДЕН:первоеПОЛНОЕfitвидео10updatesкаждой6ветви13.719с+ПОЛНОЕdevвидео9.906с;2*(fit/clip*remainingfit+dev/clip*allfinaldev)+180=2853.594<3518.047843с;price.proceed=true. НЕповторятьпервыйпроход/гейт/optimizerupdates. Первыйcheckpointfit_000.ptсoptimizer/SHA797c3c250c15a16a46f7a5c4e407a035d3ed9ea4f8a9a10aedd6cd1cc0c7fd93. ДалееcheckpointПОСЛЕкаждогоПОЛНОГОfitвидео;готовоенеповторять. Первыйdevpriceprobeнеитоговаяdevloss:послевсегоfitонпереоцениваетсясДРУГИМИвесамикаквесьfinaldev,эторазныеизмерения. Фактическийнепродлеваемыйexecution_deadline116=2026-10-28T22:06:28.844396UTC(29.10 01:06:28МСК),НЕобщий07:40. GPUPID18164+birth1793221588.861168;свежееstatus/queue/guard/всепрошлыеPIDendвb2_sparse_launch_verified.json. Следующая116a_b2_sparse_training_record600сCPUсоздастсялокальнопослеreport;каталог/62теста. ВовремяихНЕПИСАТЬплан/каталог/словарь/контролируемыеreports.

ПОСЛЕ116/116aНУЖЕНАГЕНТ:читатьполныйreport/контроллеры/фактическийPIDend/цены/checkpoints/параметры/Dnonzerograd/CиYOLOunchanged. ЭтоТОЛЬКОобучающийпилот,научнойБ2/HOTAещёНЕТ. Нужнынезависимыйаудитобученныхвесов/источников/инициализации/devloss,заморозкаДОvalи РЕАЛЬНЫЙполныйdetector+trackerнаисторическихvalсnative/zero/C/D/R1/B/last/mean. RuntimeвалидацииновойБ2ещёНЕнаписан;непритворятьсяготовым. Наqueryизменяютсяреальныеbox/scoreпередByteTrack,прочиекадрыштатные;общаяполная5queryисториясоответствуетобучению,новыйYOLOfrozenprefixобщийвсемвариантам,никакогоGTпредиктору. Старыйscripts/run_bdd_video_rate_experiment.py/cacheprotocol—первичныйnative. Преждеполнойсерииотдельныереальныекороткиегейтыцены/zero-parityсbudget/полноеПЕРВОЕвидео,готовыйGPUнеповторять. ПолныеHOTA/AssA/IDF1/FP/FN/IDSw/парные95%ДИ,Dкnative/C/R1/B/last/mean,нормальныйконтрольдобавкиядра;исправностьобучениянепольза. Историческийvalненовыйнезависимыйтест. 1эпоха/1seed/разреженные10целейнавидео—ограничение,неполноеобучениеБ2. Неостанавливатьсянаобученныхветвяхприоставшемсяокне;Б3послеБ2. ФиксированнаяP3позициянеотслеживаетдвижущийсяобъект/совмещенияобъектовнет—рабочеевозражение,недоказаннаяпричина.

ВОПРОСматрицыисторииПО-ПРЕЖНЕМУБЕЗНОВОГОЧЕЛОВЕЧЕСКОГООТВЕТА. Genericheartbeat/сменадаты/accepted:trueотинструментанеответ. Author_overrideНЕсоздан;39недопущены. ЕслиНОВЫЙПРЯМОЙответодобритматрицу,116самасохранитcheckpointиокончитсянаграницевидео;дальшеподготовленные117/120/125/128приоритетно,безповторапервого32trackerGPU/30готовыхобучений. Общий07:40непродлён,прошлыеБ1/090/095/цены/контроллерынетрогать. Б2послеостановкиповолюавторасохраняетсясостатком,самостоятельнонепродлевать3600. Граф/Проблемы232401непересобиралсярадиэтихслужебныхстадий,11гипотез/цвета/P1/index.lockсохранены. НовыхMASAветвейнет.
'''
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as out:out.write(text)
    from scripts.b123_safe_journal_20261027 import journal
    journal('b2_sparse_training_pilot_v1','Фактический первый полный стоимостной гейт Б2 пройден',
            'runs/continuation_20261028/b2_sparse_training_pilot/price.json:13,719сfit+9,906сdev, '
            '2853,594<3518,047843с,proceed=true,сохранёнпервыйполныйcheckpoint/optimizerдоостальных83fitвидео. '
            'Источники/62теста/PID+birth/endпрошлыхдеревьевпроверенывb2_sparse_launch_verified.json. '
            'Это10целевыхзапросовнакаждом104видео/1эпоха/1seed/6ветвей;неполные20441клип,неHOTA. '
            'Непродлеваемый116execution_deadline29.10 01:06:28МСК. Старыйвопроспродленияматрицыбезчеловеческогоответа,overrideне создан.')


if __name__=='__main__':main()
