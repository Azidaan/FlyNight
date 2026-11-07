"""Сохранить пределы диагностической разметки и проверяемый ночной остаток."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'runs/b123_20261027';RUN=BASE/'night_summary';ORACLE=BASE/'admission_oracle'
from scripts.b123_supported_transfer_recovery1_20261027 import write
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def f(x):return f'{x:.6f}'.replace('.',',')
def ci(x):return '['+f(x[0])+'; '+f(x[1])+']'
def main():
    previous=read(BASE/'admission_oracle_record/controller_result.json');assert previous['exit_code']==0 and previous['all_owned_ended']
    assert read(BASE/'admission_oracle_record/report.json')['catalog_tests_passed']
    diagnostic=read(ORACLE/'report.json');assert read(ORACLE/'independent_audit.json')['passed']
    totals={key:sum(row.get(key,0) for row in diagnostic['counts'].values()) for key in set().union(*(set(row) for row in diagnostic['counts'].values()))}
    coverage=dict(source='runs/b123_20261027/admission_oracle/report.json',source_sha256=hashlib.sha256((ORACLE/'report.json').read_bytes()).hexdigest(),
        total_candidates=totals['candidates'],unknown_past_identity=totals.get('no_previous_identity',0),known_past_identity=totals['candidates']-totals.get('no_previous_identity',0),
        admitted=totals.get('admitted',0),counts=totals,coverage_means_prepared_oracle_labels_not_flyvis_memory=True,unknown_does_not_automatically_mean_negative_action=True)
    coverage['known_fraction']=coverage['known_past_identity']/coverage['total_candidates'];write(ORACLE/'coverage.json',coverage)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']=='recovery_admission_oracle_v1');assert card['state']=='infrastructure'
    extra=(' Источник runs/b123_20261027/admission_oracle/coverage.json: total_candidates='+str(coverage['total_candidates'])+', unknown_past_identity='+str(coverage['unknown_past_identity'])+', known_past_identity='+str(coverage['known_past_identity'])+', admitted='+str(coverage['admitted'])+'. '
        'Доля известной прошлой личности='+f(coverage['known_fraction'])+'; это охват подготовленных диагностических меток,не память FlyVis и не вес обучающей ошибки.')
    changes=dict(obtained=card['obtained']+extra,
        interpretation=card['interpretation']+' Прошлая истинная личность установлена лишь для части кандидатов. Неизвестные случаи не были оценены как отрицательные; они не доказывают забывание объекта моделью.',
        cannot_claim=card['cannot_claim']+' У неизвестной прошлой личности нельзя автоматически ставить отрицательный исход действия. Метка допуска этого GT-контроля также не является точной меткой глобальной дельты HOTA.')
    for key,value in changes.items():
        card.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field=key,old=card[key],reason='Уточнён охват разметки прошлого объекта; он не является измерением памяти или отрицательных действий.'));card[key]=value
    card['evidence'].append(dict(kind='file',ref='runs/b123_20261027/admission_oracle/coverage.json',claim='Охват известной личности и неизвестные исходы,не измерение памяти'))
    assert path.read_bytes()==before;write(path,cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'охват GT-истории отделён от памяти и отрицательных действий',
        '*Уточнение.* '+extra+' Неизвестнаяличностьнеозначает,чтопродолжениевредно;этоограничениеразмеченногоисторическогосоответствия. '
        'КонтрольнеявляетсямаксимальнымпотолкомHOTA илиготовоймоделью. Старыеформулировкикарточкисохраненывrevisions. '
        'Ночнойостаток—подготовитьотдельныеtrain/devдляследующегообучаемогоотказа,необучатьсянаэтихval. Неизвестнуюметкунеподменятьинтерполяциейилиавтоматическимотрицанием. '
        'Классификационнаяточностьдопускане заменитпарнуюпроверкуреальнойHOTA.')
    busca=read(BASE/'busca_output_pilot/report.json');mask=read(BASE/'sam2_query_output_pilot/report.json');dev=read(BASE/'detection_supported_recovery/report.json');transfer=read(BASE/'supported_transfer/recovery2/report.json')
    def result(title,r,a,b,contrast):
        q=r['paired'][contrast] if isinstance(r['paired'],dict) and contrast in r['paired'] else r['paired']
        return title+': HOTA '+f(r['pooled'][a]['HOTA'])+' → '+f(r['pooled'][b]['HOTA'])+'; средняя парная разность '+f(q['mean_per_video'])+' п.п., 95% интервал '+ci(q['ci95'])+'.\n\n'
    text='# Итоги ночной серии 27.10.2026\n\nСводка проверенных фактов; основные источники находятся в том же каталоге runs/b123_20261027. Все времена сроков заданы по Москве.\n\n'
    text+='## 1. Граница результата\n\nЗдесь оценивается весь проект. Польза параллельного модуля допустима, но её нельзя приписывать ядру или полному модулю FlyVis. HOTA в этих отчётах усреднена по объектам через TrackEval; её нельзя смешивать со средней по классам HOTA_cls_av.\n\n'
    text+='## 2. Б1–Б3\n\nПервый пилот Б1 на сохранённых запросах не подтвердил преимущество; это не HOTA полного трекера. Б2 и Б3 прошли технические проверки, научная польза полного сопровождения не измерена. Основные гипотезы не считаются проверенными по технической исправности. Источник: completed_pilots_independent_audit.json и карточки b1_temporal_residual_training_v1, b2_predetection_gradient_v1, b3_object_memory_gradient_v1.\n\n'
    text+='## 3. Испытанные выходные адаптации\n\n'
    text+=result('Собственная выходная адаптация BUSCA',busca,'baseline','busca_output','vs_baseline')
    text+='Она не воспроизводит всю авторскую BUSCA: не меняет состояние исходного трекера, не включает весь авторский отбор памяти и компенсацию камеры. Источники: busca_output_pilot/report.json и independent_audit.json.\n\n'
    text+=result('Собственная масочная выдача SAM2 по запросу',mask,'baseline','sam2_mask','vs_baseline')
    text+='Это не точная LTR и не SGAR-MOT. Плотный перенос маски не прошёл проверку стоимости; результат выдачи по запросу нельзя переносить на него. Код и веса обученного выбора маски точной LTR не подтверждены. Источники: sam2_query_output_pilot/report.json, independent_audit.json, ltr_exact_source_lookup/report.json.\n\n'
    text+='## 4. Частная польза поддержки текущим обнаружением\n\n'
    text+=result('Калман с поддержкой текущим обнаружением на 20 видео разработки',dev,'baseline','kalman_support','kalman_vs_baseline')
    text+=result('Маска с тем же правилом на тех же 20 видео',dev,'baseline','mask_support','mask_vs_baseline')
    q=dev['paired']['mask_vs_kalman'];text+='Добавка масочного пути к сопоставимому контролю: '+f(q['mean_per_video'])+' п.п., интервал '+ci(q['ci95'])+'. Она не подтверждена; эквивалентность также не доказана. Источник: detection_supported_recovery/report.json.\n\n'
    text+='## 5. Перенос фиксированного правила\n\n'
    text+=result('Тот же Калман с поддержкой на другой 20 части прежнего val',transfer,'baseline','supported','paired')
    text+='Польза переноса не подтверждена. Эти видео не пересекаются с 20 видео разработки правила, но все 200 локальных val-видео имеют следы прежних работ. Это не новый независимый итоговый тест проекта. Десять готовых видео сохранены без повторного расчёта; конфликт времени первых изображений одного ролика исправлен по согласованным исходным пикселям трёх разрешений в том же окне поиска. Источники: supported_transfer/recovery2/report.json, independent_audit.json, service_recovery.json.\n\n'
    text+='## 6. Что показал контроль с разметкой\n\n'
    text+=result('Недоступный модели допуск с разметкой',diagnostic,'baseline','oracle','oracle_vs_baseline')
    text+='Это не прибавка реальной модели и не максимальный потолок HOTA. Контроль выбрал '+str(coverage['admitted'])+' из '+str(coverage['total_candidates'])+' исходных добавлений. У '+str(coverage['unknown_past_identity'])+' кандидатов прошлую истинную личность установить не удалось; этот счётчик описывает подготовленную разметку, а не память FlyVis. Источники: admission_oracle/report.json, independent_audit.json, coverage.json.\n\n'
    text+='## 7. Следующее обоснованное действие\n\nПодготовить раздельные данные обучения и разработки для отказа от вредных продолжений при неизменных кандидатах. Сохранять неизвестный исход отдельным состоянием, не превращать его автоматически в отрицательную метку. Метка этого диагностического допуска не является точной глобальной дельтой HOTA. Принятие будущей модели требует сравнения полного трекера с тем же правилом без обучаемого решения, одинаковых входов и положительной нижней границы парного интервала. Новая тренировка ночью не начиналась: её данные и измеренная стоимость не готовы.\n\n'
    text+='## 8. Сохранность и остановка\n\nМодуль tracking/detection_supported_kalman.py выключен по умолчанию; основной baseline не менялся. В графе обновляется одна фундаментальная задача; файлы не удаляются и не перемещаются, основные Б1–Б3 сохраняются. Изменения локальные: прежний .git/index.lock не удалялся, коммит и push не выполнены. До 07:20 сохранить остаток, до 07:29:30 завершить собственные деревья, в 07:30 приостановить автоматизацию. Продление не разрешено.\n'
    (BASE/'NIGHT_RESULTS.md').write_text(text,encoding='utf-8')
    steps=[('catalog_validation.log',['scripts/validate_project_history.py']),('catalog_tests.log',['-m','pytest','tests/test_project_history.py','-q']),
        ('graph_tests.log',['-m','pytest','tests/test_obsidian_transfer_outcome.py','tests/test_obsidian_fundamental_overview.py','-q']),('graph_sync.log',['scripts/restore_obsidian_sections_20261027.py'])]
    for name,args in steps:
        with (RUN/name).open('w',encoding='utf-8') as out:done=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        assert done.returncode==0,(name,done.returncode)
    write(RUN/'report.json',dict(complete=True,coverage_recorded=True,night_results_saved=True,catalog_validated=True,catalog_tests_passed=True,graph_tests_passed=True,graph_copy_checked_then_applied=True,new_training_started=False))
    with (BASE/'HANDOFF.md').open('a',encoding='utf-8') as out:out.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — ночная сводка и предел разметки сохранены\n\nNIGHT_RESULTS.md содержит раздельные итоги,первичныеисточники и конкретныйостаток. admission_oracle/coverage.json:3738кандидатов,2009неизвестнаяпрошлаяличность,1729известная,1226допущено. Этоохватметок,непамятьFlyVis;неизвестныенепревращатьвотрицательные. Карточкапоправленасrevisions,каталог/58тестов/5тестовграфаи провереннаякопияпрошли. Основнойbaselineнеизменён,новоеобучениенезапущено. Осталосьпроверитьконецвсехдеревьевиостановку/приостановкуавтоматизациивпрежниесроки.\n')
    print('Night results and coverage saved; checks passed; no new training')
if __name__=='__main__':main()
