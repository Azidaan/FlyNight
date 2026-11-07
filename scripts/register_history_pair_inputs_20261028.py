"""Зарегистрировать причинные пары до подготовки входов новых представлений."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now
from scripts.b123_safe_journal_20261027 import journal

def main():
    f=RUN/'history_pair_dataset';assert not (f/'protocol.json').exists()
    export=read(RUN/'history_observed_export/report.json');assert export['complete'] and len(export['completed'])==104
    settings=dict(parent_experiment='history_representations_factorial_v1',created=now(),history_items=6,max_history_frames=60,
        query_schedule='Исходные заранее сохранённые времена alignment, по реальным изображениям; не GTошибки/ID. Одинаково всем вариантам. Это запросный режим правки полных30Гцназначений, не обещание произвольного30Гцвхода.',
        tracker_config=read(RUN/'b1_tracker/protocol.json')['config'],
        sampling='84fit84обучение/20fit20разработка, по видео. Историческиеtrainдетекторвидел. Ни одного valдляподбора.',
        action='Готовить реальные пары первого назначения по исходнойгеометрическойдопустимости/тому же классу. Следующая модель будет менять стоимость пары, не выбирать прежнее full-boxдействие. Runtimeещёнеготов.',
        labels='Предыдущая реально назначенная детекция в последнем прошлом запросе и текущий кандидат. GTсовпадения HungarianIoU>=.5 одного класса, crowdисключён. Unknownобеихсторон→Y=-1/known=false,не negative. Разметка не выбирает входы или историю.',
        inputs='geometry14 и реальные history_frame/det/box/quality/valid; frame/track/detтолькоуказатели,не семантическиеIDэмбеддинги. GTтолькоtargets.',
        geometry14=['original_cost','current_conf','same_class','dx','dy','log_w_ratio','log_h_ratio','kalman_vx','kalman_vy','age_seconds','uncertainty_x','uncertainty_y','history_count_fraction','log_area_ratio'],
        budget_seconds=2400,price_rule='ПервоеПОЛНОЕвидео:2*time*103+120<остаток2400с',
        sha256={p:sha(ROOT/p) for p in ['scripts/build_history_pair_dataset_20261028.py','tracking/history_training_data.py','tracking/xywh_bytetrack.py','datasets/bdd100k_loader.py',
            'runs/continuation_20261028/history_observed_export/report.json']})
    write(f/'protocol.json',settings)
    history=read(ROOT/'experiments/project_history.json');ident='history_pair_inputs_v1';assert not any(c['id']==ident for c in history['experiments'])
    history['experiments'].append(dict(id=ident,title='Данные реальных соответствий для трёх представлений истории',area='flyvis',state='running',started='2026-10-28',
        question='Можно ли построить реальные причинные входы и проверяемые метки соответствий для одинаково обучаемой матрицы1/3/4?',
        hypothesis='Гипотезаготовности:сохранённых детекций/наблюдений/изображенийдостаточно дляпервыхназначений и реальнойистории;это не гипотезаприбавкиHOTA.',
        expected='Сначала точный CPU-повтор базового трекера,история максимум6наблюдений/60исходныхкадров,пары всех геометрическидопустимых кандидатов того же класса. Одинаковая подготовка всем будущим вариантам;GTтолько метки,unknownне negative.',
        acceptance=[dict(criterion=x,met=None) for x in ['Все104исходныхвидео подготовлены и проходят контрольные суммы/полное совпадение CPU-выходов с готовым наблюдаемым исходным трекером',
            'Все исторические кадры строго прошлые; входы не содержатGTполей, неизвестнаяличность имеет-1 и исключаетсяизошибки',
            'Обучающиеfit84и разработкаfit20непересекаются; достаточно известных положительных меток,данные не выдаются за HOTAилиобобщениедетектора']],
        evidence=[dict(kind='file',ref=f.relative_to(ROOT).as_posix()+'/protocol.json',claim='Постановка до подготовки; параметры/метки/стоимость'),
            dict(kind='file',ref='runs/continuation_20261028/history_observed_export/report.json',claim='Полный экспорт реальных наблюдений104видео')],metrics=[],confirmed_results=[]))
    write(ROOT/'experiments/project_history.json',history)
    specs=[('045_history_dataset_checks','history_dataset_checks','scripts/history_dataset_catalog_check_20261028.py',600),
        ('050_history_pair_dataset','history_pair_dataset','scripts/build_history_pair_dataset_20261028.py',2400),
        ('055_history_pair_audit','history_pair_audit','scripts/audit_history_pair_inputs_20261028.py',900)]
    for name,sub,script,budget in specs:
        folder=RUN/sub
        if sub!='history_pair_dataset':write(folder/'protocol.json',dict(budget_seconds=budget,parent_experiment=ident,sha256={script:sha(ROOT/script)}))
        write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=folder.relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(folder/'protocol.json')))
    journal(ident,'Начата подготовка реальных пар для1/3/4;GTтолько метки',
        'Источник runs/continuation_20261028/history_pair_dataset/protocol.json:точноеCPUпервое назначение,история6реальныхнаблюдений/максимум60исходныхкадров,одинаковыеquery-временаизготовойalignment. '
        'Разметка назначает толькоизвестность/личность целиошибки,невыбираетвходы;unknown=-1/known=false. ПодготовкабезGPU/обучения/HOTA. '
        '045каталог/тесты→050полные104сценыпослецены2*time*103+120<2400с→055всевременныеуказатели/маски/хеши. Это реальнаяподготовкавходовновойассоциации,неповторБ1.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as out:
        out.write('\n\nПолныеБ1/025и030/040ЗАВЕРШЕНЫ.040выдала104реальныхистории,неHOTA. Новаяготоваяцепочка045каталог/тесты→050history_pair_dataset2400сCPU→055audit900сCPU. '
            'Это новые причинные пары ПЕРВОЙреальнойассоциации:история6наблюдениймакс60кадров/тотжекласс/исходно допустимыеedge. Query-временаизalignmentвсемвариантамодинаковые,GTневлияетнавходы;меткитолькодляошибки/unknown=-1. '
            'CPUoutputsнакаждомкадреточныкготовым104историям,сохраняютсятолькоисторическиеуказатели/рамки/числоваягеометрия,targetsотдельно. '
            'После первогоПОЛНОГОвидео gate2*time*103+120<остаток2400;приfalseнедосчитыватьостальные. Готовоенеповторять.055запишеткарточкуиобязательныепроверки,смотретьreport/controllers/check*.log. '
            'СледующееагентскоеДЕЙСТВИЕпосле055:реальныеизображениядвиженияссовмещениемкамеры/банквнешности/единыйобучаемыйинтерфейсвсех8вариантовипростыхконтролей,цена,обучениеиполнаяHOTA. Новыхмасаветвейненужно,допустимтолькофиксированныйсуществующийизвлекателькаквходэтогопоручения;источники/обучениефреймовготовитьчестно,невыдаватьGTзаobserved.\n')
    print('045/050/055 prepared')

if __name__=='__main__':main()
