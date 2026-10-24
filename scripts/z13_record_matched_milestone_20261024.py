"""Verify completed matched benchmark and record the live stock recovery."""
from pathlib import Path
from datetime import datetime
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources,journal
W=ROOT/'runs/z13_weekend_20261024';B=ROOT/'runs/z13_matched_trackers_20261024';REC=B/'stock_recovery1'


def main():
    from scripts.z13_record_matched_recovery3_20261024 import ended
    from scripts.z13_continue import paired
    import numpy as np
    ended(B/'gpu_recovery3');finish=read(B/'finish_recorded.json');report=read(B/'report.json')
    assert finish['report_sha256']==sha(B/'report.json') and read(W/'matched_recovery3_verified.json')['checks_green']
    verify_sources(read(B/'gpu_recovery3/protocol.json'));verify_sources(read(REC/'protocol.json'))
    jobs=list((B/'gpu_recovery3/jobs').glob('*.json'));assert len(jobs)==49
    for marker in jobs:
        for n,h in read(marker)['outputs'].items():assert sha(ROOT/n)==h,n
    for n,h in report['output_sha256'].items():assert sha(ROOT/n)==h,n
    pv={k:{v:m['HOTA'] for v,m in values.items()} for k,values in report['per_video'].items()}
    for key,value in report['paired'].items():
        a,b=key.split('-');assert paired(pv[a],pv[b])==value
    ps=WindowsProcesses();controller=read(REC/'controller.json');monitor=read(REC/'completion.json')
    assert ps.creation(controller['pid'])==controller['birth'] and ps.creation(monitor['pid'])==monitor['birth']
    old=read(B/'stock_controls/controller_result.json');assert old['exit_code']==1 and old['all_owned_ended'] and not old['remaining']
    assert controller['save_at']==read(B/'stock_controls/controller.json')['save_at'] and controller['hard_stop']==read(B/'stock_controls/controller.json')['hard_stop']
    cardpath=ROOT/'experiments/project_history.json';cat=read(cardpath);card=next(c for c in cat['experiments'] if c['id']=='z13_matched_stock_controls_v1');assert card['state']=='ready';card['state']='running'
    card['evidence'].append(dict(kind='file',ref='runs/z13_matched_trackers_20261024/stock_recovery1/protocol.json',claim='Служебное восстановление известной формыR1,прежние3готовыхзадания и исходный срок сохранены'))
    write(cardpath,cat)
    h=W/'HANDOFF.md';prefix=f'''# Актуальное состояние — {now():%d.%m.%Y %H:%M} МСК

Основное сравнение6систем/7val завершилось03:30:48,exit0,49заданий проверены,карточка/журнал/граф записаны,58тестов зелёные: matched_recovery3_verified.json. Первичный файл runs/z13_matched_trackers_20261024/report.json,читаемаятаблица results.md и verified_summary.json. HOTAсобщейMASA:XYWH55,29000684;Byte53,16783940;BoT55,15176074;Track55,38261685;прежняяD55,36852719;центрD55,32379955. Добавка D кXYWHнеподтверждена:прежняяD средняяповидео+0,07045447[-0,06488940;0,21856380],центрD+0,01827499[0;0,05067312]. Пул и повидео не смешивать,7ранееиспользованныхvalнефинальныйтест,старые61,65непереносить.

Stock_controls автоматически стартовал03:32:15,exit1в03:41:54из-за пропущенного известногоадаптера формыR1. Ошибка сохранена. ТЕКУЩИЙ этап stock_recovery1, scripts/z13_matched_stock_recovery1_20261024.py. Контроллер{controller['pid']}/birth{controller['birth']},completionmonitor{monitor['pid']}/birth{monitor['birth']},scripts/z13_after_stock_recovery1_20261024.py. Прежний диспетчер stock_failure уже сохранён;новый монитор завершит scripts/z13_record_stock_recovery1_20261024.py,провериткаталог/тесты/граф и запишет stock_recovery1_verified.json. Меньший бюджет10800с от03:32:15НЕпродлён: сохранение{controller['save_at']},конец{controller['hard_stop']}. Действующие источники/пороги/данные неизменны;подключён старый tracking/z13_ridge_localizer_v2.py,егохеш равен recovery1_contract21.10,4тестапройдены. Старые3задания проверены,неповторяются. Цена после первого полного ролика должна включать времена3сохранённыхзаданий;если консервативный остаток не помещается,не обходить цену/не продлеватьбюджет,сохранитьчастичныйрезультат безвыводаобэффективности. R1 уже прошёл прежнее место ошибки,см.logs. Не запускать дубли.

Последняя авторизация непрерывности и отрицательный CPU-итог ниже остаются действующими. При отсутствии готовой следующей стадии выбрать обоснованный узел по плану,не продвигать проваленный descriptor_v2. При новом полном stockитоге проверить архитектурнуювеху: таблицаbenchmarkсама по себе не меняет архитектуруFlyVis.

''';h.write_text(prefix+h.read_text(encoding='utf-8'),encoding='utf-8')
    write(W/'matched_milestone_verification.json',dict(at=now().isoformat(),all49_output_hashes_valid=True,all_hota_json_hashes_valid=True,all_paired_recomputed=True,parent_report_sha256=sha(B/'report.json'),stock_controller=controller,stock_completion=monitor,stock_original_deadline_unchanged=True,complete_table='runs/z13_matched_trackers_20261024/results.md'))
    journal('проверен полный общий HOTA и продолжены контроли','Полный итог6систем уже записан карточкой z13_matched_trackers_v1 и журналом03:30. Дополнительно проверены49маркеров/хешейвыходов,всеоценочныеJSON и парныеинтервалы из report.json;исходное дерево завершеноexit0. Полнаятаблица: runs/z13_matched_trackers_20261024/results.md. ПрежняяDсMASA55,3685 противXYWH55,2900:общийсдвиг+0,07852035пункта(+0,14201544%),повидео+0,07045447[-0,06488940;0,21856380]. ЦентрD55,3238:общийсдвиг+0,03379271пункта(+0,06111902%),повидео+0,01827499[0;0,05067312]. ПриростDнеподтверждён. Это7ранееиспользованныхval,нефинальныйтест. Новаяархитектуранепринята. КонтролиC/B/R1продолжаются в stock_recovery1 с прежнимсроком;карточкапереведенаready→running. Источники: matched_milestone_verification.json и stock_recovery1/protocol.json. РанееknownошибкаR1сохранена и исправлена подключением проверенногоадаптера,безпересчётаготового.')


if __name__=='__main__':main()
