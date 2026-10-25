"""Record active hidden paired pooling queue and publish its fixed protocol."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_pooling_20261025 as stage


if __name__=='__main__':
    c=stage.read(stage.RUN/'controller.json');p=stage.read(stage.RUN/'protocol.json');ps=stage.WindowsProcesses()
    assert ps.creation(c['pid'])==c['birth'] and ps.creation(c['worker_pid'])==c['worker_birth']
    stage.handoff('ТЕКУЩИЙ z13_region_pooling_v1: runs/z13_region_pooling_20261025, scripts/z13_region_pooling_20261025.py --pipeline. '
        'Контроллер/workerPID+birth: '+str(c)+'. Запуск launch.json, все процессы скрыты, консоль не открывалась. '
        'Автор25.10 ушёл до утра и разрешил продолжать. CPU1200с+20с, save '+p['save_at']+' / hard '+p['hard_stop']+' не продлевать. '
        'Одновременно8парных условий: D/C/R1/геометрия, среднее/максимум внутри тех же рамок и тех же сохранённых полных карт. '
        'Alpha0.25 прежнего среднего зафиксирован для всех, alpha0точный контроль; никаких новых порогов, обучения/GPU/dev/HOTA/75/загрузок. '
        'Все46видео fit20+train26; средняя ветвь обязана точно воспроизвести прежние исходы на каждом запросе. '
        'Первый2замер цены обязателен; известный служебный сбой исправлять в первоначальном окне отдельной версией. '
        'Очередь после полной проверки первичных строк/хешей/интервалов автоматически ведёт каталог/журнал/тесты/защитныйграф/scopedpush. '
        'Следить controller_result.json, pipeline_failure.json, worker.stderr.log, status.json, price.json. '
        'Прежнийproduct завершёнce72d84: −0.288п.п.[−0.988;+0.109], критерийнепройден. Контекст/product/новыйpooling объединить в следующую однолистовую архитектурную веху; confirmed_results проверить. '
        'До26.10 07:35сохранение/07:39:30конец,57чужихграфправок/.gitignore сохранять.')
    stage.publish('start')
