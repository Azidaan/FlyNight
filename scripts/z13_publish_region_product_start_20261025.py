"""Record and publish the active direct-interaction queue without altering science."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_product_20261025 as run
from scripts import z13_region_product_recovery1_20261025 as recovery


if __name__=='__main__':
    p=run.read(run.RUN/'protocol.json');c=run.read(recovery.RUN/'controller.json')
    assert not (run.RUN/'start_publication.json').exists()
    ps=run.WindowsProcesses();assert ps.creation(c['pid'])==c['birth'] and ps.creation(c['worker_pid'])==c['worker_birth']
    run.handoff('ТЕКУЩИЙ z13_region_product_v1, runs/z13_region_product_20261025/recovery1, scripts/z13_region_product_recovery1_20261025.py --pipeline. '
        'PID+birth контроллера/worker: '+str(c)+'; видимая консоль/launcher вlaunch.json. '
        'CPU24модели: D/C/R1/геометрия, прямые68произведений исходных каналов против68нулей той же162входной сети; '
        'прежние6признаков назначения остаются в обоих вариантах. RMS толькоfit84; прежние метки/кэши/пакеты/1150шагов/порогиfit20/трисидa. '
        'Save '+p['save_at']+' / hard '+p['hard_stop']+';1800с+20с не продлевать. Первыйполныймодельный замер цены обязателен. '
        'Новыйреальный162входный градиент и все24искусственных исполнительных пути проверены до запуска. '
        'Готовые кэши/модели не повторять; научные настройки и допуски не менять. '
        'При полном результате очередь самостоятельно проверяет все8326выходов24моделей/хеши/интервалы, каталог/тесты/граф/scopedpush. '
        'Актуальные ошибки: recovery1/pipeline_failure.json и recovery1/worker.stderr.log. Старый root pipeline_failure/exit1 — завершённый служебный сбой; '
        'копия пространства имён исправляет адрес кэша, научная функция/SHA неизменны, выполнен настоящий fit84 forward/backward; '
        'Предыдущий контекстный узел оконченf1f86c1: −0,193п.п.[−0,579;+0,053], критерии не пройдены. '
        'Архитектуруcontext/product и confirmed_results проверить по итогам; новая диагностика не должна подменять HOTA. '
        'НикакихGPU/dev/HOTA/heldout75/загрузок. Авторизация26.10 07:40МСК,57чужихправок и.gitignore сохранять.')
    recovery.publish('start')
