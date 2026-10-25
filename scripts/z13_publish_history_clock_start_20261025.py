"""Publish the active hidden neural-clock pipeline and its exact individual deadlines."""
from pathlib import Path
import inspect
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_history_clock_20261025 as stage
from scripts import z13_history_clock_recovery1_20261025 as recovery


if __name__=='__main__':
    c=stage.read(recovery.RUN/'features/controller.json');p=stage.read(stage.RUN/'protocol.json');ps=stage.WindowsProcesses()
    assert ps.creation(c['pid'])==c['birth'] and ps.creation(c['worker_pid'])==c['worker_birth']
    from scripts import z13_region_assignment_recovery2_20261025 as original
    original.namespace()['handoff']('ТЕКУЩИЙ z13_history_clock_v1: runs/z13_history_clock_20261025/recovery1, scripts/z13_history_clock_recovery1_20261025.py --pipeline. Старый корневой pipeline_failure относится к KeyError episode до GPU; recovery1 восстанавливает только адрес эпизода по проверенному label, прежний бюджет не продлевает. '
        'Контроллер/workerPID+birth: '+str(c)+'; pipeline/launch.json. Всё скрыто, без консоли и перехвата фокуса. '
        'Новый узел: исходная нейродинамика получает delta сохранённых кадров вместо1шага на запись; delta0не меняет состояние, gaps интерполируются линейно при прежнем dt/30Гц. '
        'Это научная проба, не объявленное исправление служебной ошибки: истинных пропущенных изображений нет, изменённая динамика может ухудшить D. '
        'Clock gatee3adc343: в26600известных запросах естьgap>1; дублирующийpadding только3.954%среднепо-видео. '
        'ОдинGPUfeatures1800с, save/hard изfeatures/controller.json (не продлевать), затем CPU26replays900с. Общийsave '+p['pipeline_save_at']+' / hard '+p['pipeline_hard_stop']+'. '
        'Воскресенье25.10 не имеет ограничения07:40; авторскийконец26.10 07:40МСК сохраняется. '
        'Сохранённые C/D/R1 веса, нормы, lambda, настоящий образец, mean-region decoder/alpha0.25 неизменны. '
        'Первый2Dзамер цены и старые центры<0.5px/score<0.001 обязательны; никакой подмены допусков. CPUстарые3ветви обязаны точно воспроизвести прежние mean назначения. '
        'Очередь автоматически features->replay->первичныехеши/интервалы/карточка/тесты/сохраняющийграф/scopedpush. '
        'Смотреть features/price.json, features/worker.stderr.log, phase/controller_result.json, status.json и pipeline_failure.json. Известные служебные сбои исправлять отдельной версией в прежнем окне, готовые chunks не повторять. '
        'Никакихобучений/загрузок/dev/HOTA/75; кHOTAтолько при полномкритерии назначения, центры вторичны. '
        'Семантический readoutотрицателенf0a68099; при вехеclock объединить его в архитектуре, проверить confirmed_results.57чужихграфправок/.gitignore сохранять.')
    recovery.publish('start')
