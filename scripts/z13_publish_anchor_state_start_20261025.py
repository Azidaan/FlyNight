"""Publish the active hidden anchor-state test without interrupting its worker."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_anchor_state_20261025 as stage

if __name__=='__main__':
    c=stage.read(stage.RUN/'features/controller.json');p=stage.read(stage.RUN/'protocol.json');ps=stage.WindowsProcesses()
    assert ps.creation(c['pid'])==c['birth'] and ps.creation(c['worker_pid'])==c['worker_birth']
    from scripts import z13_region_assignment_recovery2_20261025 as notes
    notes.namespace()['handoff']('ТЕКУЩИЙ z13_anchor_state_v1: runs/z13_anchor_state_20261025; scripts/z13_anchor_state_20261025.py --pipeline. Контроллер/worker PID+birth '+str(c)+'. ОдинGPUfeatures1800с, затемCPU26replays900с; общийsave '+p['pipeline_save_at']+' / hard '+p['pipeline_hard_stop']+', меньшие бюджеты не продлевать. Только состояние: каждый запрос от того же четырёхкадрового образца, без промежуточных поисковых кадров; C/D/R1 old/anchor+геометрия. Общие исходные prefix3/4 переиспользованы точно, не отдельный GPU-проход; старые8326центров/score проверяются с прежними допусками; первый поиск exact. Нормы/веса/регуляризация/цель/mean-region/null0/alpha0.25 неизменны. Exposureэто частота поздних запросов, не доказательство вреда памяти. Все3теста и компиляция источников прошли. Первый2Dконсервативныйзамерцены обязателен, при превышении сохранитьpartial. Смотреть features/price.json, phase/controller_result.json, status.json и pipeline_failure.json. Очередь сама features→replay→первичныехеши/ДИ/карточка/журнал/тесты/защитныйграф/scopedpush. Нетдубликатов/загрузок/обученияядра/dev/HOTA/75. Ранееuncertainty/NULLотрицательны, обаavailabilitygateнеопределённы; архитектура1лист107/97опубликована2f221f09. Всёскрыто,57чужихправок/.gitignoreсохранять; авторскийконец26.10 07:40МСК.')
    stage.publish('start')
