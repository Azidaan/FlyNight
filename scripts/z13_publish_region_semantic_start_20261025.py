"""Publish the active hidden semantic-target trial and completed preflight proofs."""
from pathlib import Path
import inspect
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_semantic_20261025 as stage


if __name__=='__main__':
    c=stage.read(stage.RUN/'controller.json');p=stage.read(stage.RUN/'protocol.json');ps=stage.WindowsProcesses()
    assert ps.creation(c['pid'])==c['birth'] and ps.creation(c['worker_pid'])==c['worker_birth']
    receipt=stage.read(ROOT/'runs/z13_region_semantic_execution_check_20261025/receipt.json')
    assert receipt['all12_synthetic_execution_paths_passed'] and receipt['finisher_transforms_and_compile_passed'] and receipt['actual_protocol_matches']
    stage.original.namespace()['handoff']('ТЕКУЩИЙ z13_region_semantic_v1, runs/z13_region_semantic_20261025; scripts/z13_region_semantic_20261025.py --pipeline. '
        'Контроллер/workerPID+birth: '+str(c)+'; launch.json. Всё скрыто, без видимой консоли, автор25.10 ушёл до утра. '
        'CPU12новыхмоделей D/C/R1/геометрия,3сида;12старых utility-контролей полностью воспроизведены и не обучаются повторно. '
        'Меняется только допустимый целевой ответ: любое собственное правильное невредное действие, включая правильное бездействие, вместо только редкого улучшения. '
        'Та же маргинальная вероятность, прежние88входная модель/кэш/пакеты/1150шагов/fit20пороги. '
        'Наличие данных проверено: fit8416737правильныхневредных против1356полезных действий; это не доказанная причина. '
        'Первые12синтетических путей, фактический градиент fit84, построение завершателя и прежние12моделей проверены. '
        '1800с+20с, save '+p['save_at']+' / hard '+p['hard_stop']+' не продлевать; после первоймодели консервативнаяцена. '
        'Смотреть status.json, training/price.json, controller_result.json, worker.stderr.log, pipeline_failure.json. Известные служебные сбои чинить отдельной версией в первоначальном окне. '
        'Очередь сама проверяет12новых полных выходов/хеши/интервалы, каталог/тесты/сохраняющийграф/scopedpush. '
        'НикакихGPU/ядра/dev/HOTA/75/загрузок; кHOTA только после полного основного критерия. '
        'До этого контекст/product/pooling отрицательны, опубликованы4e0580bf;1лист103блока/93стрелки просмотрен, новых подтверждённых лучших результатов нет. '
        'При полном исходе проверить отдельные confirmed_results и архитектурную веху. Авторизация26.10 07:40МСК,57чужихграфправок/.gitignore сохранять.')
    code=inspect.getsource(stage.publish);old='    names=sorted(set(names));'
    assert code.count(old)==1;code=code.replace(old,'    names += EXTRA_OWN_FILES\n'+old)
    ns=dict(vars(stage));ns['EXTRA_OWN_FILES']=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_semantic_execution_check_20261025.py','runs/z13_region_semantic_execution_check_20261025/receipt.json']
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['publish']('start')
