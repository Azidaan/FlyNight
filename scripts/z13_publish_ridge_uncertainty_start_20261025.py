"""Register the active hidden uncertainty prototype and its verified controller."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_ridge_uncertainty_20261025 as stage

if __name__=='__main__':
    p=stage.read(stage.RUN/'protocol.json');c=stage.read(stage.RUN/'features/controller.json');ps=stage.WindowsProcesses()
    assert ps.creation(c['pid'])==c['birth'] and ps.creation(c['worker_pid'])==c['worker_birth']
    cat=stage.read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==stage.IDENT)
    assert card['state']=='ready';card['state']='running';stage.write(ROOT/'experiments/project_history.json',cat)
    stage.journal('скрытый запуск условной неопределённости',
        '*Запущено.* scripts/z13_ridge_uncertainty_20261025.py --pipeline, controller '+str(c)+'. Один GPU, без окна. '
        'Три матричных/операторных теста пройдены. Старые веса, lambda, временной ход и средние отклики сохраняются. '
        'Десять условий: C/D/R1 × прежнее/неопределённость/постоянный штраф, плюс геометрия. '
        'GPU1800с → CPU26повторов900с. Общий save '+p['pipeline_save_at']+' / hard '+p['pipeline_hard_stop']+'. Источник runs/z13_ridge_uncertainty_20261025/launch.json; меньшие бюджеты не продлевать.')
    from scripts import z13_region_assignment_recovery2_20261025 as old
    old.namespace()['handoff']('ТЕКУЩИЙ z13_ridge_uncertainty_v1, runs/z13_ridge_uncertainty_20261025; scripts/z13_ridge_uncertainty_20261025.py --pipeline. '
        'Реальный контроллер/launcher PID+birth '+str(c)+'; pipeline.json/launch.json. Всё скрыто; feature1800с→CPUreplay900с, общийsave '+p['pipeline_save_at']+' / hard '+p['pipeline_hard_stop']+'. '
        'Меняется только опора уверенности: условная ковариация гребневой модели по наблюдённому образцу, штраф1.96std; средние отклики/веса/нормы/lambda/исходное время/alpha0.25/null0 прежние. '
        '10условий C/D/R1 old/posterior/constant и геометрия; constant использует ту же среднюю неопределённость без пространственной избирательности. '
        'Три основных среднего Dposterior-база/Dold/Dconstant>=2п.п., все6ДИнижние>0. Это не калиброванная вероятность полезности илиHOTA. '
        'Все8326центров<0.5px/score<0.001; первые2Dценапрежнегоокна; все26исходныхбайтов/oldназначенийточны. '
        'Смотреть features/price.json, features/worker.stderr.log, features/replay controller_result.json, status.json и pipeline_failure.json. '
        'Очередь сама делает features→replay→первичныехеши/интервалы/карточка/каталог+61тест/сохраняющийграф/scopedpush. Известнуюслужебнуюошибку исправлять отдельнойверсией в прежнемокне, готовое не повторять. '
        'Никакихзагрузок/dev/HOTA/75/обученияядра. Старый spatial_bins gateнеопределённый396a8b28, его критерии не ослаблять и кизвлечениюнепродвигать. '
        'При полном новомитоге проверить confirmed_results и архитектурнуювеху.57чужихправок/.gitignore сохранять, до26.10 07:35сохранить/07:39:30конец.')
    source=inspect.getsource(stage.clock.publish)
    source=source.replace("'tracking/z13_history_clock.py','tests/test_z13_history_clock.py'","'tracking/z13_ridge_uncertainty.py','tests/test_z13_ridge_uncertainty.py'").replace("'tests/test_z13_history_clock.py'","'tests/test_z13_ridge_uncertainty.py'").replace('physical-frame temporal resampling inside frozen FlyVis','template-supported uncertainty in frozen FlyVis readout')
    needle='    assert not set(names)&foreign';assert source.count(needle)==1
    source=source.replace(needle,"    names += ['scripts/z13_publish_ridge_uncertainty_start_20261025.py']\n"+needle)
    ns=dict(vars(stage));exec(compile(source,__file__,'exec'),ns);ns['publish']('start')
