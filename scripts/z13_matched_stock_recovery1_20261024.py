"""Resume stock controls using the previously validated R1 singleton adapter."""
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_matched_stock_execution4_20261024 as E
S=E.S;OLD=S.STAGE;REC=S.RUN/'stock_recovery1'


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not REC.exists();E.verify();p=S.read(OLD/'protocol.json');verify_sources(p)
    e=S.read(OLD/'controller_result.json');c=S.read(OLD/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    assert ps.creation(c['pid'])!=c['birth'] and all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    contract=S.read(ROOT/'runs/z13_center_hota_20261021/recovery1_contract.json')
    assert S.sha(ROOT/'tracking/z13_ridge_localizer_v2.py')==contract['sha256']['tracking/z13_ridge_localizer_v2.py']
    jobs=list((OLD/'jobs').glob('*.json'));assert len(jobs)==3
    for path in jobs:
        for n,h in S.read(path)['outputs'].items():assert S.sha(ROOT/n)==h,n
    remaining=int((datetime.fromisoformat(c['save_at'])-S.now()).total_seconds());assert remaining>900
    names=['scripts/z13_matched_stock_recovery1_20261024.py','tracking/z13_ridge_localizer_v2.py','tests/test_z13_ridge_live_shape.py','runs/z13_center_hota_20261021/recovery1_contract.json']
    p={**p,'budget_seconds':remaining,'save_at':c['save_at'],'hard_stop':c['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**p['sha256'],**{n:S.sha(ROOT/n) for n in names}},'service_recovery':dict(previous='stock_controls',previous_result_sha256=S.sha(OLD/'controller_result.json'),science_protocol_sha256=S.sha(OLD/'protocol.json'),execution4_sha256=S.sha(OLD/'execution_v4.json'),original_started=c['started'],original_budget_seconds=10800,verified_jobs=3,reason='Actual live retina B,T,1,N versus offline B,T,N. Previously validated v2 adapter removes only singleton R1 channel, C/D delegate unchanged. The earlier known repair was omitted from this stock runner; restored without changing scientific source, samples, tau/lambda, weights or deadline. First-video price sums original saved job times as well as new jobs, not just resumed wall time.')}
    S.write(REC/'protocol.json',p)
    for path in jobs:S.write(REC/'jobs'/path.name,S.read(path))
    S.write(REC/'status.json',dict(state='ready',stage='Сохранены3задания; восстановлен прежний адаптер формыR1',updated=S.now().isoformat(),jobs_done=3,jobs_total=35))
    S.journal('исправлено повторение известной ошибки формы R1','Общее сравнение6систем завершено03:30:48exit0;49заданий и основные результаты проверены,источник runs/z13_matched_trackers_20261024/report.json. Следующие контроли автоматически стартовали03:32:15,но03:41:54остановились на centerR1:IndexError axis2size17. В новую очередь ошибочно не был подключён уже проверенный tracking/z13_ridge_localizer_v2.py;это моя служебная ошибка,не научный итог. Адаптер ранее исправлял лишнюю единичную ось живого глаза;егохеш точно равен прежнему контракту21.10,4проверки регрессии/живойформы пройдены. Исходные скрипты/веса/пороги неизменны,старые ошибка/журналы сохранены. В stock_recovery1 проверены3готовыхзадания,они не повторяются;исходный бюджет10800с от03:32:15 и конец06:32:35МСК не продлены. Цена первого полного ролика учитывает прежнее время всех сохранённых заданий,а не только остатокпослерестарта.')


def bind():
    E.verify();E.E.install()
    from scripts.z13_record_matched_recovery3_20261024 import install
    install()
    import tracking.z13_ridge_localizer as original
    from tracking.z13_ridge_localizer_v2 import RecoveredCenterRidgeLocalizer
    original.CenterRidgeLocalizer=RecoveredCenterRidgeLocalizer
    S.STAGE=REC;S.__file__=str(Path(__file__));S.T0=time.monotonic()


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=S.read(REC/'protocol.json');verify_sources(p);bind()
    for i,video in enumerate(S.VIDEOS):
        for arm in S.SYSTEMS:S.status('Штатная муха и контроли: продолжение',variant=arm+' '+video);S.child('signature',arm,video)
        S.child('masa','all',video)
        if i==0:
            first=[S.read(REC/'jobs'/f'signature_{a}_{video}.json') for a in S.SYSTEMS]+[S.read(REC/'jobs'/f'masa_all_{video}.json')]
            measured=sum(x['wall_seconds_with_imports'] for x in first);estimate=2*measured*6+600
            available=min(p['budget_seconds']-(time.monotonic()-S.T0),(datetime.fromisoformat(p['save_at'])-S.now()).total_seconds())
            S.write(REC/'price.json',dict(first_video_seconds_including_preserved_jobs=measured,conservative_remaining_seconds=estimate,available_seconds=available,proceed=estimate<available,completed_original_jobs_included=True))
            assert estimate<available,'Remaining stock work exceeds original remaining budget'
    S.evaluate();S.write(REC/'status.json',dict(state='completed',stage='Все35заданий и общий отчёт10систем готовы',updated=S.now().isoformat(),jobs_done=35,jobs_total=35))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:S.write(REC/'status.json',dict(state='failed',stage=str(e),updated=S.now().isoformat()));raise
    elif '--job' in sys.argv:
        bind();mode,arm,video=sys.argv[2:5];assert mode in ('signature','masa') and video in S.VIDEOS and arm in S.SYSTEMS+('all',)
        result=S.build_signature(arm,video) if mode=='signature' else S.build_masa(video)
        paths=[S.RUN/arm/'signatures'/video/'signatures.npz'] if mode=='signature' else [S.RUN/a/'masa'/(video+'.npz') for a in S.SYSTEMS]+[REC/'baseline_masa_replay'/(video+'.npz')]
        S.write(REC/'jobs'/f'{mode}_{arm}_{video}.json',dict(result=result,outputs={q.relative_to(ROOT).as_posix():S.sha(q) for q in paths},at=S.now().isoformat()))
    else:raise RuntimeError('Explicit recovery mode required')
