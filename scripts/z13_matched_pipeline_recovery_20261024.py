"""Service correction: price remaining six videos, preserve original total budget."""
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_matched_pipeline_20261024 as P
from scripts.z13_matched_trackers_20261024 import RUN,read,write,sha,now,journal
OLD=RUN/'gpu_stage';REC=RUN/'gpu_recovery1';T0=time.monotonic()


def prepare():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    assert not REC.exists();old=read(OLD/'protocol.json');verify_sources(old);e=read(OLD/'controller_result.json');controller=read(OLD/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    assert ps.creation(controller['pid'])!=controller['birth'] and all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    price=read(OLD/'price.json');assert price['proceed'] is False
    soft=datetime.fromisoformat(controller['save_at']);remaining=int((soft-now()).total_seconds());assert remaining>900
    corrected=2*price['first_complete_video_seconds']*(len(P.VIDEOS)-1)+600
    assert corrected<remaining,('Remaining jobs still do not fit',corrected,remaining)
    p={**old,'budget_seconds':remaining,'save_at':soft.isoformat(),'hard_stop':controller['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**old['sha256'],'scripts/z13_matched_pipeline_recovery_20261024.py':sha(Path(__file__))},'service_recovery':dict(old_result_sha256=sha(OLD/'controller_result.json'),old_price_sha256=sha(OLD/'price.json'),original_started=controller['started'],original_budget_seconds=old['budget_seconds'],corrected_remaining_videos=6,conservative_remaining_seconds=corrected,remaining_original_seconds=remaining,reason='Original technical formula2xfirstvideo*7+600 compared all7 with the remaining budget, including one already completed video. Correct remaining scope is6; safety factor2 and600seconds reserve unchanged. Science, thresholds, frozen sources, total original budget and deadline unchanged.')}
    write(REC/'protocol.json',p)
    jobs=list((OLD/'jobs').glob('*.json'));assert len(jobs)==7
    for path in jobs:
        data=read(path)
        for n,h in data['outputs'].items():assert sha(ROOT/n)==h,n
        write(REC/'jobs'/path.name,data)
    write(REC/'status.json',dict(state='ready',stage='Первый ролик проверен; продолжение оставшихся6',updated=now().isoformat()))
    journal('исправлен учёт оставшихся заданий в бюджете сравнения','Первый полный ролик всех6систем/MASA сохранён и проверен;7готовыхмаркерoв. Исходный служебный ценовой контроль остановился: runs/z13_matched_trackers_20261024/gpu_stage/price.json,стоимость всех7 ошибочно сопоставлена остатку после первого. Отдельная версия scripts/z13_matched_pipeline_recovery_20261024.py считает только оставшиеся6;множитель2 и резерв600с неизменны. Остаток и оценка записаны в gpu_recovery1/protocol.json/service_recovery. Прежниеошибка/протокол/журналы сохранены. Весь исходный бюджет18000с отсчитывается от00:53:28;не продлён,контроллер завершает до исходного05:53:48МСК. Научные данные/веса/пороги не изменены,готовое не повторяется.')


def bind():
    P.STAGE=REC;P.__file__=str(Path(__file__));P.T0=time.monotonic()


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(REC/'protocol.json');verify_sources(p);bind()
    first=P.VIDEOS[0]
    for arm in P.SYSTEMS:P.child('signature',arm,first)
    P.child('masa','all',first)
    for video in P.VIDEOS[1:]:
        for arm in P.SYSTEMS:P.status('Продолжение общего сравнения',variant=arm+' '+video);P.child('signature',arm,video)
        P.child('masa','all',video)
    P.evaluate();write(REC/'status.json',dict(state='completed',stage='Все6систем и7видео,обе общие процедуры готовы',updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--job' in sys.argv:
        bind();mode,arm,video=sys.argv[2:5];assert mode in ('signature','masa') and video in P.VIDEOS and arm in P.SYSTEMS+('all',)
        result=P.build_signature(arm,video) if mode=='signature' else P.build_masa(video)
        paths=[RUN/arm/'signatures'/video/'signatures.npz'] if mode=='signature' else [RUN/a/'masa'/(video+'.npz') for a in P.SYSTEMS]
        write(REC/'jobs'/f'{mode}_{arm}_{video}.json',dict(result=result,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in paths},at=now().isoformat()))
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(REC/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit recovery mode required')
