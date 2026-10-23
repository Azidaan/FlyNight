"""Resume verified completed jobs after absent unused-flow FPS metadata."""
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_matched_pipeline_20261024 as P
from scripts.z13_matched_trackers_20261024 import RUN,read,write,sha,now,journal
from scripts.z13_matched_fps_recovery_20261024 import install,OVERLAY
OLD=RUN/'gpu_recovery1';REC=RUN/'gpu_recovery2'


def prepare():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    assert not REC.exists();old=read(OLD/'protocol.json');verify_sources(old)
    e=read(OLD/'controller_result.json');c=read(OLD/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    assert ps.creation(c['pid'])!=c['birth'] and all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    remaining=int((datetime.fromisoformat(c['save_at'])-now()).total_seconds());assert remaining>900
    jobs=list((OLD/'jobs').glob('*.json'));assert len(jobs)==28
    for path in jobs:
        data=read(path)
        for n,h in data['outputs'].items():assert sha(ROOT/n)==h,n
    names=['scripts/z13_matched_pipeline_recovery2_20261024.py','scripts/z13_matched_fps_recovery_20261024.py',str((OVERLAY/'manifest.json').relative_to(ROOT)).replace('\\','/')]
    p={**old,'budget_seconds':remaining,'save_at':c['save_at'],'hard_stop':c['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**old['sha256'],**{n:sha(ROOT/n) for n in names}},'service_recovery2':dict(previous='gpu_recovery1',previous_result_sha256=sha(OLD/'controller_result.json'),previous_protocol_sha256=sha(OLD/'protocol.json'),verified_completed_jobs=len(jobs),remaining_original_seconds=remaining,reason=read(OVERLAY/'manifest.json')['change'])}
    write(REC/'protocol.json',p)
    for path in jobs:write(REC/'jobs'/path.name,read(path))
    write(REC/'status.json',dict(state='ready',stage='28готовых заданий проверены; восстановлены метаданные FPS',updated=now().isoformat()))
    journal('восстановлено чтение частоты кадров общего сравнения','В02:02МСК GPUэтап остановился на пятом ролике b1d22449-15fb948f: отсутствует runs/video_rate/<id>/motion/flyvis_tiled.json. Сборщик использует оттуда только fps,не массив движения. В отдельном fps_metadata_recovery частота каждого из7роликов точно сверена с контейнером исходного видео,summary и всеми существующими прежними метаданными;хеш видео проверен. Исходные данные/скрипты неизменны. Отдельная служебная версия recovery2 меняет только путь чтения fps;28готовыхзаданий проверены по хешам и сохранены. Прежниеошибка/источники/журналы сохранены. Исходныйконец05:53:48МСК и общийбюджет не продлены;CPUопыт продолжается независимо. Источники: gpu_recovery2/protocol.json и fps_metadata_recovery/manifest.json.')


def bind():
    P.STAGE=REC;P.__file__=str(Path(__file__));P.T0=time.monotonic();install()


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    verify_sources(read(REC/'protocol.json'));bind()
    for video in P.VIDEOS:
        for arm in P.SYSTEMS:P.status('Продолжение общего сравнения',variant=arm+' '+video);P.child('signature',arm,video)
        P.child('masa','all',video)
    P.evaluate();write(REC/'status.json',dict(state='completed',stage='Все6систем и7видео готовы',updated=now().isoformat()))


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
