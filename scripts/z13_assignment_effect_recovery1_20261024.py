"""Serializer-only recovery; retain both exact video jobs and original deadline."""
from pathlib import Path
import hashlib
import inspect
import shutil
import sys
import os
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
BASE=ROOT/'runs/z13_assignment_effect_20261024';REC=BASE/'recovery1';W=ROOT/'runs/z13_weekend_20261024'


def worker_source():
    from scripts import z13_assignment_effect_20261024 as O
    s=inspect.getsource(O.worker);old='proceed=estimate<available'
    assert s.count(old)==1
    return s.replace(old,'proceed=bool(estimate<available)')


def prepare():
    from datetime import datetime
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not REC.exists();p=read(BASE/'protocol.json');verify_sources(p)
    e=read(BASE/'controller_result.json');c=read(BASE/'controller.json');m=read(BASE/'completion.json');ps=WindowsProcesses()
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=b for pid,b in {**e['owned'],str(c['pid']):c['birth'],str(m['pid']):m['birth']}.items())
    assert "TypeError: Object of type bool is not JSON serializable" in (BASE/'worker.stderr.log').read_text(encoding='utf-8')
    videos=p['fit20'][:2];assert sorted(x.stem for x in (BASE/'videos').glob('*.json'))==sorted(videos)
    hashes={};seconds=[]
    for video in videos:
        path=BASE/'videos'/(video+'.json');r=read(path)
        assert all(r['checks'][k] for k in ('raw_replay_exact','all_predictions_and_identities_bound'))
        seconds.append(r['seconds']);dest=REC/'videos'/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);assert sha(dest)==sha(path);hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    estimate=2*(sum(seconds)/2)*128+180;available=(datetime.fromisoformat(c['save_at'])-now()).total_seconds();assert estimate<available
    p={**p,'budget_seconds':int(available),'save_at':c['save_at'],'hard_stop':c['hard_stop'],
       'sha256':{**p['sha256'],**hashes,Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__)),
           (BASE/'controller_result.json').relative_to(ROOT).as_posix():sha(BASE/'controller_result.json'),
           (BASE/'protocol.json').relative_to(ROOT).as_posix():sha(BASE/'protocol.json')},
       'service_recovery':dict(change='Cast only NumPy comparison flag to Python bool for JSON. No math,threshold,labels,tracker or candidate change. Two exact completed jobs copied with hashes,never replayed.',
           previous_error_sha256=sha(BASE/'worker.stderr.log'),retained_videos=videos,original_deadline_unchanged=True,
           worker_executed_source_sha256=hashlib.sha256(worker_source().encode()).hexdigest())}
    write(REC/'protocol.json',p);write(REC/'price.json',dict(first2_seconds=seconds,conservative_remaining_seconds=estimate,
        actual_remaining_seconds=available,proceed=bool(estimate<available),retained_first2=True))
    write(REC/'status.json',dict(state='ready',stage='Два точных видео сохранены;исправлена запись флага JSON',updated=now().isoformat(),done=2,total=130))


def bind():
    from scripts import z13_assignment_effect_20261024 as O
    O.RUN=REC;O.T0=time.monotonic()
    source=worker_source();assert hashlib.sha256(source.encode()).hexdigest()==read(REC/'protocol.json')['service_recovery']['worker_executed_source_sha256']
    ns=dict(vars(O));exec(compile(source,str(Path(__file__)), 'exec'),ns);O.worker=ns['worker']
    from scripts import z13_publish_assignment_effect_20261024 as P
    P.RUN=REC
    source=inspect.getsource(P.publish)
    assert source.count("names=['experiments/project_history.json'")==1 and source.count('for stage in (OLD,RUN):')==1
    source=source.replace("names=['experiments/project_history.json'", "names=['scripts/z13_assignment_effect_recovery1_20261024.py','experiments/project_history.json'")
    source=source.replace('for stage in (OLD,RUN):','for stage in (OLD,BASE,RUN):')
    ns={**vars(P),'BASE':BASE};exec(compile(source,str(Path(__file__)), 'exec'),ns)
    return O,P,ns['publish']


def header(phase):
    path=W/'HANDOFF.md'
    text=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'ТЕКУЩАЯ версия assignment_effect/recovery1: runs/z13_assignment_effect_20261024/recovery1,'
        'scripts/z13_assignment_effect_recovery1_20261024.py. Фаза '+phase+'. '
        'Смотреть её controller/completion/status/controller_result/monitor_failure/report,'
        'нестарыйошибочный корень. Первые2fit20видео уже точно повторены и сохранены. '
        'Исходный запуск остановлен при записи price:NumPy bool не JSONтип. '
        'Старыеexit1/журналы/исходник сохранены. Исправлено только приведение этого флага кbool. '
        'Остальные128видео без повторения готовых,научные условия/точный допуск/данные прежние. '
        'Исходные сохранение25.10 00:23:43.970600МСК/жёсткий конец00:24:03.970600 НЕПРОДЛЕНЫ. '
        'publication assignment_effect_recovery1_'+phase+'_publication.json;monitor самзаписываетполныйитог/проверки/граф/точныйpush. '
        'Источникисправления recovery1/protocol.json/service_recovery. При неполном результате не выводить эффективность. '
        'Прежняя ошибочная постановкаunperturbed остановлена,не повторять.\n\nНиже прежние передачи сохранены.\n\n')
    path.write_text(text+path.read_text(encoding='utf-8'),encoding='utf-8')


def start():
    O,P,publish=bind();P.handoff('start')
    O.journal('служебное восстановление записи цены без повтора двух видео',
        'runs/z13_assignment_effect_20261024/recovery1/protocol.json:оба первыхfit20видео имеют точныеrawвыходы,'
        'точныепрогнозы/идентичности. Их JSONсохранены с SHA256. Ошибка исходной попытки — '
        'NumPyboolвполеproceedпри JSONзаписиprice,после двух законченных видео. '
        'Изменено только bool(estimate<available),без вычислительных изменений. '
        'Новый независимый контроллер использует прежниеабсолютныесроки00:23:43/00:24:03МСК;'
        'старыеexit1/логи/исходникинепереписаны. Цена128остатка проверена по двум сохранённым временам.')
    header('start');publish('recovery1_start')


def monitor():
    from datetime import datetime
    O,P,publish=bind();ps=WindowsProcesses();write(REC/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    while not (REC/'controller_result.json').exists():
        if now()>datetime.fromisoformat(read(REC/'protocol.json')['hard_stop']):raise TimeoutError('No controller result before original hard deadline')
        time.sleep(3)
    e=read(REC/'controller_result.json')
    if e['exit_code']!=0 or not e['all_owned_ended']:write(REC/'monitor_failure.json',dict(at=now().isoformat(),result=e));return
    while not (W/'assignment_effect_recovery1_start_publication.json').exists():
        if now()>datetime.fromisoformat(read(REC/'protocol.json')['hard_stop']):raise TimeoutError('Startup verification not finished in original window')
        time.sleep(3)
    O.record();P.handoff('finish');header('finish');publish('recovery1_finish')


if __name__=='__main__':
    if sys.argv[1]=='--prepare':prepare()
    elif sys.argv[1]=='--worker':
        O,_,_=bind()
        try:O.worker()
        except Exception as error:write(REC/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    elif sys.argv[1]=='--start':start()
    elif sys.argv[1]=='--monitor':
        try:monitor()
        except Exception as error:write(REC/'monitor_failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    else:raise ValueError(sys.argv[1])
