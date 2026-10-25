"""Service recovery: bind map-sidecar label indices to existing episode metadata."""
from pathlib import Path
import sys
import inspect
import hashlib
import subprocess
import time
import os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_history_clock_20261025 as science
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
BASE=science.RUN
RUN=BASE/'recovery1'


def source():
    code=inspect.getsource(science.feature_worker)
    old="data=Episodes(p['evaluate26']);labels={a:read(decoder.MAPS/a/'evaluate/labels.json') for a in ('D','C','R1')}"
    new=old+"\n    for items in labels.values():\n        for label in items:\n            row=data.labels[int(label['label'])]\n            assert row[1]==label['pos'] and row[2]==label['kind'] and np.array_equal(row[6],label['pred']) and np.array_equal(row[7],label['crop'])\n            label['episode']=int(row[0])"
    assert code.count(old)==1;return code.replace(old,new)


def worker():
    phase=os.environ['Z13_HISTORY_CLOCK_PHASE']
    if phase=='replay':science.replay_worker();return
    def service_read(path):
        if Path(path)==BASE/'features/protocol.json':path=RUN/'features/protocol.json'
        return read(path)
    ns={**vars(science),'read':service_read}
    code=source();assert hashlib.sha256(code.encode()).hexdigest()==read(RUN/'features/protocol.json')['executed_feature_worker_sha256']
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['feature_worker']()


def prepare():
    assert not RUN.exists();ps=WindowsProcesses();e=read(BASE/'features/controller_result.json')
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining'] and all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    assert not list((BASE/'features').glob('*/maps_*.json'))
    assert "KeyError: 'episode'" in (BASE/'features/worker.stderr.log').read_text(encoding='utf-8')
    p=read(BASE/'features/protocol.json');science.verify_sources(p)
    from scripts.z13_train import Episodes
    import numpy as np
    data=Episodes(p['evaluate26']);bindings={}
    for arm in ('D','C','R1'):
        items=read(science.decoder.MAPS/arm/'evaluate/labels.json');assert len(items)==8326
        for label in items:
            row=data.labels[int(label['label'])]
            assert row[1]==label['pos'] and row[2]==label['kind'] and np.array_equal(row[6],label['pred']) and np.array_equal(row[7],label['crop'])
            if label['kind']==1:assert np.array_equal(row[4],label['gt'])
        bindings[arm]=dict(rows=len(items),episodes=len(set(data.labels[x['label']][0] for x in items)),positions_kinds_predictions_crops_equal=True)
    write(RUN/'metadata_binding.json',dict(at=now().isoformat(),all3arms=bindings,
        change='Only recover episode index from the existing label-indexed Episodes table. Map sidecars intentionally omit episode. No inputs, ordering, weights, targets, delta rules, thresholds or budgets change.',
        scientific_clock_sha256=sha(ROOT/'tracking/z13_history_clock.py'),old_source_sha256=sha(ROOT/'scripts/z13_history_clock_20261025.py'),
        executed_feature_worker_sha256=hashlib.sha256(source().encode()).hexdigest(),original_save_at=p['save_at'],original_hard_stop=p['hard_stop'],completed_chunks=0))
    from datetime import datetime
    assert (datetime.fromisoformat(p['save_at'])-now()).total_seconds()>600
    names=[Path(__file__).relative_to(ROOT).as_posix(),(RUN/'metadata_binding.json').relative_to(ROOT).as_posix()]
    q=dict(p,service_recovery=True,executed_feature_worker_sha256=hashlib.sha256(source().encode()).hexdigest(),sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'features/protocol.json',q);write(RUN/'features/status.json',dict(state='ready',stage='Map label -> original episode binding verified',updated=now().isoformat()))
    science.journal('служебное восстановление адреса эпизода временного узла',
        '*Ошибка.* Исходный feature worker остановился до первого GPU-блока: KeyError(episode), потому что описание карт хранит label, а номер эпизода находится в исходной таблице Episodes. '
        'Старый exit1, pipeline_failure и журналы сохранены. '
        '*Проверено.* Все8326строк каждогоC/D/R1 связываются по label с теми жеpos/kind/pred/crop и истинным центром приkind1;2188естественных эпизодов. '
        '*Исправление.* Отдельный scripts/z13_history_clock_recovery1_20261025.py добавляет только адрес episode из существующей таблицы. '
        'Математика часов, данные, исходники, веса, пороги и допуски неизменны; готовых блоков0. '
        'Прежние featuresave '+p['save_at']+' / hard '+p['hard_stop']+' не продлеваются. Источник runs/z13_history_clock_20261025/recovery1/metadata_binding.json.')


def publish(stage):
    code=inspect.getsource(science.publish);old='    assert not set(names)&foreign'
    assert code.count(old)==1;code=code.replace(old,'    names += SERVICE_FILES\n'+old)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_publish_history_clock_start_20261025.py']
    names += [p.relative_to(ROOT).as_posix() for p in RUN.rglob('*.json')]
    names += [(BASE/n).relative_to(ROOT).as_posix() for n in ('pipeline_failure.json','features/controller_result.json')]
    ns=dict(vars(science));ns['SERVICE_FILES']=names
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['publish'](stage)


def finish():
    ns={**vars(science),'publish':publish}
    exec(compile(inspect.getsource(science.finish),str(Path(__file__)),'exec'),ns);ns['finish']()


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        for phase in science.BUDGETS:
            folder=RUN/phase
            if phase=='replay':
                q=science.phase_protocol(phase);q['sha256'][Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__));write(BASE/phase/'protocol.json',q);write(folder/'protocol.json',q)
            with (folder/'controller.stdout.log').open('w',encoding='utf-8') as out,(folder/'controller.stderr.log').open('w',encoding='utf-8') as err:
                child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',str(Path(__file__))],cwd=ROOT,env={**os.environ,'Z13_HISTORY_CLOCK_PHASE':phase},stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
                while child.poll() is None:
                    if (BASE/phase/'status.json').exists():
                        try:write(RUN/'status.json',dict(read(BASE/phase/'status.json'),phase=phase))
                        except (OSError,ValueError):pass
                    time.sleep(2)
                child.wait()
            c=read(folder/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
            assert read(BASE/phase/'report.json')['complete']
        finish()
    except Exception as error:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),phase=phase,error=repr(error)));raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    else:raise RuntimeError('Explicit mode required')
