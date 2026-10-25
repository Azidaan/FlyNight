"""Recover immutable original crops; retain original failed run and deadline."""
from pathlib import Path
import inspect
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import z13_proposal_geometry_20261025 as base
from scripts.z13_region_choice_gate_recovery1_20261025 import crop_metadata
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=base.RUN/'recovery1'


def hydrate(queries,p):
    metadata=crop_metadata(p);count=0
    for video in p['videos']:
        for q in queries[video]:
            l=q['label'];m=metadata[l['label']]
            assert all(l[k]==m[k] for k in ('video','pos','kind','pred')) and q['frame']==m['frame']
            l['crop']=m['crop'];count+=1
    assert count==8326
    write(RUN/'crop_binding.json',dict(at=now().isoformat(),rows=count,all_original_crops_bound=True))


def worker_source():
    s=inspect.getsource(base.worker)
    a="queries,centers=original.load_queries(read(OLD/'protocol.json'));wanted=set(p['wanted_uids'])"
    assert s.count(a)==1
    return s.replace(a,"queries,centers=original.load_queries(read(OLD/'protocol.json'));hydrate(queries,p);wanted=set(p['wanted_uids'])")


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    assert not RUN.exists()
    p,e=read(base.RUN/'protocol.json'),read(base.RUN/'controller_result.json')
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    assert not list((base.RUN/'videos').glob('*.json'))
    base.verify_sources(p);meta=crop_metadata(p);assert len(meta)==8326
    extra=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_choice_gate_recovery1_20261025.py',
           (base.RUN/'protocol.json').relative_to(ROOT).as_posix(),(base.RUN/'controller_result.json').relative_to(ROOT).as_posix(),(base.RUN/'failure.json').relative_to(ROOT).as_posix()]
    extra += [f'runs/z13_data/{v}.npz' for v in p['videos']]
    q=dict(p,service_recovery=dict(at=now().isoformat(),error="KeyError('crop'): compact labels lack original crop",completed_videos=0,
        change='Hydrate original fr_crop through verified label/video/position/frame/pred binding. No changed action, tolerance, outcome, data split, thresholds or deadline. Prior sources/logs/exit1 retained.',
        original_save_at=p['save_at'],original_hard_stop=p['hard_stop']),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        executed_worker_sha256=hashlib.sha256(worker_source().encode()).hexdigest(),sha256={**p['sha256'],**{n:sha(ROOT/n) for n in extra}})
    write(RUN/'protocol.json',q);write(RUN/'crop_preflight.json',dict(at=now().isoformat(),rows=8326,all_original_crops_bound=True))
    write(RUN/'status.json',dict(state='ready',stage='Исходные вырезки привязаны; прежний бюджет',done=0,total=26,updated=now().isoformat()))
    base.journal('служебное восстановление геометрической проверки','*Ошибка агента.* Повторно не учтено отсутствие crop в коротком reserve-списке. Исходный exit1 и журналы сохранены, готовых видео0.\n\n*Исправление.* Проверенная привязка8326исходных fr_crop по video/episode/pos/frame/pred, recovery1/crop_preflight.json. Источники отдельной версии, критерии и сроки '+p['save_at']+' / '+p['hard_stop']+' прежние. Для следующих адаптаций обязательна проверка фактических ключей меток до запуска.')


def namespace():
    ns={**vars(base),'RUN':RUN,'hydrate':hydrate}
    exec(compile(inspect.getsource(base.summarize),__file__,'exec'),ns)
    return ns


def worker():
    p=read(RUN/'protocol.json');s=worker_source();assert hashlib.sha256(s.encode()).hexdigest()==p['executed_worker_sha256']
    ns=namespace();exec(compile(s,__file__,'exec'),ns);ns['worker']()


def finish():
    ns=namespace();exec(compile(inspect.getsource(base.finish),__file__,'exec'),ns);ns['finish']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--finish' in sys.argv:finish()
    else:raise RuntimeError('Explicit mode required')
