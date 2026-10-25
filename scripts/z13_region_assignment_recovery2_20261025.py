"""Service-only resume: restore CUDA after importing the CPU replay helper.

All completed labels are reused. Original feature and global deadlines persist.
"""
from pathlib import Path
import inspect
import hashlib
import os
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
BASE=ROOT/'runs/z13_region_assignment_20261025'
RUN=BASE/'recovery2'


def source():
    s=(ROOT/'scripts/z13_region_assignment_20261025.py').read_text(encoding='utf-8')
    replacements=[
        ("RUN=ROOT/'runs/z13_region_assignment_20261025'","RUN=ROOT/'runs/z13_region_assignment_20261025/recovery2'"),
        ("for number,phase in enumerate(BUDGETS):","for number,phase in enumerate(('features','training'),start=1):"),
        ("queries,_=queries_and_crops();byuid=","queries,_=queries_and_crops();os.environ['CUDA_VISIBLE_DEVICES']='0';byuid="),
        ("read(RUN/'labels/report.json')","read(ROOT/'runs/z13_region_assignment_20261025/labels/report.json')"),
        ("path=RUN/'labels/videos'/(v+'.json')","path=ROOT/'runs/z13_region_assignment_20261025/labels/videos'/(v+'.json')")]
    for old,new in replacements:
        assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
    old="ns={**vars(pub),'RUN':RUN};exec"
    new="source=source.replace(\"names=['tracking/z13_region_assignment.py'\",\"names=RECOVERY_FILES+['tracking/z13_region_assignment.py'\");ns={**vars(pub),'RUN':RUN,'RECOVERY_FILES':RECOVERY_FILES};exec"
    assert s.count(old)==1;s=s.replace(old,new)
    s=s.replace('runs/z13_region_assignment_20261025/training/report.json','runs/z13_region_assignment_20261025/recovery2/training/report.json')
    s=s.replace('runs/z13_region_assignment_20261025/independent_audit.json','runs/z13_region_assignment_20261025/recovery2/independent_audit.json')
    s=s.replace('runs/z13_region_assignment_20261025/publication.json','runs/z13_region_assignment_20261025/recovery2/publication.json')
    s=s.replace("'runs/z13_region_assignment_20261025/'+", "'runs/z13_region_assignment_20261025/recovery2/'+")
    return s


def namespace():
    ns={'__name__':'region_assignment_service_recovery','__file__':str(Path(__file__))}
    exec(compile(source(),str(Path(__file__)),'exec'),ns)
    default=ns['phase_protocol']
    def phase_protocol(phase):
        if phase!='features':return default(phase)
        p=read(RUN/'protocol.json');old=read(BASE/'features/protocol.json')
        q=dict(p,device='cuda',budget_seconds=old['budget_seconds'],save_at=old['save_at'],hard_stop=old['hard_stop'],stop_request_path=(RUN/'features/stop.request').relative_to(ROOT).as_posix())
        write(RUN/'features/protocol.json',q);write(RUN/'features/status.json',dict(state='ready',stage='Восстановление CUDA после CPU-импорта',updated=now().isoformat()));return q
    ns['phase_protocol']=phase_protocol
    extra=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_assignment_recovery2_console.ps1','scripts/z13_region_assignment_device_check_20261025.py']
    extra += [(BASE/n).relative_to(ROOT).as_posix() for n in ('device_service_check.json','pipeline_failure.json','features/protocol.json','features/controller_result.json','features/status.json')]
    extra += [p.relative_to(ROOT).as_posix() for p in (BASE/'labels').rglob('*.json')]
    ns['RECOVERY_FILES']=extra
    return ns


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not RUN.exists();p=read(BASE/'protocol.json');verify_sources(p)
    ps=WindowsProcesses()
    for phase in ('labels','features'):
        e=read(BASE/phase/'controller_result.json');assert e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    assert read(BASE/'labels/controller_result.json')['exit_code']==0 and read(BASE/'labels/report.json')['complete']
    assert read(BASE/'features/controller_result.json')['exit_code']==1
    assert not list((BASE/'features').glob('*/**/chunk_*.json'))
    assert read(BASE/'device_service_check.json')['passed']
    code=source();compile(code,str(Path(__file__)),'exec')
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_assignment_recovery2_console.ps1','scripts/z13_region_assignment_device_check_20261025.py',
        'runs/z13_region_assignment_20261025/device_service_check.json','runs/z13_region_assignment_20261025/features/protocol.json','runs/z13_region_assignment_20261025/features/controller_result.json','runs/z13_region_assignment_20261025/labels/report.json']
    recovered=dict(p,service_recovery=dict(at=now().isoformat(),cause='queries_and_crops imports z13_assignment_effect, whose CPU-only module masks CUDA with an empty environment string. It happened after CUDA phase setup but before network loading.',fix='Restore device0 only after CPU metadata helper returns, before loading any network. Tiny CUDA tensor test passed in a fresh process. Labels unchanged and all130 reused; no completed GPU chunks.',feature_save_at=read(BASE/'features/protocol.json')['save_at'],feature_hard_stop=read(BASE/'features/protocol.json')['hard_stop'],global_save_at=p['pipeline_save_at'],global_hard_stop=p['pipeline_hard_stop']),executed_service_sha256=hashlib.sha256(code.encode()).hexdigest(),sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',recovered)
    for name in ('report.json','controller_result.json','verified.json'):
        src=BASE/'labels'/name;dst=RUN/'labels'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(src)==sha(dst)
    write(RUN/'labels/reused.json',dict(source=(BASE/'labels').relative_to(ROOT).as_posix(),report_sha256=sha(BASE/'labels/report.json'),controller_result_sha256=sha(BASE/'labels/controller_result.json'),new_label_computations=0))
    write(RUN/'status.json',dict(state='ready',phase='features',stage='Все130меток сохранены; восстановление устройства',updated=now().isoformat()))
    ns=namespace()
    ns['journal']('исправление видимости GPU после CPU-импорта','*Сбой.* В фазе features импорт производителя CPU-меток сбросил CUDA_VISIBLE_DEVICES, после чего штатный загрузчик сети получил device_count0. Старые exit1/журналы/источники сохранены. Все37296строк130видео уже сохранены; их не повторяем.\n\n*Проверено.* device_service_check.json воспроизводит переход0 -> пусто при импорте и успешный тензор cuda0 после восстановления0 в новом процессе.\n\n*Исправление.* Только восстановление видимости перед загрузкой замороженной сети. Научные признаки, веса, метки, критерии, паритет и бюджеты прежние. Исходные save/hard GPU и global сохранены в recovery2/protocol.json. Завершённых GPU-блоков0. Все исходные130меток используются через ссылки на прежние первичные файлы.')


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        ns=namespace();p=read(RUN/'protocol.json');assert hashlib.sha256(source().encode()).hexdigest()==p['executed_service_sha256']
        if '--pipeline' in sys.argv:
            try:ns['pipeline']()
            except Exception as exc:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(exc)));ns['handoff']('Служебное recovery2 регионального узла остановилось: '+repr(exc)+'. Проверить новые логи и первоначальный остаток времени; готовые130меток и GPU-блоки сохранять, не повторять.');raise
        elif '--worker' in sys.argv:
            phase=os.environ['Z13_REGION_PHASE']
            try:ns[{'features':'feature_worker','training':'train_worker'}[phase]]();write(RUN/phase/'status.json',dict(state='completed',stage='Этап сохранён',updated=now().isoformat()))
            except Exception as exc:write(RUN/phase/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
        else:raise RuntimeError('Explicit mode required')
