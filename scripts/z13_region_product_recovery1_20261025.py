"""Service recovery: isolate worker globals from the frozen cache loader namespace."""
from pathlib import Path
import sys
import inspect
import hashlib
import subprocess
import os
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_product_20261025 as science
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
BASE=science.RUN
RUN=BASE/'recovery1'


def worker_source():
    code=inspect.getsource(science.worker);old='    ns.update(RUN=RUN,'
    assert code.count(old)==1
    return code.replace(old,'    ns=dict(ns)\n    ns.update(RUN=RUN,')


def worker():
    ns=dict(vars(science));exec(compile(worker_source(),str(Path(__file__)),'exec'),ns);ns['worker']()


def publish(stage):
    code=inspect.getsource(science.publish);old='    names=sorted(set(names));'
    assert code.count(old)==1
    code=code.replace(old,'    names += SERVICE_FILES\n'+old)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_product_recovery1_console.ps1','scripts/z13_publish_region_product_start_20261025.py']
    names += [p.relative_to(ROOT).as_posix() for p in RUN.glob('*.json')]
    names += [(BASE/n).relative_to(ROOT).as_posix() for n in ('pipeline_failure.json','training/controller_result.json')]
    ns=dict(vars(science));ns['SERVICE_FILES']=names
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['publish'](stage)


def finish():
    def service_read(path):
        if Path(path)==BASE/'training/controller_result.json':path=RUN/'controller_result.json'
        return read(path)
    ns=dict(vars(science));ns.update(read=service_read,publish=publish)
    exec(compile(inspect.getsource(science.finish),str(Path(__file__)),'exec'),ns);ns['finish']()


def prepare():
    assert not RUN.exists()
    old=read(BASE/'training/controller_result.json');assert old['exit_code']==1 and old['all_owned_ended'] and not old['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=birth for pid,birth in old['owned'].items())
    assert 'in load_split' in (BASE/'training/worker.stderr.log').read_text(encoding='utf-8')
    assert not list((BASE/'training').glob('*/seed*/done.json')) and not list((BASE/'training').glob('*/seed*/checkpoint.pt'))
    p=read(BASE/'protocol.json');science.verify_sources(p)
    # Actual frozen-data loader preflight, not just synthetic tensor dimensions.
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    ns,loader=science.build_loader(torch)
    clone=dict(ns);clone.update(RUN=BASE)
    d=loader(torch,'D','fit84','product')
    assert len(d['uid'])==24787 and int((d['kind']>=0).sum())==17859 and d['geo'].shape[-1]==82
    model=science.make_product_ranker(torch).cpu();mask=d['mask'][:2]
    z=model(d['x'][:2],d['seed'][:2],d['geo'][:2],mask)
    loss=science.context.signed_action_loss(torch,z,d['good'][:2],d['harm'][:2],mask);loss.backward()
    assert bool(loss.isfinite()) and all(bool(v.grad.isfinite().all()) for v in model.parameters() if v.grad is not None) and not torch.cuda.is_initialized()
    from datetime import datetime
    remaining=(datetime.fromisoformat(p['save_at'])-now()).total_seconds();assert remaining>600
    write(RUN/'service_check.json',dict(at=now().isoformat(),cause='Mutating ns.update(RUN=new_run) also changed load_split.__globals__; frozen cache lookup became empty. Copy the execution namespace before update.',
        actual_fit84_rows=24787,known=17859,actual_forward_backward_finite=True,scientific_training_sha256=p['executed_training_sha256'],scientific_training_unchanged=True,
        completed_models=0,remaining_original_seconds=remaining,original_save_at=p['save_at'],original_hard_stop=p['hard_stop'],patched_worker_sha256=hashlib.sha256(worker_source().encode()).hexdigest()))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_product_recovery1_console.ps1',(RUN/'service_check.json').relative_to(ROOT).as_posix()]
    write(RUN/'protocol.json',dict(p,service_only=True,sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}}))
    write(RUN/'status.json',dict(state='ready',stage='Namespace-isolated cache loader; original budget',updated=now().isoformat()))
    science.journal('служебное исправление загрузчика прямого взаимодействия',
        '*Ошибка.* Исходный worker завершился до первой модели: изменение RUN в общем пространстве имён перенаправило load_split с готового кэша на пустой каталог. '
        'Старые pipeline_failure.json, training/controller_result.json exit1 и журналы сохранены. '
        '*Исправление.* Отдельный scripts/z13_region_product_recovery1_20261025.py копирует исполнительное пространство имён перед заменой RUN; '
        'источники научного опыта и SHA функции обучения неизменны. На настоящих24787строках fit84 (17859известных) пройдены загрузка82входных геометрических/прямых признаков, прямой/обратный проход162входной сетиCPU. '
        '*Бюджет.* Прежние save '+p['save_at']+' / hard '+p['hard_stop']+'; завершённых моделей0, повторения нет. '
        'Источник runs/z13_region_product_20261025/recovery1/service_check.json. Искусственная предыдущая проба не проверяла фактический адрес кэша; этот пробел исправлен реальной пробой.')


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        with (RUN/'controller.stdout.log').open('w',encoding='utf-8') as out,(RUN/'controller.stderr.log').open('w',encoding='utf-8') as err:
            child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(RUN),'--script',str(Path(__file__))],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
            while child.poll() is None:
                status=BASE/'training/status.json'
                if status.exists():
                    try:write(RUN/'status.json',read(status))
                    except (OSError,ValueError):pass
                time.sleep(2)
            child.wait()
        finish()
    except Exception as error:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(error)));raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    elif '--publish-start' in sys.argv:publish('start')
    else:raise RuntimeError('Explicit mode required')
