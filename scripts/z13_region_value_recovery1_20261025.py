"""Resume saved signed-utility steps; supply missing existing namespace helpers."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import time
import hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_value_20261025 as science
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
CONTROL=science.RUN/'recovery1'


def namespace():
    ns=science.namespace();helpers=science.original.namespace()
    for name in ('load_split','predict','policy_rows','gc'):ns[name]=helpers[name]
    ns['signed_action_loss']=science.signed_action_loss
    # Only the controller receipt moves. All scientific reads/writes stay in the
    # original run, including the complete seed0/step1150 checkpoint.
    def routed_read(path):
        target=Path(path)
        if target.resolve()==(science.RUN/'controller_result.json').resolve():target=CONTROL/'controller_result.json'
        return read(target)
    ns['read']=routed_read
    return ns


def prepare():
    assert not CONTROL.exists()
    p=read(science.RUN/'protocol.json');verify_sources(p)
    e=read(science.RUN/'controller_result.json');ps=WindowsProcesses()
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    assert "name 'predict' is not defined" in (science.RUN/'worker.stderr.log').read_text(encoding='utf-8')
    import torch
    checkpoint=science.RUN/'training/D/seed0/checkpoint.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['step']==1150 and saved['protocol_sha256']==sha(science.RUN/'training/protocol.json')
    assert not list((science.RUN/'training').glob('*/seed*/done.json'))
    ns=namespace();code=science.training_source()
    assert hashlib.sha256(code.encode()).hexdigest()==p['executed_training_sha256']
    for name in ('predict','policy_rows','load_split'):assert callable(ns[name])
    names=[Path(__file__).relative_to(ROOT).as_posix(),(science.RUN/'controller_result.json').relative_to(ROOT).as_posix()]
    q=dict(p,service_recovery=dict(at=now().isoformat(),missing=['predict','policy_rows','gc'],
       source='Existing unchanged original namespace helpers',saved_step1150_reused=True,
       checkpoint_sha256=sha(checkpoint),scientific_training_sha256=hashlib.sha256(code.encode()).hexdigest(),
       all_scientific_paths_unchanged=True,only_controller_receipt_redirected=True),
       stop_request_path=(CONTROL/'stop.request').relative_to(ROOT).as_posix(),sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    assert q['save_at']==p['save_at'] and q['hard_stop']==p['hard_stop']
    write(CONTROL/'protocol.json',q)
    science.namespace()['journal']('служебное восстановление пространства функций знаковой цели',
       'Первый worker сохранил D/seed0 на1150шагах, затем завершился NameError: predict. '
       'В новом отдельном служебном модуле подключены прежние predict/policy_rows/gc/load_split; '
       'скомпилированная научная функция обучения и её хеш прежние. Контрольная точка используется без повтора1150шагов. '
       'Научные чтения/выходы и первоначальный save/hard не меняются, перенаправлена только квитанция нового контроллера. '
       'Старые ошибка, источники и журналы сохранены. Источник runs/z13_region_value_20261025/recovery1/protocol.json.')


def worker():
    p=read(CONTROL/'protocol.json');verify_sources(p)
    assert hashlib.sha256(science.training_source().encode()).hexdigest()==p['executed_training_sha256']
    ns=namespace();exec(compile(science.training_source(),str(Path(__file__)),'exec'),ns)
    try:ns['train_worker']()
    except Exception as error:
        write(science.RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    write(science.RUN/'status.json',dict(state='completed',stage='All12 signed-utility models saved after service recovery',updated=now().isoformat()))


def monitor():
    try:
        p=read(CONTROL/'protocol.json')
        while not (CONTROL/'controller_result.json').exists():
            if now()>datetime.fromisoformat(p['hard_stop'])+timedelta(seconds=30):raise TimeoutError('Recovery controller missing')
            time.sleep(2)
        namespace()['finish']()
    except Exception as error:write(CONTROL/'monitor_failure.json',dict(at=now().isoformat(),error=repr(error)));raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
    elif '--monitor' in sys.argv:monitor()
    else:raise RuntimeError('Explicit mode required')
