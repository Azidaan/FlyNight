"""Preserve the stopped run and its completed artifacts before resume."""
from common import *
start('cpu',timeout=600)
import psutil,subprocess
p=read(OUT/'execution_protocol.json');verify(p)
old=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/af64c78618de4ea7901fd1b61d35de02'
guard=read(old/'memory_guard.json');assert guard['returncode']==1 and guard['reason'] is None
for pid,birth in guard['owned'].items():
    try:assert psutil.Process(int(pid)).create_time()!=birth,'Old owned process still live'
    except psutil.NoSuchProcess:pass
receipts={};artifacts={}
for arm in ['zero_disabled','zero_empty','G7','R134_spatial']:
    for path in (OUT/'rollouts'/arm).glob('*.receipt.json'):
        r=read(path);assert r['passed'] and r['execution_sha256']==sha(OUT/'execution_protocol.json')
        receipts[path.relative_to(ROOT).as_posix()]=sha(path)
        for n,h in r['artifacts'].items():assert sha(ROOT/n)==h;artifacts[n]=h
        for n,h in r.get('feature_files',{}).items():assert sha(ROOT/n)==h;artifacts[n]=h
assert len([n for n in receipts if '/zero_' in n])==40
files=[OUT/n for n in ['status_write_recovery.py','register_status_recovery.py','recovery1_queue.py',
    'status_retry_test.py','execution_protocol.json','deployment.json','zero_complete.json']]
files += [old/'memory_guard.json',old/'memory_guard.worker.log',OUT/'arm_queue_status.json',OUT/'evaluation_queue.err.log']
pending=OUT/'status.json.pending'
if pending.exists():
    saved=OUT/'failed_status_pending.json'
    with saved.open('xb') as f:f.write(pending.read_bytes())
    files.append(saved)
files += [OUT/n for n in ['native_run.py','native_runtime.py','common.py','register_evaluation.py','deployment_audit.py',
    'evaluate_native.py','native_statistics.py','id_fate.py']]
write('status_recovery_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
    failure='PermissionErrorWinError5 replacing status.json.pending->status.json atR134_spatial/b1e3e9f5-92377424 frame300. Scientific stage guard reportedexit1, no memory stop. Exact handle owner unknown; do not blame a specific reader.',
    correction='Wrap only common.status with bounded30x0.1sec PermissionError retry, retaining the original event timestamp and logging every retry. Scientific sources/params/inputs/decisions unchanged. Same verified memory supervisor reentered, no nested worker.',
    resume='Reuse every complete receipt and map. Resume ONLY unfinished video from its beginning; no tracker checkpoint was saved midvideo. Retained feature caches are reused. Then finish temporal/C34 and official metrics/fate. Old stopped queues/logs remain unchanged.',
    completed_receipts=receipts,verified_artifacts=artifacts,
    sources={x.relative_to(ROOT).as_posix():sha(x) for x in files},scientific_execution_sha256=sha(OUT/'execution_protocol.json'),
    save_stop=STOP.isoformat()))
journal('Техническое восстановление записи прогресса реального пилота',
    'НаR134_spatial/b1e3e9f5-92377424кадр300записьstatus.json.pendingне смогла заменитьstatus.json:PermissionErrorWinError5. Сохранён ошибочныйguardaf64c78618de4ea7901fd1b61d35de02/exit1; остановкапамятьюнебыла. Исходные научные файлы не меняются. Перед продолжением проверены все40нулевыхконтролей,20готовыхG7и завершённыеR134_spatial,ихартефакты/карты. Толькослужебнаязаписьstatusполучает30ограниченныхповторов с прежнимвременемсобытия иотдельнымжурналом. Незавершённоевидео будетпродолжено с началаприготовыхкартах, готовыевидеонеповторяются. Паспорт `runs/full_template_native_20261109/status_recovery_protocol.json`.')
print('STATUS RECOVERY REGISTERED',len(receipts),flush=True)
