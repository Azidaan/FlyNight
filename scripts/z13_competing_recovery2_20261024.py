"""Verified CPU visibility repair, preserving original scientific code/deadline."""
from pathlib import Path
from datetime import datetime
import os,sys,time,faulthandler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_competing_candidates_20261024 as S
OLD=S.RUN;PREVIOUS=OLD/'recovery1';REC=OLD/'recovery2'


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    assert not REC.exists();p=S.read(PREVIOUS/'protocol.json');S.P.verify_sources(p);e=S.read(PREVIOUS/'controller_result.json');c=S.read(PREVIOUS/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']==3221225477 and e['all_owned_ended'] and not e['remaining']
    assert ps.creation(c['pid'])!=c['birth'] and all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    diagnostic=OLD/'cpu_visibility_diagnostic.json';proof=S.read(diagnostic);assert proof['rows']['negative']['exit_code']!=0 and proof['rows']['empty']['exit_code']==0
    remaining=int((datetime.fromisoformat(c['save_at'])-S.now()).total_seconds());assert remaining>300
    names=['scripts/z13_competing_recovery2_20261024.py','scripts/z13_cpu_visibility_diagnostic_20261024.py',diagnostic.relative_to(ROOT).as_posix()]
    p={**p,'budget_seconds':remaining,'save_at':c['save_at'],'hard_stop':c['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**p['sha256'],**{n:S.sha(ROOT/n) for n in names}},'service_recovery2':dict(previous_result_sha256=S.sha(PREVIOUS/'controller_result.json'),previous_protocol_sha256=S.sha(PREVIOUS/'protocol.json'),diagnostic_sha256=S.sha(diagnostic),reason='Confirmed minimum CPU scalar autograd fails with CUDA_VISIBLE_DEVICES=-1 but succeeds with empty visibility, gradient4 at x2, CUDA uninitialized. Set empty visibility before torch import only; original two threads/arithmetic/data/losses unchanged. Re-run original controls and32loss guard; no new full models were completed. Original1800second budget still ends04:38:48/04:39:08MSK.')}
    S.write(REC/'protocol.json',p);S.write(REC/'status.json',dict(state='ready',stage='Проверенное CPUвосстановление видимостиCUDA',updated=S.now().isoformat()))
    S.journal('подтверждена причина CPUсбоя и исправлена видимостьCUDA','Системныйсбой повторён в исходномBCEbackward ещё до новойошибки;одинпоток/копирование/MKLDNNнепомогли. Минимальная проверка безмодели/данных: градиентx*xприCPUx=2 падает с CUDA_VISIBLE_DEVICES=-1;пустаявидимость и0даютградиент4приcuda_initialized=false. Проверка cpu_visibility_diagnostic.json сохранена. Новаяслужебнаяверсия recovery2 заменяеттолько-1напустуюстроку доимпортаTorch,явноCPUвсеобъекты. Исходные2потока/код/математика/данные/допуски прежние;старыеошибки сохранены. Прежний32шаговыйконтроль обязателен. Исходный1800сбюджет от04:08:48непродлён,конец04:39:08МСК.')


def bind():
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='0'
    import torch
    torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0;torch.set_default_device('cpu');torch.set_num_threads(2)
    assert not torch.cuda.is_initialized() and torch.empty(1).device.type=='cpu'
    S.RUN=REC;S.T0=time.monotonic();faulthandler.enable(all_threads=True)


def monitor():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_weekend_driver_20261024 import ready,execute,verify_and_graph
    import msvcrt
    lock=(REC/'completion.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ps=WindowsProcesses();S.write(REC/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=S.now().isoformat(),source_sha256=S.sha(Path(__file__))))
    while not (REC/'controller_result.json').exists():
        if S.now()>=datetime.fromisoformat(S.read(REC/'protocol.json')['hard_stop']):raise TimeoutError('Original CPU deadline')
        time.sleep(5)
    while not ready(REC):time.sleep(1)
    if not (REC/'finish_recorded.json').exists():execute('record_competing_recovery2',['scripts/z13_competing_recovery2_20261024.py','--record'])
    verify_and_graph('competing_recovery2');S.write(ROOT/'runs/z13_weekend_20261024/competing_recovery2.json',dict(at=S.now().isoformat(),complete=True))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        bind()
        try:S.worker()
        except Exception as e:S.write(REC/'status.json',dict(state='failed',stage=repr(e),updated=S.now().isoformat()));raise
    elif '--record' in sys.argv:
        bind()
        from scripts import z13_record_competing_candidates_20261024 as F
        F.S.RUN=REC;F.main()
    elif '--monitor' in sys.argv:
        try:monitor()
        except Exception as e:S.write(ROOT/'runs/z13_weekend_20261024/competing_recovery2_failure.json',dict(at=S.now().isoformat(),error=repr(e)));raise
    else:raise RuntimeError('Explicit recovery mode required')
