"""One diagnosed CPU-native retry, preserving original budget and frozen math."""
from pathlib import Path
from datetime import datetime
import os,sys,time,inspect,faulthandler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_competing_candidates_20261024 as S
OLD=S.RUN;REC=OLD/'recovery1'


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    assert not REC.exists();p=S.read(OLD/'protocol.json');S.P.verify_sources(p);e=S.read(OLD/'controller_result.json');c=S.read(OLD/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']==3221225477 and e['all_owned_ended'] and not e['remaining']
    assert ps.creation(c['pid'])!=c['birth'] and all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    assert not list(OLD.rglob('model.pt')) and not (OLD/'timing.json').exists()
    remaining=int((datetime.fromisoformat(c['save_at'])-S.now()).total_seconds());assert remaining>600
    names=['scripts/z13_competing_recovery1_20261024.py','scripts/z13_record_competing_candidates_20261024.py','tests/test_z13_competing_execution.py']
    p={**p,'budget_seconds':remaining,'save_at':c['save_at'],'hard_stop':c['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**p['sha256'],**{n:S.sha(ROOT/n) for n in names}},'service_recovery':dict(previous_result_sha256=S.sha(OLD/'controller_result.json'),previous_protocol_sha256=S.sha(OLD/'protocol.json'),reason='Native Windows access violation0xC0000005 after all18controls reproduced exactly, before timing/model outputs. Cause unproven; synthetic actual-ranker64step gradient test passes. One technical retry with one CPU library thread, faulthandler and before/after32step stage markers. Scientific arithmetic/data/optimizer unchanged, full replay/32loss tolerances retained. No completed training exists to repeat.',original_started=c['started'],original_budget_seconds=1800)}
    S.write(REC/'protocol.json',p);S.write(REC/'status.json',dict(state='ready',stage='Один диагностический повтор CPU, один поток',updated=S.now().isoformat()))
    S.journal('диагностический повтор после системного сбоя CPU','Новая функцияошибки: все6ансамблей/18старыхмоделей точно воспроизведены,но затем процесс завершён0xC0000005 безPythonисключения и безготовыхновыхмоделей. Стекнесохранён;причина неустановлена. Тест64шагов фактическоговыборщика на смешанныхнулевых/множественных/маскированныхкандидатах пройден. Единственныйдиагностическийповтор: recovery1,одинпотокCPUбиблиотек,faulthandler,маркерыдо/после32шагов. Старыеисточники/коды/контроли сохранены;исходный конец04:39:08МСК и1800с от04:08:48непродлены,допускинеослаблены. Это служебнаягипотезапроверки,необъяснениеотрицательныхнаучныхрезультатов.')


def single_cpu():
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    import torch
    original=torch.set_num_threads;original(1);torch.set_num_threads=lambda n:original(1)
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)


def worker():
    faulthandler.enable(all_threads=True);S.RUN=REC;S.T0=time.monotonic()
    def mark(stage):S.write(REC/'native_stage.json',dict(at=S.now().isoformat(),stage=stage,threads=1))
    source=inspect.getsource(S.worker)
    a="loss,_=P.train_steps(torch,model,tensors,nref,0,ref['pos_weight'],maxsteps=32)"
    b="train(torch,model,tensors,nref,0,ref['pos_weight'],maxsteps=32)"
    assert source.count(a)==1 and source.count(b)==1
    source=source.replace(a,"_mark('before_original32');"+a+";_mark('after_original32')")
    source=source.replace(b,"_mark('before_joint32');"+b+";_mark('after_joint32')")
    ns=vars(S).copy();ns['_mark']=mark
    with single_cpu():exec(compile(source,str(Path(__file__)),'exec'),ns);ns['worker']()


def record():
    from scripts import z13_record_competing_candidates_20261024 as F
    F.S.RUN=REC
    with single_cpu():F.main()


def monitor():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_weekend_driver_20261024 import ready,execute,verify_and_graph
    import msvcrt
    lock=(REC/'completion.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ps=WindowsProcesses();S.write(REC/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=S.now().isoformat(),source_sha256=S.sha(Path(__file__))))
    while not (REC/'controller_result.json').exists():
        if S.now()>=datetime.fromisoformat(S.read(REC/'protocol.json')['hard_stop']):raise TimeoutError('Recovery result absent before hard deadline')
        time.sleep(5)
    while not ready(REC):time.sleep(1)
    if not (REC/'finish_recorded.json').exists():execute('record_competing_recovery1',['scripts/z13_competing_recovery1_20261024.py','--record'])
    verify_and_graph('competing_recovery1');S.write(ROOT/'runs/z13_weekend_20261024/competing_recovery1.json',dict(at=S.now().isoformat(),complete=True))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:S.write(REC/'status.json',dict(state='failed',stage=repr(e),updated=S.now().isoformat()));raise
    elif '--record' in sys.argv:record()
    elif '--monitor' in sys.argv:
        try:monitor()
        except Exception as e:S.write(ROOT/'runs/z13_weekend_20261024/competing_recovery1_failure.json',dict(at=S.now().isoformat(),error=repr(e)));raise
    else:raise RuntimeError('Explicit recovery mode required')
