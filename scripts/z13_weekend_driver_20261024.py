"""Event queue without model calls: finalize stages, verify, start dependent CPU work."""
from pathlib import Path
from datetime import datetime
import os,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=ROOT/'runs/z13_weekend_20261024';BENCH=ROOT/'runs/z13_matched_trackers_20261024';FIXED=ROOT/'runs/z13_filter_descriptor_fixed_20261024';PY=ROOT/'.venv/Scripts/python.exe'


def execute(name,args,allowed_failure=False):
    log=RUN/'logs'/name;log.parent.mkdir(parents=True,exist_ok=True)
    with log.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,log.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        code=subprocess.call([str(PY),*args],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
    if code and not allowed_failure:raise RuntimeError((name,code,str(log)))
    return code


def ready(folder):
    result=folder/'controller_result.json'
    if not result.exists():return False
    e=read(result);c=read(folder/'controller.json');ps=WindowsProcesses()
    if ps.creation(c['pid'])==c['birth']:return False
    if e['exit_code']!=0 or not e['all_owned_ended'] or e['remaining']:raise RuntimeError(('Dependent stage not complete',str(folder),e['exit_code']))
    assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items());return True


def verify_and_graph(name):
    execute(name+'_catalog',['scripts/validate_project_history.py'])
    execute(name+'_tests',['-m','pytest','tests/test_project_history.py','-q'])
    execute(name+'_vault',['scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/(name+'_vault.json'))])
    # Publication is prepared as an explicit receipt, never stages the entire
    # worktree. The active agent/authorized scheduled continuation publishes it.
    write(RUN/(name+'_verified.json'),dict(at=now().isoformat(),checks_green=True,publication_pending=True))


def handle(name,action):
    done=RUN/(name+'.json');fail=RUN/(name+'_failure.json')
    if done.exists() or fail.exists():return
    try:
        if action():write(done,dict(at=now().isoformat(),complete=True))
    except Exception as e:write(fail,dict(at=now().isoformat(),error=repr(e)))


def main():
    import msvcrt
    lock=(RUN/'driver.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    auth=read(RUN/'authorization.json');deadline=datetime.fromisoformat(auth['save_at']);ps=WindowsProcesses();write(RUN/'driver.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat(),source_sha256=sha(Path(__file__))))
    def parent():
        if not ready(BENCH/'gpu_recovery1'):return False
        if not (BENCH/'finish_recorded.json').exists():execute('record_matched',['scripts/z13_record_matched_trackers_20261024.py'])
        verify_and_graph('matched');return True
    def stock():
        if not ready(BENCH/'stock_controls'):return False
        if not (BENCH/'stock_controls/finish_recorded.json').exists():execute('record_stock',['scripts/z13_record_matched_stock_20261024.py'])
        verify_and_graph('stock');return True
    def training():
        if not ready(FIXED/'cache'):return False
        assert read(FIXED/'cache/report.json')['parity_pass']
        if not (FIXED/'train/protocol.json').exists():execute('prepare_fixed_training',['scripts/z13_filter_descriptor_fixed_train_20261024.py','--prepare'])
        if (FIXED/'train/controller.json').exists():return True
        # Separate visible console for the new training stage. CPU worker and
        # independent controller are detached and remain at Idle priority.
        folder=FIXED/'train';script=ROOT/'scripts/z13_filter_descriptor_fixed_train_20261024.py'
        with (folder/'launcher.stdout.log').open('w',encoding='utf-8') as out,(folder/'launcher.stderr.log').open('w',encoding='utf-8') as err:
            proc=subprocess.Popen([str(PY),'-u',str(ROOT/'scripts/z13_bounded_recovery_controller_20261022.py'),'--run',str(folder),'--script',str(script)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
        command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/z13_filter_descriptor_fixed_train_console.ps1'),'-Run',str(folder)]
        view=subprocess.Popen(command,cwd=ROOT,creationflags=subprocess.CREATE_NEW_CONSOLE)
        write(folder/'launch.json',dict(at=now().isoformat(),launcher_pid=proc.pid,launcher_birth=ps.creation(proc.pid),console_pid=view.pid,console_birth=ps.creation(view.pid),console_alive=view.poll() is None))
        return True
    def result():
        if not ready(FIXED/'train'):return False
        if not (FIXED/'train/finish_recorded.json').exists():execute('record_fixed_training',['scripts/z13_filter_descriptor_fixed_train_20261024.py','--record'])
        verify_and_graph('fixed');return True
    while now()<deadline:
        handle('matched',parent);handle('stock',stock);handle('fixed_training_launch',training);handle('fixed_result',result)
        write(RUN/'status.json',dict(state='running',stage='Локальная очередь: расчёты,запись,проверки и зависимое обучение',updated=now().isoformat(),model_calls_for_waiting=0,ready_actions=[n for n in ('matched','stock','fixed_training_launch','fixed_result') if (RUN/(n+'.json')).exists()]))
        if all((RUN/(n+'.json')).exists() or (RUN/(n+'_failure.json')).exists() for n in ('matched','stock','fixed_training_launch','fixed_result')):break
        time.sleep(5)
    write(RUN/'driver_result.json',dict(at=now().isoformat(),deadline_reached=now()>=deadline,independent_worker_controllers_still_enforce_their_own_budgets=True))


if __name__=='__main__':main()
