"""Complete only already registered conditional stages; no model polling or new experiment choice."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_temporal_identity_trial_20261025 as S
from scripts.z13_temporal_identity_trial_recovery1_20261025 import retry_write
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    out=S.BASE/'continuation2.json';assert not out.exists();ps=WindowsProcesses()
    retry_write(out,dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=S.now().isoformat(),script=Path(__file__).relative_to(ROOT).as_posix()))
    service=S.RUN/'recovery2';p=S.read(service/'protocol.json')
    try:
        while not (service/'controller_result.json').exists():
            if S.now()>datetime.fromisoformat(p['hard_stop'])+timedelta(seconds=30):raise TimeoutError('Vectorized controller missing past original deadline')
            time.sleep(2)
        c=S.read(service/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining']
        assert all(ps.creation(int(pid))!=birth for pid,birth in c['owned'].items())
        r=S.read(S.RUN/'report.json')
        if r['internal_pass']:
            result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_temporal_identity_final_20261025.py','--prepare'],cwd=ROOT,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
            assert result.returncode==0,result.stderr.decode('utf8',errors='replace')
            final=S.RUN/'final_execution'
            with (final/'controller.stdout.log').open('w',encoding='utf8') as log,(final/'controller.stderr.log').open('w',encoding='utf8') as err:
                child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(final),'--script',str(ROOT/'scripts/z13_temporal_identity_final_20261025.py')],cwd=ROOT,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
                retry_write(final/'launch.json',dict(pid=child.pid,birth=ps.creation(child.pid),at=S.now().isoformat()))
                assert child.wait()==0,'Final phase failed'
        with (S.BASE/'record_result.stdout.log').open('w',encoding='utf8') as log:
            result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_record_temporal_identity_result_20261025.py','--record'],cwd=ROOT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        assert result.returncode==0,'Independent result audit/recording failed;models preserved'
        retry_write(S.BASE/'continuation2_result.json',dict(at=S.now().isoformat(),state='recorded',recorded=True,internal_pass=S.read(S.RUN/'report.json')['internal_pass'],final_evaluated=S.read(S.RUN/'report.json')['final_evaluated']))
    except Exception as e:
        retry_write(S.BASE/'continuation2_failure.json',dict(at=S.now().isoformat(),error=repr(e)));raise


if __name__=='__main__':main()
