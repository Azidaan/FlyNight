"""Independent records-only morning save; never launches scientific work."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import RUN,read,write,now,sha
from scripts.z13_hota_deadline_guard import WindowsProcesses
PY=ROOT/'.venv/Scripts/python.exe'


def ended(result,processes,controller):
    assert result['all_own_workers_ended'] and not result['remaining_owned_processes']
    alive={pid:born for pid,born in result['known_owned_processes'].items() if processes.creation(int(pid))==born}
    assert not alive,alive
    assert processes.creation(controller['pid'])!=controller['birth'],'Controller still alive'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    identity=read(RUN/'execution_identity.json');assert sha(RUN/'protocol.json')==identity['protocol_sha256'] and sha(RUN/'technical.json')==identity['technical_sha256']
    processes=WindowsProcesses();control=read(RUN/'full_controller.json');birth=processes.creation(control['pid']);assert birth is not None
    if args.preflight:print('Read-only preflight: execution identity and live full controller verified');return
    import msvcrt
    lock=(RUN/'morning_save.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    marker=RUN/'morning_save_result.json'
    if marker.exists():raise RuntimeError('Existing morning record must not be overwritten')
    controller=dict(pid=control['pid'],birth=birth);write(RUN/'morning_save_launch.json',dict(pid=os.getpid(),controller=controller,started=now().isoformat(),scope='Records/tests/graph/Git only; no scientific job; preserve foreign edits'))
    try:
        while True:
            result_path=RUN/'full_controller_result.json'
            if result_path.exists():
                result=read(result_path)
                if result['state']=='failed' and now()<datetime.fromisoformat('2026-10-21T07:35:00+03:00'):
                    write(RUN/'morning_save_attention.json',dict(reason='Execution failed before morning; agent must inspect/recover, no final result recorded',result=result,at=now().isoformat()));return
                if processes.creation(controller['pid'])!=birth:break
            if now()>datetime.fromisoformat('2026-10-21T07:39:55+03:00'):raise RuntimeError('No verified controller end by morning; no record or publication')
            time.sleep(2)
        ended(result,processes,controller)
        log=RUN/'morning_save.log';env={**os.environ,'GIT_OPTIONAL_LOCKS':'0','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
        def run(command):
            with log.open('a',encoding='utf-8') as out:subprocess.run([str(x) for x in command],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        if not (RUN/'finish_recorded.json').exists():run([PY,'scripts/z13_record_center_hota_20261021.py','--phase','finish'])
        run([PY,'scripts/validate_project_history.py']);run([PY,'-m','pytest','tests/test_project_history.py','-q'])
        run([PY,'scripts/z13_build_vault_preserving_edits.py','--report','runs/z13_center_hota_20261021/vault_morning_save.json'])
        ended(result,processes,controller)
        write(RUN/'morning_save_audit.json',dict(verified=now().isoformat(),completed_jobs=result['completed_jobs'],total_jobs=result['total_jobs'],remaining=len(result['remaining']),all_owned_ended=True,controller_ended=True,checks_passed=True,partial_no_hota=result['state']!='completed'))
        # Exact own paths only. No trajectories, caches, weights or execution logs.
        paths=['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','ObsidianFlyVis/09 Запланировано/z13_center_hota_v1.md']
        names=['full_controller_result.json','queue_status.json','save_checkpoint.json','night_finished.json','finish_recorded.json','morning_save_audit.json','morning_save_launch.json']
        paths.extend('runs/z13_center_hota_20261021/'+name for name in names if (RUN/name).exists())
        paths.extend(p.relative_to(ROOT).as_posix() for p in (RUN/'completed_jobs').glob('*.json'))
        paths.extend(p.relative_to(ROOT).as_posix() for arm in ('C','D','R1') for p in (RUN/arm/'signatures').glob('*/signatures.json'))
        paths.extend(p.relative_to(ROOT).as_posix() for p in (RUN/'masa_details').glob('*.json'))
        if result['state']=='completed':paths.append('runs/z13_center_hota_20261021/eval/report.json')
        journal=ROOT/'ObsidianFlyVis/06 Журнал'
        paths.extend(p.relative_to(ROOT).as_posix() for p in journal.glob('2026-10-21 *итог или сохранение очереди центрированного трекера*.md'))
        for name in paths:(ROOT/name).resolve().relative_to(ROOT.resolve())
        staged=subprocess.check_output(['git','--no-optional-locks','diff','--cached','--name-only','-z'],cwd=ROOT,env=env).decode('utf-8').split('\0')
        assert all(not name or name in paths for name in staged),'Foreign staged files; publication stopped'
        index_lock=ROOT/'.git/index.lock'
        if index_lock.exists():
            # Earlier explicit user authorization: copy/remove empty stale lock only after fresh Git process check.
            check="@(Get-CimInstance Win32_Process -Filter \"Name = 'git.exe'\").Count"
            count=subprocess.check_output(['powershell.exe','-NoProfile','-Command',check],text=True).strip();assert count=='0','Live Git; keep lock'
            stat=index_lock.stat();assert stat.st_size==0
            backup=RUN/('index_lock_'+now().strftime('%H%M%S_%f')+'.backup');backup.write_bytes(index_lock.read_bytes())
            assert backup.read_bytes()==b'' and index_lock.stat().st_mtime_ns==stat.st_mtime_ns and index_lock.stat().st_size==0
            index_lock.unlink()
        run(['git','add','-f','--',*paths]);run(['git','commit','-m','Save bounded tracker queue and verified morning remainder','-m','Co-authored-by: Codex <noreply@openai.com>'])
        run(['git','push','origin','main']);head=subprocess.check_output(['git','--no-optional-locks','rev-parse','HEAD'],cwd=ROOT,env=env,text=True).strip()
        write(marker,dict(state='saved_and_pushed',commit=head,at=now().isoformat(),remaining=len(result['remaining'])))
    except Exception as exc:
        write(RUN/'morning_save_error.json',dict(at=now().isoformat(),error=str(exc)));raise
    finally:lock.close()


if __name__=='__main__':main()
