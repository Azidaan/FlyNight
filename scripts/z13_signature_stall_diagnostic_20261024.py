"""Bounded instrumented replay of one unfinished signature; no arithmetic changes."""
from pathlib import Path
from datetime import datetime
import inspect
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
OLD=ROOT/'runs/z13_val20_matched_20261024/common_authorized_recovery1'
RUN=OLD.parent/'signature_stall_diagnostic'


def prepare():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_val20_common_20261024 import journal
    assert not RUN.exists();p=read(OLD/'protocol.json');verify_sources(p)
    e=read(OLD/'controller_result.json');c=read(OLD/'controller.json');m=read(OLD/'completion.json');ps=WindowsProcesses()
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    identities={**e['owned'],str(c['pid']):c['birth'],str(m['pid']):m['birth']}
    assert all(ps.creation(int(pid))!=birth for pid,birth in identities.items())
    job=read(OLD/'live_child.json');assert job['mode']=='signature' and job['arm']=='centerR1' and job['video']=='b23adb0d-72704b27'
    markers=list((OLD/'jobs').glob('*.json'));assert len(markers)==180
    for marker in markers:
        for name,digest in read(marker)['outputs'].items():assert sha(ROOT/name)==digest
    stdout=OLD/'logs'/('signature_'+job['arm']+'_'+job['video']+'.stdout.log')
    assert '950 кадров' in stdout.read_text(encoding='utf-8')
    assert not (OLD.parent/job['arm']/'signatures'/job['video']/'signatures.npz').exists()
    p={**p,'created':now().isoformat(),'budget_seconds':600,'save_at':c['save_at'],'hard_stop':c['hard_stop'],
        'stop_request_path':(RUN/'stop.request').relative_to(ROOT).as_posix(),
        'sha256':{**p['sha256'],'scripts/z13_signature_stall_diagnostic_20261024.py':sha(Path(__file__)),
                  '.venv/Scripts/py-spy.exe':sha(ROOT/'.venv/Scripts/py-spy.exe')},
        'diagnostic_job':job,'diagnostic_scope':'One unfinished centerR1 signature only. Same builder/data/math/precision/config;only destination directory differs. Native stack samples every45s,own PID+birth checked,no locals.600s bound inside unchanged original25Oct06:00deadline. No automatic retry,training,threshold change or effectiveness claim. Retain180completed jobs.',
        'old_failed_stdout_sha256':sha(stdout),'old_result_sha256':sha(OLD/'controller_result.json')}
    write(RUN/'protocol.json',p)
    write(RUN/'status.json',dict(state='ready',stage='Диагностика R1:180готовых заданий сохранены',updated=now().isoformat(),jobs_done=0,jobs_total=1))
    journal('повторная остановка требует инструментального разбора',
        'common_authorized_recovery1/controller_result.json:exit1/all_owned_ended,18:57:02МСК. '
        'На этот раз не MASA,а centerR1/b23adb0d-72704b27 перестал продвигаться после950кадров;'
        '180заданий и12полныхвидео проверены по хешам. Причина пока не установлена. '
        'Старый ночной CUDA-стек другого задания показывает ожидание cuStreamSynchronize '
        'в hex_render,но не доказывает причину нынешней остановки. '
        'signature_stall_diagnostic/protocol.json:один ограниченный600с повтор '
        'незавершённой сигнатуры со снимками Python/нативного стека каждые45с. '
        'Меняется только каталог вывода/служебное наблюдение,не данные/математика. '
        'Все прежние ошибки сохранены;нового вывода об эффективности нет.')
    print('Diagnostic prepared:one unfinished signature,180completed jobs retained.',flush=True)


def diagnostic_job():
    import numpy as np
    from scripts import z13_val20_common_authorized_20261024 as R
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(RUN/'protocol.json');verify_sources(p);ps=WindowsProcesses()
    write(RUN/'actual_job.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    S=R.S;S.STAGE=RUN;S.T0=time.monotonic();job=p['diagnostic_job']
    def status(stage,**kw):
        assert now()<datetime.fromisoformat(p['save_at']) and not (RUN/'stop.request').exists()
        write(RUN/'status.json',dict(state='running',stage='Диагностический повтор R1 без смены математики',updated=now().isoformat(),jobs_done=0,jobs_total=1,**kw))
    S.status=status
    source=inspect.getsource(R.ORIGINAL_SIGNATURE)
    expression="output=(STAGE/'probe' if probe else RUN)/arm/'signatures'"
    assert source.count(expression)==1
    source=source.replace(expression,"output=DIAGNOSTIC_OUTPUT/arm/'signatures'")
    ns={**vars(S),'DIAGNOSTIC_OUTPUT':RUN};exec(compile(source,str(Path(__file__)),'exec'),ns)
    result=ns['build_signature'](job['arm'],job['video'],False)
    path=RUN/job['arm']/'signatures'/job['video']/'signatures.npz'
    with np.load(path) as a:
        assert int(a['frame_count'])==p['frames'][job['video']] and np.isfinite(a['outputs']).all()
        rows=int(len(a['outputs']))
    write(RUN/'job_result.json',dict(at=now().isoformat(),result=result,
        outputs={path.relative_to(ROOT).as_posix():sha(path)},complete_video_frames=p['frames'][job['video']],
        output_rows=rows,passed_previous_frame950=True,output_directory_only_changed=True))
    write(RUN/'status.json',dict(state='completed',stage='Полный повтор R1 завершён;это не HOTA',updated=now().isoformat(),jobs_done=1,jobs_total=1))


def worker():
    ps=WindowsProcesses();began=time.monotonic();samples=[]
    with (RUN/'job.stdout.log').open('w',encoding='utf-8') as out,(RUN/'job.stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(Path(__file__)),'--job'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
        birth=ps.creation(child.pid);write(RUN/'job_launcher.json',dict(pid=child.pid,birth=birth))
        next_sample=45.
        try:
            while child.poll() is None:
                elapsed=time.monotonic()-began
                if elapsed>=next_sample and (RUN/'actual_job.json').exists():
                    target=read(RUN/'actual_job.json');assert ps.creation(target['pid'])==target['birth']
                    dump=subprocess.run([str(ROOT/'.venv/Scripts/py-spy.exe'),'dump','--native','--full-filenames','--pid',str(target['pid'])],cwd=ROOT,capture_output=True,timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
                    path=RUN/('stack_'+str(len(samples)+1)+'.txt');path.write_bytes(dump.stdout+dump.stderr)
                    samples.append(dict(at=now().isoformat(),elapsed_seconds=elapsed,exit_code=dump.returncode,path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
                    write(RUN/'stack_samples.json',samples);next_sample=elapsed+45.
                assert elapsed<590,'Bounded diagnostic near600s limit;no automatic retry'
                time.sleep(.5)
        except BaseException:
            if child.poll() is None and ps.creation(child.pid)==birth:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                child.wait(timeout=20)
            raise
        assert child.returncode==0,child.returncode
    report={**read(RUN/'job_result.json'),'stack_samples':samples,
        'wall_seconds_including_instrumentation':time.monotonic()-began,
        'root_cause_established':False,'not_effectiveness_result':True}
    write(RUN/'report.json',report)


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--job' in sys.argv:diagnostic_job()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit diagnostic mode required')
