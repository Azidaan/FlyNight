"""One fresh CUDA interpreter after verified stream stall; same jobs and deadline."""
from pathlib import Path
from datetime import datetime
import sys,time,inspect,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_matched_pipeline_recovery2_20261024 as R
from scripts.z13_matched_trackers_20261024 import RUN,read,write,sha,now,journal
OLD=RUN/'gpu_recovery2';REC=RUN/'gpu_recovery3';P=R.P


def prepare():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    assert not REC.exists();old=read(OLD/'protocol.json');verify_sources(old)
    e=read(OLD/'controller_result.json');c=read(OLD/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']!=0 and e['all_owned_ended'] and not e['remaining']
    assert ps.creation(c['pid'])!=c['birth'] and all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    jobs=list((OLD/'jobs').glob('*.json'));assert len(jobs)==43
    for path in jobs:
        for n,h in read(path)['outputs'].items():assert sha(ROOT/n)==h,n
    remaining=int((datetime.fromisoformat(c['save_at'])-now()).total_seconds());assert remaining>900
    p={**old,'budget_seconds':remaining,'save_at':c['save_at'],'hard_stop':c['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**old['sha256'],'scripts/z13_matched_pipeline_recovery3_20261024.py':sha(Path(__file__))},'service_recovery3':dict(previous='gpu_recovery2',previous_result_sha256=sha(OLD/'controller_result.json'),previous_protocol_sha256=sha(OLD/'protocol.json'),verified_jobs=43,stack_sha256=sha(OLD/'cuda_stall_stack.txt'),reason='Repeated samples show same render scalar check waiting in CUDA stream synchronization while frame counter unchanged for >8min. Only own identified stalled process was ended. One fresh interpreter retry, same science and absolute deadline. 600s per-child lack-of-progress guard stops own tree; no endless retry.')}
    write(REC/'protocol.json',p)
    for path in jobs:write(REC/'jobs'/path.name,read(path))
    write(REC/'status.json',dict(state='ready',stage='43готовых задания проверены; новый CUDA-процесс',updated=now().isoformat()))
    journal('восстановление после ожидания CUDA на последнем видео','На centerD/b251064f-e7a165fd счётчик оставался25/1205 с03:03:41. Повторные снимки py-spy показали одну точку hex_render,нативный стек cuStreamSynchronize/memcpy_and_sync;CPUвремя росло без кадров. Стек сохранён в gpu_recovery2/cuda_stall_stack.txt. Собственный PID8944/birth134372737753743918 проверен и завершён,прежний ненулевой код сохранён. В gpu_recovery3 проверены43готовых задания по хешам;повторяется только незавершённое задание в новом интерпретаторе. Научные настройки/данные/веса/срок05:53:48МСК неизменны. Добавлен только служебный контроль отсутствия продвижения600с,завершающий своё дерево;бесконечныеповторы запрещены. CPUкэш3720блоков завершён сexit0 и parity_pass;отдельное обучение36моделей автоматически запущено.')


def bind():
    R.REC=REC;R.__file__=str(Path(__file__));R.bind()
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    source=inspect.getsource(P.child)
    old='while p.poll() is None:check();time.sleep(.5)'
    assert source.count(old)==1
    new='''birth=_ps.creation(p.pid)
        while p.poll() is None:
            check()
            stamp=read(STAGE/'status.json')['updated']
            if (now()-datetime.fromisoformat(stamp)).total_seconds()>600:
                if _ps.creation(p.pid)==birth:
                    subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,check=False,creationflags=subprocess.CREATE_NO_WINDOW)
                p.wait(timeout=20)
                raise TimeoutError('600seconds without frame/stage progress; own child ended, no automatic repeat')
            time.sleep(.5)'''
    ns=vars(P).copy();ns['_ps']=WindowsProcesses();exec(compile(source.replace(old,new),str(Path(__file__)),'exec'),ns);P.child=ns['child']


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        from scripts.z13_localization_labels_20261023 import verify_sources
        verify_sources(read(REC/'protocol.json'));bind()
        try:
            for video in P.VIDEOS:
                for arm in P.SYSTEMS:P.status('Продолжение после CUDA-задержки',variant=arm+' '+video);P.child('signature',arm,video)
                P.child('masa','all',video)
            P.evaluate();write(REC/'status.json',dict(state='completed',stage='Все6систем и7видео готовы',updated=now().isoformat()))
        except Exception as e:write(REC/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    elif '--job' in sys.argv:
        bind();mode,arm,video=sys.argv[2:5];assert mode in ('signature','masa') and video in P.VIDEOS and arm in P.SYSTEMS+('all',)
        result=P.build_signature(arm,video) if mode=='signature' else P.build_masa(video)
        paths=[RUN/arm/'signatures'/video/'signatures.npz'] if mode=='signature' else [RUN/a/'masa'/(video+'.npz') for a in P.SYSTEMS]
        write(REC/'jobs'/f'{mode}_{arm}_{video}.json',dict(result=result,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in paths},at=now().isoformat()))
    else:raise RuntimeError('Explicit recovery mode required')
