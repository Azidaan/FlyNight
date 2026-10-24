"""Local bounded input->raw comparison queue; waiting uses no model calls."""
from pathlib import Path
from datetime import datetime
import os,sys,time,subprocess,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_val20_inputs_20261024 as I
from scripts import z13_val20_raw_20261024 as R
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_weekend_driver_20261024 import execute,verify_and_graph
W=ROOT/'runs/z13_weekend_20261024';PY=ROOT/'.venv/Scripts/python.exe'


def ended(stage):
    ps=WindowsProcesses()
    while not (stage/'controller_result.json').exists():
        assert now()<datetime.fromisoformat(read(W/'authorization.json')['hard_stop']);time.sleep(3)
    c=read(stage/'controller.json');e=read(stage/'controller_result.json')
    while ps.creation(c['pid'])==c['birth']:time.sleep(.5)
    assert e['worker_ended'] and e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    return e


def update(stage,ident,ok,obtained,interpretation,decision):
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==ident);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    fields=dict(summary=interpretation,obtained=obtained,interpretation=interpretation,decision=decision)
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=key,old=card[key],reason='Проверенное окончание локального этапа20val.'))
        card[key]=value
    card.update(state=('infrastructure' if ident==I.IDENT else 'completed_positive') if ok else 'inconclusive',done=f'{now():%Y-%m-%d}')
    for a in card['acceptance']:a.update(met=True if ok else False,evidence=obtained)
    card['evidence'].append(dict(kind='file',ref=(stage/'controller_result.json').relative_to(ROOT).as_posix(),claim='Код завершения и конец своего дерева'))
    write(cp,cat);I.journal('итог '+ident,'*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision)
    write(stage/'finish_recorded.json',dict(at=now().isoformat(),complete=ok,report_sha256=sha(stage/'report.json') if (stage/'report.json').exists() else None))


def record_inputs(e):
    stage=I.STAGE;count=len(list((stage/'jobs').glob('*.json')));ok=e['exit_code']==0 and (stage/'report.json').exists()
    if ok:
        from scripts.z13_localization_labels_20261023 import verify_sources
        p=read(stage/'protocol.json');verify_sources(p);r=read(stage/'report.json');assert r['jobs']==21 and count==21 and r['videos']==p['videos'] and r['detector_replay_exact']
        for path in (stage/'jobs').glob('*.json'):
            for n,h in read(path)['outputs'].items():assert sha(ROOT/n)==h
    update(stage,I.IDENT,ok,f"runs/z13_val20_matched_20261024/inputs/controller_result.json; jobs={count}/21; report.json; price.json",'Входы20val полностью подготовлены и проверены;HOTAещё нет.' if ok else 'Подготовка20val неполна;сохранён только остаток,научного исхода нет.','Автоматически перейти к заранее зарегистрированным4исходным трекерам.' if ok else 'Следующий этап не запускать;сохранить готовые входы и ошибку без повторения готового.');return ok


def record_raw(e):
    stage=R.STAGE;ok=e['exit_code']==0 and (stage/'report.json').exists();count=len(list((stage/'jobs').glob('*.json')))
    if ok:
        from scripts.z13_localization_labels_20261023 import verify_sources
        from scripts.z13_continue import paired
        from evaluation.hota_eval import evaluate_hota
        p=read(stage/'protocol.json');verify_sources(p);r=read(stage/'report.json');assert count==84 and len(r['videos'])==20
        for path in (stage/'jobs').glob('*.json'):
            for n,h in read(path)['outputs'].items():assert sha(ROOT/n)==h
        for arm in R.M.ARMS:
            assert evaluate_hota(stage/arm/'raw',p['videos'],split='val')==r['pooled'][arm]
        pv={a:{v:x['HOTA'] for v,x in rows.items()} for a,rows in r['per_video'].items()}
        for key,val in r['paired'].items():a,b=key.split('-');assert paired(pv[a],pv[b])==val
    obtained=f"runs/z13_val20_matched_20261024/raw/controller_result.json; jobs={count}/84; report.json -> pooled/per_video/paired"+(('; '+json.dumps(read(stage/'report.json')['pooled'],ensure_ascii=False)) if ok else '')
    update(stage,R.IDENT,ok,obtained,'Полное исходное сравнение4трекеров на20val. completed_positive означает готовое сравнение,а не выигрыш FlyVis или какого-либо трекера. Значимость каждого контраста определяется его отдельным парнымДИ;MASA/Dещё не измерены.' if ok else 'Исходное сравнение20val неполно;готовые задания сохранены,эффективность не оценивается.','Далее отдельная стадия общей MASA/принятых FlyVis с собственным протоколом и ценой;не смешивать raw20 с MASA7.' if ok else 'Не продолжать дорогую ступень до устранения проверенной причины в допустимом бюджете.');return ok


def launch():
    stage=R.STAGE;assert not (stage/'controller.json').exists();ps=WindowsProcesses()
    with (stage/'launcher.stdout.log').open('w',encoding='utf-8') as out,(stage/'launcher.stderr.log').open('w',encoding='utf-8') as err:
        p=subprocess.Popen([str(PY),'-u',str(ROOT/'scripts/z13_bounded_recovery_controller_20261022.py'),'--run',str(stage),'--script',str(Path(R.__file__))],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
    v=subprocess.Popen(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/z13_val20_raw_console.ps1'),'-Run',str(stage)],cwd=ROOT,creationflags=subprocess.CREATE_NEW_CONSOLE)
    write(stage/'launch.json',dict(at=now().isoformat(),launcher_pid=p.pid,launcher_birth=ps.creation(p.pid),console_pid=v.pid,console_birth=ps.creation(v.pid)))


def main():
    import msvcrt
    stage=I.STAGE;lock=(stage/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ps=WindowsProcesses();write(stage/'queue.json',dict(at=now().isoformat(),pid=os.getpid(),birth=ps.creation(os.getpid()),sha256=sha(Path(__file__))))
    if not record_inputs(ended(stage)):verify_and_graph('val20_inputs_partial');return
    verify_and_graph('val20_inputs');execute('prepare_val20_raw',['scripts/z13_val20_raw_20261024.py','--prepare']);launch()
    record_raw(ended(R.STAGE));verify_and_graph('val20_raw');write(W/'val20_chain_result.json',dict(at=now().isoformat(),ended=True,publication_pending=True))


if __name__=='__main__':
    try:main()
    except Exception as e:write(W/'val20_chain_failure.json',dict(at=now().isoformat(),error=repr(e)));raise
