"""Record the user's latest14:00MSK work window, without editing frozen sources."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main():
    import psutil
    roots=[]
    for name,script in [('manager','full_precision_manager.py'),('post','full_post.py')]:
        receipt=read(RUN/'full_precision'/(name+'_started.json'));p=psutil.Process(receipt['pid'])
        assert abs(p.create_time()-receipt['created'])<.01 and any(script in a for a in p.cmdline())
        roots.append(dict(**receipt,script='research_identity_assignment_run_20261108/'+script))
    write(RUN/'author_window_20261108.json',dict(registered=datetime.now(MSK).isoformat(),deadline='2026-11-08T14:00:00+03:00',
        user_request='Я спать хочу пойти, тебе хватит материала от Гемини для работы до14:00поМСК? Или ещё собрать?',
        interpretation='Continue authorized four-problem work; no more Gemini material required. Apply14:00MSK as current work-window boundary.',
        scientific_sources_unchanged=True,watch_source_sha256=sha(ROOT/'research_identity_assignment_run_20261108/author_deadline_watch.py')),new=True)
    write(RUN/'deadline_roots.json',dict(roots=roots),new=True)
    with (RUN/'author_deadline.log').open('x',encoding='utf-8') as out,(RUN/'author_deadline.err').open('x',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','research_identity_assignment_run_20261108/author_deadline_watch.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    write(RUN/'author_deadline_started.json',dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time()),new=True)
    journal=dict(entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Автор сообщил об уходе спать и работе до14:00МСК08.11.2026. МатериаловGemini достаточно; дополнительный сбор не требуется. Граница текущего окна записана в `runs/identity_assignment_20261108/author_window_20261108.json`. Независимый скрытый контроллер сохраняет идентичность зарегистрированных своих деревьев и завершит их до14:00; чужие процессы не затрагиваются. Замороженные научные настройки и готовые результаты не менялись.\n\n")
    write(RUN/'author_window_journal.json',journal,new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+journal['entry'],1),encoding='utf-8');print(proc.pid)


if __name__=='__main__':main()
