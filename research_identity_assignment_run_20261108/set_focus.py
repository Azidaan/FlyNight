"""Record the author's sequential priority without altering the running experiment."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main():
    focus=dict(registered=datetime.now(MSK).isoformat(),authorization='Не берись сразу за все4. Лучше какую-то одну проблему решить до конца и заниматься потом другими.',
        active_problem=1,active_question='Отказ при одной геометрически допустимой рамке: устранить необоснованный отказ без увеличения чужих назначений.',
        deferred_problems=[2,3,4],full_objective_preserved=True,
        running_pilot='Finish the already-running frozen full comparison; do not add arms, retune or change its criteria.',
        next_work='Analyze sole-candidate refusals and the full no-FlyVis geometry control first. New interventions must address problem1 only.',
        scope_boundary='Do not force a candidate already owned by another track or relabel history as a shortcut; ownership conflict remains problem2.',
        completion='A working, verified sole-candidate rule; zero-intervention parity; matched no-FlyVis and threshold/cost controls; full-tracker benefit and wrong-assignment safeguards, or continue replacing the failed rule. No completion from a one-frame gain alone.',
        deadline=read(RUN/'author_window_20261108.json')['deadline'])
    write(RUN/'focus_order_20261108.json',focus,new=True)
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Автор уточнил порядок: сначала довести до конца одну проблему, затем переходить к остальным. Текущий фокус — отказ при единственной геометрически допустимой рамке; проблемы2–4 отложены. Уже запущенный полный пилот завершается с неизменными настройками и контролями; новых ветвей для остальных проблем не добавлять. Общая цель четырёх исправлений сохранена. Выходной критерий первой проблемы — проверенная правка с контролем безFlyVis и ограничением чужих назначений, не один условный положительный исход. Запись `runs/identity_assignment_20261108/focus_order_20261108.json`. Окно до14:00МСК08.11.2026 сохранено.\n\n"
    write(RUN/'focus_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')


if __name__=='__main__':main()
