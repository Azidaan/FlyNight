"""Record an execution-only recovery, keeping the previous decision in revisions."""
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/z13_target_center_20261018'


def main():
    marker=RUN/'io_recovery_recorded.json'
    if marker.exists():
        raise RuntimeError('Already recorded')
    failure=json.loads((RUN/'recovery_full_result.json').read_text())
    recovery=json.loads((RUN/'io_recovery_controller.json').read_text(encoding='utf-8'))
    assert failure['exit_code']==1 and failure['worker_ended']
    now=datetime.now(timezone(timedelta(hours=3)))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    path=ROOT/'experiments/project_history.json'
    catalog=json.loads(path.read_text(encoding='utf-8'))
    card=next(c for c in catalog['experiments'] if c['id']=='z13_target_center_v1')
    card.setdefault('revisions',[]).append(dict(date=now.date().isoformat(),commit=head,field='decision',
        old=card['decision'],reason='Служебное восстановление после PermissionError при атомарной записи статуса; условия опыта неизменны.'))
    card['decision']='Продолжить только незавершённый R1 в прежнем окне до 04:05:19 МСК 18.10 с завершением до 04:05:49. D/C сохранены и не повторяются. Служебная версия повторяет временно заблокированную замену JSON не более 5 секунд; замороженные научные источники неизменны. Текущий контроллер: io_recovery_controller.json, окончание: io_recovery_result.json в runs/z13_target_center_20261018. Предыдущая ошибка сохранена.'
    card['evidence'].append(dict(kind='json',ref='runs/z13_target_center_20261018/io_recovery_controller.json',claim='Оставшийся бюджет и хеши сохранённых D/C'))
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf-8'); anchor='## Журнал изменений\n';assert anchor in plan
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — служебное восстановление `z13_target_center_v1`\n\n'
    entry+='*Получено.* Предыдущий процесс завершился с кодом 1 после WinError 5 при замене status.json.tmp → status.json (runs/z13_target_center_20261018/recovery_full.stderr.log и recovery_full_result.json). Полные D/C сохранены; оба reproduction.json прошли исходные допуски. Сбой обнаружен при просьбе автора открыть консоль; автоматизация ранее отключена по его требованию. Служебный запуск первоначально отклонён из-за семизначной доли секунды в ISO-дате Python 3.10; ошибка сохранена в io_launcher.stderr.log, разбор даты исправлен до запуска вычислений.\n\n'
    entry+='*Решение.* scripts/z13_target_center_io_recovery.py заменяет только запись JSON на ограниченные повторы атомарной замены. Проверка с тремя искусственными отказами прошла (io_recovery_write_test.json). Консоль читает с разрешением совместной замены файла; конкретный процесс, державший файл при первоначальном отказе, не установлен. Возобновлён только R1; хеши D/C записаны в io_recovery_controller.json, срок прежний. Научные настройки и 61 исходный хеш сохранены; GPU, обучение ядра, dev, HOTA и отложенные видео не запускались. Консоль открыта заново, расписание чата не включалось.\n'
    path.write_text(plan.replace(anchor,anchor+entry,1),encoding='utf-8')
    marker.write_text(json.dumps(dict(recorded_at=now.isoformat(),remaining_budget_seconds=recovery['remaining_budget_seconds'])),encoding='utf-8')


if __name__=='__main__':
    main()
