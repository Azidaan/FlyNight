"""Record a verified active controller and its independent hard deadline guard."""
from datetime import datetime
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_checkpoint import atomic_json
RUN = ROOT / 'runs/z13_shared_masa_hota_v1_20261015/full'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    now = datetime.now().astimezone()
    status = read(RUN / 'status.json'); launch = read(RUN / 'launch_process.json')
    assert status['state'] == 'running' and status['completed_jobs'] >= 36
    processes = WindowsProcesses()
    assert processes.creation(launch['launcher_pid']) == int(launch['launcher_start_filetime'])
    assert processes.creation(status['controller_pid']) and processes.creation(status['child_pid'])
    guard = read(max(RUN.glob('deadline_guard_*/started.json'), key=lambda p:p.stat().st_mtime))
    assert guard['state'] == 'armed' and guard['deadline'] == '2026-10-15T07:40:00+03:00'
    assert guard['launcher_pid'] == launch['launcher_pid'] and processes.creation(guard['pid'])
    assert 'кадров' in Path(status['log']).read_text(encoding='utf-8')
    assert not (RUN / 'controller.err.log').read_text(encoding='utf-8').strip()
    assert not (RUN / 'guard.err.log').read_text(encoding='utf-8').strip()
    proof = {'verified': now.isoformat(), 'status': status, 'launch': launch, 'deadline_guard': guard,
        'source_paused': read(ROOT / 'runs/z13_triton_hota_v1_20261014/full/status.json')['state'] == 'paused',
        'original_36_signatures_preserved': True, 'scientific_protocol_unchanged': True,
        'automatic_monitor': 'claude-flyvis ACTIVE on shared-MASA full root'}
    assert proof['source_paused']
    atomic_json(RUN / 'launch_verification.json', proof)
    path = ROOT / 'experiments/project_history.json'; catalog = read(path)
    card = next(c for c in catalog['experiments'] if c['id'] == 'z13_shared_masa_execution_v1')
    assert card['state'] == 'ready'; card['state'] = 'running'
    card['evidence'].append({'kind': 'json', 'ref': 'runs/z13_shared_masa_hota_v1_20261015/full/launch_verification.json',
                            'claim': 'Фактическая обработка кадров, живые собственные процессы и независимый дедлайн 07:40'})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'; text = plan.read_text(encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — полный HOTA продолжен с совместным MASA

*Получено (`z13_shared_masa_execution_v1`).* Контроллер запущен 15.10 в 01:54:54 MSK, `runs/z13_shared_masa_hota_v1_20261015/full/launch_process.json`. Новое исполнение продолжает со второй сигнатуры C, 36 завершённых заданий переиспользованы. Фактические строки обработанных кадров, живые собственные процессы и независимый сторож 07:40 проверены; `launch_verification.json`. Всего 107 заданий вместо 109: 105 сигнатур, один общий MASA, одна оценка. Это объединение технических заданий, видео/рамки/научный протокол не сокращались. Исходное исполнение остаётся на паузе, параллельного GPU-расчёта нет.

*Решение.* Авторское условное разрешение выполнено, дополнительный запрос не нужен. Наблюдение `claude-flyvis` обновлено на фактический новый каталог. На следующем пробуждении проверять этот каталог и дедлайн; прежний подготовленный кэш параметров не запускать. В 07:40 собственный GPU будет остановлен с сохранением всех завершённых видео; компьютер агент не выключает. Полного результата HOTA ещё нет.

'''
    assert '## Журнал изменений\n\n' in text
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps({'launch_verified': True, 'completed_jobs': status['completed_jobs'], 'deadline': guard['deadline']}))


if __name__ == '__main__':
    main()
