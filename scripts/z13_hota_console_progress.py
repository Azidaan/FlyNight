"""Read-only HOTA inference progress console."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_console_progress import read_json, bar
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def render():
    status = read_json(RUN / 'status.json')
    authorization = read_json(RUN / 'full_authorization.json')
    names = {'authorized': 'подготовка запуска', 'running': 'работает',
             'completed': 'расчёт завершён', 'paused': 'приостановлен', 'failed': 'ОШИБКА'}
    lines = ['З13 — проверка HOTA', f'Обновление: {datetime.now():%d.%m %H:%M:%S}',
             'Состояние: ' + names.get(status.get('state'), 'нет сведений'), '']
    for arm in ('B', 'C', 'D'):
        markers = list((RUN / 'completed_jobs').glob(f'{arm}_sig_*.json')) if (RUN / 'completed_jobs').exists() else []
        lines.append(f'{arm}: выходы трекера {bar(len(markers), 35)} {len(markers)}/35 видео')
        masa = len(list((RUN / arm / 'masa').glob('*.npz'))) if (RUN / arm / 'masa').exists() else 0
        lines.append(f'   MASA: сохранены признаки {masa}/35 видео')
    job = status.get('job', {})
    lines.extend(['', f"Текущий этап: {job.get('stage', 'ожидает')} {job.get('arm', '')} {job.get('video', '')}"])
    if authorization.get('deadline'):
        deadline = datetime.fromisoformat(authorization['deadline'])
        remaining = max(0, (deadline - datetime.now().astimezone()).total_seconds())
        lines.append(f'Окно заканчивается {deadline:%d.%m %H:%M} МСК; осталось {remaining/3600:.2f} ч')
    if status.get('reason'):
        lines.append(str(status['reason']))
    lines.extend(['', 'B — без FlyVis; C — замороженная; D — обучаемая.',
                  'После выходов трекера — MASA, сшивка и HOTA.',
                  'Закрытие этого окна не останавливает расчёт. Ctrl+C — закрыть наблюдение.'])
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--once', action='store_true'); args = ap.parse_args()
    if not args.once and os.name == 'nt':
        os.system('')
    while True:
        value = render()
        if args.once:
            print(value); return
        sys.stdout.write('\033[H\033[2J' + value + '\n'); sys.stdout.flush(); time.sleep(2)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
