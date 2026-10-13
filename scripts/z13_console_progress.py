"""Read-only live console for Z13. No torch imports or training processes."""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_spatial_v2_20261013'


def read_json(path):
    # Allow the trainer to atomically replace a file even while we read it.
    try:
        if os.name == 'nt':
            import msvcrt
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_void_p]
            kernel.CreateFileW.restype = ctypes.c_void_p
            handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                return {}
            try:
                fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
            except Exception:
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel.CloseHandle(handle)
                raise
            with os.fdopen(fd, 'r', encoding='utf-8') as stream:
                return json.load(stream)
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def bar(done, total, width=30):
    fraction = min(1, max(0, done / total)) if total else 0
    filled = int(width * fraction)
    return '[' + '#' * filled + '-' * (width - filled) + f'] {100 * fraction:5.1f}%'


def render():
    protocol = read_json(RUN / 'protocol.json')
    controller = read_json(RUN / 'status.json')
    jobs = protocol.get('jobs', [])
    steps = protocol.get('steps', 4000)
    statuses = [read_json(RUN / job['arm'] / job['name'] / 'status.json') for job in jobs]
    done = sum(min(s.get('step', 0), steps) for s in statuses)
    state_names = {'running': 'работает', 'failed': 'ОСТАНОВЛЕН: ошибка',
        'paused': 'приостановлен', 'retrying': 'повтор после сбоя',
        'training_completed': 'обучение и выбор порогов завершены'}
    state = controller.get('state', 'нет сведений')
    lines = ['З13 v2 — обучение B / C / D',
        f'Время: {datetime.now():%d.%m.%Y %H:%M:%S}  |  {state_names.get(state, state)}', '',
        'Сумма шагов обучения положения (без времени расчёта предсказаний):',
        f'{bar(done, len(jobs) * steps)}  {done:,} / {len(jobs) * steps:,}', '',
        'B — без FlyVis; C — замороженная FlyVis; D — обучаемая FlyVis.',
        'Часть 1/2 — обучение на половинах train; полный — на всех 104 видео.', '']
    names = {'fold0': 'часть 1', 'fold1': 'часть 2', 'full': 'полный '}
    active = controller.get('stage', '')
    for job, status in zip(jobs, statuses):
        key = f"{job['arm']}_{job['name']}"
        current = status.get('step', 0)
        stage = status.get('stage', '')
        suffix = 'готово' if stage == 'predictions_saved' else 'ожидает'
        if key == active and state in ('running', 'retrying'):
            suffix = 'обучение' if stage == 'localization' else 'подготовка'
            if stage == 'prediction':
                suffix = f"предсказания {status.get('episodes_done', 0)}/{status.get('episodes_total', '?')}"
        lines.append(f"{'>' if key == active else ' '} {job['arm']} {names.get(job['name'], job['name'])} {bar(current, steps, 22)} {current:4}/{steps}  {suffix}")
        if key == active and status.get('updated'):
            try:
                age = (datetime.now().astimezone() - datetime.fromisoformat(status['updated'])).total_seconds()
                lines.append(f'    Последняя запись: {max(0, int(age))} с назад' + (' — давно нет обновлений' if age > 120 else ''))
            except ValueError:
                pass
            if stage == 'localization' and status.get('remaining_seconds') is not None:
                lines.append(f"    До конца этого обучения: ~{status['remaining_seconds'] / 60:.0f} мин; предсказания затем.")
    lines.extend(['', 'Обучение оценки качества и выбор порогов:'])
    for arm in protocol.get('arms', []):
        quality = read_json(RUN / arm / 'quality/status.json')
        tau = read_json(RUN / arm / 'tau.json')
        label = 'готово' if quality.get('stage') == 'model_ready' else 'ожидает'
        if active == f'{arm}_quality' and state == 'running':
            label = 'идёт обучение / выбор порога'
        lines.append(f"  {arm}: {bar(quality.get('step', 0), protocol.get('quality_steps', 1500), 18)}  {label}")
        if tau:
            lines.append(f"     Порог: {tau.get('tau')} (это ещё не результат HOTA)")
    if controller.get('error') or state == 'failed':
        lines.extend(['', str(controller.get('error', controller.get('reason', 'Смотри журнал расчёта.')))])
    lines.extend(['', '100% шагов не означает завершение обработки предсказаний и выбора порогов.',
        'Окно только читает прогресс. Его закрытие НЕ останавливает обучение.',
        'Обновление каждые 2 секунды. Ctrl+C — закрыть наблюдение.'])
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    if os.name == 'nt' and not args.once:
        kernel = ctypes.windll.kernel32
        kernel.SetConsoleTitleW('З13 — прогресс обучения B / C / D')
        os.system('')  # Enable Windows terminal escape processing.
    while True:
        content = render()
        if args.once:
            print(content)
            return
        sys.stdout.write('\033[H\033[2J' + content + '\n')
        sys.stdout.flush()
        time.sleep(2)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
