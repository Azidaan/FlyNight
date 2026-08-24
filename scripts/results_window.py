"""Окно «Результаты»: таблица прогонов и сводка по трём метрикам.

Зачем отдельное окно. Прогонов набралось два десятка, и решать, какие из
них участвуют в расчёте, в узкой колонке панели неудобно. Плюс сюда ведут
два пути — кнопка в панели и щелчок по строке легенды в окне графика,
— а одно окно на оба пути лучше двух одинаковых экранов.

Что внутри. Слева таблица всех прогонов с переключателем «учитывать».
Справа три величины, которые нас интересуют: общая точность, целевая
клетка (мелкие объекты ночью) и трекинг. Каждая считается ДВАЖДЫ — с
учётом отключений и по всем прогонам, — и если отключения меняют вывод,
окно говорит об этом прямо.

Запуск: python scripts/results_window.py
        python scripts/results_window.py --selftest
"""
import argparse
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import re
import subprocess

from evaluation.results_store import (  # noqa: E402
    collect_runs, metric_blocks, toggle,
)

HTML = PROJECT_ROOT / 'ui' / 'results.html'
TITLE = 'Результаты — YOLO26 + DRB'
JOB_LOG = PROJECT_ROOT / 'runs' / 'results_job.log'

# Строка «=== имя ... ===» печатается на каждую считаемую модель — по ним
# считается, сколько сделано.
RE_MODEL = re.compile(r'^===\s*(.+?)\s*===')
# Счётчик внутри модели. Форматов два: «трекинг 29/71:» с двоеточием и
# «45/892 6.4it/s» с пробелом. Прежний шаблон требовал пробел после и на
# трекинге не срабатывал вовсе — полоса стояла на нуле до конца модели.
RE_STEP = re.compile(r'(\d+)/(\d+)(?=[\s:,]|$)')
RE_TRACK = re.compile(r'трекинг\s+(\d+)/(\d+)')


def training_busy() -> str:
    """Идёт ли обучение: оценка займёт ту же видеокарту.

    Через PowerShell, а НЕ через wmic: в Windows 11 wmic удалён, вызов
    молча падал, и проверка возвращала «свободно» при идущем обучении.
    При любой ошибке считаем, что занято — пропустить второй расчёт на
    занятой карте хуже, чем лишний раз не дать его запустить.
    """
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Select-Object -ExpandProperty CommandLine")
    try:
        r = subprocess.run(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps],
            capture_output=True, text=True, timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return 'проверку занятости выполнить не удалось'
    if r.returncode != 0:
        return 'проверку занятости выполнить не удалось'
    for line in (r.stdout or '').splitlines():
        if 'train_drb.py' in line or 'train_baseline.py' in line:
            return 'обучение'
    return ''


class ResultsApi:
    """Мост в окно: состояние, переключение учёта и запуск расчёта."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.job_block = ''
        self.job_total = 0

    def state(self) -> dict:
        runs = collect_runs()
        return {
            'runs': runs,
            'blocks': metric_blocks(runs),
            'stamp': 'обновлено ' + datetime.now().strftime('%H:%M:%S'),
            'job': self.job_status(),
        }

    def toggle_run(self, key: str) -> bool:
        return toggle(key)

    def compute(self, block_id: str) -> dict:
        """Запустить расчёт величины. Считаются только недостающие модели."""
        if self.proc is not None and self.proc.poll() is None:
            return {'ok': False, 'text': 'Расчёт уже идёт'}
        busy = training_busy()
        if busy:
            return {'ok': False,
                    'text': f'Идёт {busy} — оценка займёт ту же видеокарту'}
        # Ключ вида «target_full»: величина и длина обучения. Считать их
        # вместе нельзя — у каждой длины свой контроль и своё расписание.
        runs = collect_runs()
        target = None
        for b in metric_blocks(runs):
            for L in b['lens']:
                if L['id'] == block_id:
                    target = L
        if target is None or not target['state']['job']:
            return {'ok': False, 'text': 'Здесь считать нечего'}
        models = target['state']['missing'] + target['state']['stale']
        if not models:
            return {'ok': False, 'text': 'Всё уже посчитано'}
        cmd = [sys.executable, str(PROJECT_ROOT / 'scripts' / 'evaluate_probes.py'),
               '--models', *models, '--subsets', 'night',
               '--out', f'{block_id}_night.csv']
        if target['state']['job'] == 'track':
            cmd.append('--tracking')
        JOB_LOG.parent.mkdir(parents=True, exist_ok=True)
        self.job_block, self.job_total = block_id, len(models)
        with open(JOB_LOG, 'w', encoding='utf-8') as log:
            self.proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               | subprocess.DETACHED_PROCESS))
        return {'ok': True, 'text': f'Считаю {len(models)} моделей'}

    def job_status(self) -> dict:
        """Доля выполненного: пройденные модели плюс счётчик внутри текущей.

        Что именно идёт, пишется словами: у трекинга это видео, у оценки по
        поднаборам — пачки кадров. Без этого полоса была немой.
        """
        if self.proc is None:
            return {'running': False}
        alive = self.proc.poll() is None
        done, cur, tot, what, name = 0, 0, 0, '', ''
        try:
            with open(JOB_LOG, encoding='utf-8', errors='replace') as f:
                for raw in f:
                    for line in raw.replace('\r', '\n').split('\n'):
                        m = RE_MODEL.match(line)
                        if m:
                            done += 1
                            name = m.group(1).split('/')[0].strip()
                            cur, tot, what = 0, 0, ''
                            continue
                        t = RE_TRACK.search(line)
                        if t:
                            cur, tot, what = int(t.group(1)), int(t.group(2)), 'видео'
                            continue
                        st = RE_STEP.search(line)
                        if st:
                            cur, tot, what = int(st.group(1)), int(st.group(2)), 'кадры'
        except OSError:
            pass
        inner = (cur / tot) if tot else 0.0
        n = max(1, self.job_total)
        pct = min(99.0, 100.0 * (max(0, done - 1) + inner) / n) if alive else 100.0
        if not alive:
            text = 'расчёт завершён'
        else:
            text = f'модель {min(max(done, 1), n)} из {n}'
            if name:
                text += f' · {name}'
            if tot:
                text += f' · {what} {cur} из {tot}'
        return {'running': alive, 'block': self.job_block,
                'percent': round(pct, 1), 'text': text}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    import webview

    class _Quiet(logging.Filter):
        def filter(self, record):
            return 'window.native' not in record.getMessage()
    logging.getLogger('pywebview').addFilter(_Quiet())

    # Служебному окну — свой заголовок, иначе снимок находит уже открытое
    # пользователем окно и проверяет не ту страницу.
    title = TITLE + ' · проверка' if args.selftest else TITLE
    api = ResultsApi()
    webview.create_window(
        title, html=HTML.read_text(encoding='utf-8'), js_api=api,
        width=1080, height=720, min_size=(820, 520))

    if args.selftest:
        def snap() -> None:
            time.sleep(3.0)
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    'launcher', PROJECT_ROOT / 'Запусти меня.py')
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod._capture_window_by_title(
                    title, PROJECT_ROOT / 'ui' / 'results_screenshot.png')
                print('SELFTEST OK: скриншот сохранён')
            except Exception as e:                       # noqa: BLE001
                print(f'SELFTEST: снимок не удался: {e}')
            finally:
                try:
                    webview.windows[0].destroy()
                except Exception:                        # noqa: BLE001
                    pass
        threading.Thread(target=snap, daemon=True).start()

    webview.start()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
