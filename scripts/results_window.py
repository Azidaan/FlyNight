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

from evaluation.results_store import (  # noqa: E402
    collect_runs, metric_blocks, toggle,
)

HTML = PROJECT_ROOT / 'ui' / 'results.html'
TITLE = 'Результаты — YOLO26 + DRB'


class ResultsApi:
    """Мост в окно: состояние и переключение учёта прогона."""

    def state(self) -> dict:
        runs = collect_runs()
        return {
            'runs': runs,
            'blocks': metric_blocks(runs),
            'stamp': 'обновлено ' + datetime.now().strftime('%H:%M:%S'),
        }

    def toggle_run(self, key: str) -> bool:
        return toggle(key)


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
