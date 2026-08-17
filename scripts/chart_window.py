"""Отдельное окно с подробным графиком обучения.

Своё окно и свой профиль WebView2 — работает параллельно с панелью и не
мешает ей. Данные читаются прямо с диска (results.csv, который Ultralytics
пишет по эпохам), поэтому окно живое: обновляется само, пока идёт обучение.

Запуск: python scripts/chart_window.py
Служебный режим: --selftest — открыть, снять ui/chart_screenshot.png, закрыть.
"""
import argparse
import csv
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VENV_PY = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
HTML = PROJECT_ROOT / 'ui' / 'chart.html'
TITLE = 'График обучения — YOLO26 + DRB'

# палитра совпадает с панелью (ui/launcher.html)
COL_ACCENT, COL_OK, COL_BLUE = '#d97757', '#3f9277', '#3f6ea8'
COL_REF, COL_SEC, COL_INK = '#a39b8c', '#6b6457', '#2f2a24'

PROBE_LABELS = [('w0', 'Вклад 50% (план)', COL_ACCENT),
                ('w2', 'Вклад 12%', COL_OK),
                ('w4', 'Вклад 2%', COL_BLUE)]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def read_run(path: Path) -> dict[str, list[float]]:
    """results.csv Ultralytics -> {колонка: значения}. Строка = эпоха."""
    if not path.exists():
        return {}
    try:
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return {}
    out: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r.items():
            if k is None:
                continue
            n = _num(v)
            if n is not None:
                out.setdefault(k.strip(), []).append(n)
    return out


def col(run: dict, needle: str) -> list[float]:
    for k, v in run.items():
        if needle in k:
            return v
    return []


def series(name, colour, xs, ys, dash=False, width=2.0) -> dict:
    return {'name': name, 'color': colour, 'x': xs, 'y': ys,
            'dash': dash, 'width': width}


class ChartApi:
    """Данные для графика. Ничего не запускает — только читает файлы."""

    def curves(self) -> dict:
        views = []
        runs_dir = PROJECT_ROOT / 'runs'

        # ——— вид 1: проба инициализации DRB ———
        probes = []
        for pid, label, colour in PROBE_LABELS:
            r = read_run(runs_dir / 'init_probe' / pid / 'seed42' / 'results.csv')
            m = col(r, 'mAP50(B)')
            if m:
                probes.append(series(label, colour, col(r, 'epoch'), m))
        if probes:
            base = read_run(runs_dir / 'baseline' / 'seed42' / 'results.csv')
            n = max(len(s['y']) for s in probes)
            bm = col(base, 'mAP50(B)')[:n]
            ref = [series('baseline (без DRB)', COL_REF,
                          col(base, 'epoch')[:n], bm, dash=True, width=1.4)] if bm else []
            views.append({
                'id': 'probe', 'title': 'Проба инициализации DRB',
                'sub': 'с какого вклада ветвей стартовать блок · выше — лучше',
                'ylabel': 'mAP@0.50', 'xlabel': 'эпоха',
                'series': ref + probes})

        # ——— вид 2: точность всех обученных конфигураций ———
        acc = []
        palette = [COL_ACCENT, COL_OK, COL_BLUE, COL_SEC, COL_INK]
        for i, p in enumerate(sorted(runs_dir.glob('*/seed*/results.csv'))):
            if 'init_probe' in p.parts or 'smoke' in p.parts[-3]:
                continue
            r = read_run(p)
            m = col(r, 'mAP50(B)')
            if len(m) < 2:
                continue
            acc.append(series(f'{p.parts[-3]}/{p.parts[-2]}',
                              palette[i % len(palette)], col(r, 'epoch'), m))
        if acc:
            views.append({
                'id': 'acc', 'title': 'Точность по эпохам',
                'sub': 'mAP@0.50 на проверочной выборке · выше — лучше',
                'ylabel': 'mAP@0.50', 'xlabel': 'эпоха', 'series': acc})

        # ——— виды 3-4: свежайший прогон подробно ———
        newest, mt = None, 0.0
        for p in runs_dir.glob('*/seed*/results.csv'):
            if 'smoke' in p.parts[-3]:
                continue
            if p.stat().st_mtime > mt:
                newest, mt = p, p.stat().st_mtime
        if newest is not None:
            r = read_run(newest)
            tag = f'{newest.parts[-3]}/{newest.parts[-2]}'
            x = col(r, 'epoch')
            views.append({
                'id': 'loss', 'title': f'Ошибки обучения · {tag}',
                'sub': 'чем ниже, тем лучше · рамки, классы, границы',
                'ylabel': 'ошибка', 'xlabel': 'эпоха', 'norm': True,
                'series': [
                    series('рамки (box)', COL_ACCENT, x, col(r, 'train/box_loss')),
                    series('классы (cls)', COL_OK, x, col(r, 'train/cls_loss')),
                    series('границы (dfl)', COL_BLUE, x, col(r, 'train/dfl_loss')),
                ]})
            vcls = col(r, 'val/cls_loss')
            m = col(r, 'mAP50(B)')
            if vcls and m:
                views.append({
                    'id': 'overfit', 'title': f'Переобучение · {tag}',
                    'sub': 'ошибка на обучении падает, а на проверке растёт — '
                           'модель начинает запоминать',
                    'ylabel': 'нормировано к своему размаху', 'xlabel': 'эпоха',
                    'norm': True,
                    'series': [
                        series('ошибка на обучении', COL_SEC, x,
                               col(r, 'train/cls_loss'), dash=True, width=1.5),
                        series('ошибка на проверке', COL_ACCENT, x, vcls),
                        series('точность (mAP@0.50)', COL_OK, x, m),
                    ]})

        return {'views': views,
                'stamp': datetime.now().strftime('обновлено %H:%M:%S')}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    import webview

    class _Quiet(logging.Filter):
        def filter(self, record):
            return 'window.native' not in record.getMessage()
    logging.getLogger('pywebview').addFilter(_Quiet())

    api = ChartApi()
    win = webview.create_window(
        TITLE, html=HTML.read_text(encoding='utf-8'), js_api=api,
        width=1180, height=760, min_size=(760, 520))

    if args.selftest:
        def snap() -> None:
            from importlib import util as _u
            spec = _u.spec_from_file_location('panel', PROJECT_ROOT / 'Запусти меня.py')
            mod = _u.module_from_spec(spec)
            spec.loader.exec_module(mod)
            time.sleep(7)
            try:
                mod._capture_window_by_title(
                    TITLE, PROJECT_ROOT / 'ui' / 'chart_screenshot.png')
                print('CHART SELFTEST OK: снимок сохранён')
            finally:
                win.destroy()
        threading.Thread(target=snap, daemon=True).start()

    storage = PROJECT_ROOT / '.cache' / (
        'webview2_chart_selftest' if args.selftest else 'webview2_chart')
    storage.mkdir(parents=True, exist_ok=True)
    webview.start(storage_path=str(storage), private_mode=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
