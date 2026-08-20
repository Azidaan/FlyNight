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
                ('w4', 'Вклад 2%', COL_BLUE),
                ('boost', 'Тождественный старт + разгон', COL_INK),
                ('ft_drb', 'Дообучение · с блоком', COL_ACCENT),
                ('ft_base', 'Дообучение · без блока', COL_SEC)]
# Эталон для проб — короткий прогон БЕЗ блока на том же расписании.
CONTROL = ('base12', 'Контроль · без DRB, равные условия', COL_REF)


def shade(hex_colour: str, step: int) -> str:
    """Оттенок того же цвета: сиды одного семейства различимы, но родственны."""
    if step <= 0:
        return hex_colour
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    k = min(0.42, 0.21 * step)                 # осветляем к фону, не выбеливая
    r, g, b = (round(c + (255 - c) * k) for c in (r, g, b))
    return f'#{r:02x}{g:02x}{b:02x}'


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


def series(name, colour, xs, ys, dash=False, width=2.0, group='other') -> dict:
    return {'name': name, 'color': colour, 'x': xs, 'y': ys,
            'dash': dash, 'width': width, 'group': group}


# группы конфигураций — по ним в окне включаются и выключаются линии
GROUPS = [('baseline', 'Baseline'), ('drb', 'DRB'), ('aug', 'DRB + аугментации'),
          ('ablation', 'Ablation'), ('probe', 'Пробы')]


def group_of(run_name: str) -> str:
    if run_name.startswith('drb_aug'):
        return 'aug'
    if run_name.startswith('drb'):
        return 'drb'
    if run_name.startswith('ablation'):
        return 'ablation'
    if run_name.startswith('baseline'):
        return 'baseline'
    return 'other'


def read_alphas(path: Path) -> dict[str, list[float]]:
    """drb_alphas.csv -> средние по уровням пирамиды вклады ветвей по эпохам."""
    if not path.exists():
        return {}
    try:
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return {}
    by_epoch: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        e = _num(r.get('epoch'))
        a1, a2 = _num(r.get('alpha_lqa')), _num(r.get('alpha_fga'))
        if e is None or a1 is None or a2 is None:
            continue
        by_epoch.setdefault(int(e), []).append((a1, a2))
    xs = sorted(by_epoch)
    if not xs:
        return {}
    return {
        'x': [float(e) for e in xs],
        'lqa': [sum(v[0] for v in by_epoch[e]) / len(by_epoch[e]) for e in xs],
        'fga': [sum(v[1] for v in by_epoch[e]) / len(by_epoch[e]) for e in xs],
    }


class ChartApi:
    """Данные для графика. Ничего не запускает — только читает файлы."""

    def curves(self) -> dict:
        views = []
        runs_dir = PROJECT_ROOT / 'runs'

        # ——— вид 1: проба инициализации DRB, точность ———
        probes, alphas = [], []
        for pid, label, colour in PROBE_LABELS:
            run = runs_dir / 'init_probe' / pid / 'seed42'
            r = read_run(run / 'results.csv')
            m = col(r, 'mAP50(B)')
            if m:
                probes.append(series(label, colour, col(r, 'epoch'), m,
                                     group='probe'))
            a = read_alphas(run / 'drb_alphas.csv')
            if a:
                alphas.append((label, colour, a))
        if probes:
            # Эталоном может быть только прогон на ТОМ ЖЕ расписании.
            # Стосерийный baseline на 12-й эпохе идёт с втрое большей
            # скоростью обучения и с ещё включённой мозаикой — рядом с
            # пробами он создавал ложную картину, поэтому здесь его нет.
            cid, clabel, ccol = CONTROL
            cr = read_run(runs_dir / 'init_probe' / cid / 'seed42' / 'results.csv')
            cm = col(cr, 'mAP50(B)')
            if cm:
                ref = [series(clabel, ccol, col(cr, 'epoch'), cm,
                              dash=True, width=1.6, group='baseline')]
                sub = ('с какого вклада ветвей стартовать блок · выше — лучше · '
                       'пунктир — контроль без DRB на том же расписании')
            else:
                ref = []
                sub = ('с какого вклада ветвей стартовать блок · выше — лучше · '
                       'эталона пока нет: контрольный прогон без DRB '
                       'не посчитан (узел Е)')
            views.append({
                'id': 'probe', 'title': 'Проба инициализации DRB · точность',
                'sub': sub,
                'ylabel': 'mAP@0.50', 'xlabel': 'эпоха',
                'series': ref + probes})

        # ——— вид 2: как «включаются» ветви блока ———
        if alphas:
            sr = []
            for label, colour, a in alphas:
                sr.append(series(f'{label} · LQA', colour, a['x'], a['lqa'],
                                 group='probe'))
                sr.append(series(f'{label} · FGA', colour, a['x'], a['fga'],
                                 dash=True, width=1.5, group='probe'))
            views.append({
                'id': 'alpha', 'title': 'Вклад ветвей DRB по эпохам',
                'sub': 'сплошная — пространственная ветвь, пунктир — частотная; '
                       'растёт ли вклад, который дали на старте',
                'ylabel': 'вклад ветви', 'xlabel': 'эпоха', 'series': sr})

        # ——— вид 2: точность всех обученных конфигураций ———
        # цвет по семейству конфигураций, оттенок — по сиду: сравнивать
        # baseline с DRB нужно на глаз, а сиды внутри семейства не путать
        family_col = {'baseline': COL_SEC, 'drb': COL_ACCENT, 'aug': COL_OK,
                      'ablation': COL_BLUE, 'other': COL_INK}
        acc, seen = [], {}
        for p in sorted(runs_dir.glob('*/seed*/results.csv')):
            if 'init_probe' in p.parts or 'smoke' in p.parts[-3]:
                continue
            r = read_run(p)
            m = col(r, 'mAP50(B)')
            if len(m) < 2:
                continue
            run_name = p.parts[-3]
            g = group_of(run_name)
            k = seen.get(g, 0)
            seen[g] = k + 1
            acc.append(series(f'{run_name}/{p.parts[-2]}',
                              shade(family_col.get(g, COL_SEC), k),
                              col(r, 'epoch'), m, dash=(k % 2 == 1),
                              width=2.0, group=g))
        if acc:
            views.append({
                'id': 'acc', 'title': 'Точность по эпохам · все конфигурации',
                'sub': 'mAP@0.50 на проверочной выборке · выше — лучше · '
                       'кнопками ниже можно оставить только нужные',
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

        present = {s['group'] for v in views for s in v['series']}
        return {'views': views,
                'groups': [{'id': g, 'label': lab} for g, lab in GROUPS
                           if g in present],
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

    # У служебного окна ОТДЕЛЬНЫЙ заголовок: иначе поиск окна по имени
    # находит уже открытое пользователем окно графика и снимает его — то
    # есть показывает старую страницу вместо проверяемой.
    title = TITLE + ' · проверка' if args.selftest else TITLE
    api = ChartApi()
    win = webview.create_window(
        title, html=HTML.read_text(encoding='utf-8'), js_api=api,
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
                    title, PROJECT_ROOT / 'ui' / 'chart_screenshot.png')
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
