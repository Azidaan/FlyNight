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
                ('drb12', 'DRB · условия контроля', COL_BLUE),
                ('drb_bp', 'DRB · поправка + полоса', COL_SEC),
                ('drb_ad', 'DRB · подъём по кадру', COL_ACCENT),
                ('drb_fix', 'DRB · фиксированный фильтр', COL_INK),
                ('lp_ft', 'DRB · ступенчато', COL_OK),
                ('ft_drb', 'Дообучение · с блоком', COL_OK),
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


def series(name, colour, xs, ys, dash=False, width=2.0, group='other',
           live=False) -> dict:
    return {'name': name, 'color': colour, 'x': xs, 'y': ys,
            'dash': dash, 'width': width, 'group': group, 'live': live}


# Сколько минут без записи в results.csv считать прогон остановившимся.
# Эпоха идёт около двадцати минут, так что порог должен быть заметно больше.
LIVE_WINDOW_MIN = 45


def live_run_dir() -> Path | None:
    """Каталог прогона, который считается прямо сейчас, или None.

    Определяется по свежести results.csv: если в него писали недавно, значит
    обучение идёт. Точнее без обращения к списку процессов не выйдет, а
    лезть туда из окна графика незачем.
    """
    newest, mt = None, 0.0
    for p in (PROJECT_ROOT / 'runs').glob('*/seed*/results.csv'):
        try:
            t = p.stat().st_mtime
        except OSError:
            continue
        if t > mt:
            newest, mt = p.parent, t
    if newest is None:
        return None
    return newest if (time.time() - mt) < LIVE_WINDOW_MIN * 60 else None


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

    def open_results(self) -> None:
        """Окно результатов — тем же путём, что и из панели."""
        import subprocess
        try:
            subprocess.Popen(
                [sys.executable,
                 str(PROJECT_ROOT / 'scripts' / 'results_window.py')],
                cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               | subprocess.DETACHED_PROCESS))
        except OSError:
            pass

    def curves(self) -> dict:
        views = []
        runs_dir = PROJECT_ROOT / 'runs'
        live = live_run_dir()
        # Отключённые в окне результатов не рисуются нигде — иначе снятая
        # там галочка меняла бы числа, но не картинку.
        try:
            from evaluation.results_store import load_excluded, run_key
            excluded = set(load_excluded())
        except Exception:                                   # noqa: BLE001
            excluded, run_key = set(), lambda g, s: ''

        def skip(run: Path) -> bool:
            return run_key(run.parent.name, int(run.name[4:])) in excluded

        # ——— вид 1: проба инициализации DRB, точность ———
        # Сидов у пробы может быть несколько: у контроля их три (замер шума),
        # у подъёма по кадру два (проверка на повторяемость). Раньше читался
        # жёстко seed42, и второй сид на график не попадал вовсе.
        def seed_runs(pid: str) -> list[tuple[int, Path]]:
            root = runs_dir / 'init_probe' / pid
            out = []
            for d in sorted(root.glob('seed*')):
                if d.is_dir() and d.name[4:].isdigit():
                    out.append((int(d.name[4:]), d))
            return sorted(out)

        probes, alphas = [], []
        for pid, label, colour in PROBE_LABELS:
            runs = seed_runs(pid)
            for k, (seed, run) in enumerate(runs):
                # Подпись с сидом — только когда сидов больше одного, иначе
                # у одиннадцати проб из двенадцати легенда распухла бы зря.
                name = label if len(runs) == 1 else f'{label} · сид {seed}'
                if skip(run):
                    continue
                r = read_run(run / 'results.csv')
                m = col(r, 'mAP50(B)')
                if m:
                    probes.append(series(name, shade(colour, k),
                                         col(r, 'epoch'), m,
                                         dash=(k % 2 == 1), group='probe',
                                         live=(run == live)))
                a = read_alphas(run / 'drb_alphas.csv')
                if a:
                    alphas.append((name, shade(colour, k), a))
        if probes:
            # Эталоном может быть только прогон на ТОМ ЖЕ расписании.
            # Стосерийный baseline на 12-й эпохе идёт с втрое большей
            # скоростью обучения и с ещё включённой мозаикой — рядом с
            # пробами он создавал ложную картину, поэтому здесь его нет.
            #
            # Все сиды контроля рисуются отдельными линиями: расстояние между
            # ними и есть разброс от случайного зерна, и любая проба, лежащая
            # внутри этого пучка, о блоке ничего не говорит. Пока была видна
            # одна контрольная линия, разброс был не виден вообще.
            cid, clabel, ccol = CONTROL
            cruns = seed_runs(cid)
            ref = []
            for k, (seed, run) in enumerate(cruns):
                if skip(run):
                    continue
                cm = col(read_run(run / 'results.csv'), 'mAP50(B)')
                if not cm:
                    continue
                cname = clabel if len(cruns) == 1 else f'{clabel} · сид {seed}'
                ref.append(series(cname, shade(ccol, k),
                                  col(read_run(run / 'results.csv'), 'epoch'),
                                  cm, dash=True, width=1.6, group='baseline',
                                  live=(run == live)))
            if ref:
                sub = ('выше — лучше · пунктир — контроль без DRB на том же '
                       'расписании; расстояние между его сидами — это разброс '
                       'от случайного зерна')
            else:
                sub = ('выше — лучше · эталона пока нет: контрольный прогон '
                       'без DRB не посчитан (узел Е)')
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
            if skip(p.parent):
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
                              width=2.0, group=g, live=(p.parent == live)))
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
