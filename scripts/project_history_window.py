"""Окно «История проекта»: эксперименты, коммиты, хронология единого плана.

Только просмотр. Окно не запускает обучений, не меняет Git и не пишет в
план — оно собирает индекс из трёх источников (Git, ЕДИНЫЙ_ПЛАН.md,
кураторский каталог experiments/project_history.json) и показывает его.

Отдельный процесс и отдельный профиль WebView2 — как у окон результатов и
графика: падение одного окна не уносит панель.

Запуск: python scripts/project_history_window.py
        python scripts/project_history_window.py --selftest [--narrow]
Селфтест открывает служебное окно с отдельным заголовком, ждёт загрузки
данных, снимает по скриншоту каждой вкладки в ui/ и закрывается.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.project_history import filters, render  # noqa: E402
from evaluation.project_history.git_reader import GitError, GitReader  # noqa: E402
from evaluation.project_history.index import (  # noqa: E402
    Index, build_index, load_index)
from evaluation.project_history.model import (  # noqa: E402
    EVENT_KIND_LABELS, RESULT_LABELS, STATE_LABELS)
from evaluation.project_history.plan_parser import parse_plan  # noqa: E402
from evaluation.project_history.safe_paths import resolve_inside  # noqa: E402

HTML = PROJECT_ROOT / 'ui' / 'project_history.html'
TITLE = 'История проекта — эксперименты, коммиты, план'
SCREENSHOT = PROJECT_ROOT / 'ui' / 'project_history_screenshot.png'
# план идёт до вкладок с фильтрами: именно такой порядок ловил ошибку затирания рейки
TABS = ('overview', 'plan', 'experiments', 'commits', 'timeline', 'compare', 'glossary')
DIFF_BYTES = 200_000
_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


class HistoryApi:
    """Мост в окно. Каждый метод возвращает JSON-совместимые словари."""

    def __init__(self, root: Path = PROJECT_ROOT, cache_dir: Path | None = None):
        self.root = Path(root)
        self.cache_dir = cache_dir
        self.idx: Index | None = None
        self.git = GitReader(self.root)
        self.boot_event = threading.Event()
        self._plan_html: str | None = None
        self._lock = threading.Lock()

    # ——— служебное ———
    def boot_done(self) -> None:
        self.boot_event.set()

    def _index(self) -> Index:
        with self._lock:
            if self.idx is None:
                self.idx = load_index(self.root, cache_dir=self.cache_dir)
            return self.idx

    def refresh(self) -> dict:
        """Перечитать источники и пересобрать индекс (кнопка «Обновить»)."""
        with self._lock:
            self.idx = build_index(self.root, cache_dir=self.cache_dir, write_cache=True)
            self._plan_html = None
        return self.overview()

    def labels(self) -> dict:
        return {'states': STATE_LABELS, 'results': RESULT_LABELS,
                'kinds': EVENT_KIND_LABELS}

    # ——— обзор ———
    def overview(self) -> dict:
        idx = self._index()
        recent = [t.to_dict() for t in idx.timeline
                  if t.kind in ('plan_changelog', 'plan_chronology', 'manual')][:12]
        return {
            'built_at': idx.built_at, 'from_cache': idx.from_cache,
            'working_tree': idx.working_tree, 'remote_refs': idx.remote_refs,
            'stats': idx.stats, 'warnings': idx.warnings, 'errors': idx.errors,
            'areas': [a.to_dict() for a in idx.areas],
            'recent': recent,
            'plan': {'path': idx.plan.get('path', ''), 'n_lines': idx.plan.get('n_lines', 0),
                     'changelog': idx.plan.get('changelog', [])[:6]},
            'labels': self.labels(),
        }

    # ——— эксперименты ———
    def experiments(self, f: dict | None = None) -> dict:
        idx = self._index()
        items = [e.to_dict() for e in idx.experiments]
        shown = filters.apply(items, f)
        shown.sort(key=lambda e: (e.get('finished') or e.get('started') or ''), reverse=True)
        return {'items': shown, 'total': len(items), 'shown': len(shown),
                'facets': filters.facets(items),
                'areas': {a.id: a.to_dict() for a in idx.areas}}

    def experiment(self, exp_id: str) -> dict:
        idx = self._index()
        e = next((x for x in idx.experiments if x.id == exp_id), None)
        if e is None:
            return {'error': f'эксперимент {exp_id!r} не найден'}
        d = e.to_dict()
        by_short = {c.hash[:7]: c for c in idx.commits}
        d['commit_info'] = [
            {'short': h[:7], 'subject': by_short[h[:7]].subject, 'date': by_short[h[:7]].date,
             'is_empty': by_short[h[:7]].is_empty} if h[:7] in by_short
            else {'short': h[:7], 'subject': '— нет в истории —', 'date': '', 'missing': True}
            for h in e.commits]
        d['events'] = [t.to_dict() for t in idx.timeline
                       if t.experiment_id == e.id and t.kind != 'commit'][:30]
        d['area_title'] = next((a.title for a in idx.areas if a.id == e.area), e.area)
        d['compare'] = self._compare(e)
        return d

    @staticmethod
    def _compare(e) -> dict:
        """Сравнение «ожидалось — сделано»: только по полям каталога, без догадок."""
        acc = e.acceptance or []
        met = [a for a in acc if a.get('met') is True]
        unmet = [a for a in acc if a.get('met') is False]
        unknown = [a for a in acc if a.get('met') not in (True, False)]
        if not e.expected or not (e.obtained or e.done):
            verdict = 'unconfirmed'
        elif acc:
            verdict = ('match' if met and not unmet else
                       'mismatch' if unmet and not met else
                       'partial' if met and unmet else 'unconfirmed')
        else:
            verdict = {'completed_positive': 'match', 'completed_negative': 'mismatch',
                       'closed': 'mismatch', 'inconclusive': 'partial'}.get(e.state, 'unconfirmed')
        return {'verdict': verdict, 'met': met, 'unmet': unmet, 'unknown': unknown,
                'evidence_ok': sum(1 for v in e.evidence if v.exists),
                'evidence_missing': sum(1 for v in e.evidence if v.exists is False)}

    # ——— коммиты ———
    def commits(self, f: dict | None = None) -> dict:
        idx = self._index()
        items = []
        for c in idx.commits:
            d = c.to_dict()
            d.pop('files', None)
            items.append(d)
        shown = filters.apply(items, f)
        return {'items': shown, 'total': len(items), 'shown': len(shown),
                'facets': filters.facets(items)}

    def commit(self, hash_: str) -> dict:
        idx = self._index()
        try:
            c = self.git.commit_detail(hash_)
        except GitError as e:
            return {'error': str(e)}
        d = c.to_dict()
        listed = next((x for x in idx.commits if x.hash == c.hash), None)
        if listed:
            d['experiments'] = listed.experiments
            d['plan_events'] = listed.plan_events
            d['result'] = listed.result
            d['result_source'] = listed.result_source
            d['branch'] = listed.branch
        d['experiment_titles'] = {e.id: e.title for e in idx.experiments
                                  if e.id in d.get('experiments', [])}
        d['events'] = [t.to_dict() for t in idx.timeline if t.id in d.get('plan_events', [])]
        if c.is_empty:
            d['diff'], d['diff_truncated'] = '', False
        else:
            try:
                d['diff'], d['diff_truncated'] = self.git.diff_preview(c.hash, max_bytes=DIFF_BYTES)
            except GitError as e:
                d['diff'], d['diff_truncated'] = f'diff не прочитан: {e}', False
        d['diff_limit_kb'] = DIFF_BYTES // 1000
        return d

    def commit_file_diff(self, hash_: str, path: str) -> dict:
        try:
            text, trunc = self.git.diff_preview(hash_, path=path, max_bytes=DIFF_BYTES)
        except GitError as e:
            return {'error': str(e)}
        return {'diff': text, 'truncated': trunc}

    # ——— хронология ———
    def timeline(self, f: dict | None = None) -> dict:
        idx = self._index()
        items = [t.to_dict() for t in idx.timeline]
        shown = filters.apply(items, f)
        return {'items': shown, 'total': len(items), 'shown': len(shown),
                'facets': filters.facets(items)}

    # ——— план и словарь ———
    def plan(self) -> dict:
        idx = self._index()
        if self._plan_html is None:
            p = self.root / 'dissertation' / 'ЕДИНЫЙ_ПЛАН.md'
            try:
                doc = parse_plan(p.read_text(encoding='utf-8'))
                self._plan_html = render.plan_html(doc)
            except OSError as e:
                self._plan_html = f'<p>План не прочитан: {render.escape(str(e))}</p>'
        return {'html': self._plan_html, 'headings': idx.plan.get('headings', []),
                'path': idx.plan.get('path', ''), 'n_lines': idx.plan.get('n_lines', 0)}

    def glossary(self) -> dict:
        idx = self._index()
        return idx.glossary

    # ——— действия, разрешённые окну ———
    def open_path(self, rel: str) -> dict:
        """Открыть существующий файл или каталог внутри проекта системным способом."""
        p = resolve_inside(self.root, str(rel))
        if p is None:
            return {'ok': False, 'text': 'путь вне проекта — открывать нельзя'}
        if not p.exists():
            return {'ok': False, 'text': f'нет на диске: {rel}'}
        try:
            os.startfile(str(p))  # noqa: S606 — путь проверен resolve_inside
        except OSError as e:
            return {'ok': False, 'text': str(e)}
        return {'ok': True, 'text': f'открыто: {rel}'}

    def copy_text(self, text: str) -> dict:
        """Буфер обмена через clip.exe — без оболочки, текст только в stdin."""
        try:
            subprocess.run(['clip'], input=str(text).encode('utf-16le'),
                           timeout=5, creationflags=_NO_WINDOW, check=True)
        except (OSError, subprocess.SubprocessError) as e:
            return {'ok': False, 'text': f'не скопировано: {e}'}
        return {'ok': True, 'text': 'скопировано'}


def _launcher_module():
    """Снимок окна — той же функцией, что и у панели (без дублирования)."""
    from importlib import util as _u
    spec = _u.spec_from_file_location('panel', PROJECT_ROOT / 'Запусти меня.py')
    mod = _u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--narrow', action='store_true',
                    help='селфтест при минимальном размере окна')
    ap.add_argument('--tabs', default=','.join(TABS),
                    help='какие вкладки снимать в селфтесте')
    args = ap.parse_args()

    import webview

    class _Quiet(logging.Filter):
        def filter(self, record):
            return 'window.native' not in record.getMessage()
    logging.getLogger('pywebview').addFilter(_Quiet())

    # У служебного окна свой заголовок: иначе снимок найдёт рабочее окно.
    title = TITLE + ' · проверка' if args.selftest else TITLE
    api = HistoryApi()
    size = (960, 600) if args.narrow else (1320, 840)
    win = webview.create_window(
        title, html=HTML.read_text(encoding='utf-8'), js_api=api,
        width=size[0], height=size[1], min_size=(960, 600))

    if args.selftest:
        def snap() -> None:
            mod = _launcher_module()
            ok = api.boot_event.wait(timeout=60)
            time.sleep(2.0)
            try:
                if not ok:
                    print('SELFTEST: страница не сообщила о загрузке за 60 с')
                tabs = [t for t in args.tabs.split(',') if t in TABS]
                failures: list[str] = []
                for k, tab in enumerate(tabs):
                    win.evaluate_js(f'ui.selectTab({tab!r}); ui.selftestPrepare({tab!r})')
                    time.sleep(1.4)
                    out = (SCREENSHOT if k == 0 else
                           SCREENSHOT.with_name(f'project_history_screenshot_{tab}.png'))
                    if args.narrow:
                        out = out.with_name(out.stem + '_narrow.png')
                    mod._capture_window_by_title(title, out)
                    # снимок не доказывает, что вкладка отрисовалась: сверяем
                    # заголовок содержимого и строку статуса с ошибками JS
                    probe = win.evaluate_js(
                        "JSON.stringify({tab: S.tab, status: document.getElementById('status').textContent})")
                    bad = ('"tab":"%s"' % tab) not in (probe or '') or '"status":"JS:' in (probe or '')
                    print(f'SELFTEST {"FAIL" if bad else "OK"}: {out.relative_to(PROJECT_ROOT)} {probe}')
                    failures.append(tab) if bad else None
                if failures:
                    print(f'SELFTEST: вкладки с ошибкой: {", ".join(failures)}')
            except Exception as e:                       # noqa: BLE001
                print(f'SELFTEST: снимок не удался: {e}')
            finally:
                try:
                    win.destroy()
                except Exception:                        # noqa: BLE001
                    pass
        threading.Thread(target=snap, daemon=True).start()

    storage = PROJECT_ROOT / '.cache' / (
        'webview2_history_selftest' if args.selftest else 'webview2_history')
    storage.mkdir(parents=True, exist_ok=True)
    webview.start(storage_path=str(storage), private_mode=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
