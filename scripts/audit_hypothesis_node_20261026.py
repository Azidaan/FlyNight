"""Проверка изолированного узла и русских имён трёх новых гипотез."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VAULT = ROOT / 'ObsidianFlyVis'
RUN = ROOT / 'runs/flyvis_temporal_residual_20261026'


def main():
    from obsidian_checked_hypotheses import NOTE, TESTED
    history = json.loads((ROOT / 'experiments/project_history.json').read_text(encoding='utf-8'))
    path = VAULT / (NOTE + '.md')
    body = path.read_text(encoding='utf-8')
    outgoing = re.findall(r'\[\[([^\]]+)\]\]', body)
    incoming = []
    for candidate in VAULT.rglob('*.md'):
        if candidate == path:
            continue
        text = candidate.read_text(encoding='utf-8', errors='replace')
        for link in re.findall(r'\[\[([^\]]+)\]\]', text):
            target = link.split('|', 1)[0].split('#', 1)[0].removesuffix('.md')
            if target in {NOTE, 'Проверенные гипотезы'}:
                incoming.append(candidate.relative_to(VAULT).as_posix())
        # Ordinary internal Markdown links also create edges in Obsidian.
        for target in re.findall(r'\]\(([^)]+)\)', text):
            from urllib.parse import unquote
            if '://' not in target and NOTE in unquote(target):
                incoming.append(candidate.relative_to(VAULT).as_posix())
    assert not outgoing and not incoming, (outgoing, incoming)
    assert not any('://' not in target for target in re.findall(r'\]\(([^)]+)\)', body))
    expected = {c['id'] for c in history['experiments'] if c['state'] in TESTED and c.get('hypothesis') and c.get('obtained')}
    actual = set(re.findall(r'\*\*Код:\*\* `([^`]+)`', body))
    assert actual == expected
    new = [c for c in history['experiments'] if any(t in {'HY26-1', 'HY26-2', 'HY26-3'} for t in c.get('tags', []))]
    named = []
    for card in new:
        note = VAULT / '09 Запланировано' / (card['title'] + '.md')
        assert note.exists(), note
        text = note.read_text(encoding='utf-8')
        assert card['id'] in text and card['title'] in text
        assert not re.search('[A-Za-z]', note.stem), note.stem
        assert card['id'] not in actual
        named.append(note.relative_to(VAULT).as_posix())
    graph = json.loads((VAULT / '.obsidian/graph.json').read_text(encoding='utf-8'))
    assert graph.get('showOrphans') is True
    report = dict(passed=True, note=path.relative_to(ROOT).as_posix(), incoming_edges=len(incoming),
                  outgoing_edges=len(outgoing), checked_cards=len(actual), new_hypotheses_excluded=True,
                  russian_filenames=named, show_orphans=True, note_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  scope='Static generated files and graph settings; live Obsidian window not visually inspected')
    (RUN / 'graph_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    now = datetime.now(timezone(timedelta(hours=3)))
    entry = (f'**{now:%d.%m.%Y, %H:%M} MSK** — отдельный узел проверенных гипотез и русские названия Б1–Б3\n\n'
             f'*Сделано.* Генератор создал заметку `ObsidianFlyVis/{NOTE}.md`, без привязки к оглавлениям и другим узлам. '
             f'В неё входят {len(actual)} завершённых карточек с гипотезой и результатом; это число карточек, не число независимых подтверждений. '
             'Новые Б1–Б3 исключены, поскольку научно ещё не проверены. Имена трёх новых заметок, заголовки и подписи ссылок соответствуют русским названиям автора; коды сохранены в свойствах.\n\n'
             '*Проверено.* `runs/flyvis_temporal_residual_20261026/graph_audit.json`: входящих и исходящих рёбер нет, показ одиночных узлов включён, имена новых файлов без латиницы. '
             'Визуальная проверка открытого окна Obsidian не выполнялась. Изменённая до работы чужая заметка `z13_distribution_pilot_v1.md` исключена из перезаписи через `--preserve-note`.\n\n'
             '*Границы.* Это навигация по прежним источникам и техническая подготовка Б1; новых научных результатов нет. HOTA не измерялась, процент к baseline и доля потолка не определяются.\n\n')
    from register_flyvis_hypotheses_20261026 import save
    plan_path = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    plan = plan_path.read_text(encoding='utf-8')
    assert 'отдельный узел проверенных гипотез и русские названия Б1–Б3' not in plan
    save(plan_path, plan.replace('## Журнал изменений\n', '## Журнал изменений\n\n' + entry, 1))


if __name__ == '__main__':
    main()
