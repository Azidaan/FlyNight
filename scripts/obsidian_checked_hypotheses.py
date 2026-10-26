"""Один изолированный узел проверенных гипотез; переходы не создают рёбер."""
from __future__ import annotations

import re

NOTE = 'Проверенные гипотезы — результаты и ограничения'
TESTED = {'completed_positive', 'completed_negative', 'inconclusive'}


def plain(value):
    # Source text may itself contain wiki-links; render them as plain labels.
    text = re.sub(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', lambda m: m[2] or m[1], str(value))
    return re.sub(r'\[([^\]]+)\]\((?!https?://|obsidian://)[^)]+\)', r'\1', text)


def build(vault, history, card_paths, ref, fm, states):
    cards = [c for c in history['experiments'] if c['state'] in TESTED
             and c.get('hypothesis') and c.get('obtained')]
    areas = {a['id']: a['title'] for a in history['areas']}
    body = (f'# {NOTE}\n\n'
            'Это отдельный узел без структурных связей. Здесь только карточки с завершённой проверкой '
            'и записанным результатом. Положительный исход относится к критерию своей карточки: '
            'он не означает автоматически улучшения HOTA или всей FlyVis. Неопределённый исход '
            'сохранён отдельно от отрицательного. Планы, инфраструктура и закрытия без измерения не включены.\n\n'
            'Переходы ↗ открывают подробности и не создают рёбер графа. Источник — '
            '`experiments/project_history.json`; первичные результаты указаны в каждой карточке.\n\n')
    for area in history['areas']:
        group = [c for c in cards if c['area'] == area['id']]
        if not group:
            continue
        body += f"## {areas[area['id']]}\n\n"
        for c in sorted(group, key=lambda c: (c.get('started', ''), c['id']), reverse=True):
            body += f"### {plain(c['title'])}\n\n"
            body += f"**Исход проверки:** {states[c['state']]}. **Код:** `{c['id']}`.\n\n"
            for field, title in [('hypothesis', 'Проверяли'), ('obtained', 'Получено'),
                                 ('interpretation', 'Смысл результата'), ('cannot_claim', 'Границы вывода')]:
                if c.get(field):
                    body += f'**{title}.** {plain(c[field])}\n\n'
            body += ref('', card_paths[c['id']], 'Карточка, критерии и первичные источники') + '\n\n'
    vault.write(NOTE + '.md', fm({'aliases': ['Проверенные гипотезы'],
                                'tags': ['проверенные-гипотезы'], 'проверок': len(cards)}) + body)
    return len(cards)
