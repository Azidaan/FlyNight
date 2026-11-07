"""Отдельная группа Б1–Б3 и сводка завершённых проверок."""
from __future__ import annotations

import re

NOTE = 'Проверенные гипотезы — результаты и ограничения'
HUB_NOTE = 'Гипотезы'
TESTED = {'completed_positive', 'completed_negative', 'inconclusive'}
HYPOTHESIS_TAGS = {'HY26-1', 'HY26-2', 'HY26-3'}


def is_branch(card):
    return bool(HYPOTHESIS_TAGS.intersection(card.get('tags', [])))


def verification_tag(card):
    """Цвет обозначает завершённую проверку, включая отрицательную."""
    if not card.get('hypothesis'):
        return None
    measured = (card.get('state') in TESTED | {'closed', 'superseded'}
                and bool(card.get('obtained')) and bool(card.get('evidence')))
    return 'проверка-гипотезы/завершена' if measured else 'проверка-гипотезы/не-завершена'


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
                                'tags': ['проверенные-гипотезы', 'проверка-гипотезы/завершена'], 'проверок': len(cards)}) + body)
    hub = (f'# {HUB_NOTE}\n\n'
           'Центр отдельной группы: три луча ведут к Б1, Б2 и Б3. '
           'С остальными группами графа структурных связей нет.\n\n'
           '## Новые гипотезы\n\n')
    for card in history['experiments']:
        if not is_branch(card):
            continue
        hub += f"### {plain(card['title'])}\n\n"
        hub += f"**Статус:** {states[card['state']]}.\n\n{plain(card['hypothesis'])}\n\n"
        hub += f"[[{card_paths[card['id']]}|{plain(card['title'])}]]\n\n"
    hub += '## Проверенные гипотезы\n\n'
    hub += ref('', NOTE, 'Завершённые проверки, результаты и ограничения') + '\n'
    vault.write(HUB_NOTE + '.md', fm({'tags': ['гипотезы-узел']}) + hub)
    return len(cards)
