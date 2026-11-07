"""Прямо порученная гипотеза динамики ядра; запись не разрешает эксперимент."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'experiments/flyvis_core_dynamics_hypothesis.json'


def load():
    if not REGISTRY.is_file():
        return None
    data = json.loads(REGISTRY.read_text(encoding='utf-8'))
    assert data['version'] == 1 and data['status'] == 'planned'
    assert data['author_request'] and data['level'] == 'ядро FlyVis'
    assert data['sections'] and data['papers'] and data['authorization_scope']
    assert all((ROOT / name).is_file() for name in data['local_sources'])
    for result in data.get('diagnostic_results', []):
        source = ROOT / result['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == result['sha256']
        audit = json.loads(source.read_text(encoding='utf-8'))
        assert audit['complete'] and audit['passed']
    return data


def paths():
    data = load()
    return {'Гипотезы — ядро FlyVis/' + data['title']} if data else set()


def root_section():
    data = load()
    return f'[[{next(iter(paths()))}|{data["title"]}]]\n\n' if data else ''


def build(vault, generator):
    data = load()
    if not data:
        return
    body = f'# {data["title"]}\n\n↑ [[Гипотезы|Гипотезы]]\n\n'
    body += '**Статус:** непроверенная гипотеза; красный узел. Изменение динамики ядра не реализовано; его научная проверка не завершена.\n\n'
    body += '**Уровень изменения:** ядро FlyVis.\n\n' + data['authorization_scope'] + '\n\n'
    for section in data['sections']:
        body += '## ' + section['title'] + '\n\n' + section['text'] + '\n\n'
    for result in data.get('diagnostic_results', []):
        body += '## ' + result['title'] + '\n\n' + result['summary'] + '\n\n'
        body += f'[Проверенный источник]({(ROOT / result["source"]).as_uri()})\n\n'
        body += f'[Числа и условия решения]({(ROOT / result["analysis"]).as_uri()})\n\n'
        body += 'Завершение этого диагностического этапа не завершает проверку адаптивных временных констант.\n\n'
    body += '## Первичные источники\n\n'
    for paper in data['papers']:
        body += f'[{paper["title"]}]({paper["url"]})\n\n'
    for name in data['local_sources']:
        body += f'[{name}]({(ROOT / name).as_uri()})\n\n'
    vault.write(next(iter(paths())) + '.md', generator.fm({
        'tags': ['гипотезы-подход', 'проверка-гипотезы/не-завершена'],
        'aliases': [data['id'], 'Я1'], 'уровень': data['level'],
        'родитель': 'Гипотезы', 'роль': 'кандидат на изменение динамики ядра',
    }) + body)
