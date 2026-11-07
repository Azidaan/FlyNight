"""Отдельная карта проблем всего проекта, построенная по проверяемым источникам."""
from pathlib import Path
from urllib.parse import quote
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
HUB = 'Обзор проекта/Проблемы'
FOLDER = 'Проблемы проекта'
STATUS = {'open': 'открыта', 'partial': 'частично решена', 'resolved': 'решена'}
START = '<!-- SYSTEM_PROBLEM_LINKS_START -->'
END = '<!-- SYSTEM_PROBLEM_LINKS_END -->'


def load():
    data = json.loads((ROOT/'experiments/system_problems.json').read_text(encoding='utf-8'))
    ids = [p['id'] for p in data['problems']]
    assert len(ids) == len(set(ids))
    assert all(p['status'] in STATUS for p in data['problems'])
    for item in [*data['problems'], *data['patterns']]:
        for source in item['evidence']:
            path = (ROOT/source['source']).resolve()
            assert path.is_relative_to(ROOT) and path.is_file(), source
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], source
    for pattern in data['patterns']:
        assert set(pattern['problems']).issubset(ids)
        assert len({e['source'] for e in pattern['evidence']}) >= 2
    return data


def problem_path(item):
    return FOLDER+'/'+item['id']+' — '+item['title']


def pattern_path(item):
    return FOLDER+'/Повторения/'+item['title']


def paths(data=None):
    data = load() if data is None else data
    return {HUB, *(problem_path(p) for p in data['problems']),
            *(pattern_path(p) for p in data['patterns'])}


def stealth(path, label):
    return '[↗ '+label+'](obsidian://open?vault=ObsidianFlyVis&file='+quote(path, safe='')+')'


def link(path, label):
    return '[['+path+'|'+label+']]\n\n'


def sources(item):
    return '\n\n'.join('['+e['claim']+']('+ (ROOT/e['source']).as_uri()+')' for e in item['evidence'])+'\n\n'


def build(V, G, history, card_paths):
    from scripts.obsidian_fly_problems import PROBLEMS
    data = load()
    by_id = {p['id']: p for p in data['problems']}
    backlinks = {}

    def write(path, body, tags, alias=None):
        meta = '---\ntags: ['+', '.join(tags)+']\n'
        if alias:
            meta += 'aliases: ['+alias+']\n'
        V.write(path+'.md', meta+'---\n\n'+body)

    for p in data['problems']:
        path = problem_path(p)
        body = '# '+p['title']+'\n\n'+link(HUB, 'Проблемы')
        body += 'Статус: **'+STATUS[p['status']]+'**. Компонент: '+p['component']+'.\n\n'
        body += '## 1. Что установлено\n\n'+p['fact']+'\n\n'
        body += '## 2. Рабочие объяснения\n\n'+'\n\n'.join(p['hypotheses'])+'\n\n'
        body += '## 3. Следующая различающая проверка\n\n'+p['next_check']+'\n\n'
        body += '## 4. Что позволит считать проблему решённой\n\n'+p['closure']+'\n\n'
        if p.get('limitations'):
            body += '## Ограничения вывода\n\n'+p['limitations']+'\n\n'
        related = [x for x in data['patterns'] if p['id'] in x['problems']]
        if related:
            body += '## 5. Повторяющиеся сочетания\n\n'
            for pattern in related:
                body += link(pattern_path(pattern), pattern['title'])
        body += '## 6. Основания и история случаев\n\n'+sources(p)
        if len(p['cases'])>1:
            body += 'Последние содержательные случаи:\n\n'
            for case in p['cases'][-3:]:
                body += case['date']+' — '+case['fact']+'\n\n'
        for card in p['cards']:
            target = card_paths.get(card)
            if target:
                body += stealth(target, 'Связанный результат или постановка')+'\n\n'
                backlinks.setdefault(target, []).append((path, p['title']))
        for ident in p['legacy']:
            target = '10 Проблемы мухи/'+ident+' — '+PROBLEMS[ident]['title']
            body += stealth(target, 'Прежняя запись '+ident+' — статус не пересматривался')+'\n\n'
        body += 'Внесено в карту '+p['first_seen']+'. История случаев и контрольные суммы источников: [реестр проблем]('+ (ROOT/'experiments/system_problems.json').as_uri()+').\n'
        write(path, body, ['проблема-проекта', 'проблема/'+p['status']], p['id'])

    for p in data['patterns']:
        body = '# '+p['title']+'\n\n'+link(HUB, 'Проблемы')
        body += p['status']+'\n\n'+p['observation']+'\n\n'
        body += '## 1. Связанные проблемы\n\n'
        for ident in p['problems']:
            body += link(problem_path(by_id[ident]), by_id[ident]['title'])
        body += '## 2. Как проверить общее объяснение\n\n'+p['next_check']+'\n\n'
        body += '## 3. Первичные основания\n\n'+sources(p)
        write(pattern_path(p), body, ['повторение-проблем'], p['id'])

    body = '# Проблемы\n\nОтдельная карта проблем всего проекта мухи: ядро FlyVis, полный зрительный модуль и остальные компоненты различаются. Связи наружу открываются стелс-переходами без рёбер между группами.\n\n'
    body += '## 1. Нерешённые вопросы\n\n'
    for p in data['problems']:
        body += link(problem_path(p), p['title'])+'Статус: '+STATUS[p['status']]+'.\n\n'
    body += '## 2. Повторяющиеся сочетания\n\nПовторение результата ещё не устанавливает общую причину.\n\n'
    for p in data['patterns']:
        body += link(pattern_path(p), p['title'])
    body += '## 3. Как ведётся карта\n\nНовый случай добавляется в историю существующей проблемы с первичным источником. Отдельный узел нужен для содержательной повторяемой проблемы, а не для каждого отказа запуска. Факт, рабочее объяснение, проверка и условие решения записываются отдельно.\n\nКрасный — открытая проблема, жёлтый — частично решённая, зелёный — решённая; фиолетовый — наблюдаемое повторение без установленной общей причины. Это статусы решения проблемы. У дерева гипотез цвет по-прежнему означает завершённость проверки.\n\nПрежние P1–P11 сохранены без изменений и доступны из связанных записей. Они скрыты из обзора; файлы не удаляются и не переносятся. Реестр: [system_problems.json]('+ (ROOT/'experiments/system_problems.json').as_uri()+').\n'
    write(HUB, body, ['проблемы-узел'])

    # Обратные переходы добавляются только в уже выбранные опорные задачи.
    # Защищённое дерево гипотез и старые P1–P11 не переписываются.
    for target, links in backlinks.items():
        name = target+'.md'
        if name not in V.written:
            continue
        file = V.path/name
        text = file.read_text(encoding='utf-8')
        if START in text:
            before, tail = text.split(START, 1)
            text = before+tail.split(END, 1)[1]
        section = '\n\n'+START+'\n\n## Связанные проблемы\n\n'
        section += '\n\n'.join(stealth(path, title) for path, title in sorted(set(links)))
        V.write(name, text.rstrip()+section+'\n\n'+END+'\n')
    return paths(data)
