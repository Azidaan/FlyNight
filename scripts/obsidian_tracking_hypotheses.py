"""Прямо порученная ветка Трекинг: кандидаты, не результаты опыта."""
from pathlib import Path
import json,hashlib

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/'experiments/tracking_hypotheses.json'
HUB='Гипотезы — трекинг/Трекинг — память личности и надёжное сопоставление'


def load():
    if not REGISTRY.is_file():return None
    data=json.loads(REGISTRY.read_text(encoding='utf-8'))
    assert data['author_request'] and data['version']==1
    assert [m['id'] for m in data['methods']]==['T'+str(i) for i in range(1,13)]
    assert all(m['status'] in ('planned','completed') and m['limitation'] and m['check'] for m in data['methods'])
    for method in data['methods']:
        if method['status']=='completed':
            result=method['result'];path=ROOT/result['source']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==result['sha256']
            evidence=json.loads(path.read_text(encoding='utf-8'))
            assert evidence['complete'] and evidence['passed']
            assert result['card_id'] and result['verdict']
    return data


def method_path(method):return 'Гипотезы — трекинг/'+method['title']


def paths():
    data=load()
    return {HUB,*[method_path(m) for m in data['methods']]} if data else set()


def root_section():
    return f'[[{HUB}|Трекинг — кандидаты Т1–Т12]]\n\n' if load() else ''


def build(V,G):
    data=load()
    if not data:return
    review=(ROOT/data['source_review']).as_uri()
    body='# Трекинг — память личности и надёжное сопоставление\n\n↑ [[Гипотезы|Гипотезы]]\n\n'
    body+='Кандидаты для развития всей системы. Наличие статьи и исправного авторского кода не подтверждает пользу на нашем стенде.\n\n'
    body+=data['numbering']+'\n\n'
    body+='Приоритет для первой изолированной проверки — управление записью памяти с сильным простым контролем. Раздельные роли памяти и обучение различению конкурентов — следующие возможные ступени; запуск этой записью не разрешается.\n\n'
    for m in data['methods']:body+=f'[[{method_path(m)}|{m["title"]}]]\n\n'
    body+=f'[Разбор первичных источников и совместимости]({review})\n'
    V.write(HUB+'.md',G.fm({'tags':['гипотезы-узел'],'aliases':['tracking-candidates']})+body)
    for m in data['methods']:
        body=f'# {m["title"]}\n\n↑ [[{HUB}|Трекинг]]\n\n'
        complete=m['status']=='completed'
        if complete:
            result=m['result']
            body+='**Статус:** проверка завершена. Зелёный означает завершение проверки, а не подтверждение пользы.\n\n'
            body+=f'**Научный исход:** {result["verdict"]}\n\n{result["summary"]}\n\n'
            body+=f'Карточка: `{result["card_id"]}`. [Полный проверенный источник]({(ROOT/result["source"]).as_uri()})\n\n'
        else:
            body+='**Статус:** кандидат для проверки; научный результат в нашем стенде не получен.\n\n'
        for key,title in [('role','Роль'),('mechanism','Механизм'),('question','Проверяемый вопрос'),('limitation','Ограничения переноса'),('check','Нужная проверка'),('code_review','Проверка реализации')]:
            body+=f'## {title}\n\n{m[key]}\n\n'
        body+=f'[Первичная публикация]({m["paper"]})\n\n'
        if m['code']:body+=f'[Официальный код]({m["code"]})\n\n'
        body+=f'[Общий разбор]({review})\n\n'
        body+='Пользу всей системы и вклад ядра FlyVis измерять раздельно. При реализации нужны одинаковые данные и согласованные контроли, полная HOTA и парные интервалы; пороги по проверочным видео не подбирать.\n'
        V.write(method_path(m)+'.md',G.fm({'tags':['гипотезы-подход','проверка-гипотезы/завершена' if complete else 'проверка-гипотезы/не-завершена'],
            'aliases':['tracking-'+m['id']], 'родитель':HUB,'роль':m['role']})+body)
