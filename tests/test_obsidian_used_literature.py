import json
from urllib.parse import quote

from scripts import obsidian_level_graph as graph


def test_used_literature_links_follow_renumbered_targets_without_graph_edges(tmp_path,monkeypatch):
    experiments=tmp_path/'experiments'
    experiments.mkdir()
    record=dict(id='paper',title='Проверка возврата',problem_group='Личность',method_group='Возврат',
        publication='Авторы. Название. 2010.',doi='10.example/test',url='https://example.org/paper',
        sections='Раздел 2',established='Измеряется согласованность.',problem='Проблема открыта.',
        application='Основание нашей проверки.',limitation='Согласованность не доказывает личность.',
        evidence=['runs/check/result.json'],related=['exp:test','sys:test'])
    (experiments/'useful_literature.json').write_text(json.dumps(dict(sources=[record])),encoding='utf8')
    monkeypatch.setattr(graph,'ROOT',tmp_path)
    tree=graph.Tree(tmp_path/'vault')
    tree.add('Исследования','Исследования')
    tree.add('Эксперименты','Эксперименты')
    tree.add('Проблемы мухи','Проблемы мухи')
    branch=tree.add('topic:checks','Проверки','Эксперименты')
    tree.used[branch]=99
    tree.add('exp:test','Опыт',branch,'Метод и результат.')
    tree.add('sys:test','Проблема','Проблемы мухи','Описание препятствия.')
    old_path=tree.nodes['exp:test']['path']
    graph.used_literature(tree)
    tree.save()
    new_path=tree.nodes['exp:test']['path']
    assert old_path!=new_path
    paper=tree.nodes['literature:paper']
    text=(tree.vault/paper['path']).read_text(encoding='utf8')
    assert quote(new_path[:-3],safe='') in text
    assert quote(old_path[:-3],safe='') not in text
    assert '[[' not in text
    for target in ['exp:test','sys:test']:
        body=(tree.vault/tree.nodes[target]['path']).read_text(encoding='utf8')
        assert quote(paper['path'][:-3],safe='') in body
        assert 'Литературное основание' in body


def test_literature_registry_is_optional_for_other_checkouts(tmp_path,monkeypatch):
    monkeypatch.setattr(graph,'ROOT',tmp_path)
    tree=graph.Tree(tmp_path/'vault')
    tree.add('Исследования','Исследования')
    graph.used_literature(tree)
    assert list(tree.nodes)==['Исследования']
