from scripts.obsidian_level_graph import ACTIVE, Tree


def test_new_sibling_keeps_existing_numbers_and_renamed_note_survives(tmp_path):
    def generate(children):
        tree = Tree(tmp_path)
        tree.add('Термины', 'Термины')
        for key, title in children:
            tree.add(key, title, 'Термины')
        tree.save()
        return tree.nodes

    old = generate([('a', 'А'), ('b', 'Б')])
    new = generate([('z', 'Добавление'), ('b', 'Новое название'), ('a', 'А')])
    assert new['z']['number'] == '3.3'
    assert new['b']['number'] == old['b']['number']
    assert new['a']['number'] == old['a']['number']
    assert ACTIVE not in (tmp_path / old['b']['path']).read_text(encoding='utf8')
    assert ACTIVE in (tmp_path / new['b']['path']).read_text(encoding='utf8')


def test_original_keeps_method_limitations_and_external_sources(tmp_path):
    from scripts.obsidian_level_graph import original
    (tmp_path/'Исходник.md').write_text(
        '---\ntags: [архив]\n---\n# Исходник\n\n↑ [[Родитель]]\n\n'
        '**Статус:** кандидат.\n\n## Механизм\nСравниваем историю цели.\n\n'
        '## Ограничения\nНет готовых весов.\n\n[Статья](https://example.org/paper)\n',
        encoding='utf8')
    text=original(tmp_path,'Исходник')
    assert 'Сравниваем историю цели.' in text and 'Нет готовых весов.' in text
    assert '[Статья](https://example.org/paper)' in text
    assert '[[' not in text


def test_color_distinguishes_failed_finished_and_unfinished_work():
    from scripts.obsidian_level_graph import card_color
    assert card_color({'state':'completed_positive'})=='выполнено'
    for state in ['completed_negative','inconclusive','closed']:
        assert card_color({'state':state,'done':'2026-10-01'})=='без-положительного-итога'
    assert card_color({'state':'planned'})=='ожидает'
    assert card_color({'state':'running'})=='в-работе'
    assert card_color({'state':'infrastructure'}) is None


def test_semantic_branch_has_one_owner_and_is_reused(tmp_path):
    from scripts.obsidian_level_graph import semantic_parent
    tree=Tree(tmp_path)
    tree.add('Исследования','Исследования')
    first=semantic_parent(tree,'Исследования',['Память','Внешний вид'])
    second=semantic_parent(tree,'Исследования',['Память','Внешний вид'])
    assert first==second
    assert tree.nodes[first]['number']=='6.1.1'
    assert tree.nodes[tree.nodes[first]['parent']]['title']=='Память'


def test_hypothesis_week_uses_month_and_monday_boundaries():
    from datetime import date
    from scripts.obsidian_level_graph import hypothesis_week
    assert hypothesis_week(date(2026,11,1))==(1,date(2026,11,1),date(2026,11,1))
    assert hypothesis_week(date(2026,11,8))==(2,date(2026,11,2),date(2026,11,8))
    assert hypothesis_week(date(2026,11,9))==(3,date(2026,11,9),date(2026,11,15))
    assert hypothesis_week(date(2026,10,31))==(5,date(2026,10,26),date(2026,10,31))
