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
