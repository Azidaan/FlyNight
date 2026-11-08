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
