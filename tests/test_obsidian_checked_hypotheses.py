from scripts.obsidian_checked_hypotheses import build, NOTE
from scripts.build_obsidian_vault import ref, fm, STATE_RU


def test_checked_node_is_isolated_and_excludes_unrun_or_empty_cards():
    class Vault:
        def __init__(self):
            self.notes = {}
        def write(self, name, body):
            self.notes[name] = body
    vault = Vault()
    sample = dict(id='checked', title='Проверка памяти', area='flyvis',
                  state='completed_negative', hypothesis='[[Чужой узел|Проверяем память]]',
                  obtained='Результат [источник](local.md)', cannot_claim='Не HOTA')
    history = dict(areas=[dict(id='flyvis', title='Муха')], experiments=[sample,
                   dict(sample, id='future', state='planned'), dict(sample, id='empty', obtained='')])
    count = build(vault, history, {'checked': '01 Муха/Эксперименты/checked'}, ref, fm, STATE_RU)
    text = vault.notes[NOTE + '.md']
    assert count == 1 and '[[Чужой' not in text and '[[' not in text
    assert 'obsidian://open?' in text and 'local.md' not in text
    assert '`future`' not in text and '`empty`' not in text
    assert 'отрицательный' in text and 'Не HOTA' in text
