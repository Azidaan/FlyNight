from scripts.obsidian_checked_hypotheses import build, NOTE, HUB_NOTE
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
    hub = vault.notes[HUB_NOTE + '.md']
    assert '# Гипотезы\n' in hub and '[[' not in hub
    assert 'obsidian://open?' in hub


def test_hypotheses_hub_has_three_rays_and_no_link_to_checked_summary():
    import re
    class Vault:
        def __init__(self):
            self.notes = {}
        def write(self, name, body):
            self.notes[name] = body
    vault = Vault()
    cards = [dict(id=f'b{i}', title=f'Б{i} — Проверка памяти', area='flyvis', state='planned',
                  hypothesis='Рабочая гипотеза', tags=[f'HY26-{i}']) for i in (1, 2, 3)]
    paths = {c['id']: '09 Запланировано/' + c['title'] for c in cards}
    build(vault, dict(areas=[dict(id='flyvis', title='Муха')], experiments=cards), paths, ref, fm, STATE_RU)
    links = re.findall(r'\[\[([^|\]]+)', vault.notes[HUB_NOTE + '.md'])
    assert set(links) == set(paths.values()) and len(links) == 3


def test_other_groups_cannot_attach_hypothesis_branches():
    from scripts.build_obsidian_vault import HYPOTHESIS_PATHS
    path = '09 Запланировано/Б1 — Проверка памяти'
    previous = HYPOTHESIS_PATHS.copy()
    try:
        HYPOTHESIS_PATHS.add(path)
        assert '[[' not in ref('09', path, 'Б1', structural=True)
        assert 'obsidian://open?' in ref('09', path, 'Б1', structural=True)
    finally:
        HYPOTHESIS_PATHS.clear()
        HYPOTHESIS_PATHS.update(previous)


def test_verification_color_means_finished_not_positive_and_needs_evidence():
    from scripts.obsidian_checked_hypotheses import verification_tag
    base = dict(hypothesis='Проверяем память', obtained='Измеренный результат', evidence=[{'ref': 'result.json'}])
    for state in ('completed_positive', 'completed_negative', 'inconclusive', 'closed', 'superseded'):
        assert verification_tag(dict(base, state=state)) == 'проверка-гипотезы/завершена'
    for state in ('planned', 'ready', 'running', 'unknown', 'infrastructure'):
        assert verification_tag(dict(base, state=state)) == 'проверка-гипотезы/не-завершена'
    assert verification_tag(dict(base, state='completed_negative', obtained='')) == 'проверка-гипотезы/не-завершена'
    assert verification_tag(dict(base, state='completed_positive', evidence=[])) == 'проверка-гипотезы/не-завершена'
    assert verification_tag({'state': 'planned'}) is None
