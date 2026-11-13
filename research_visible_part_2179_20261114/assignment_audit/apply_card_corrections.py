import copy
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(r'E:\Dissertation')
OUT = ROOT / 'research_visible_part_2179_20261114/assignment_audit'
PATH = ROOT / 'experiments/project_history.json'
REPORT = ROOT / 'research_memory_2179_20261113/РЕЗУЛЬТАТ.md'
raw = PATH.read_bytes()
before = json.loads(raw.decode('utf-8'))
after = copy.deepcopy(before)
cards = {c['id']: c for c in after['experiments']}
proposals = json.loads((OUT / 'suggested_corrections.json').read_text(encoding='utf-8'))
selected = [p for p in proposals['corrections'] if not p['card_id'].startswith('report:')]
assert len(selected) == 13
assert len({(p['card_id'], p['field']) for p in selected}) == 13
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
now = datetime.datetime.now().astimezone()
report_hash = hashlib.sha256(REPORT.read_bytes()).hexdigest()
changes = []
for p in selected:
    c = cards[p['card_id']]
    field = p['field']
    old = c[field]
    new = p['new']
    merged_root_addition = False
    if old != p['old']:
        assert field == 'cannot_claim'
        assert p['card_id'] in ['association_weights_2179_copy_20261112', 'history_representation_2179_20261113']
        assert old.startswith(p['old']) and 'Уточнение14.11:' in old
        # Keep the parent's full already-published caution verbatim, adding only
        # limits absent from that caution instead of overwriting or repeating it.
        if p['card_id'] == 'association_weights_2179_copy_20261112':
            new = old + ' Полная перестановка идентификаторов GT также не доказана.'
        else:
            new = old + ' Не доказаны ошибочное запоминание чужого объекта, полная перестановка GT-идентификаторов и правильность принудительной передачи 110260 номеру 2179.'
        merged_root_addition = True
    assert old != new
    revision = dict(date=now.date().isoformat(), commit=head, field=field, old=old,
                    reason='Независимая сверка 14.11: отделено назначение по прямоугольникам GT от физической принадлежности видимых пикселей случая 2179; численные результаты и состояния сохранены.')
    c.setdefault('revisions', []).append(revision)
    c[field] = new
    changes.append(dict(card_id=p['card_id'], field=field, old=old, new=new,
                        merged_root_addition=merged_root_addition, revision=revision))

# Verify every other key, including state, metrics, criteria, and the new card,
# remains byte-for-byte equivalent after JSON decoding.
reconstructed = copy.deepcopy(after)
rcards = {c['id']: c for c in reconstructed['experiments']}
for ch in reversed(changes):
    c = rcards[ch['card_id']]
    c[ch['field']] = ch['old']
    assert c['revisions'].pop() == ch['revision']
oldcards = {c['id']: c for c in before['experiments']}
for cid in {p['card_id'] for p in selected}:
    if 'revisions' not in oldcards[cid]:
        assert rcards[cid]['revisions'] == []
        del rcards[cid]['revisions']
assert reconstructed == before
text = json.dumps(after, ensure_ascii=False, indent=2) + '\n'
assert json.loads(text) == after
assert PATH.read_bytes() == raw, 'Concurrent catalog modification; abort before writing'
PATH.write_text(text, encoding='utf-8', newline='\n')
written = PATH.read_bytes()
assert json.loads(written) == after
assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == report_hash
receipt = dict(finished_at=now.isoformat(), head=head, changed_fields=len(changes),
               changed_cards=len({c['card_id'] for c in changes}),
               source='suggested_corrections.json; report proposals not applied',
               catalog_before_sha256=hashlib.sha256(raw).hexdigest(),
               catalog_after_sha256=hashlib.sha256(written).hexdigest(),
               historical_report_sha256=report_hash, historical_report_unchanged=True,
               all_other_fields_unchanged=True, original_numeric_metrics_states_and_acceptance_unchanged=True,
               visible_components_card_unchanged=True, root_cannot_claim_additions_preserved=True,
               changes=changes,
               validation='Structural and exact-field preservation checks passed; root reruns project validator and tests v2 after this edit.')
(OUT / 'card_corrections_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
journal = ('Уточнены 13 полей шести прежних карточек проверки весов, памяти и независимых случаев. '
           'Убрано необоснованное отождествление назначения рамки по GT с физической принадлежностью её видимых пикселей: '
           'предпочтение 2445 для области 110260 на 1165 само по себе не доказывает ошибочное узнавание чужой машины. '
           'Численные результаты, состояния, метрики и критерии прежних опытов сохранены; старый текст каждого изменённого поля добавлен в revisions. '
           'Ранее добавленные ограничения двух карточек сохранены. Исторический research_memory_2179_20261113/РЕЗУЛЬТАТ.md и новая карточка visible_components не изменены. '
           'Источник точной сверки: research_visible_part_2179_20261114/assignment_audit/card_corrections_receipt.json. '
           'Проверена неизменность всех остальных полей; общие проверки каталога и тесты повторяет основной агент после этой правки.\n')
(OUT / 'card_corrections_journal.txt').write_text(journal, encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='changes'}, ensure_ascii=False))
