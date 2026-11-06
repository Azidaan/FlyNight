"""Prepare HEAD-based audit-only ledger files, without writing working files or Git.

Optional --final-spec JSON names one additional card/journal/receipt artifact and
its expected id. Every included record must already exist exactly in the working
ledger. Run in a new --out directory after subsequent ledger or HEAD changes.
"""
from pathlib import Path
from datetime import datetime
import argparse
import ast
import difflib
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/flyvis_repairs_20261106'
PLAN = 'dissertation/ЕДИНЫЙ_ПЛАН.md'
CAT = 'experiments/project_history.json'
GLOSSARY = 'experiments/metrics_glossary.json'
IDS = ('flyvis_system_lifecycle_repair_20261106', 'flyvis_system_repairs_batch1_20261106',
       'flyvis_system_integrity_stable_packages_20261106')

def sha(value): return hashlib.sha256(value).hexdigest()
def git(*args): return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True).stdout
def readjson(path): return json.loads(Path(path).read_bytes())

def literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict' and not node.args:
        assert all(k.arg is not None for k in node.keywords)
        return {k.arg: literal(k.value) for k in node.keywords}
    if isinstance(node, ast.List): return [literal(n) for n in node.elts]
    if isinstance(node, ast.Tuple): return tuple(literal(n) for n in node.elts)
    return ast.literal_eval(node)

def assigned(tree, name):
    matches = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(matches) == 1, name
    return matches[0]

def lifecycle_artifacts():
    # The original recording script did not persist its card/journal separately.
    # Reconstruct literals only; never execute that mutating script.
    tree = ast.parse((ROOT / 'scripts/record_system_audit_20261106.py').read_text(encoding='utf-8'))
    card = literal(assigned(tree, 'card'))
    node = assigned(tree, 'entry')
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'encode'
    joined = node.func.value
    assert isinstance(joined, ast.JoinedStr)
    receipt = readjson(RUN / 'record_receipt.json')
    now = datetime.fromisoformat(receipt['at'])
    pieces = []
    for value in joined.values:
        if isinstance(value, ast.Constant): pieces.append(value.value)
        else:
            assert isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name) and value.value.id == 'now'
            assert value.conversion == -1 and isinstance(value.format_spec, ast.JoinedStr)
            pieces.append(format(now, ''.join(literal(n) for n in value.format_spec.values)))
    return card, ''.join(pieces).encode('utf-8'), receipt

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--final-spec')
    args = ap.parse_args()
    out = (ROOT / args.out).resolve(); out.relative_to(RUN.resolve())
    assert not out.exists(), 'Use a new directory; previous overlay evidence is immutable'
    head = git('rev-parse', 'HEAD').decode().strip()
    before = {name: (ROOT / name).read_bytes() for name in (PLAN, CAT)}
    base = {name: git('show', head + ':' + name) for name in (PLAN, CAT, GLOSSARY)}
    current = json.loads(before[CAT]); original = json.loads(base[CAT])
    own0, journal0, receipt0 = lifecycle_artifacts()
    cards = [own0, readjson(RUN / 'geometry_correction_card.json'), readjson(RUN / 'stable_packages_card.json')]
    entries = [(datetime.fromisoformat(receipt0['at']), journal0, 'reconstructed lifecycle literals'),
        (datetime.fromisoformat(readjson(RUN / 'batch1_record_receipt.json')['at']), (RUN / 'batch1_journal.txt').read_bytes(), 'batch1_journal.txt'),
        (datetime.fromisoformat(readjson(RUN / 'geometry_correction_receipt.json')['at']), (RUN / 'geometry_correction_journal.txt').read_bytes(), 'geometry_correction_journal.txt'),
        (datetime.fromisoformat(readjson(RUN / 'stable_packages_record_receipt.json')['at']), (RUN / 'stable_packages_journal.txt').read_bytes(), 'stable_packages_journal.txt')]
    selected_ids = list(IDS)
    assert [c['id'] for c in cards] == selected_ids
    if args.final_spec:
        spec_path = (ROOT / args.final_spec).resolve(); spec_path.relative_to(RUN.resolve())
        spec = readjson(spec_path)
        assert set(spec) == {'id', 'card', 'journal', 'receipt'}
        assert spec['id'] not in selected_ids and spec['id'].startswith('flyvis_system_')
        resolved = {key: (ROOT / spec[key]).resolve() for key in ('card', 'journal', 'receipt')}
        for path in resolved.values(): path.relative_to(RUN.resolve())
        card = readjson(resolved['card']); assert card['id'] == spec['id']
        cards.append(card); selected_ids.append(card['id'])
        entries.append((datetime.fromisoformat(readjson(resolved['receipt'])['at']), resolved['journal'].read_bytes(), spec['journal']))
    old_by_id = {c['id']: c for c in original['experiments']}
    live_by_id = {c['id']: c for c in current['experiments']}
    assert len(old_by_id) == len(original['experiments']) and len(live_by_id) == len(current['experiments'])
    for card in cards:
        assert card['id'] not in old_by_id, 'HEAD already contains a selected card; review ownership explicitly'
        assert live_by_id[card['id']] == card, 'Working card differs from its own saved artifact'
        assert card['metrics'] == [], 'This narrow ledger extraction must not require new glossary entries'
    # Corrections preserve each prior final field in the card's revision history.
    first_batch = readjson(RUN / 'batch1_card.json')
    corrected = cards[1]
    for field in ('interpretation', 'decision', 'cannot_claim'):
        assert any(r['field'] == field and r['old'] == first_batch[field] for r in corrected['revisions'])
    ordered = sorted(entries, key=lambda item: item[0], reverse=True)
    for _, entry, name in ordered:
        assert entry.startswith(b'\n\n**') and entry.endswith(b'\n'), name
        assert before[PLAN].count(entry) == 1, 'Journal artifact must occur exactly once: ' + name
        assert base[PLAN].count(entry) == 0
    marker = '## Журнал изменений'.encode('utf-8')
    assert base[PLAN].count(marker) == 1
    insertion = b''.join(entry for _, entry, _ in ordered)
    plan_after = base[PLAN].replace(marker, marker + insertion, 1)
    assert plan_after.replace(marker + insertion, marker, 1) == base[PLAN]

    text = base[CAT].decode('utf-8')
    start = re.search(r'"experiments"\s*:\s*\[', text).end() - 1
    _, end = json.JSONDecoder().raw_decode(text, start)
    offset = len(text[:end - 1].rstrip())
    added = ''.join(',\n' + '\n'.join('    ' + line for line in json.dumps(card, ensure_ascii=False, indent=2).splitlines()) for card in cards)
    cat_after = (text[:offset] + added + text[offset:]).encode('utf-8')
    parsed = json.loads(cat_after)
    assert parsed == dict(original, experiments=[*original['experiments'], *cards])
    assert (cat_after.decode()[:offset] + cat_after.decode()[offset + len(added):]).encode() == base[CAT]
    changed_other_ids = [key for key in old_by_id if key in live_by_id and old_by_id[key] != live_by_id[key]]
    added_other_ids = [key for key in live_by_id if key not in old_by_id and key not in selected_ids]
    removed_other_ids = [key for key in old_by_id if key not in live_by_id]

    out.mkdir(parents=True)
    files = []
    for name, data in ((PLAN, plan_after), (CAT, cat_after)):
        target = out / 'after' / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        saved = out / 'baseline' / name; saved.parent.mkdir(parents=True, exist_ok=True); saved.write_bytes(base[name])
        diff = ''.join(difflib.unified_diff(base[name].decode().splitlines(keepends=True), data.decode().splitlines(keepends=True), fromfile='a/' + name, tofile='b/' + name))
        patch = out / (Path(name).name + '.patch'); patch.write_bytes(diff.encode())
        files.append(dict(path=name, baseline_sha256=sha(base[name]), after_sha256=sha(data),
            staged_candidate=str(target.relative_to(ROOT).as_posix()), patch=str(patch.relative_to(ROOT).as_posix())))
    (out / 'baseline' / GLOSSARY).parent.mkdir(parents=True, exist_ok=True)
    (out / 'baseline' / GLOSSARY).write_bytes(base[GLOSSARY])
    (out / 'selected_cards.json').write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'selected_journal_entries.txt').write_bytes(insertion)
    assert git('rev-parse', 'HEAD').decode().strip() == head, 'HEAD changed during extraction'
    assert all((ROOT / name).read_bytes() == before[name] for name in before), 'Working ledger changed concurrently'
    result = dict(base_commit=head, selected_card_ids=selected_ids, files=files,
        selected_journal_entries=[dict(recorded_at=when.isoformat(), source=name, sha256=sha(entry)) for when, entry, name in ordered],
        counts=dict(HEAD_cards=len(original['experiments']), working_cards=len(current['experiments']), overlay_cards=len(parsed['experiments'])),
        verification=dict(all_HEAD_cards_unchanged=True, non_experiment_HEAD_fields_unchanged=True,
            HEAD_catalogue_original_bytes_recoverable=True, HEAD_plan_original_bytes_recoverable=True,
            working_ledger_unchanged=True, git_writes_performed=False),
        excluded_working_changes=dict(added_other_card_ids=added_other_ids, changed_other_card_ids=changed_other_ids,
            removed_other_card_ids=removed_other_ids, other_plan_changes='All working-plan differences except the exact selected journal insertions are excluded.'),
        working_ledger_sha256={name: sha(data) for name, data in before.items()},
        HEAD_glossary_sha256=sha(base[GLOSSARY]), script_sha256=sha(Path(__file__).read_bytes()),
        boundary='Prepared files only. No index staging, branch changes, commits or working-ledger edits. Subsequent index-only staging must verify base_commit and candidate hashes.')
    (out / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('base_commit', 'counts', 'verification', 'selected_card_ids')}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
