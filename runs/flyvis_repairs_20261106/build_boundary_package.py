"""Build a pinned payload for the final bounded repair addendum."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUN = 'runs/flyvis_repairs_20261106'
OUT = ROOT / RUN / 'boundary_package_v1'
PARENT = '2c3cb2644f66eedfa70c7820ae29070bf2738794'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', *args], cwd=ROOT)

def main():
    assert not (OUT / 'manifest.json').exists()
    gate=json.loads((ROOT/RUN/'cache_completion_20261107/publication_gate.json').read_bytes())
    assert gate['passed'] and gate['publication_tests']==60
    for name, digest in gate['source_sha256'].items():
        assert sha((ROOT/name).read_bytes())==digest, 'Validated source changed: '+name
    prefix = OUT.relative_to(ROOT).as_posix()
    baseline = {}
    for row in git('ls-tree', '-r', '-z', PARENT).split(b'\0'):
        if row:
            meta, name = row.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            baseline[name.decode('utf-8')] = (mode, kind, oid)
    targets = set((OUT / 'candidate_paths.txt').read_text(encoding='utf-8').splitlines())
    assert targets and all(targets)
    assert all(name.startswith((RUN+'/', 'tracking/', 'tests/', 'scripts/', 'evaluation/project_history/')) for name in targets)
    overlay_name = RUN + '/boundary_ledger_overlay_v2/manifest.json'
    overlay_raw = (ROOT / overlay_name).read_bytes()
    overlay = json.loads(overlay_raw)
    assert overlay['base_commit'] == PARENT and len(overlay['files']) == 2
    overrides = {}
    for row in overlay['files']:
        name = row['path']
        assert sha(git('cat-file', 'blob', baseline[name][2])) == row['baseline_sha256']
        assert sha((ROOT / row['staged_candidate']).read_bytes()) == row['after_sha256']
        assert row['after_sha256']==gate['ledger_sha256'][name]
        overrides[name] = row['staged_candidate']
        targets.add(name)
    targets.update(prefix + '/' + name for name in ('candidate_paths.txt', 'source_paths.txt',
        'artifact_paths.txt', 'commit_message.txt', 'verification_inputs.json', 'manifest.json'))
    targets.add('runs/.gitattributes')
    sources = sorted(name for name in targets if not name.startswith('runs/'))
    artifacts = sorted(targets.difference(sources))
    (OUT / 'source_paths.txt').write_text('\n'.join(sources)+'\n', encoding='utf-8')
    (OUT / 'artifact_paths.txt').write_text('\n'.join(artifacts)+'\n', encoding='utf-8')
    old_attrs = git('cat-file', 'blob', baseline['runs/.gitattributes'][2])
    existing = set(old_attrs.decode('utf-8').splitlines())
    additions = []
    for name in artifacts:
        relative = name[len('runs/'):]
        assert not any(c in relative for c in ' \t*?[]\\')
        rule = '/' + relative + ' -text'
        if rule not in existing:
            additions.append(rule)
    attrs = old_attrs + ('' if old_attrs.endswith(b'\n') else '\n').encode() + ('\n'.join(sorted(additions))+'\n').encode()
    assert attrs.startswith(old_attrs)
    candidate = OUT / 'gitattributes_candidate.txt'
    candidate.write_bytes(attrs)
    overrides['runs/.gitattributes'] = candidate.relative_to(ROOT).as_posix()
    report = dict(parent=PARENT, source_paths=sources, files=len(targets),
        old_attributes_sha256=sha(old_attrs), candidate_attributes_sha256=sha(attrs),
        exact_attribute_rules_added=len(additions), shared_working_attributes_not_written=True,
        ledger_overlay_sha256=sha(overlay_raw))
    (OUT/'verification_inputs.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = []
    for name in sorted(targets):
        source = overrides.get(name, name)
        previous = baseline.get(name)
        assert previous is None or previous[1] == 'blob'
        row = dict(target_path=name, input_path=source, mode=previous[0] if previous else '100644',
                   expected_parent_blob=previous[2] if previous else None)
        if name == prefix + '/manifest.json':
            assert previous is None
            row.update(sha256=None, size=None, self_manifest=True)
        else:
            raw = (ROOT/source).read_bytes()
            assert previous is None or raw != git('cat-file', 'blob', previous[2]), 'Unchanged: ' + name
            row.update(sha256=sha(raw), size=len(raw))
        rows.append(row)
    manifest = dict(schema_version=1, base_commit=PARENT, new_ref='refs/codex/audit-boundary-20261107',
                    ledger_manifest=dict(path=overlay_name, sha256=sha(overlay_raw)), files=rows)
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    (OUT/'manifest.json').write_bytes(raw)
    print(json.dumps(dict(files=len(rows), sources=len(sources), manifest_sha256=sha(raw),
                         message_sha256=sha((OUT/'commit_message.txt').read_bytes()))))

if __name__ == '__main__':
    main()
