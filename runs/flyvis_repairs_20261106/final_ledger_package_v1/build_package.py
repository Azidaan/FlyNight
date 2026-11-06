"""Prepare an exact, isolated ledger publication; do not write Git or worktree inputs."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PARENT = "f451455074126d3d4a9f082e4872186a45c3d0d5"
RUN = "runs/flyvis_repairs_20261106"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=ROOT)


def write(name, text):
    (OUT / name).write_bytes(text.encode("utf-8"))


def main():
    assert not (OUT / "manifest.json").exists()
    entries = {}
    for record in git("ls-tree", "-r", "-z", PARENT).split(b"\0"):
        if record:
            meta, path = record.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            entries[path.decode("utf-8")] = (mode, kind, oid)
    overlay_path = f"{RUN}/audit_ledger_overlay_v2/manifest.json"
    overlay_raw = (ROOT / overlay_path).read_bytes()
    assert digest(overlay_raw) == "b14444d71ee5f2a2617249857d29d5a31d9f55cd5b2d9d96ac403dd4bb848ba9"
    overlay = json.loads(overlay_raw)
    assert overlay["base_commit"] == PARENT
    assert len(overlay["selected_card_ids"]) == 4 and len(overlay["selected_journal_entries"]) == 5
    assert "58 passed" in (ROOT / f"{RUN}/catalogue_tests_after_final.log").read_text(encoding="utf-8")
    assert "OK: каталог согласован с источниками" in (ROOT / f"{RUN}/catalogue_after_final.log").read_text(encoding="utf-8")
    assert "58 passed" in (ROOT / f"{RUN}/audit_ledger_overlay_v2/tests.log").read_text(encoding="utf-8")
    targets = set((OUT / "candidate_paths.txt").read_text(encoding="utf-8").splitlines())
    assert targets and all(name.startswith((RUN + "/", "scripts/record_")) for name in targets)
    overrides = {}
    for row in overlay["files"]:
        target = row["path"]
        assert digest(git("cat-file", "blob", entries[target][2])) == row["baseline_sha256"]
        assert digest((ROOT / row["staged_candidate"]).read_bytes()) == row["after_sha256"]
        overrides[target] = row["staged_candidate"]
        targets.add(target)
    prefix = OUT.relative_to(ROOT).as_posix()
    metadata = ["build_package.py", "candidate_paths.txt", "README.txt", "commit_message.txt",
                "source_paths.txt", "artifact_paths.txt", "verification_inputs.json", "manifest.json"]
    targets.update(prefix + "/" + name for name in metadata)
    targets.add("runs/.gitattributes")
    sources = sorted(name for name in targets if name in overrides or name.startswith("scripts/"))
    artifacts = sorted(targets.difference(sources))
    assert len(sources) == 7
    assert all(name.startswith("runs/") for name in artifacts)
    write("source_paths.txt", "\n".join(sources) + "\n")
    write("artifact_paths.txt", "\n".join(artifacts) + "\n")
    attrs_original = git("cat-file", "blob", entries["runs/.gitattributes"][2])
    assert attrs_original == (ROOT / "runs/.gitattributes").read_bytes()
    old_rules = set(attrs_original.decode("utf-8").splitlines())
    rules = []
    for name in artifacts:
        relative = name[len("runs/"):]
        assert not any(char in relative for char in " \t*?[]\\")
        rule = "/" + relative + " -text"
        if rule not in old_rules:
            rules.append(rule)
    attrs_candidate = attrs_original.rstrip(b"\n") + b"\n" + ("\n".join(sorted(rules)) + "\n").encode("utf-8")
    assert attrs_candidate.startswith(attrs_original)
    (OUT / "gitattributes_candidate.txt").write_bytes(attrs_candidate)
    overrides["runs/.gitattributes"] = prefix + "/gitattributes_candidate.txt"
    write("commit_message.txt", "Record verified system integrity audit without unrelated workspace changes\n\n"
          "Add only four audit cards and five historical journal entries to the pinned baseline.\n"
          "Preserve existing working files and concurrent research through an isolated Git index.\n"
          "Validation: 747 repair tests passed, 2 explicitly skipped; 58 ledger tests passed in both working and isolated views.\n")
    write("README.txt", "1. Состав записи\n\n"
          "В журнал и каталог входят только четыре карточки и пять блоков этого аудита поверх проверенного исходного коммита. "
          "Остальные пользовательские записи остаются в рабочих файлах и не включаются этим пакетом. "
          "Канонические версии двух документов берутся из проверенной audit_ledger_overlay_v2, а не из полного рабочего дерева.\n\n"
          "2. Изоляция Git\n\n"
          "Пакет предназначен для prepare_private_ledger_commit.py: отдельный индекс, неизменяемые объекты Git, "
          "commit-tree с зафиксированным родителем и создание нового refs/codex/audit-ledger-20261107 без замены существующей ссылки. "
          "Текущая ветка, общий индекс и рабочие документы не изменяются. Полные локальные временные копии и приватный индекс не публикуются. "
          "Публикация GitLab использует только изменённые файлы; чужой исследовательский коммит не переносится.\n\n"
          "3. Сохранность доказательств\n\n"
          "Виртуальная версия runs/.gitattributes сохраняет все прежние строки и добавляет только точные правила -text для новых артефактов. "
          "Сам рабочий файл атрибутов не заменяется. Это обеспечивает сохранение байтов патчей и журналов после публикации. "
          "manifest.json хешируется внешним аргументом команды, остальные строки содержат точные SHA256, размеры и исходные объекты Git.\n\n"
          "4. Проверки и границы\n\n"
          "Штатный валидатор и 58 тестов прошли в полном рабочем каталоге и в изолированном представлении V2. "
          "Два старых предупреждения о шаблонных путях сохранены. Это запись технического результата; новой оценки HOTA нет. "
          "Идентификаторы фактического локального и удалённого коммитов и проверка публикации записываются отдельным последующим подтверждением.\n")
    verification = dict(parent=PARENT, ledger_manifest_sha256=digest(overlay_raw),
                        source_paths=sources, artifact_count=len(artifacts), total_files=len(targets),
                        working_attributes_preserved=True, attributes_old_sha256=digest(attrs_original),
                        attributes_candidate_sha256=digest(attrs_candidate), exact_rules_added=len(rules),
                        ledger_candidates={row["path"]: row["after_sha256"] for row in overlay["files"]})
    write("verification_inputs.json", json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    rows = []
    for name in sorted(targets):
        source = overrides.get(name, name)
        baseline = entries.get(name)
        mode = baseline[0] if baseline else "100644"
        assert baseline is None or baseline[1] == "blob"
        row = dict(target_path=name, input_path=source, expected_parent_blob=baseline[2] if baseline else None, mode=mode)
        if name == prefix + "/manifest.json":
            row.update(sha256=None, size=None, self_manifest=True)
        else:
            raw = (ROOT / source).read_bytes()
            if baseline:
                assert raw != git("cat-file", "blob", baseline[2]), "Unchanged payload: " + name
            row.update(sha256=digest(raw), size=len(raw))
        rows.append(row)
    manifest = dict(schema_version=1, base_commit=PARENT, new_ref="refs/codex/audit-ledger-20261107",
                    ledger_manifest=dict(path=overlay_path, sha256=digest(overlay_raw)), files=rows)
    write("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(files=len(rows), sources=len(sources), artifacts=len(artifacts),
                         manifest_sha256=digest((OUT / "manifest.json").read_bytes()),
                         message_sha256=digest((OUT / "commit_message.txt").read_bytes()))))


if __name__ == "__main__":
    main()
