"""Prepare a pinned commit with a PRIVATE index and a new non-branch ref.

Default is read-only. Only the root agent may run --apply after reviewing the
final V2 candidates, package manifests and commit message. Shared worktree,
index, HEAD, refs/heads and configuration are never written by this helper.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/flyvis_repairs_20261106"
PARENT = "f451455074126d3d4a9f082e4872186a45c3d0d5"
NEW_REF = "refs/codex/audit-ledger-20261107"
LEDGERS = {"dissertation/ЕДИНЫЙ_ПЛАН.md", "experiments/project_history.json"}
EXTRA_PACKAGE_PATHS = {
    "scripts/record_system_audit_20261106.py",
    "scripts/record_system_repairs_batch1_20261106.py",
    "scripts/record_geometry_audit_correction_20261106.py",
    "scripts/record_stable_audit_packages_20261106.py",
    "scripts/record_system_integrity_final_20261106.py",
    "runs/.gitattributes",
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local(relative):
    require(isinstance(relative, str) and "\\" not in relative, "Use repository-relative POSIX paths")
    name = PurePosixPath(relative)
    require(not name.is_absolute() and str(name) == relative and ".." not in name.parts,
            "Noncanonical relative path: " + relative)
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def under_run(relative):
    path = local(relative)
    path.relative_to(RUN.resolve())
    return path


class Git:
    def __init__(self, private_index):
        forbidden = [key for key in ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES") if os.environ.get(key)]
        require(not forbidden, "Unexpected Git environment overrides: " + ", ".join(forbidden))
        self.plain_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        self.private_env = dict(self.plain_env, GIT_INDEX_FILE=str(private_index))

    def run(self, *args, data=None, shared_read=False, allow_missing=False):
        # shared_read is used only by the fixed read-only snapshot() calls.
        if shared_read:
            require(args[0] in {"rev-parse", "for-each-ref"}, "Shared Git command must be read-only")
        process = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, input=data,
            capture_output=True, env=self.plain_env if shared_read else self.private_env)
        if allow_missing and process.returncode == 1:
            return None
        require(process.returncode == 0, f"Git {args[0]} failed ({process.returncode}): "
                + process.stderr.decode("utf-8", errors="replace"))
        return process.stdout


def tree_entries(git, tree):
    result = {}
    for row in git.run("ls-tree", "-r", "-z", tree).split(b"\0"):
        if row:
            metadata, name = row.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split()
            require(kind in {"blob", "commit"}, "Unexpected recursive tree entry")
            result[name.decode("utf-8")] = (mode, oid)
    return result


def snapshot(git):
    def file_hash(path):
        return sha(path.read_bytes()) if path.is_file() else None
    index = Path(git.run("rev-parse", "--git-path", "index", shared_read=True).decode().strip())
    if not index.is_absolute():
        index = ROOT / index
    config = Path(git.run("rev-parse", "--git-path", "config", shared_read=True).decode().strip())
    if not config.is_absolute():
        config = ROOT / config
    return dict(head=git.run("rev-parse", "HEAD", shared_read=True).decode().strip(),
        heads_sha256=sha(git.run("for-each-ref", "--format=%(refname) %(objectname)", "refs/heads", shared_read=True)),
        shared_index_sha256=file_hash(index), config_sha256=file_hash(config),
        working_ledger_sha256={name: file_hash(ROOT / name) for name in sorted(LEDGERS)})


def plan(args, git):
    require(git.run("rev-parse", PARENT + "^{commit}").decode().strip() == PARENT, "Pinned parent is unavailable")
    require(git.run("rev-parse", "--verify", "--quiet", NEW_REF, allow_missing=True) is None,
            "Reserved new ref already exists; never overwrite it")
    require(git.run("symbolic-ref", "--quiet", NEW_REF, allow_missing=True) is None,
            "Reserved ref already exists as a symbolic ref")
    root = Path(git.run("rev-parse", "--show-toplevel").decode().strip()).resolve()
    require(root == ROOT.resolve(), "Unexpected repository root")
    baseline = tree_entries(git, PARENT)
    pins, rows = [], []

    def pinned(relative, digest):
        path = under_run(relative)
        raw = path.read_bytes()
        require(sha(raw) == digest, "Pinned input hash mismatch: " + relative)
        pins.append((path, raw))
        return raw

    manifest_raw = pinned(args.manifest, args.manifest_sha256)
    package = json.loads(manifest_raw)
    require(package["schema_version"] == 1 and package["base_commit"] == PARENT
            and package["new_ref"] == NEW_REF, "Unexpected package schema, parent or new ref")
    ledger_spec = package["ledger_manifest"]
    ledger = json.loads(pinned(ledger_spec["path"], ledger_spec["sha256"]))
    require(ledger["base_commit"] == PARENT, "Ledger overlay uses a different parent")
    require(len(ledger["files"]) == 2 and {row["path"] for row in ledger["files"]} == LEDGERS,
            "Ledger manifest must name exactly the two approved paths")
    ledger_rows = {row["path"]: row for row in ledger["files"]}
    files = package["files"]
    require(isinstance(files, list) and files, "Package files[] is required")
    self_rows = 0
    for item in files:
        name, source = item["target_path"], item["input_path"]
        local(name)  # Validate canonical target spelling even for a virtual input.
        allowed = name.startswith("runs/flyvis_repairs_20261106/") or name in LEDGERS | EXTRA_PACKAGE_PATHS
        require(allowed, "Publication path outside approved audit scope: " + name)
        require(not any(part in {"private_ledger_commit", "private_ledger_backups"}
                        for part in PurePosixPath(name).parts + PurePosixPath(source).parts),
                "Private recovery files must never be published")
        path = local(source)
        if source not in EXTRA_PACKAGE_PATHS:
            path.relative_to(RUN.resolve())
        if name == "runs/.gitattributes":
            require(source != name, "Attributes must use the separately reviewed virtual candidate")
            path.relative_to(RUN.resolve())
        require("expected_parent_blob" in item, "Missing explicit parent guard: " + name)
        expected_parent_blob = item["expected_parent_blob"]
        mode = item["mode"]
        require(mode in {"100644", "100755"}, "Publication requires a regular file: " + name)
        if expected_parent_blob is None:
            require(name not in baseline, "Expected new path already exists in parent: " + name)
        else:
            require(name in baseline and baseline[name] == (mode, expected_parent_blob),
                    "Expected parent blob or mode mismatch: " + name)
            require(git.run("cat-file", "-t", expected_parent_blob).strip() == b"blob", "Parent is not a blob")
        if item.get("self_manifest", False):
            self_rows += 1
            require(name == source == args.manifest and item["sha256"] is None and item["size"] is None
                    and expected_parent_blob is None and mode == "100644", "Invalid self-manifest row")
            raw = manifest_raw  # Exact bytes are pinned by the independent CLI SHA256.
        else:
            raw = path.read_bytes()
            require(sha(raw) == item["sha256"], "Publication input SHA256 mismatch: " + name)
            size = item["size"]
            require(isinstance(size, int) and not isinstance(size, bool) and len(raw) == size,
                    "Publication input byte size mismatch: " + name)
        if name in LEDGERS:
            row = ledger_rows[name]
            require(source == row["staged_candidate"] and sha(raw) == row["after_sha256"],
                    "Ledger input is not the reviewed V2 candidate: " + name)
            require(name in baseline and sha(git.run("cat-file", "blob", baseline[name][1])) == row["baseline_sha256"],
                    "Ledger baseline differs from V2: " + name)
        pins.append((path, raw))
        rows.append(dict(path=name, mode=mode, bytes=raw, source=source))
    require(self_rows == 1, "Exactly one externally pinned self-manifest row is required")
    require(LEDGERS.issubset({row["path"] for row in rows}), "Both reviewed ledger candidates must be included")
    require(len({row["path"] for row in rows}) == len(rows), "Duplicate publication path")
    message = pinned(args.message_file, args.message_sha256)
    require(bool(message.strip()) and b"\0" not in message, "Commit message is empty or contains NUL")
    for row in rows:
        row["oid"] = git.run("hash-object", "-t", "blob", "--no-filters", "--stdin", data=row["bytes"]).decode().strip()
    return dict(rows=rows, pins=pins, baseline=baseline, message=message)


def apply(args, git, private_dir, prepared, before):
    private_dir.mkdir(parents=True, exist_ok=False)
    receipt = dict(at=datetime.now(timezone.utc).isoformat(), parent=PARENT, new_ref=NEW_REF,
        private_directory=str(private_dir), shared_before=before, completed=False,
        publish_private_directory=False, commit=None, ref_creation_attempted=False,
        files=[dict(path=r["path"], mode=r["mode"], oid=r["oid"], sha256=sha(r["bytes"]), bytes=len(r["bytes"])) for r in prepared["rows"]])
    receipt_path = private_dir / "receipt.json"
    def save():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    def pinned_unchanged():
        require(all(path.read_bytes() == raw for path, raw in prepared["pins"]), "A pinned input changed; stop without publishing a ref")
    try:
        save()
        git.run("read-tree", PARENT)  # No -u: private index only, never checkout.
        expected = dict(prepared["baseline"])
        for row in prepared["rows"]:
            oid = git.run("hash-object", "-t", "blob", "-w", "--no-filters", "--stdin", data=row["bytes"]).decode().strip()
            require(oid == row["oid"] and git.run("cat-file", "blob", oid) == row["bytes"], "Written blob bytes differ")
            git.run("update-index", "--add", "--cacheinfo", row["mode"], oid, row["path"])
            expected[row["path"]] = (row["mode"], oid)
        tree = git.run("write-tree").decode().strip()
        require(tree_entries(git, tree) == expected, "Private tree differs from exact allowlist overlay")
        for row in prepared["rows"]:
            require(git.run("cat-file", "blob", tree + ":" + row["path"]) == row["bytes"], "Tree bytes differ")
        pinned_unchanged()
        message_path = private_dir / "commit_message.txt"
        message_path.write_bytes(prepared["message"])
        commit = git.run("commit-tree", tree, "-p", PARENT, "-F", str(message_path)).decode().strip()
        require(git.run("rev-parse", commit + "^@").decode().split() == [PARENT], "Unexpected commit parent")
        require(git.run("rev-parse", commit + "^{tree}").decode().strip() == tree, "Unexpected commit tree")
        receipt.update(commit=commit, tree=tree)
        save()
        pinned_unchanged()
        receipt["ref_creation_attempted"] = True
        save()
        git.run("update-ref", "--no-deref", NEW_REF, commit, "0" * 40)  # CAS create only; never refs/heads.
        require(git.run("rev-parse", "--verify", NEW_REF).decode().strip() == commit, "New ref verification failed")
        receipt["completed"] = True
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["automatic_rollback_performed"] = False
        receipt["instruction"] = "Inspect this private index/receipt and the new ref. Do not reset or replace shared Git state. Unreferenced created objects are harmless; do not delete other actors' refs."
        raise
    finally:
        receipt["shared_after"] = snapshot(git)
        receipt["shared_snapshot_unchanged"] = receipt["shared_after"] == before
        receipt["concurrency_note"] = "The helper issued no writes to shared index/HEAD/heads/config/worktree. Independent actors may legitimately change the recorded snapshots; such drift is reported, not reverted."
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--message-file", required=True)
    parser.add_argument("--message-sha256", required=True)
    parser.add_argument("--private-dir", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    private_dir = under_run(args.private_dir)
    private_root = (RUN / "private_ledger_commit").resolve()
    private_dir.relative_to(private_root)
    require(private_dir != private_root and not private_dir.exists(), "Use a new private child directory")
    git = Git(private_dir / "index")
    before = snapshot(git)
    prepared = plan(args, git)
    if args.apply:
        result = apply(args, git, private_dir, prepared, before)
    else:
        result = dict(mode="read_only_preflight", passed=True, parent=PARENT, new_ref=NEW_REF,
            private_directory_created=False, shared_before=before, shared_after=snapshot(git),
            files=[dict(path=r["path"], oid=r["oid"], sha256=sha(r["bytes"]), bytes=len(r["bytes"])) for r in prepared["rows"]])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
