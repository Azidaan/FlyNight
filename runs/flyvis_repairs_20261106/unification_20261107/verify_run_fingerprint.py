"""Verify a saved run from its source archive without modifying its original receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(stream):
    hasher = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return digest(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def changed_files(base, expected):
    changes = []
    for name, checksum in expected.items():
        path = (base / name).resolve()
        path.relative_to(base.resolve())
        if not path.is_file() or file_digest(path) != checksum:
            changes.append(name)
    return changes


def verify(out, expected_sha256=None):
    manifest_path = out / "run_checksums.json"
    checksum = file_digest(manifest_path)
    sidecar = (out / "run_checksums.sha256").read_text(encoding="ascii").split()
    require(sidecar == [checksum, manifest_path.name], "Manifest digest or filename differs")
    if expected_sha256 is not None:
        require(checksum == expected_sha256.lower(), "Manifest differs from the supplied digest")

    sums = json.loads(manifest_path.read_bytes())
    protocol = json.loads((out / "run_protocol.json").read_bytes())
    require(sums["schema"] == protocol["schema"] == "flynight-run-fingerprint-v1", "Unknown schema")
    require(sums["run_id"] == protocol["run_id"] == out.name, "Run ID differs")
    require(sums["branch"] == protocol["branch"] and sums["commit"] == protocol["commit"], "Revision differs")
    require(sums["passed"] and sums["sources_unchanged"] and sums["inputs_unchanged"], "Run failed")
    sources, inputs = protocol["source_sha256"], protocol["input_sha256"]
    outputs = sums["outputs_sha256"]
    require(sums["source_count"] == len(sources) > 0 and sums["input_count"] == len(inputs) > 0,
            "File counts differ")
    result_file = protocol.get("result_file", "pipeline_real_clip.json")
    require({"frozen_sources.zip", "run_protocol.json", result_file} <= outputs.keys(),
            "Required outputs missing from manifest")
    require(protocol["source_archive_sha256"] == outputs["frozen_sources.zip"], "Archive digests differ")
    for kind, base, files in (("outputs", out, outputs), ("inputs", ROOT, inputs)):
        changed = changed_files(base, files)
        require(not changed, f"Missing or changed {kind}: {changed}")

    with zipfile.ZipFile(out / "frozen_sources.zip") as archive:
        require(sorted(archive.namelist()) == sorted(sources), "Archived source list differs")
        for name, checksum_source in sources.items():
            with archive.open(name) as stream:
                require(digest(stream) == checksum_source, f"Archived source differs: {name}")

    guard = json.loads((out / "memory_guard.json").read_bytes())
    require(guard["returncode"] == 0 and guard["reason"] is None and not guard["stop_errors"],
            "Memory supervisor did not record normal completion")
    require(json.loads((out / result_file).read_bytes())["passed"], "Pipeline failed")
    require(changed_files(out, {"run_protocol.json": "0" * 64}) == ["run_protocol.json"],
            "Negative digest control failed")
    commit = protocol["commit"]
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Invalid recorded commit")
    tree = subprocess.check_output(["git", "rev-parse", "--verify", commit + "^{tree}"],
                                   cwd=ROOT, text=True).strip()
    require(tree == protocol["tree"], "Recorded Git tree differs")
    drift = changed_files(ROOT, sources)
    return dict(passed=True, run_id=out.name, run_commit=commit, manifest_sha256=checksum,
                sources_checked=len(sources), inputs_checked=len(inputs), outputs_checked=len(outputs),
                source_archive_checked=True, current_sources_match=not drift, current_source_changes=drift,
                negative_digest_control_passed=True, minimum_available_gib=guard["minimum_available_bytes"] / 1024**3,
                verifier_sha256=file_digest(Path(__file__)), torch_imported="torch" in sys.modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--expected-sha256", help="Trusted digest from the publication")
    parser.add_argument("--require-current-sources", action="store_true")
    parser.add_argument("--report", type=Path, help="Write a new report at an unused path")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.run_id):
        parser.error("Expected the 32-character run ID")
    report = verify(HERE / "uncapped_checks" / args.run_id, args.expected_sha256)
    if args.require_current_sources and not report["current_sources_match"]:
        raise ValueError(f"Current source differences: {report['current_source_changes']}")
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
