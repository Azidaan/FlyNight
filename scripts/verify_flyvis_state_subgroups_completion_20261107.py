"""Read-only completion review; never reruns inference, metrics or old recorders."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import subprocess
import zipfile

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/flyvis_state_subgroups_ram_stream4_20261107"
DEST = OUT / "goal_completion_review.json"
# These were still owned by the chain when the historical finalizer hashed them.
OPEN_AT_OLD_FINALIZATION = {"finalize.log", "postprocess.stage.json"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError("Preserve the earlier review; choose a new report path.")
    p, a, old, recovery = [load(OUT / f) for f in
        ("protocol.json", "independent_audit.json", "completion_verified.json", "recovery_manifest.json")]
    if psutil.virtual_memory().available < 12 * 1024**3:
        raise RuntimeError("System memory reserve below the user's 12 GiB threshold.")
    checks = {}
    checks["full_scope"] = a["passed"] and a["cases"] == 64 and a["independent_scenes"] == 8
    checks["protocol_exact"] = a["protocol_sha256"] == sha(OUT / "protocol.json")
    checks["inference_exact"] = a["inference_sha256"] == sha(OUT / "inference_completed.json")
    case_ids = {x["id"] for x in p["cases"]}
    checks["all_cases_audited"] = case_ids == {x["case"]["id"] for x in a["rows"]} and len(a["rows"]) == 64
    checks["case_shape"] = all(len(x["metrics"]) == 2 and all(len(v) == 24 for v in x["metrics"].values()) for x in a["rows"])
    case_sums = {}
    for row in a["rows"]:
        key = row["case"]["id"]
        actual = sha(OUT / "cases" / (key + ".npz"))
        marker = load(OUT / "cases" / (key + ".json"))
        case_sums[key] = actual
        if actual != row["sha256"] or actual != old["cases"][key] or actual != marker["sha256"]:
            raise ValueError("Case checksum mismatch: " + key)
        if marker["protocol_sha256"] != a["protocol_sha256"]:
            raise ValueError("Case protocol mismatch: " + key)
    checks["all_64_case_checksums"] = len(case_sums) == 64
    checks["three_finished_cases_reused"] = len(recovery["reused_cases"]) == 3 and all(
        case_sums[k] == v["sha256"] for k, v in recovery["reused_cases"].items())
    sources = {**p["frozen_sources"], **recovery["new_sources"]}
    archive = OUT / "frozen_sources.zip"
    checks["source_archive_exact"] = sha(archive) == old["source_archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        checks["archived_sources_exact"] = all(hashlib.sha256(z.read(f)).hexdigest() == h
            for f, h in sources.items() if f.endswith(".py"))
        checks["archived_protocol_exact"] = hashlib.sha256(z.read("protocol.json")).hexdigest() == a["protocol_sha256"]
    drift = {f: {"at_run": h, "current": sha(ROOT / f)} for f, h in sources.items() if sha(ROOT / f) != h}
    checks["weights_and_connectome_exact"] = all(sha(ROOT / f) == h for f, h in sources.items() if not f.endswith(".py"))
    differences = [f for f, h in old["artifacts"].items() if sha(OUT / f) != h]
    checks["closed_historical_artifacts_exact"] = set(differences) <= OPEN_AT_OLD_FINALIZATION
    probe = load(OUT / "batch_equivalence.json")
    checks["accepted_numerical_probe"] = probe["passed"] and probe["boundary_byte_exact"] and probe["fields_byte_exact"] and probe["evaluation_signs_exact"]
    checks["encoder_tolerance"] = a["maximum_eye_error"] <= 1e-5
    controllers = {}
    identities = set()
    for f in ["execution_probe/controller_result.json", "execution_run/controller_result.json", "execution_audit/controller_result.json"] + [x.name for x in OUT.glob("*.ended.json")]:
        d = load(OUT / f)
        controllers[f] = {k: d.get(k) for k in ("exit_code", "reason", "all_owned_ended")}
        identities.update((int(pid), birth) for pid, birth in d.get("owned", {}).items())
    chain = load(OUT / "chain.launch.json")
    identities.add((chain["pid"], chain["birth"]))
    live = []
    for pid, birth in identities:
        try:
            if abs(psutil.Process(pid).create_time() - birth) < .01:
                live.append({"pid": pid, "birth": birth})
        except psutil.NoSuchProcess:
            pass
    checks["controllers_successful"] = all(d["exit_code"] == 0 and not d["reason"] and d["all_owned_ended"] for d in controllers.values())
    checks["actual_processes_ended"] = not live
    chain_result = load(OUT / "chain_result.json")
    checks["chain_complete"] = chain_result["complete"] and chain_result["all_stages_ended"]
    checks["before_registered_deadline"] = datetime.fromisoformat(chain_result["ended_msk"]) < datetime.fromisoformat(p["deadline_msk"])
    memory = load(OUT / "final_memory_usage.json")
    execution = load(OUT / "execution_run/controller_result.json")
    checks["registered_ram_safeguards"] = memory["job"]["limit_active"] and memory["job"]["limit_bytes"] == 4 * 1024**3 and memory["job"]["peak_job_bytes"] < 4 * 1024**3 and execution["minimum_system_available"] >= 12 * 1024**3
    checks["recorded_tests_passed"] = "71 passed" in (OUT / "record_tests.log").read_text() and "OK: каталог согласован с источниками" in (OUT / "catalog_check.log").read_text(encoding="utf-8-sig")
    snapshot = load(OUT / "record_snapshot.json")
    card = next(x for x in load(ROOT / "experiments/project_history.json")["experiments"] if x["id"] == p["id"])
    problem = next(x for x in load(ROOT / "experiments/system_problems.json")["problems"] if x["id"] == "С7")
    checks["catalog_and_problem_record_exact"] = card == snapshot["card"] and snapshot["appended_problem_case"] in problem["cases"]
    checks["negative_verdict_preserved"] = not a["primary_met"] and not any(v["passed"] for v in a["candidates"].values()) and card["state"] == "completed_negative"
    checks["chronology_and_roadmap_recorded"] = all(OUT.name in (ROOT / f).read_text(encoding="utf-8") for f in ("dissertation/ЕДИНЫЙ_ПЛАН.md", "FLYVIS_TRACKER_RESEARCH_ROADMAP.md"))
    graph = load(OUT / "graph_audit.json")
    checks["graph_preserved_at_recording"] = graph["passed"] and graph["protected_hypotheses_byte_exact"] and not graph["deleted_files"]
    protected = [x for x in (OUT / "vault_before").rglob("*.md") if x.relative_to(OUT / "vault_before").parts[0].startswith(("Гипотезы", "10 Проблемы мухи")) or x.name.startswith(("Б1 —", "Б2 —", "Б3 —"))]
    checks["protected_notes_currently_exact"] = len(protected) >= 36 and all(sha(x) == sha(ROOT / "ObsidianFlyVis" / x.relative_to(OUT / "vault_before")) for x in protected)
    figure = load(OUT / "figure_provenance.json")
    checks["figure_exact"] = figure["source_sha256"] == sha(OUT / "independent_audit.json") and figure["figure_sha256"] == sha(OUT / "state_groups_tradeoff.png")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    checks["main_branch_selected"] = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "codex/system-integrity-20261106-e8f674ce"
    scientific_paths = ["protocol.json", "independent_audit.json", "РЕЗУЛЬТАТ.md", "record_snapshot.json"]
    checks["scientific_evidence_in_main_commit"] = all(subprocess.check_output(["git", "show", head + ":" + (OUT / f).relative_to(ROOT).as_posix()], cwd=ROOT) == (OUT / f).read_bytes() for f in scientific_paths)
    report = dict(passed=all(checks.values()), checked_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat(), checks=checks,
        head=head, original_manifest_sha256=sha(OUT / "completion_verified.json"), historical_open_file_differences=differences,
        explanation="The original finalizer hashed finalize.log and postprocess.stage.json before its chain closed them. Preserve that historical manifest; this review verifies all other old hashes and binds the closed chain files separately.",
        chain_closed_sha256={f: sha(OUT / f) for f in ("finalize.log", "postprocess.stage.json", "chain_result.json", "finalize.ended.json")},
        current_source_drift=drift, controllers=controllers, process_identities_checked=len(identities), live=live,
        protected_notes_checked=len(protected), case_sha256=case_sums,
        audit_sha256=sha(OUT / "independent_audit.json"), scope="Evidence completion review only; no inference, metric recomputation, tests or scientific intervention repeated.")
    DEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "current_source_drift": list(drift), "report": str(DEST)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
