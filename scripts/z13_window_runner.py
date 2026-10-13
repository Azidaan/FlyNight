"""One GPU child at a time, with an independent deadline guard for our tree.

This runner never downloads data or opens held-out val. T2 must pass before T3;
T3 runs only the two approved technical dev videos. Full training stays at 4000
steps. Every training process receives a cooperative save/stop request.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json


def stop_owned_child(proc):
    """Caller supplies only a Popen it created and still owns."""
    if proc is not None and proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            creationflags=subprocess.CREATE_NO_WINDOW, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def accept_t3(run_root):
    """Check approved technical outputs and the pre-recorded price limit."""
    records = []
    for video in ("019efe88-bbc2a4ac", "023b022b-723860c9"):
        folder = run_root / "t3/signatures" / video
        summary = json.loads((folder / "signatures.json").read_text(encoding="utf-8"))
        with np.load(folder / "signatures.npz") as archive:
            outputs = archive["outputs"]
            frames = int(archive["frame_count"])
        with np.load(ROOT / "runs/fly_signatures" / video / "signatures.npz") as baseline:
            expected_frames = int(baseline["frame_count"])
        events = summary["z13_events"]
        good = (frames == expected_frames and np.isfinite(outputs).all()
                and events["steps"] > 0 and events["births"] > 0
                and events["steps"] == events["interventions"] + events["refusals"])
        records.append({"video": video, "frames": frames, "expected_frames": expected_frames,
                        "events": events, "seconds": summary["seconds"], "passed": bool(good)})
    seconds_per_video = sum(r["seconds"] for r in records) / len(records)
    # The CURRENT card has a hard 16 h budget (1 technical + 3 training
    # + 12 dev). Do not revive the earlier 1.5 multiplier.
    projected_dev_hours = seconds_per_video * 35 * 3 / 3600
    t2 = json.loads((run_root / "t2/report.json").read_text(encoding="utf-8"))
    technical_hours = (sum(t2[a]["seconds"] for a in ("B", "C", "D"))
                       + sum(r["seconds"] for r in records)) / 3600
    price_ok = projected_dev_hours <= 12 and technical_hours <= 1
    report = {"videos": records, "projected_dev_gpu_hours": projected_dev_hours,
              "measured_technical_gpu_hours": technical_hours,
              "remaining_training_budget_hours": 3, "total_gpu_budget_hours": 16,
              "within_price_limit": price_ok,
              "passed": all(r["passed"] for r in records) and price_ok,
              "caveat": "Technical acceptance only; dev extrapolation covers eye collection, DINO/MASA overhead remains to be measured. No HOTA or localization quality claim"}
    atomic_json(run_root / "t3_acceptance.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--save-at", required=True)
    parser.add_argument("--hard-stop", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    save_at, hard_stop = (datetime.fromisoformat(s) for s in (args.save_at, args.hard_stop))
    if save_at.tzinfo is None or hard_stop.tzinfo is None or hard_stop <= save_at:
        raise ValueError("Explicit timezone and ordered save/hard deadlines required")
    run_root = args.run_root.resolve()
    run_root.relative_to(ROOT)
    run_root.mkdir(parents=True, exist_ok=True)
    if datetime.now().astimezone() >= save_at:
        raise RuntimeError("Window already closed; no new GPU process permitted")
    # Keep an OS lock for the entire controller lifetime, including child work.
    import msvcrt
    lock = (run_root / "controller.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0"); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    t2_manifest = run_root / "t2_manifest.json"
    if args.resume and t2_manifest.exists():
        from tracking.z13_checkpoint import provenance
        from scripts.z13_train import DATA
        saved = json.loads(t2_manifest.read_text(encoding="utf-8"))
        if saved != provenance(ROOT, DATA, saved["videos"], saved["config"]):
            raise ValueError("Resume rejected: technical code or data changed")
    job = run_root / ("job_" + datetime.now().strftime("%H%M%S"))
    job.mkdir()
    stop_file = job / "stop.request"
    python = ROOT / ".venv/Scripts/python.exe"
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.IDLE_PRIORITY_CLASS
    current = {"process": None}
    finished = threading.Event()
    ownership_lock = threading.Lock()

    def status(**values):
        atomic_json(job / "status.json", {
            "controller_pid": os.getpid(), "updated": datetime.now().astimezone().isoformat(),
            "save_at": args.save_at, "hard_stop": args.hard_stop, **values,
        })

    def guard():
        while not finished.wait(0.5):
            now = datetime.now().astimezone()
            if now >= save_at and not stop_file.exists():
                stop_file.write_text("Save state and stop at next safe boundary", encoding="utf-8")
            if now >= hard_stop:
                # Only our still-running child PID and descendants. A live Popen
                # handle prevents confusing this child with a reused PID.
                with ownership_lock:
                    proc = current["process"]
                    stop_owned_child(proc)
                return

    watchdog = threading.Thread(target=guard, daemon=True)
    watchdog.start()
    training_flags = ["--output-root", str(run_root), "--deadline", args.save_at,
                      "--stop-file", str(stop_file)] + (["--resume"] if args.resume else [])

    def run(stage, command):
        if stop_file.exists() or datetime.now().astimezone() >= save_at:
            status(state="paused", stage=stage)
            return 75
        with (job / "log.txt").open("a", encoding="utf-8") as log:
            log.write(f"\n[{stage}] {datetime.now().astimezone().isoformat()}\n"); log.flush()
            with ownership_lock:
                proc = subprocess.Popen([str(python), "-u", *command], cwd=ROOT,
                    creationflags=flags, stdout=log, stderr=subprocess.STDOUT)
                current["process"] = proc
            status(state="running", stage=stage, child_pid=proc.pid, command=command)
            code = proc.wait()
            with ownership_lock:
                current["process"] = None
            status(state="completed" if code == 0 else "paused" if code == 75 or stop_file.exists() else "failed",
                   stage=stage, exit_code=code)
            return code

    try:
        t2_report = run_root / "t2/report.json"
        if not (args.resume and t2_report.exists()):
            code = run("T2", ["scripts/z13_train.py", "--t2", *training_flags])
            if code != 0:
                return code
        if not all(json.loads(t2_report.read_text(encoding="utf-8"))["conditions"].values()):
            status(state="failed", stage="T2", reason="pre-recorded technical conditions failed")
            return 2
        for video in ("019efe88-bbc2a4ac", "023b022b-723860c9"):
            output = run_root / "t3/signatures"
            if args.resume and (output / video / "signatures.json").exists():
                continue
            code = run("T3_" + video, ["scripts/build_fly_signatures.py", "--skip-fly",
                "--z13", str(run_root / "t2/model_D.pt"), "--output", str(output),
                "--video-id", video, "--max-eyes", "16"])
            if code != 0:
                return code
        technical = accept_t3(run_root)
        if not technical["passed"]:
            status(state="failed", stage="T3", reason="technical execution or original price gate failed")
            return 2
        for arm in ("B", "C", "D"):
            if args.resume and (run_root / arm / "model.pt").exists():
                continue
            code = run("MAIN_" + arm, ["scripts/z13_train.py", "--arm", arm, *training_flags])
            if code != 0:
                return code
        # Full dev/MASA evaluation needs the trained models; never substitute
        # T2 models or report technical localization as a HOTA improvement.
        status(state="training_completed", stage="B_C_D",
               reason="All models ready; full dev/MASA evaluation remains")
        return 0
    finally:
        with ownership_lock:
            stop_owned_child(current["process"])
        finished.set()
        watchdog.join(timeout=2)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
