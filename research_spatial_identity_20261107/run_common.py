"""Utilities for the authorized spatial-map comparison, no import-time execution."""
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/spatial_identity_maps_20261107"
MSK = timezone(timedelta(hours=3))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value, *, new=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if new:
        with path.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
    else:
        tmp = path.with_suffix(path.suffix + ".pending")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)


def status(stage, **values):
    write(RUN / "status.json", {"stage": stage, "updated": datetime.now(MSK).isoformat(), **values})


def verified_protocol():
    p = read(RUN / "execution_protocol.json")
    for name, expected in p["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Frozen source changed: " + name)
    return p


def guard():
    import sys
    directory = ROOT / "runs/flyvis_repairs_20261106/unification_20261107"
    sys.path.insert(0, str(directory))
    from system_memory_guard import guard_entrypoint
    seconds = max(1, int((datetime(2026, 11, 9, 7, 40, tzinfo=MSK) - datetime.now(MSK)).total_seconds()))
    return guard_entrypoint(timeout=seconds)
