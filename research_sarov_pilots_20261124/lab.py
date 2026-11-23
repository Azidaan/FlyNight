from pathlib import Path
import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT / "research_memory_gap_transfer_20261114"
GAPS = ROOT / "research_natural_gap_ownership_20261114"
for env, sub in {"TEMP":"tmp", "TMP":"tmp", "TMPDIR":"tmp", "MPLCONFIGDIR":"matplotlib",
                 "TORCH_HOME":"torch", "CUDA_CACHE_PATH":"cuda", "YOLO_CONFIG_DIR":"ultralytics_yolo26seg",
                 "TORCHINDUCTOR_CACHE_DIR":"torchinductor", "TRITON_CACHE_DIR":"triton"}.items():
    dest = ROOT / ".cache" / sub
    dest.mkdir(parents=True, exist_ok=True)
    os.environ[env] = str(dest)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for env in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[env] = "1"
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(f)


def _hash_stream(f):
    h = hashlib.sha256()
    for piece in iter(lambda: f.read(2**20), b""):
        h.update(piece)
    return h.hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=3))).isoformat(timespec="seconds")


def fingerprint(paths):
    import subprocess
    return {"msk":now(), "git":subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(),
            "sha256":{Path(p).relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}}
