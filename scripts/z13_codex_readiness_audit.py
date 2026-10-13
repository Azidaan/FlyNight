"""Read-only Z13 readiness audit; synthetic loader reproduction, no GPU/training.

Writes only its own audit directory. Does not repair or relabel Claude's data.
"""
import hashlib
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/z13_codex_readiness_20261013"


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    fixtures = OUT / "synthetic"
    fixtures.mkdir(parents=True, exist_ok=True)
    for name, count, base in (("a", 3, 0), ("b", 2, 3)):
        offsets = np.arange(count + 1) * 5
        frames = np.repeat(np.arange(base, base + count), 5)
        np.savez_compressed(
            fixtures / f"{name}.npz",
            ep_offset=offsets,
            ep_pert=np.zeros(count, bool),
            fr_hex=np.repeat(frames[:, None], 721, axis=1).astype(np.float16),
            ex_mask=np.ones((count, 721), bool),
            lab_ep=np.arange(count), lab_pos=np.zeros(count, int),
            lab_kind=np.ones(count, int), lab_col=np.zeros(count, int),
            lab_center=np.zeros((count, 2)), lab_gtbox=np.ones((count, 4)),
            lab_pred=np.zeros((count, 2)), lab_crop=np.ones((count, 4)),
        )
    source = ROOT / "scripts/z13_train.py"
    spec = importlib.util.spec_from_file_location("z13_train_readiness", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DATA = fixtures
    data = mod.Episodes(["a", "b"])
    actual = [int(row[0]) for row in data.labels]
    expected = list(range(5))
    rep = {
        "time_local": datetime.now().astimezone().isoformat(),
        "training_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "synthetic_episode_indices_expected": expected,
        "synthetic_episode_indices_actual": actual,
        "synthetic_loader_passed": actual == expected,
        "data_modified": False,
        "gpu_started": False,
    }
    for key, relative in (("t1", "runs/z13_data/t1.json"),
                          ("skipfly", "runs/z13/skipfly_check.json"),
                          ("extraction", "runs/z13_data/summary.json")):
        path = ROOT / relative
        rep[key] = {"source": relative, "result": json.loads(path.read_text(encoding="utf-8"))}
    (OUT / "readiness.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("skipfly", "extraction")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
