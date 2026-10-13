"""Check corrected episode ownership on all 130 train files, without GPU."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_train import DATA, Episodes, split_videos
from tracking.z13_checkpoint import atomic_json


def main():
    fit, tau = split_videos()
    videos = sorted(fit + tau)
    data = Episodes(videos)
    episode_base = label_base = mismatches = 0
    for video in videos:
        with np.load(DATA / (video + ".npz")) as z:
            episodes = len(z["ep_offset"]) - 1
            labels = z["lab_ep"]
            source_centers = z["lab_center"]
            source_hex = z["fr_hex"]
            offset = z["ep_offset"]
        merged = data.labels[label_base:label_base + len(labels)]
        for i, row in enumerate(merged):
            mismatches += int(row[0] != episode_base + int(labels[i]))
            mismatches += int(not np.array_equal(row[4], source_centers[i], equal_nan=True))
        for e in range(episodes):
            merged_start = data.off[episode_base + e]
            mismatches += int(not np.array_equal(data.hex[merged_start], source_hex[offset[e]]))
        label_base += len(labels)
        episode_base += episodes
    t1 = json.loads((DATA / "t1.json").read_text(encoding="utf-8"))
    result = {"videos": len(videos), "fit_videos": len(fit), "tau_videos": len(tau),
              "episodes": episode_base, "labels": label_base, "ownership_mismatches": mismatches,
              "t1": t1, "passed": mismatches == 0 and t1["passed"] and len(videos) == 130}
    atomic_json(ROOT / "runs/z13_codex_repair_20261013/data_check.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
