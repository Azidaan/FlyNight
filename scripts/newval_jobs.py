"""Задания сигнатур мухи для новых val-видео (life_filter_newval_v1): те же параметры, что у val7 (runs/jobs_signatures.json)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ids = json.loads((ROOT / "runs/newval_cache/selection.json").read_text(encoding="utf-8"))["videos"]
jobs = [dict(builder="scripts/build_fly_signatures.py", marker="signatures.npz", retina="lamina", retina_downscale=4, output="runs/fly_signatures_newval",
             video_id=v, videos="data/bdd100k/videos/val", detections_root="runs/newval_cache", flow_root="runs/newval_cache") for v in ids]
(ROOT / "runs/jobs_signatures_newval.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
print(f"[JOBS] {len(jobs)}")
