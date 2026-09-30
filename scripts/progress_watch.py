"""Монитор прогресса фоновых прогонов (сторож run_fovea_jobs.py): % готовности, скорость, прогноз, сбои.

Запуск: python scripts/progress_watch.py --jobs runs/jobs_signatures.json --status runs/jobs_signatures.status.json runs/jobs_signatures_rev.status.json
Только читает файлы; ничего не запускает и не меняет. Ctrl+C — выход.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--status", type=Path, nargs="*", default=[])
    parser.add_argument("--every", type=float, default=30.0)
    return parser.parse_args()


def gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        u, m, t = (x.strip() for x in out.split(","))
        return f"GPU {u}% | память {int(m) / 1024:.1f}/{int(t) / 1024:.1f} ГБ"
    except Exception:  # noqa: BLE001
        return "GPU —"


def main() -> int:
    args = parse_args()
    jobs = json.loads((PROJECT_ROOT / args.jobs).read_text(encoding="utf-8"))
    total = len(jobs)
    history: list[tuple[float, int]] = []
    while True:
        done = sum((PROJECT_ROOT / j["output"] / j["video_id"] / j.get("marker", "fovea_flow.npz")).exists() for j in jobs)
        now = time.time(); history.append((now, done)); history = [h for h in history if now - h[0] <= 1800]
        rate = (history[-1][1] - history[0][1]) / max(history[-1][0] - history[0][0], 1.0) * 60 if len(history) > 1 else 0.0
        eta = (datetime.now() + timedelta(minutes=(total - done) / rate)).strftime("%H:%M") if rate > 0 else "—"
        pct = 100.0 * done / max(total, 1)
        bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Прогон: {args.jobs.name}   {datetime.now():%H:%M:%S}\n")
        print(f"  [{bar}] {pct:5.1f} %   {done}/{total} роликов")
        print(f"  скорость {rate:.2f} ролика/мин (за последние 30 мин)   прогноз окончания: {eta}")
        print(f"  {gpu()}\n")
        for st in args.status:
            p = PROJECT_ROOT / st
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            beat = d.get("heartbeat", "?")
            print(f"  сторож {st.name}: сердцебиение {beat}, готово {d.get('done')}, сбоев {d.get('failed')}, в очереди {d.get('queued')}")
            for a in d.get("active", []):
                warn = "  ← молчит!" if a["silent_seconds"] > 60 else ""
                print(f"      {a['video_id'][:17]}  идёт {a['seconds']:>4} с, тишина {a['silent_seconds']:>3} с{warn}")
        print("\n  (окно только показывает прогресс; закрыть — Ctrl+C или крестик)")
        if done >= total:
            print("\n  ГОТОВО.")
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
