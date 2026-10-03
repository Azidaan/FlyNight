"""Окно прогресса Г3 (scripts/gate_fly_eye_frontend.py). Только читает runs/gate_g3/status.json. Ctrl+C — выход."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

STATUS = Path(__file__).resolve().parents[1] / "runs" / "gate_g3" / "status.json"


def bar(frac, width=40):
    frac = max(0.0, min(1.0, frac)); n = int(round(frac * width))
    return "█" * n + "░" * (width - n) + f" {frac * 100:5.1f}%"


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        u, m, t = (x.strip() for x in out.split(","))
        return f"GPU {u}% | память {int(m) / 1024:.1f}/{int(t) / 1024:.1f} ГБ"
    except Exception:  # noqa: BLE001
        return "GPU —"


def main():
    os.system("title Г3: глаз мухи перед детектором")
    while True:
        os.system("cls")
        print("Г3 — замороженный YOLO26 на кадрах: сырой / накопление 3–5 кадров / фоторецептор мухи\n")
        if not STATUS.exists():
            print("ждём старта…"); time.sleep(10); continue
        try:
            st = json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            time.sleep(3); continue
        total, done = st.get("total", 1), st.get("done", 0)
        workers = {k: v for k, v in st.items() if k.startswith("w") and isinstance(v, dict)}
        part = sum(w["done_frames"] / max(w["frames"], 1) for w in workers.values() if w["done_frames"] < w["frames"])
        frac = min((done + part) / max(total, 1), 1.0)
        el = time.time() - st.get("started", time.time())
        eta = el / frac * (1 - frac) if frac > 0.02 else None
        print(f"Видео: {done}/{total}")
        print(f"Всего: {bar(frac)}\n")
        for k, w in sorted(workers.items()):
            if w["done_frames"] < w["frames"]:
                print(f"  {w['video']}: {bar(w['done_frames'] / max(w['frames'], 1), 25)}")
        print(f"\nПрошло: {timedelta(seconds=int(el))}" + (f" | осталось ≈ {timedelta(seconds=int(eta))} | конец ≈ {(datetime.now() + timedelta(seconds=eta)):%H:%M}" if eta else ""))
        print(gpu())
        print(f"\nСтадия: {st.get('stage', '—')}")
        if st.get("stage") == "detect_done":
            print("Детекции готовы — идёт оценка.")
        time.sleep(15)


if __name__ == "__main__":
    main()
