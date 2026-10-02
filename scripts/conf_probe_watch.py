"""Окно прогресса пробы Г2 (scripts/conf_target_probe.py). Только читает файлы. Ctrl+C — выход.

Запуск: python scripts/conf_probe_watch.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS = PROJECT_ROOT / "runs" / "conf_probe" / "status.json"
CACHES = PROJECT_ROOT / "runs" / "conf_probe_cache"
VAL_VIDEOS = 200
# доли времени одного задания: обучение 2 эпохи ≈ 45 мин, кэш val ≈ 22 мин
W_TRAIN, W_CACHE = 45.0, 22.0


def bar(frac: float, width: int = 40) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n) + f" {frac * 100:5.1f}%"


def gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        u, m, t, c = (x.strip() for x in out.split(","))
        return f"GPU {u}% | память {int(m) / 1024:.1f}/{int(t) / 1024:.1f} ГБ | {c}°C"
    except Exception:  # noqa: BLE001
        return "GPU —"


def main() -> int:
    os.system("title Проба Г2: цель уверенности без IoU")
    while True:
        os.system("cls")
        print("Проба Г2 — дообучение YOLO26: штатная цель (ctl) против цели без IoU (iou_free), 3 сида\n")
        if not STATUS.exists():
            print("ждём старта…")
            time.sleep(10)
            continue
        st = json.loads(STATUS.read_text(encoding="utf-8"))
        jobs, done = st.get("jobs", []), st.get("done", [])
        cur, stage = st.get("current", "—"), st.get("stage", "—")
        part = 0.0
        if stage == "train" and st.get("batches"):
            ep, eps = st.get("epoch", 1), st.get("epochs", 2)
            part = W_TRAIN / (W_TRAIN + W_CACHE) * (((ep - 1) + st.get("batch", 0) / st["batches"]) / eps)
            sub = f"обучение, эпоха {ep}/{eps}, батч {st.get('batch', 0)}/{st['batches']}"
            subfrac = ((ep - 1) + st.get("batch", 0) / st["batches"]) / eps
        elif stage == "cache":
            n = len(list((CACHES / cur / "val").glob("*.npz"))) if (CACHES / cur / "val").exists() else 0
            part = W_TRAIN / (W_TRAIN + W_CACHE) + W_CACHE / (W_TRAIN + W_CACHE) * n / VAL_VIDEOS
            sub, subfrac = f"кэш детекций val 960: {n}/{VAL_VIDEOS} видео", n / VAL_VIDEOS
        else:
            sub, subfrac = stage, 0.0
        total = (len(done) + (part if cur not in done else 0)) / max(len(jobs), 1)
        el = time.time() - st.get("started", time.time())
        eta = el / total * (1 - total) if total > 0.01 else None
        print(f"Всего ({len(done)}/{len(jobs)} заданий): {bar(total)}")
        print(f"Сейчас: {cur} — {sub}")
        print(f"         {bar(subfrac)}\n")
        print(f"Прошло: {timedelta(seconds=int(el))}" + (f" | осталось ≈ {timedelta(seconds=int(eta))} | конец ≈ {(datetime.now() + timedelta(seconds=eta)):%H:%M}" if eta else ""))
        print(gpu())
        print("\nГотово: " + (", ".join(done) if done else "—"))
        if stage in ("error", "finished", "eval"):
            print(f"\nСостояние: {stage} {st.get('error', '')}")
        age = time.time() - st.get("updated", time.time())
        if age > 600 and stage == "train":
            print(f"\n⚠ статус не обновлялся {int(age // 60)} мин — возможно, зависло")
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
