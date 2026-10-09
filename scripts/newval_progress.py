"""Прогресс конвейера life_filter_newval_v1: полоса по каждому шагу и оценка оставшегося времени.

Считает готовые файлы шагов (выравнивание, детекции, сигнатуры мухи, DINO, MASA, оценка). Оставшееся время — по
измеренной на пробе скорости (мин на видео): выравнивание 0,17, детекции 1,6, сигнатуры 2,5 (3 процесса), DINO 0,2, MASA 2,9.

Запуск: python scripts/newval_progress.py [--watch 30]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOTAL = 104
RATE = {"выравнивание": 0.17, "детекции": 1.6, "сигнатуры мухи": 2.5, "DINO": 0.2, "MASA": 2.9}


def counts():
    sel = ROOT / "runs/newval_cache/selection.json"
    man = ROOT / "runs/newval_videos/manifest.json"
    aligned = sum(1 for d in (ROOT / "runs/newval_videos").glob("*/alignment.json"))
    ok = None
    if sel.exists():
        ok = len(json.loads(sel.read_text(encoding="utf-8"))["videos"])
    elif man.exists():
        try:
            ok = sum(1 for v in json.loads(man.read_text(encoding="utf-8"))["videos"].values() if v.get("alignment_ok"))
        except Exception:
            ok = None
    n = ok or TOTAL
    return [("выравнивание", TOTAL if sel.exists() else aligned, TOTAL),
            ("детекции", sum(1 for _ in (ROOT / "runs/newval_cache").glob("*/detections/raw_video.npz")), n),
            ("сигнатуры мухи", sum(1 for _ in (ROOT / "runs/fly_signatures_newval").glob("*/signatures.npz")), n),
            ("DINO", sum(1 for _ in (ROOT / "runs/dino_embeddings_newval").glob("*.npz")), n),
            ("MASA", sum(1 for _ in (ROOT / "runs/masa_embeddings_newval").glob("*.npz")), n)], ok


def bar(done, total, width=30):
    f = int(width * done / max(total, 1))
    return "█" * f + "░" * (width - f)


def show():
    rows, ok = counts()
    lines = [f"Проверка на новых val-видео — {time.strftime('%H:%M:%S')}" + (f"  (выравнивание сошлось у {ok} из {TOTAL}; остальные исключены по правилу)" if ok else "")]
    left = 0.0
    for name, d, t in rows:
        d = min(d, t)
        lines.append(f"  {name:15s} {bar(d, t)} {d:3d}/{t:<3d} {100 * d / max(t, 1):5.1f} %")
        left += (t - d) * RATE[name]
    log = ROOT / "runs/newval_eval_log.txt"
    done = log.exists() and "[OK]" in log.read_text(encoding="utf-8", errors="ignore")
    lines.append(f"  {'оценка':15s} {bar(1 if done else 0, 1)} {'готово' if done else 'ждёт'}")
    total_work = sum(t * RATE[n] for n, _, t in rows)
    lines.append(f"  всего {bar(total_work - left, total_work)} {100 * (total_work - left) / total_work:5.1f} %, осталось ≈ {left / 60:.1f} ч")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--watch", type=int, default=0); a = ap.parse_args()
    while True:
        show()
        if not a.watch:
            break
        time.sleep(a.watch); print()
