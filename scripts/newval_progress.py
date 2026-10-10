"""Прогресс конвейера life_filter_newval_v1: полоса по каждому шагу и оценка оставшегося времени.

Считает готовые файлы шагов (выравнивание, детекции, сигнатуры мухи, DINO, MASA, оценка). Оставшееся время — по
измеренной на пробе скорости (мин на видео): выравнивание 0,17, детекции 1,6, сигнатуры 2,5 (3 процесса), DINO 0,2, MASA 2,9.
Режим --watch: полный блок один раз, затем одна строка, перезаписываемая на месте (без очистки экрана — панель
терминала приложения её плохо переносит).

Запуск: python scripts/newval_progress.py [--watch 60]
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


def eval_done():
    log = ROOT / "runs/newval_eval_log.txt"
    return log.exists() and "[OK]" in log.read_text(encoding="utf-8", errors="ignore")


def show():
    rows, ok = counts()
    lines = [f"Проверка на новых val-видео — {time.strftime('%H:%M:%S')}" + (f"  (выравнивание сошлось у {ok} из {TOTAL}; остальные исключены по правилу)" if ok else "")]
    left = 0.0
    for name, d, t in rows:
        d = min(d, t)
        lines.append(f"  {name:15s} {bar(d, t)} {d:3d}/{t:<3d} {100 * d / max(t, 1):5.1f} %")
        left += (t - d) * RATE[name]
    done = eval_done()
    lines.append(f"  {'оценка':15s} {bar(1 if done else 0, 1)} {'готово' if done else 'ждёт'}")
    total_work = sum(t * RATE[n] for n, _, t in rows)
    lines.append(f"  всего {bar(total_work - left, total_work)} {100 * (total_work - left) / total_work:5.1f} %, осталось ≈ {left / 60:.1f} ч")
    print("\n".join(lines), flush=True)


def line():
    rows, _ = counts()
    cur = next(((n, d, t) for n, d, t in rows if d < t), None)
    left = sum((t - min(d, t)) * RATE[n] for n, d, t in rows)
    total = sum(t * RATE[n] for n, _, t in rows)
    step = f"{cur[0]} {cur[1]}/{cur[2]}" if cur else ("оценка: готово" if eval_done() else "оценка: идёт")
    return f"{time.strftime('%H:%M')}  сейчас: {step:20s} всего {bar(total - left, total, 20)} {100 * (total - left) / total:5.1f} %  осталось ≈ {left / 60:.1f} ч"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    show()
    if a.watch:
        print()
        while True:
            print("\r" + line().ljust(110), end="", flush=True)
            time.sleep(a.watch)
