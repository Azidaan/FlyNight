"""Проба Г2: дообучение YOLO26 с целью уверенности без умножения на IoU (карточка conf_target_probe_v1).

Плечи (меняется ровно одно условие — патч цели):
  ctl      — дообучение от weights/baseline_seed42/best.pt, штатная функция потерь;
  iou_free — то же дообучение, цель уверенности без умножения на IoU (training/conf_target.py).
Общие условия: данные full_step5 (тот же train, что у baseline), 2 эпохи при 640, batch 16,
SGD lr0 0.002 без разогрева, остальное — по умолчанию Ultralytics (мозаика включена в обоих плечах:
при 2 эпохах close_mosaic=10 не срабатывает). Сиды 42, 123, 2024; порядок чередуется
(ctl42, iou42, ctl123, …), чтобы частичный результат был парным.
После обучения — кэш детекций val при 960, conf 0.01 (scripts/build_full_val_cache.py),
затем оценка scripts/eval_conf_probe.py. Статус для окна прогресса — runs/conf_probe/status.json.

Запуск: python scripts/conf_target_probe.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PY = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
INIT = PROJECT_ROOT / "weights" / "baseline_seed42" / "best.pt"
DATA = PROJECT_ROOT / "training" / "configs" / "generated" / "full_step5.yaml"
RUNS = PROJECT_ROOT / "runs" / "conf_probe"
CACHES = PROJECT_ROOT / "runs" / "conf_probe_cache"
STATUS = RUNS / "status.json"
ARMS = ("ctl", "iou_free")
SEEDS = (42, 123, 2024)


def plan() -> list[tuple[str, int]]:
    return [(a, s) for s in SEEDS for a in ARMS]


def write_status(**kw) -> None:
    st = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    st.update(kw, updated=time.time())
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATUS)


def on_batch(trainer) -> None:
    i = getattr(trainer, "_probe_i", 0) + 1
    trainer._probe_i = i
    if i % 25 == 0:
        nb = len(trainer.train_loader)
        write_status(stage="train", epoch=int(trainer.epoch) + 1, epochs=int(trainer.epochs), batch=i - int(trainer.epoch) * nb, batches=nb)


def train(arm: str, seed: int, epochs: int) -> Path:
    from ultralytics import YOLO

    from training import conf_target
    tag = f"{arm}_s{seed}"
    best = RUNS / tag / "weights" / "last.pt"
    if best.exists():
        return best
    if arm == "iou_free":
        conf_target.install()
    model = YOLO(str(INIT))
    model.add_callback("on_train_batch_end", on_batch)
    model.train(data=str(DATA), epochs=epochs, imgsz=640, batch=16, workers=4, seed=seed, deterministic=True, device=0,
                optimizer="SGD", lr0=0.002, warmup_epochs=0, val=False, plots=False, verbose=False,
                project=str(RUNS), name=tag, exist_ok=True)
    if arm == "iou_free":
        print(f"[conf_target] вызовов назначателя с патчем: {conf_target.CALLS['n']}", flush=True)
        conf_target.uninstall()
        assert conf_target.CALLS["n"] > 0, "патч не сработал"
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    jobs = plan()
    write_status(stage="start", jobs=[f"{a}_s{s}" for a, s in jobs], done=[], started=time.time())
    done = []
    for k, (arm, seed) in enumerate(jobs, 1):
        tag = f"{arm}_s{seed}"
        write_status(current=tag, job=k, njobs=len(jobs), stage="train", epoch=0, batch=0)
        print(f"=== [{k}/{len(jobs)}] обучение {tag} ===", flush=True)
        # обучение — в отдельном процессе: патч и состояние Ultralytics не протекают между плечами
        r = subprocess.run([str(PY), "-u", __file__, "--one", arm, str(seed), str(args.epochs)], cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            write_status(stage="error", error=f"обучение {tag} упало, код {r.returncode}")
            return r.returncode
        w = RUNS / tag / "weights" / "last.pt"
        cache = CACHES / tag
        write_status(stage="cache", batch=0)
        print(f"=== [{k}/{len(jobs)}] кэш val 960 {tag} ===", flush=True)
        subprocess.run([str(PY), "-u", "scripts/build_full_val_cache.py", "--cache", str(cache.relative_to(PROJECT_ROOT)),
                        "--imgsz", "960", "--model", str(w.relative_to(PROJECT_ROOT))], cwd=str(PROJECT_ROOT), check=True)
        done.append(tag)
        write_status(done=done, stage="job_done")
    write_status(stage="eval")
    subprocess.run([str(PY), "-u", "scripts/eval_conf_probe.py"], cwd=str(PROJECT_ROOT), check=False)
    write_status(stage="finished")
    print("[OK] проба завершена", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        train(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        raise SystemExit(0)
    raise SystemExit(main())
