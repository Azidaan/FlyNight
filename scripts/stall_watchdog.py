"""Сторож молчащих заданий fly_persistent_eye (ночной прогон 11.10): несколько CUDA-процессов на Windows изредка зависают.

Каждые 60 с: для каждого процесса scripts/fly_persistent_eye.py (по командной строке: --video-id, --out) смотрит время
последней записи его лога <out>/<id>.log; если лог молчит дольше --stall секунд и процесс живёт дольше того же срока —
завершает процесс с потомками. Оркестратор (scripts/fly_persistent_eye_run.py) повторяет задание (вторая попытка).
Работает, пока жив оркестратор. Журнал — runs/fly_persistent_eye/watchdog.txt.

Запуск: python scripts/stall_watchdog.py [--stall 600]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG = PROJECT_ROOT / "runs/fly_persistent_eye/watchdog.txt"


def arg(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--stall", type=float, default=600.0); a = ap.parse_args()
    while True:
        orch = [p for p in psutil.process_iter(["cmdline"]) if any("fly_persistent_eye_run.py" in (c or "") for c in (p.info["cmdline"] or []))]
        if not orch:
            LOG.open("a", encoding="utf-8").write(f"{time.strftime('%H:%M:%S')} оркестратор завершён — сторож выходит\n"); return 0
        now = time.time()
        for p in psutil.process_iter(["cmdline", "create_time"]):
            cmd = p.info["cmdline"] or []
            if not any(c.endswith("fly_persistent_eye.py") for c in cmd):
                continue
            vid, out = arg(cmd, "--video-id"), arg(cmd, "--out")
            if not vid or not out:
                continue
            log = Path(out) / f"{vid}.log"
            last = log.stat().st_mtime if log.exists() else p.info["create_time"]
            if now - last > a.stall and now - p.info["create_time"] > a.stall:
                try:
                    for ch in p.children(recursive=True):
                        ch.kill()
                    p.kill()
                    LOG.open("a", encoding="utf-8").write(f"{time.strftime('%H:%M:%S')} завершён молчащий {vid} ({out}), молчал {now - last:.0f} с\n")
                except psutil.Error as e:
                    LOG.open("a", encoding="utf-8").write(f"{time.strftime('%H:%M:%S')} не удалось завершить {vid}: {e}\n")
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
