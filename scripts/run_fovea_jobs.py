"""Параллельный запуск фовеальных прогонов со сторожем (ночная работа 30.09.2026).

Каждое задание — один ролик в одном процессе `build_foveated_readout_cache.py --decode-flow`
с потолком видеопамяти. Сторож убивает процесс, если его вывод молчит дольше --stall
секунд или ролик идёт дольше --timeout, и перезапускает до --retries раз. Готовые
ролики (есть fovea_flow.npz) пропускаются. Сердцебиение и состояние пишутся в
<status>.json — по нему видно, что прогон жив.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "scripts" / "build_foveated_readout_cache.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True, help="JSON: список {retina, output, video_id, videos, detections_root, flow_root}")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu-mem-fraction", type=float, default=0.25)
    parser.add_argument("--max-eyes", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2, help="потоков CPU на процесс (torch/OpenCV/BLAS); без ограничения 6 процессов × 16 потоков душат друг друга")
    parser.add_argument("--stall", type=float, default=900.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--status", type=Path, required=True)
    return parser.parse_args()


def stamp() -> str:
    return time.strftime("%H:%M:%S")


class Runner:
    def __init__(self, args: argparse.Namespace, jobs: list[dict]) -> None:
        self.args = args
        self.queue = [j for j in jobs if not (PROJECT_ROOT / j["output"] / j["video_id"] / "fovea_flow.npz").exists()]
        self.skipped = len(jobs) - len(self.queue)
        self.total = len(jobs)
        self.lock = threading.Lock()
        self.done: list[dict] = []
        self.failed: list[dict] = []
        self.active: dict[int, dict] = {}
        self.started = time.time()

    def log(self, text: str) -> None:
        with self.lock:
            print(f"{stamp()} {text}", flush=True)

    def write_status(self) -> None:
        with self.lock:
            elapsed = time.time() - self.started
            finished = len(self.done)
            rate = elapsed / finished if finished else None
            left = len(self.queue) + len(self.active)
            status = {"heartbeat": time.strftime("%Y-%m-%d %H:%M:%S"), "total": self.total, "skipped_ready": self.skipped, "done": finished, "failed": len(self.failed), "queued": len(self.queue),
                      "active": [{"video_id": a["job"]["video_id"], "retina": a["job"]["retina"], "seconds": round(time.time() - a["start"]), "silent_seconds": round(time.time() - a["last_output"])} for a in self.active.values()],
                      "eta_hours": round(rate * left / max(self.args.workers, 1) / 3600, 2) if rate else None, "failures": self.failed}
            self.args.status.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")

    def command(self, job: dict) -> list[str]:
        cmd = [sys.executable, "-u", str(BUILDER), "--decode-flow", "--retina", job["retina"], "--output", job["output"], "--video-id", job["video_id"], "--gpu-mem-fraction", str(self.args.gpu_mem_fraction), "--max-eyes", str(self.args.max_eyes)]
        for key, flag in (("videos", "--videos"), ("detections_root", "--detections-root"), ("flow_root", "--flow-root")):
            if job.get(key):
                cmd += [flag, job[key]]
        return cmd

    def run_one(self, worker: int, job: dict) -> bool:
        for attempt in range(1, self.args.retries + 2):
            start = time.time()
            import os
            env = {**os.environ, **{k: str(self.args.threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "FOVEA_THREADS")}}
            proc = subprocess.Popen(self.command(job), cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
            state = {"job": job, "start": start, "last_output": start}
            with self.lock:
                self.active[worker] = state
            ok_line = None

            def reader() -> None:
                nonlocal ok_line
                for line in proc.stdout:
                    state["last_output"] = time.time()
                    if line.startswith("[OK]") and job["video_id"] in line:
                        ok_line = line.strip()
                    elif "Traceback" in line or "Error" in line:
                        self.log(f"[w{worker}] {job['video_id']} {job['retina']}: {line.strip()[:200]}")

            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            reason = None
            while proc.poll() is None:
                time.sleep(5)
                now = time.time()
                if now - state["last_output"] > self.args.stall:
                    reason = f"молчит {now - state['last_output']:.0f} с"
                elif now - start > self.args.timeout:
                    reason = f"дольше {self.args.timeout:.0f} с"
                if reason:
                    proc.kill()
                    break
            proc.wait()
            thread.join(timeout=10)
            with self.lock:
                self.active.pop(worker, None)
            if proc.returncode == 0 and ok_line:
                self.log(f"[w{worker}] {ok_line} ({time.time() - start:.0f} с)")
                return True
            self.log(f"[w{worker}] СБОЙ {job['video_id']} {job['retina']} попытка {attempt}: {reason or f'код {proc.returncode}'}")
        return False

    def worker(self, worker: int) -> None:
        while True:
            with self.lock:
                if not self.queue:
                    return
                job = self.queue.pop(0)
            ok = self.run_one(worker, job)
            with self.lock:
                (self.done if ok else self.failed).append({"video_id": job["video_id"], "retina": job["retina"], "output": job["output"]})

    def run(self) -> int:
        self.log(f"[PLAN] заданий {self.total}, готово ранее {self.skipped}, в очереди {len(self.queue)}, процессов {self.args.workers}, потолок памяти {self.args.gpu_mem_fraction}")
        threads = [threading.Thread(target=self.worker, args=(k,), daemon=True) for k in range(self.args.workers)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            self.write_status()
            time.sleep(15)
        self.write_status()
        self.log(f"[DONE] готово {len(self.done)}, сбоев {len(self.failed)}")
        return 0 if not self.failed else 1


def main() -> int:
    args = parse_args()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    return Runner(args, jobs).run()


if __name__ == "__main__":
    raise SystemExit(main())
