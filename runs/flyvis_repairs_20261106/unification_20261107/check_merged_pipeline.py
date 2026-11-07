"""Sequential acceptance checks for the merged checkout on the local Windows bench."""
import os
import json
from pathlib import Path
import sys

from system_memory_guard import guard_entrypoint

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    job = guard_entrypoint()
    mode, *files = sys.argv[1:]
    if mode not in ("replay", "cpu", "cuda"):
        raise ValueError("Expected replay, cpu or cuda")
    if mode == "replay" and files:
        raise ValueError("Replay uses all seven registered videos")
    if mode != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(2)
    if mode == "replay":
        from scripts import freeze_bench

        sys.argv = [str(freeze_bench.__file__), "--output", str(job.run_dir / "replay")]
        return freeze_bench.main()

    import pytest

    args = [*files, "-v", "-ra", "--tb=short", "-p", "no:cacheprovider",
            "--basetemp", str(job.run_dir / "pytest_temp"),
            "--junitxml", str(job.run_dir / "pytest.xml")]
    if not files or (files != ["tests"] and any(
        not f.startswith("tests/test_") or not f.endswith(".py") or Path(f).parent != Path("tests")
        for f in files
    )):
        raise ValueError("Supply tests or explicit test files under tests/")
    if mode == "cuda":
        from tracking.gpu_lock import CANONICAL_GPU_LOCK_NAME, GpuLock

        with GpuLock(CANONICAL_GPU_LOCK_NAME):
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA checks require a working GPU")
            print("CUDA:", torch.cuda.get_device_name(), flush=True)
            return pytest.main(args)

    from unittest.mock import patch
    from evaluation.project_history.index import fingerprint
    from run_fingerprint import sha

    torch.set_default_device("cpu")
    def snapshot():
        files = [path for folder in ("scripts", "tracking", "models", "evaluation", "datasets", "tests")
                 for path in sorted((ROOT / folder).rglob("*.py"))]
        return {"index": fingerprint(ROOT), "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files}}

    before = snapshot()
    (job.run_dir / "checkout_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    with patch.object(torch.cuda, "is_available", return_value=False), \
         patch.object(torch.cuda, "device_count", return_value=0), \
         patch.object(torch.cuda, "_lazy_init", side_effect=RuntimeError("CPU-only acceptance check")):
        code = pytest.main(args)
    after = snapshot()
    stable = after == before
    (job.run_dir / "checkout_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    print(f"Checkout and project-history fingerprint unchanged: {stable}", flush=True)
    return code or int(torch.cuda.is_initialized() or not stable)


if __name__ == "__main__":
    sys.exit(main())
