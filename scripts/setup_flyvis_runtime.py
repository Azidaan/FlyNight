"""Установить изолированный runtime и официальные веса FlyVis 1.2.0.

Сценарий намеренно не устанавливает Torch и torchvision: используются уже
проверенные версии из основного .venv проекта. Остальные зависимости и код
FlyVis помещаются в исключённый из Git каталог tools/flyvis-runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "tools" / "flyvis-runtime"
DATA_DIR = PROJECT_ROOT / "runs" / "flyvis"
ARCHIVE = DATA_DIR / "results_pretrained_models.zip"
ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=13cJr2nMn89j-jBAd5RduYRJpBcXwoNrC&export=download&confirm=t"
)
ARCHIVE_SHA256 = "71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1"

# Прямые зависимости FlyVis, кроме torch и torchvision. Pip разрешит их
# транзитивные зависимости только внутри RUNTIME_DIR.
RUNTIME_REQUIREMENTS = (
    "flyvis==1.2.0",
    "ipython",
    "numba",
    "datamate==1.0.0",
    "umap-learn",
    "ffmpeg-python",
    "pytest",
    "python-dotenv",
    "hydra-core",
    "toolz",
    "google-api-python-client",
    "xarray",
    "cachetools",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_runtime() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Основной Torch не видит CUDA; FlyVis будет непригоден для замеров GPU")
    if not (RUNTIME_DIR / "flyvis").is_dir():
        raise FileNotFoundError(f"Код FlyVis не найден: {RUNTIME_DIR}")
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Архив весов не найден: {ARCHIVE}")
    actual = file_sha256(ARCHIVE)
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(
            f"Контрольная сумма архива не совпала: {actual}, ожидалась {ARCHIVE_SHA256}"
        )
    models = sorted((DATA_DIR / "results" / "flow" / "0000").glob("*/best_chkpt"))
    if len(models) != 50:
        raise RuntimeError(f"Ожидалось 50 checkpoint, найдено {len(models)}")
    print(f"[OK] Torch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"[OK] FlyVis runtime: {RUNTIME_DIR}")
    print(f"[OK] официальный архив: {actual}")
    print(f"[OK] checkpoint: {len(models)}")


def install_runtime() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    # FlyVis ставится отдельно без зависимостей, иначе pip может загрузить
    # другой Torch и разрушить сопоставимость с окружением YOLO26.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            RUNTIME_REQUIREMENTS[0],
            "--no-deps",
            "--target",
            str(RUNTIME_DIR),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *RUNTIME_REQUIREMENTS[1:],
            "--target",
            str(RUNTIME_DIR),
        ],
        check=True,
    )


def download_archive() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and file_sha256(ARCHIVE) == ARCHIVE_SHA256:
        print(f"[SKIP] архив уже загружен и проверен: {ARCHIVE}")
        return
    partial = ARCHIVE.with_suffix(".zip.part")
    urllib.request.urlretrieve(ARCHIVE_URL, partial)
    actual = file_sha256(partial)
    if actual != ARCHIVE_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Контрольная сумма загрузки не совпала: {actual}, ожидалась {ARCHIVE_SHA256}"
        )
    partial.replace(ARCHIVE)
    print(f"[OK] архив загружен: {ARCHIVE}")


def extract_archive() -> None:
    destination = DATA_DIR.resolve()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Небезопасный путь в архиве: {member.filename}")
        archive.extractall(destination)
    print(f"[OK] веса распакованы: {DATA_DIR / 'results'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="только проверить уже установленный runtime и веса",
    )
    args = parser.parse_args()
    if not args.check_only:
        install_runtime()
        download_archive()
        extract_archive()
    verify_runtime()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

