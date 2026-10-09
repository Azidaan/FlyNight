"""Выборочно скачать исходные видео BDD100K по диапазонам байт из bdd100k_videos.zip (цель 1а).

По умолчанию — 1 400 MOT-train роликов (ID из labels/box_track_20/train) в
data/bdd100k/videos/train. Каждый ролик проверяется по CRC32 и размеру из
центрального каталога, пишется во временный файл и переименовывается; в
манифест попадают sha256, размер, число кадров и частота по контейнеру.
Повторный запуск докачивает недостающее.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
import urllib.request
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.zip_range_fetch import Entry, RangeZip  # noqa: E402

URL = "http://dl.yf.io/bdd100k/bdd100k_videos.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--split", default="train")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos")
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "runs" / "bdd100k_videos_zip_index.pkl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ids-file", type=Path, default=None, help="JSON со списками ID (все строковые списки объединяются) — скачать только их (life_filter_newval_v1)")
    return parser.parse_args()


def http_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as response:
        return int(response.headers["Content-Length"])


def make_reader(url: str, retries: int):
    def read(start: int, end: int) -> bytes:
        delay = 2.0
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=180) as response:
                    if response.status not in (206, 200):
                        raise OSError(f"HTTP {response.status}")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise OSError(f"получено {len(data)} байт вместо {end - start + 1}")
                return data
            except Exception as error:  # noqa: BLE001 — сеть: повторяем с паузой
                if attempt == retries - 1:
                    raise
                print(f"[RETRY] {error}; пауза {delay:.0f} с", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")
    return read


def probe_video(path: Path) -> dict:
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        info = {"frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": float(capture.get(cv2.CAP_PROP_FPS)), "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), "opened": bool(capture.isOpened())}
        capture.release()
        return info
    except Exception as error:  # noqa: BLE001
        return {"opened": False, "error": str(error)}


def file_ok(path: Path, entry: Entry) -> bool:
    if not path.exists() or path.stat().st_size != entry.uncompressed_size:
        return False
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return (crc & 0xFFFFFFFF) == entry.crc32


def main() -> int:
    args = parse_args()
    ids = sorted(p.stem for p in (args.labels / args.split).glob("*.json"))
    if args.ids_file:
        spec = json.loads(args.ids_file.read_text(encoding="utf-8"))
        wanted = {v for val in spec.values() if isinstance(val, list) for v in val if isinstance(v, str)}
        ids = [i for i in ids if i in wanted]
    if args.limit:
        ids = ids[: args.limit]
    size = http_size(args.url)
    read = make_reader(args.url, args.retries)
    if args.index.exists():
        raw = pickle.load(open(args.index, "rb"))
        archive = RangeZip.__new__(RangeZip)
        archive._read, archive.size = read, size
        archive.entries = {name: Entry(name, *values) for name, values in raw.items()}
    else:
        archive = RangeZip(read, size)
        args.index.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({n: (e.method, e.crc32, e.compressed_size, e.uncompressed_size, e.local_header_offset) for n, e in archive.entries.items()}, open(args.index, "wb"))
    found = archive.find_videos(ids, args.split)
    missing = [i for i in ids if i not in found]
    out_dir = args.output / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / f"{args.split}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"source": args.url, "archive_bytes": size, "videos": {}}
    total = sum(archive.entries[n].uncompressed_size for n in found.values())
    print(f"[PLAN] {len(found)} роликов ({total / 1e9:.2f} ГБ), отсутствуют в архиве: {len(missing)}", flush=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pending = []
    done = 0
    for video_id in sorted(found):
        entry = archive.entries[found[video_id]]
        if video_id in manifest["videos"] and file_ok(out_dir / f"{video_id}.mov", entry):
            done += 1
        else:
            pending.append(video_id)
    print(f"[PLAN] уже есть {done}, к загрузке {len(pending)}, потоков {args.workers}", flush=True)
    started = time.perf_counter()
    downloaded_bytes = 0

    def fetch_one(video_id: str) -> tuple[str, dict, int]:
        entry = archive.entries[found[video_id]]
        target = out_dir / f"{video_id}.mov"
        data = archive.extract(found[video_id])
        tmp = target.with_suffix(".mov.part")
        tmp.write_bytes(data)
        tmp.replace(target)
        record = {"member": found[video_id], "bytes": len(data), "crc32": entry.crc32, "sha256": hashlib.sha256(data).hexdigest(), **probe_video(target)}
        return video_id, record, len(data)

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = {pool.submit(fetch_one, video_id): video_id for video_id in pending}
        for index, future in enumerate(as_completed(futures), 1):
            video_id, record, nbytes = future.result()
            manifest["videos"][video_id] = record
            downloaded_bytes += nbytes
            done += 1
            if index % 10 == 0 or index == len(pending):
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
                elapsed = time.perf_counter() - started
                print(f"[FETCH] {index}/{len(pending)} {video_id}; всего {downloaded_bytes / 1e9:.2f} ГБ за {elapsed / 60:.1f} мин @ {downloaded_bytes / max(elapsed, 1e-6) / 1e6:.1f} МБ/с", flush=True)
    manifest["missing_in_archive"] = missing
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] готово {done}/{len(found)}; манифест {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
