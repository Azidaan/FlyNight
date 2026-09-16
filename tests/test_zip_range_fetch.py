"""Цель 1а: выборочное чтение записей из большого zip по диапазонам байт."""

from __future__ import annotations

import zipfile
import zlib
from pathlib import Path

import numpy as np

from scripts.zip_range_fetch import RangeZip


class FileRanges:
    """Заменитель HTTP: читает диапазоны из локального файла и считает запросы."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.size = path.stat().st_size
        self.calls = 0

    def read(self, start: int, end: int) -> bytes:
        self.calls += 1
        with self.path.open("rb") as handle:
            handle.seek(start)
            return handle.read(end - start + 1)


def _make_zip(path: Path, *, stored: bool) -> dict[str, bytes]:
    rng = np.random.default_rng(0)
    members = {f"bdd100k/videos/train/{i:04d}-abc.mov": rng.integers(0, 256, 5000 + 700 * i, dtype=np.uint8).tobytes() for i in range(6)}
    members["bdd100k/videos/val/zz.mov"] = b"val-video-bytes" * 100
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return members


def test_central_directory_lists_members_with_offsets(tmp_path: Path) -> None:
    members = _make_zip(tmp_path / "a.zip", stored=True)
    source = FileRanges(tmp_path / "a.zip")
    archive = RangeZip(source.read, source.size)
    assert set(archive.entries) == set(members)
    entry = archive.entries["bdd100k/videos/val/zz.mov"]
    assert entry.uncompressed_size == len(members["bdd100k/videos/val/zz.mov"])
    assert entry.method == 0
    assert source.calls <= 3  # хвост + каталог, без чтения тел


def test_extract_member_verifies_crc_for_stored_and_deflated(tmp_path: Path) -> None:
    for stored in (True, False):
        members = _make_zip(tmp_path / f"{int(stored)}.zip", stored=stored)
        source = FileRanges(tmp_path / f"{int(stored)}.zip")
        archive = RangeZip(source.read, source.size)
        name = "bdd100k/videos/train/0003-abc.mov"
        data = archive.extract(name)
        assert data == members[name]
        assert zlib.crc32(data) & 0xFFFFFFFF == archive.entries[name].crc32


def test_extract_only_reads_the_member_bytes(tmp_path: Path) -> None:
    _make_zip(tmp_path / "a.zip", stored=True)
    source = FileRanges(tmp_path / "a.zip")
    archive = RangeZip(source.read, source.size)
    before = source.calls
    archive.extract("bdd100k/videos/train/0001-abc.mov")
    assert source.calls - before <= 2  # локальный заголовок + тело (или одним запросом)


def test_member_names_by_video_id(tmp_path: Path) -> None:
    _make_zip(tmp_path / "a.zip", stored=True)
    source = FileRanges(tmp_path / "a.zip")
    archive = RangeZip(source.read, source.size)
    found = archive.find_videos(["0002-abc", "zz", "missing"], split="train")
    assert found == {"0002-abc": "bdd100k/videos/train/0002-abc.mov"}
