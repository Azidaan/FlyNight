from __future__ import annotations

import binascii
import struct

import pytest

from scripts.extract_remote_zip_member import CENTRAL, find_member


def central_record(name: str, offset: int = 123) -> bytes:
    encoded = name.encode("utf-8")
    fixed = CENTRAL.pack(
        b"PK\x01\x02",
        20,
        20,
        0,
        0,
        0,
        0,
        binascii.crc32(b"payload") & 0xFFFFFFFF,
        7,
        7,
        len(encoded),
        0,
        0,
        0,
        0,
        0,
        offset,
    )
    return fixed + encoded


def test_find_member_by_basename() -> None:
    directory = central_record("videos/first.mov") + central_record("videos/second.mov")
    match = find_member(directory, "second.mov")
    assert match["name"] == "videos/second.mov"
    assert match["local_offset"] == 123


def test_find_member_rejects_missing_name() -> None:
    with pytest.raises(RuntimeError, match="0"):
        find_member(central_record("videos/first.mov"), "missing.mov")


def test_find_member_reads_zip64_local_offset() -> None:
    name = b"videos/large.mov"
    actual_offset = 9_000_000_000
    extra_payload = actual_offset.to_bytes(8, "little")
    extra = struct.pack("<HH", 0x0001, len(extra_payload)) + extra_payload
    fixed = CENTRAL.pack(
        b"PK\x01\x02",
        45,
        45,
        0,
        0,
        0,
        0,
        0,
        7,
        7,
        len(name),
        len(extra),
        0,
        0,
        0,
        0,
        0xFFFFFFFF,
    )
    match = find_member(fixed + name + extra, "large.mov")
    assert match["local_offset"] == actual_offset
