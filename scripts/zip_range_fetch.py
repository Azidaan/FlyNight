"""Выборочное чтение записей из большого zip (в том числе Zip64) по диапазонам байт.

Нужен для цели 1а: официальный архив bdd100k_videos.zip весит 1,97 ТБ, а
требуются только 1 400 роликов MOT-train. Центральный каталог читается с
конца архива, тела записей — по их смещениям; CRC32 проверяется.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Callable

RangeReader = Callable[[int, int], bytes]  # read(start, end_inclusive) -> bytes

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CENTRAL_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"


@dataclass(frozen=True)
class Entry:
    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


class RangeZip:
    def __init__(self, read: RangeReader, size: int, *, tail_bytes: int = 1 << 16) -> None:
        self._read = read
        self.size = int(size)
        self.entries = self._read_central_directory(tail_bytes)

    # --- каталог -----------------------------------------------------------------
    def _read_central_directory(self, tail_bytes: int) -> dict[str, Entry]:
        tail_start = max(self.size - tail_bytes, 0)
        tail = self._read(tail_start, self.size - 1)
        eocd_pos = tail.rfind(EOCD_SIG)
        if eocd_pos < 0:
            raise ValueError("EOCD не найден в хвосте архива")
        cd_size, cd_offset, entry_count = None, None, None
        loc_pos = tail.rfind(EOCD64_LOCATOR_SIG, 0, eocd_pos)
        if loc_pos >= 0:
            eocd64_offset = struct.unpack_from("<Q", tail, loc_pos + 8)[0]
            rel = eocd64_offset - tail_start
            if rel < 0:
                block = self._read(eocd64_offset, eocd64_offset + 56 - 1)
            else:
                block = tail[rel : rel + 56]
            if block[:4] == EOCD64_SIG:
                entry_count = struct.unpack_from("<Q", block, 32)[0]
                cd_size = struct.unpack_from("<Q", block, 40)[0]
                cd_offset = struct.unpack_from("<Q", block, 48)[0]
        if cd_size is None:
            entry_count = struct.unpack_from("<H", tail, eocd_pos + 10)[0]
            cd_size = struct.unpack_from("<I", tail, eocd_pos + 12)[0]
            cd_offset = struct.unpack_from("<I", tail, eocd_pos + 16)[0]
        directory = self._read(cd_offset, cd_offset + cd_size - 1)
        entries: dict[str, Entry] = {}
        position = 0
        while position + 46 <= len(directory) and directory[position : position + 4] == CENTRAL_SIG:
            method, = struct.unpack_from("<H", directory, position + 10)
            crc, comp, uncomp = struct.unpack_from("<III", directory, position + 16)
            name_len, extra_len, comment_len = struct.unpack_from("<HHH", directory, position + 28)
            local_offset, = struct.unpack_from("<I", directory, position + 42)
            name = directory[position + 46 : position + 46 + name_len].decode("utf-8", "replace")
            extra = directory[position + 46 + name_len : position + 46 + name_len + extra_len]
            comp, uncomp, local_offset = self._zip64_fields(extra, comp, uncomp, local_offset)
            entries[name] = Entry(name, method, crc, comp, uncomp, local_offset)
            position += 46 + name_len + extra_len + comment_len
        if entry_count is not None and len(entries) != entry_count:
            raise ValueError(f"Прочитано {len(entries)} записей, ожидалось {entry_count}")
        return entries

    @staticmethod
    def _zip64_fields(extra: bytes, comp: int, uncomp: int, offset: int) -> tuple[int, int, int]:
        position = 0
        while position + 4 <= len(extra):
            header_id, data_size = struct.unpack_from("<HH", extra, position)
            if header_id == 0x0001:
                cursor = position + 4
                if uncomp == 0xFFFFFFFF:
                    uncomp, = struct.unpack_from("<Q", extra, cursor); cursor += 8
                if comp == 0xFFFFFFFF:
                    comp, = struct.unpack_from("<Q", extra, cursor); cursor += 8
                if offset == 0xFFFFFFFF:
                    offset, = struct.unpack_from("<Q", extra, cursor); cursor += 8
                break
            position += 4 + data_size
        return comp, uncomp, offset

    # --- тела --------------------------------------------------------------------
    def extract(self, name: str, *, header_probe: int = 512) -> bytes:
        entry = self.entries[name]
        start = entry.local_header_offset
        # локальный заголовок вместе с началом тела одним запросом
        head = self._read(start, start + header_probe + entry.compressed_size - 1)
        if head[:4] != LOCAL_SIG:
            raise ValueError(f"{name}: локальный заголовок не найден по смещению {start}")
        name_len, extra_len = struct.unpack_from("<HH", head, 26)
        data_start = 30 + name_len + extra_len
        body = head[data_start : data_start + entry.compressed_size]
        if len(body) < entry.compressed_size:
            body += self._read(start + len(head), start + data_start + entry.compressed_size - 1)
        if entry.method == 0:
            data = body
        elif entry.method == 8:
            data = zlib.decompress(body, -15)
        else:
            raise ValueError(f"{name}: неподдерживаемый метод сжатия {entry.method}")
        if len(data) != entry.uncompressed_size or (zlib.crc32(data) & 0xFFFFFFFF) != entry.crc32:
            raise ValueError(f"{name}: размер или CRC32 не совпали")
        return data

    def find_videos(self, video_ids, split: str, *, prefixes=("bdd100k/videos/100k", "bdd100k/videos")) -> dict[str, str]:
        """ID видео -> имя записи; официальный архив хранит ролики под bdd100k/videos/100k/<split>/."""
        found = {}
        for video_id in video_ids:
            for prefix in prefixes:
                name = f"{prefix}/{split}/{video_id}.mov"
                if name in self.entries:
                    found[video_id] = name
                    break
        return found
