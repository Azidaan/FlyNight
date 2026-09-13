"""Извлечь один файл из большого удалённого ZIP через HTTP Range."""

from __future__ import annotations

import argparse
import binascii
import struct
import urllib.request
import zlib
from pathlib import Path


EOCD = struct.Struct("<4s4H2LH")
ZIP64_LOCATOR = struct.Struct("<4sIQI")
ZIP64_EOCD = struct.Struct("<4sQ2H2I4Q")
CENTRAL = struct.Struct("<4s6H3I5H2I")
LOCAL = struct.Struct("<4s5H3I2H")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("member", help="точное имя или окончание пути внутри ZIP")
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def request(url: str, *, start: int | None = None, stop: int | None = None) -> bytes:
    headers = {"User-Agent": "BDD100K-range-extractor/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if stop is None else stop}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        data = response.read()
        if start is not None and response.status != 206:
            raise RuntimeError(f"Сервер проигнорировал Range: HTTP {response.status}")
        return data


def content_length(url: str) -> int:
    headers = {"User-Agent": "BDD100K-range-extractor/1.0"}
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers, method="HEAD")
    ) as response:
        return int(response.headers["Content-Length"])


def central_directory(url: str, total_size: int) -> bytes:
    tail_size = min(total_size, 4 * 1024 * 1024)
    tail_start = total_size - tail_size
    tail = request(url, start=tail_start, stop=total_size - 1)
    eocd_index = tail.rfind(b"PK\x05\x06")
    if eocd_index < 0 or eocd_index + EOCD.size > len(tail):
        raise RuntimeError("Конец центрального каталога ZIP не найден")
    fields = EOCD.unpack_from(tail, eocd_index)
    central_size = int(fields[5])
    central_offset = int(fields[6])
    if central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        locator_index = eocd_index - ZIP64_LOCATOR.size
        if locator_index < 0:
            raise RuntimeError("Указатель ZIP64 не найден")
        locator = ZIP64_LOCATOR.unpack_from(tail, locator_index)
        if locator[0] != b"PK\x06\x07":
            raise RuntimeError("Сигнатура указателя ZIP64 не совпала")
        zip64_offset = int(locator[2])
        if tail_start <= zip64_offset <= total_size - ZIP64_EOCD.size:
            begin = zip64_offset - tail_start
            zip64_data = tail[begin : begin + ZIP64_EOCD.size]
        else:
            zip64_data = request(
                url,
                start=zip64_offset,
                stop=zip64_offset + ZIP64_EOCD.size - 1,
            )
        zip64 = ZIP64_EOCD.unpack(zip64_data)
        if zip64[0] != b"PK\x06\x06":
            raise RuntimeError("Сигнатура каталога ZIP64 не совпала")
        central_size = int(zip64[8])
        central_offset = int(zip64[9])
    if central_offset >= tail_start and central_offset + central_size <= total_size:
        begin = central_offset - tail_start
        return tail[begin : begin + central_size]
    return request(
        url,
        start=central_offset,
        stop=central_offset + central_size - 1,
    )


def _zip64_values(row: dict, extra: bytes) -> dict:
    position = 0
    while position + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, position)
        payload = extra[position + 4 : position + 4 + size]
        position += 4 + size
        if kind != 0x0001:
            continue
        cursor = 0
        for key, sentinel, width in (
            ("uncompressed_size", 0xFFFFFFFF, 8),
            ("compressed_size", 0xFFFFFFFF, 8),
            ("local_offset", 0xFFFFFFFF, 8),
        ):
            if row[key] != sentinel:
                continue
            if cursor + width > len(payload):
                raise RuntimeError("Неполное расширенное поле ZIP64")
            row[key] = int.from_bytes(payload[cursor : cursor + width], "little")
            cursor += width
        return row
    if any(row[key] == 0xFFFFFFFF for key in ("compressed_size", "uncompressed_size", "local_offset")):
        raise RuntimeError("Для записи требуется отсутствующее расширенное поле ZIP64")
    return row


def find_member(directory: bytes, requested: str) -> dict:
    position = 0
    candidates = []
    while position + CENTRAL.size <= len(directory):
        fields = CENTRAL.unpack_from(directory, position)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError(f"Повреждён центральный каталог у байта {position}")
        name_length, extra_length, comment_length = fields[10:13]
        name_start = position + CENTRAL.size
        name = directory[name_start : name_start + name_length].decode("utf-8")
        extra_start = name_start + name_length
        extra = directory[extra_start : extra_start + extra_length]
        row = {
            "name": name,
            "flags": int(fields[3]),
            "method": int(fields[4]),
            "crc32": int(fields[7]),
            "compressed_size": int(fields[8]),
            "uncompressed_size": int(fields[9]),
            "local_offset": int(fields[16]),
        }
        row = _zip64_values(row, extra)
        if name == requested or name.endswith("/" + requested) or name.endswith(requested):
            candidates.append(row)
        position = name_start + name_length + extra_length + comment_length
    if len(candidates) != 1:
        names = [row["name"] for row in candidates]
        raise RuntimeError(f"Найдено совпадений: {len(candidates)}; {names[:10]}")
    return candidates[0]


def extract(url: str, member: dict, output: Path) -> None:
    local_offset = member["local_offset"]
    header = request(url, start=local_offset, stop=local_offset + LOCAL.size - 1)
    fields = LOCAL.unpack(header)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError("Локальный заголовок ZIP не найден")
    name_length, extra_length = int(fields[9]), int(fields[10])
    data_start = local_offset + LOCAL.size + name_length + extra_length
    compressed = request(
        url,
        start=data_start,
        stop=data_start + member["compressed_size"] - 1,
    )
    if member["method"] == 0:
        payload = compressed
    elif member["method"] == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Метод сжатия ZIP не поддержан: {member['method']}")
    if len(payload) != member["uncompressed_size"]:
        raise RuntimeError("Размер извлечённого файла не совпал с каталогом ZIP")
    actual_crc = binascii.crc32(payload) & 0xFFFFFFFF
    if actual_crc != member["crc32"]:
        raise RuntimeError("CRC32 извлечённого файла не совпал")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)


def main() -> int:
    args = parse_args()
    total = content_length(args.url)
    directory = central_directory(args.url, total)
    member = find_member(directory, args.member)
    print(
        f"[FOUND] {member['name']}: {member['uncompressed_size']} bytes, "
        f"method={member['method']}",
        flush=True,
    )
    extract(args.url, member, args.output)
    print(f"[OK] {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
