"""Decoder for the binary FIT protocol used by Garmin and most sports watches.

A FIT file is a header, a stream of records and a CRC. Each record is either a
definition (the layout of a message type) or data laid out by the last definition for
its local type. This decoder keeps only the messages the caller asks for, but walks
every record so that an unknown or developer field can never shift the read position.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

FIT_EPOCH_OFFSET = 631065600  # seconds from 1970-01-01 to the FIT epoch, 1989-12-31

# base type id -> (struct code, size, invalid value)
BASE_TYPES = {
    0x00: ("B", 1, 0xFF),  # enum
    0x01: ("b", 1, 0x7F),
    0x02: ("B", 1, 0xFF),
    0x83: ("h", 2, 0x7FFF),
    0x84: ("H", 2, 0xFFFF),
    0x85: ("i", 4, 0x7FFFFFFF),
    0x86: ("I", 4, 0xFFFFFFFF),
    0x07: ("s", 1, None),  # string
    0x88: ("f", 4, None),
    0x89: ("d", 8, None),
    0x0A: ("B", 1, 0x00),  # uint8z
    0x8B: ("H", 2, 0x0000),
    0x8C: ("I", 4, 0x00000000),
    0x0D: ("B", 1, 0xFF),  # byte
    0x8E: ("q", 8, 0x7FFFFFFFFFFFFFFF),
    0x8F: ("Q", 8, 0xFFFFFFFFFFFFFFFF),
    0x90: ("Q", 8, 0x0000000000000000),
}

TIMESTAMP_FIELD = 253

_CRC_TABLE = (
    0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
    0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
)  # fmt: skip


class FitError(ValueError):
    pass


@dataclass(frozen=True)
class FieldDef:
    number: int
    size: int
    base_type: int


@dataclass(frozen=True)
class MessageDef:
    global_number: int
    big_endian: bool
    fields: tuple[FieldDef, ...]
    developer_size: int  # bytes of developer fields to skip


class FitDecoder:
    """Decode a FIT file into (global message number, {field number: value}) pairs."""

    def __init__(self, data: bytes, keep: set[int] | None = None) -> None:
        self.data = data
        self.keep = keep
        self.definitions: dict[int, MessageDef] = {}
        self.last_timestamp: int | None = None

    def messages(self) -> list[tuple[int, dict[int, object]]]:
        start, end = self._check_header_and_crc()
        out = []
        pos = start
        while pos < end:
            header = self.data[pos]
            pos += 1
            if header & 0x80:
                pos = self._compressed_data(header, pos, out)
            elif header & 0x40:
                pos = self._definition(header, pos)
            else:
                pos = self._data(header & 0x0F, pos, out, None)
        if pos != end:
            raise FitError("record stream overruns the data size in the header")
        return out

    def _check_header_and_crc(self) -> tuple[int, int]:
        if len(self.data) < 12:
            raise FitError("file is too short for a FIT header")
        header_size = self.data[0]
        if self.data[8:12] != b".FIT":
            raise FitError("missing .FIT signature")
        data_size = struct.unpack_from("<I", self.data, 4)[0]
        end = header_size + data_size
        if len(self.data) < end + 2:
            raise FitError("file is truncated")
        if header_size == 14:
            header_crc = struct.unpack_from("<H", self.data, 12)[0]
            if header_crc not in (0, crc16(self.data[:12])):
                raise FitError("header CRC mismatch")
        if struct.unpack_from("<H", self.data, end)[0] != crc16(self.data[:end]):
            raise FitError("file CRC mismatch")
        return header_size, end

    def _definition(self, header: int, pos: int) -> int:
        local = header & 0x0F
        has_developer = bool(header & 0x20)
        big_endian = self.data[pos + 1] == 1
        global_number = struct.unpack_from(">H" if big_endian else "<H", self.data, pos + 2)[0]
        count = self.data[pos + 4]
        pos += 5
        fields = []
        for _ in range(count):
            number, size, base_type = self.data[pos : pos + 3]
            fields.append(FieldDef(number, size, base_type))
            pos += 3
        developer_size = 0
        if has_developer:
            dev_count = self.data[pos]
            pos += 1
            developer_size = sum(self.data[pos + 3 * i + 1] for i in range(dev_count))
            pos += 3 * dev_count
        self.definitions[local] = MessageDef(
            global_number, big_endian, tuple(fields), developer_size
        )
        return pos

    def _compressed_data(self, header: int, pos: int, out: list) -> int:
        if self.last_timestamp is None:
            raise FitError("compressed timestamp before any full timestamp")
        offset = header & 0x1F
        timestamp = (self.last_timestamp & ~0x1F) + offset
        if offset < (self.last_timestamp & 0x1F):
            timestamp += 0x20  # the 5-bit offset rolled over
        return self._data((header >> 5) & 0x03, pos, out, timestamp)

    def _data(self, local: int, pos: int, out: list, timestamp: int | None) -> int:
        definition = self.definitions.get(local)
        if definition is None:
            raise FitError(f"data record for undefined local message {local}")
        values: dict[int, object] = {}
        for field_def in definition.fields:
            values[field_def.number] = _decode_field(self.data, pos, field_def, definition)
            pos += field_def.size
        pos += definition.developer_size
        if timestamp is not None:
            values[TIMESTAMP_FIELD] = timestamp
        if isinstance(values.get(TIMESTAMP_FIELD), int):
            self.last_timestamp = values[TIMESTAMP_FIELD]
        if self.keep is None or definition.global_number in self.keep:
            out.append((definition.global_number, values))
        return pos


def crc16(data: bytes, crc: int = 0) -> int:
    """The FIT CRC: CRC-16 with polynomial 0xA001, processed a nibble at a time."""
    for byte in data:
        for nibble in (byte & 0x0F, byte >> 4):
            tmp = _CRC_TABLE[crc & 0x0F]
            crc = (crc >> 4) & 0x0FFF
            crc = crc ^ tmp ^ _CRC_TABLE[nibble]
    return crc


def _decode_field(data: bytes, pos: int, field_def: FieldDef, definition: MessageDef):
    """One field value, or None when it holds the type's invalid marker. Arrays and
    unknown base types come back as None too: the readers here only need scalars."""
    spec = BASE_TYPES.get(field_def.base_type)
    if spec is None:
        return None
    code, size, invalid = spec
    raw = data[pos : pos + field_def.size]
    if code == "s":
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace") or None
    if field_def.size != size:
        return None
    value = struct.unpack((">" if definition.big_endian else "<") + code, raw)[0]
    if invalid is not None and value == invalid:
        return None
    if code in "fd" and value != value:  # NaN is the float invalid marker
        return None
    return value
