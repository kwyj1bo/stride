import struct

import fitdecode
import numpy as np
import pandas as pd
import pytest

from stride.io.fit import read_fit
from stride.io.fit_protocol import FitDecoder, FitError, crc16
from stride.io.fit_writer import write_fit


@pytest.fixture
def samples():
    t = pd.date_range("2026-11-01T01:00:00Z", periods=5, freq="1s")
    return pd.DataFrame(
        {
            "t": t,
            "lat": [19.0760, 19.07605, 19.0761, np.nan, 19.0762],
            "lon": [72.8777, 72.87775, 72.8778, np.nan, 72.8779],
            "ele": [10.0, 10.2, 10.4, 10.6, -3.0],
            "hr": [120, 121, np.nan, 123, 124],
            "cadence": [80, 81, 82, 83, 84],
            "distance": [0.0, 5.55, 11.1, 16.65, 22.2],
            "speed": [5.5, 5.55, 5.6, 5.65, 5.7],
        }
    )


def test_round_trip_keeps_values_within_fit_resolution(tmp_path, samples):
    path = tmp_path / "a.fit"
    write_fit(path, samples, sport="running")
    raw = read_fit(path)
    s = raw.samples
    assert raw.sport == "running"
    assert (s["t"] == samples["t"]).all()
    np.testing.assert_allclose(s["lat"], samples["lat"], atol=1e-6)  # 1 semicircle ~ 8e-8 deg
    np.testing.assert_allclose(s["ele"], samples["ele"], atol=0.1)
    np.testing.assert_allclose(s["distance"], samples["distance"], atol=0.005)
    np.testing.assert_allclose(s["speed"], samples["speed"], atol=0.0005)
    assert np.isnan(s["hr"].iloc[2]) and np.isnan(s["lat"].iloc[3])


def test_independent_decoder_agrees(tmp_path, samples):
    path = tmp_path / "a.fit"
    write_fit(path, samples)
    with fitdecode.FitReader(str(path)) as reader:
        records = [
            frame
            for frame in reader
            if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "record"
        ]
    assert len(records) == 5
    ours = read_fit(path).samples
    for frame, (_, row) in zip(records, ours.iterrows(), strict=True):
        assert frame.get_value("timestamp") == row["t"].to_pydatetime()
        assert frame.get_value("enhanced_altitude") == pytest.approx(row["ele"])
        assert frame.get_value("distance") == pytest.approx(row["distance"])


def test_corrupted_byte_fails_crc(tmp_path, samples):
    path = tmp_path / "a.fit"
    write_fit(path, samples)
    data = bytearray(path.read_bytes())
    data[40] ^= 0xFF
    with pytest.raises(FitError, match="CRC"):
        FitDecoder(bytes(data)).messages()


def test_truncated_file_is_rejected(tmp_path, samples):
    path = tmp_path / "a.fit"
    write_fit(path, samples)
    with pytest.raises(FitError, match="truncated"):
        FitDecoder(path.read_bytes()[:-10]).messages()


def _wrap(body: bytes) -> bytes:
    header = struct.pack("<BBHI4s", 12, 0x10, 100, len(body), b".FIT")
    content = header + body
    return content + struct.pack("<H", crc16(content))


def test_compressed_timestamps_and_developer_fields():
    # local 0: record with timestamp + heart rate, plus one 2-byte developer field
    body = struct.pack("<BBBHB", 0x60, 0, 0, 20, 2)
    body += struct.pack("<BBBBBB", 253, 4, 0x86, 3, 1, 0x02)
    body += struct.pack("<BBBB", 1, 0, 2, 0)  # one developer field, 2 bytes
    body += struct.pack("<BIB", 0x00, 1030, 100) + b"\xaa\xbb"
    # local 1: record with heart rate only, for compressed-timestamp headers
    body += struct.pack("<BBBHB", 0x41, 0, 0, 20, 1) + struct.pack("<BBB", 3, 1, 0x02)
    # 1030 has low bits 6; offset 2 means the 5-bit counter rolled over
    body += struct.pack("<BB", 0x80 | (1 << 5) | 2, 101)
    body += struct.pack("<BB", 0x80 | (1 << 5) | 3, 102)
    messages = FitDecoder(_wrap(body)).messages()
    assert [values[253] for _, values in messages] == [1030, 1058, 1059]
    assert [values[3] for _, values in messages] == [100, 101, 102]


def test_big_endian_definition():
    body = struct.pack("<BBB", 0x40, 0, 1) + struct.pack(">H", 20) + struct.pack("<B", 1)
    body += struct.pack("<BBB", 253, 4, 0x86)
    body += struct.pack("<B", 0) + struct.pack(">I", 123456)
    assert FitDecoder(_wrap(body)).messages() == [(20, {253: 123456})]
