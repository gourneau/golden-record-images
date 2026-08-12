"""Memory-mapped reader for the 384 kHz Voyager master WAV.

The file is 1.4 GB of IEEE float32 stereo at 384 kHz (473.86 s). We never load
it whole -- every consumer wants a few seconds around one frame -- so this
parses the RIFF chunks once and hands back a numpy memmap view.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3


@dataclass(frozen=True)
class WavInfo:
    path: Path
    data_offset: int
    data_bytes: int
    sample_rate: int
    channels: int
    bits: int
    fmt_tag: int

    @property
    def dtype(self) -> np.dtype:
        if self.fmt_tag == WAVE_FORMAT_IEEE_FLOAT and self.bits == 32:
            return np.dtype("<f4")
        if self.fmt_tag == WAVE_FORMAT_PCM and self.bits == 16:
            return np.dtype("<i2")
        raise ValueError(f"unsupported format tag={self.fmt_tag} bits={self.bits}")

    @property
    def n_frames(self) -> int:
        return self.data_bytes // (self.dtype.itemsize * self.channels)

    @property
    def duration(self) -> float:
        return self.n_frames / self.sample_rate


def probe(path: str | Path) -> WavInfo:
    path = Path(path)
    with path.open("rb") as f:
        if f.read(4) != b"RIFF":
            raise ValueError(f"{path} is not a RIFF file")
        f.seek(8)
        if f.read(4) != b"WAVE":
            raise ValueError(f"{path} is not a WAVE file")

        fmt = None
        while True:
            head = f.read(8)
            if len(head) < 8:
                raise ValueError(f"{path}: no data chunk found")
            cid, size = struct.unpack("<4sI", head)
            if cid == b"fmt ":
                fmt = struct.unpack("<HHIIHH", f.read(16)[:16])
                f.seek(size - 16, 1)
            elif cid == b"data":
                if fmt is None:
                    raise ValueError(f"{path}: data chunk precedes fmt chunk")
                tag, channels, rate, _byte_rate, _align, bits = fmt
                return WavInfo(
                    path=path,
                    data_offset=f.tell(),
                    data_bytes=size,
                    sample_rate=rate,
                    channels=channels,
                    bits=bits,
                    fmt_tag=tag,
                )
            else:
                f.seek(size + (size & 1), 1)


def memmap(info: WavInfo, *, allow_truncated: bool = False) -> np.ndarray:
    """Return a (n_frames, channels) memmap. Read-only.

    `allow_truncated` lets us work against a partially-downloaded file: the
    header advertises the full length long before the bytes have landed.
    """
    on_disk = info.path.stat().st_size - info.data_offset
    usable = info.data_bytes
    if on_disk < info.data_bytes:
        if not allow_truncated:
            raise ValueError(
                f"{info.path} is short: {on_disk} of {info.data_bytes} data bytes present"
            )
        usable = on_disk

    itemsize = info.dtype.itemsize * info.channels
    n = usable // itemsize
    return np.memmap(
        info.path, dtype=info.dtype, mode="r", offset=info.data_offset, shape=(n, info.channels)
    )


def read(info: WavInfo, channel: int, start: int, count: int, *, mm: np.ndarray | None = None):
    """Read `count` samples of one channel starting at absolute sample `start`."""
    if mm is None:
        mm = memmap(info, allow_truncated=True)
    start = max(0, start)
    stop = min(len(mm), start + count)
    return np.asarray(mm[start:stop, channel], dtype=np.float32)
