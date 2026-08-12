"""Download the 384 kHz master digitisation of the Golden Record image track.

1.4 GB, fetched once, by whoever builds the web assets. Visitors to the site
never see this file -- they get ~540 KB FLAC slices cut from it.

Source: https://archive.org/details/voyager.decode
        "Voyager Golden Record Images Waveform", Ron Barry and David Pescovitz.

Note the files live in a subdirectory whose name contains a space and an
ampersand, so the bare filename 404s. That tripped me up and is why the URL
below is spelled out rather than assembled.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = (
    "https://archive.org/download/voyager.decode/"
    "Voyager%20Video%20Audio%20%26%20Decode"
)
MASTER = f"{BASE}/384kHzStereo.wav"
EXPECTED_BYTES = 1_455_685_720

DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "master" / "384kHzStereo.wav"


def download(url: str, dest: Path, expected: int | None = None) -> Path:
    """Fetch `url` to `dest`, resuming if a partial file is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0

    if expected and have == expected:
        print(f"{dest.name} already complete ({have:,} bytes)")
        return dest
    if have:
        print(f"resuming at {have:,} bytes")

    req = urllib.request.Request(url, headers={"User-Agent": "golden-record-decode/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(req) as r:  # noqa: S310 - fixed https URL
        if have and r.status != 206:
            # Server ignored the range request; start over rather than corrupt.
            print("server refused to resume, restarting", file=sys.stderr)
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        done = have
        with dest.open(mode) as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {done/1e6:8.1f} / {total/1e6:.1f} MB  {pct:5.1f}%",
                          end="", flush=True)
        print()

    got = dest.stat().st_size
    if expected and got != expected:
        raise SystemExit(
            f"{dest} is {got:,} bytes, expected {expected:,}. Re-run to resume."
        )
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()

    path = download(MASTER, args.dest, EXPECTED_BYTES)

    # Confirm it is what we think it is before anyone builds against it.
    from . import wav

    info = wav.probe(path)
    print(
        f"{path}\n  {info.sample_rate} Hz, {info.channels} ch, {info.bits}-bit "
        f"(format tag {info.fmt_tag}), {info.duration:.2f} s"
    )
    if info.sample_rate != 384_000 or info.channels != 2:
        raise SystemExit("unexpected format; the pipeline assumes 384 kHz stereo")


if __name__ == "__main__":
    main()
