"""Precompute per-frame waveform peaks for the gallery lightbox.

The lightbox must never wait on audio to draw its waveform strip: the FLAC for
one frame is ~500 KiB and decoding it costs a round trip plus a decode, while
the strip itself is only ever a few hundred pixels wide. So every frame gets a
min/max peak envelope computed here, once, at build time, and the page fetches
a few KiB of JSON instead.

The envelope is computed from **the shipped FLAC**, not from the master. That is
deliberate: the strip is a picture of the audio the browser will actually play,
so it has to be a picture of the asset the browser actually fetches. The FLACs
are bit-exact int16 (pipeline/build.py verifies the round trip), so this is also
the same data the decoder sees.

Output, one file per frame, at web/public/data/waves/<id>.json:

    id            frame id, e.g. "L000"
    source        catalog-relative path of the audio this was measured from
    sampleRate    96000, read from the decoded asset, not assumed
    samples       length of the asset in samples
    leadIn        samples of run-up before trace 0 (2000 in the current build)
    period        line period in samples, from the catalog's per-frame estimate
    traces        traces in the picture (512)
    buckets       number of min/max pairs below
    min, max      the envelope, int8, i.e. -127..127 of the asset's own full
                  scale. The assets are peak-normalised, so +-127 is the loudest
                  sample in that frame and nothing else.
    scale         multiply an envelope value by this to get master units:
                  peak / 127, where peak is the frame's pre-normalisation peak.

x-position to trace index, which is the whole point of shipping period and
leadIn alongside the envelope:

    sample = (bucket + 0.5) / buckets * samples
    trace  = (sample - leadIn) / period

Traces below 0 are the run-up; traces at or past `traces` are the tail lines
build.py keeps past the last trace start.

Usage:
    python -m pipeline.waveform                 # all 156 frames
    python -m pipeline.waveform --frames L000,R012
    python -m pipeline.waveform --buckets 1200 --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import wav

REPO = Path(__file__).resolve().parent.parent
WEB_DATA = REPO / "web" / "public" / "data"
CATALOG = WEB_DATA / "catalog.json"
WAVES = WEB_DATA / "waves"

DEFAULT_BUCKETS = 1200
INT8_FULL = 127.0


# --------------------------------------------------------------------------
# audio access
# --------------------------------------------------------------------------


def _decoder() -> list[str] | None:
    """The command that turns a FLAC into a WAV on stdout-to-file, or None."""
    if shutil.which("flac"):
        return ["flac", "--silent", "--decode", "--force"]
    if shutil.which("ffmpeg"):
        return ["ffmpeg", "-v", "error", "-y", "-i"]
    return None


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return (mono float32 samples in [-1, 1], sample rate).

    A .wav is parsed directly by pipeline.wav. A .flac is decoded to a temporary
    .wav by the `flac` CLI (or ffmpeg) and then parsed the same way, so exactly
    one WAV parser is in play and it is the one the rest of the pipeline uses.
    """
    if path.suffix.lower() == ".wav":
        return _read_wav(path)

    sibling = path.with_suffix(".wav")
    if sibling.exists():
        return _read_wav(sibling)

    cmd = _decoder()
    if cmd is None:
        raise RuntimeError(
            f"cannot read {path.name}: neither the `flac` CLI nor ffmpeg is on PATH, "
            "and there is no .wav sibling"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / (path.stem + ".wav")
        if cmd[0] == "flac":
            argv = [*cmd, "-o", str(out), str(path)]
        else:
            argv = [*cmd, str(path), str(out)]
        proc = subprocess.run(argv, capture_output=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"{cmd[0]} failed on {path.name}: "
                f"{proc.stderr.decode(errors='replace').strip()[:400]}"
            )
        return _read_wav(out)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    info = wav.probe(path)
    mm = wav.memmap(info)
    x = np.asarray(mm[:, 0])
    if info.dtype == np.dtype("<i2"):
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
    return x, info.sample_rate


# --------------------------------------------------------------------------
# envelope
# --------------------------------------------------------------------------


def envelope(x: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max of `x` over `buckets` contiguous, near-equal spans.

    reduceat over integer edges rather than a reshape: the frames are not a
    round multiple of the bucket count and dropping the remainder would put the
    right-hand end of the strip a few hundred samples out of register with the
    audio, which is exactly the alignment the playhead depends on.
    """
    n = len(x)
    if n == 0:
        raise ValueError("empty audio")
    buckets = max(1, min(buckets, n))
    edges = np.floor(np.arange(buckets, dtype=np.float64) * n / buckets).astype(np.int64)
    # reduceat requires strictly increasing edges; with buckets <= n they are.
    lo = np.minimum.reduceat(x, edges)
    hi = np.maximum.reduceat(x, edges)
    return lo, hi


def quantise(v: np.ndarray) -> list[int]:
    """Envelope floats to int8, clipped. The assets are peak-normalised, so this
    keeps ~0.8% of full scale per step -- invisible at any strip width we draw."""
    q = np.rint(np.clip(v, -1.0, 1.0) * INT8_FULL)
    return [int(t) for t in np.clip(q, -INT8_FULL, INT8_FULL)]


def build_frame(frame: dict, cat: dict, buckets: int, data_dir: Path) -> dict:
    src = frame.get("file") or f"frames/{frame['id']}.flac"
    path = data_dir / src
    if not path.exists():
        raise FileNotFoundError(f"{frame['id']}: {path} is missing")

    x, rate = read_audio(path)
    expected = frame.get("durationSamples")
    if expected and len(x) != expected:
        raise ValueError(
            f"{frame['id']}: asset is {len(x)} samples, catalog says {expected}"
        )
    cat_rate = cat.get("sampleRate", 96000)
    if rate != cat_rate:
        raise ValueError(f"{frame['id']}: asset is {rate} Hz, catalog says {cat_rate} Hz")

    lo, hi = envelope(x, buckets)
    peak = float(frame.get("peak") or 0.0)

    return {
        "id": frame["id"],
        "source": src,
        "sampleRate": int(rate),
        "samples": int(len(x)),
        "leadIn": int(frame.get("leadIn", 0)),
        "period": float(frame.get("periodGuess") or cat.get("nominalPeriod")),
        "traces": int(cat.get("tracesPerFrame", 512)),
        "buckets": int(len(lo)),
        "peak": round(peak, 6),
        "scale": round(peak / INT8_FULL, 9) if peak else 0.0,
        "min": quantise(lo),
        "max": quantise(hi),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--out", type=Path, default=WAVES)
    ap.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    ap.add_argument("--frames", default="", help="comma-separated frame ids")
    ap.add_argument("--force", action="store_true", help="rewrite files that already exist")
    args = ap.parse_args(argv)

    cat = json.loads(args.catalog.read_text())
    data_dir = args.catalog.parent
    wanted = {s.strip() for s in args.frames.split(",") if s.strip()}
    frames = [f for f in cat["frames"] if not wanted or f["id"] in wanted]
    if wanted:
        missing = wanted - {f["id"] for f in frames}
        if missing:
            print(f"unknown frame ids: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written, skipped, failed = 0, 0, []
    total_bytes = 0
    index = []

    for f in frames:
        dest = args.out / f"{f['id']}.json"
        if dest.exists() and not args.force:
            skipped += 1
            total_bytes += dest.stat().st_size
            continue
        try:
            rec = build_frame(f, cat, args.buckets, data_dir)
        except Exception as e:  # one bad frame must not lose the other 155
            failed.append((f["id"], str(e)))
            print(f"  {f['id']}: FAILED {e}", file=sys.stderr)
            continue
        dest.write_text(json.dumps(rec, separators=(",", ":")))
        size = dest.stat().st_size
        total_bytes += size
        written += 1
        index.append({"id": rec["id"], "buckets": rec["buckets"], "bytes": size})

    if written or skipped:
        (args.out / "_index.json").write_text(
            json.dumps(
                {
                    "what_this_is": (
                        "Per-frame min/max waveform envelopes for the gallery lightbox, "
                        "measured from the shipped FLACs by pipeline/waveform.py."
                    ),
                    "buckets": args.buckets,
                    "frames": len(frames),
                    "written": written,
                    "files": sorted(index, key=lambda r: r["id"]),
                },
                indent=1,
            )
        )

    dt = time.time() - t0
    print(
        f"waveform: {written} written, {skipped} already present, {len(failed)} failed "
        f"in {dt:.1f}s; {total_bytes / 1024:.0f} KiB total, "
        f"{total_bytes / max(1, written + skipped) / 1024:.1f} KiB/frame"
    )
    for fid, msg in failed:
        print(f"  FAILED {fid}: {msg}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
