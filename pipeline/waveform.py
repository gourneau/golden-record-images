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
# the grid bundle
# --------------------------------------------------------------------------
# The per-frame files above are right for the lightbox, which draws one frame at
# a time and wants every bucket. They were wrong for the grid, and the failure
# was reported as "the waveforms only load for the first few images".
#
# The grid drew each strip from its own fetch, fired when the card was painted.
# That is 116 requests racing 232 thumbnails, and the page cached a failure as a
# permanent null -- one lost request meant that strip never appeared again, no
# matter how much you scrolled, while the thumbnail beside it loaded fine. The
# tail of a slow or flaky connection is exactly the part that goes missing, and
# exactly the part you reach by scrolling down.
#
# So the grid gets ONE file instead of 116. It is a fourfold downsample of the
# envelopes already committed here, base64 int8 rather than JSON numbers, which
# is about 90 KiB for the whole gallery -- less than one thumbnail. There is no
# per-card request left to lose.
#
# Built from the per-frame JSON, not from the audio: it needs no FLACs, no
# master and no `flac` binary, so it can be regenerated in any clone and cannot
# drift away from the strips the lightbox draws.

CARD_BUCKETS = 300

# NOT "_cards.json". GitHub Pages runs Jekyll, which silently drops every path
# beginning with an underscore -- the bundle deployed and returned 404, and so
# does the _index.json manifest beside it, which nothing had ever fetched so
# nobody noticed. The repo now carries a .nojekyll, and this name does not
# depend on that file being honoured.
CARDS = "cards.json"


def downsample(vals: list[int], buckets: int, reduce) -> np.ndarray:
    """Reduce an envelope to `buckets` spans, the same way `envelope` cut it.

    Min and max must be reduced with their own operator: averaging them, or
    subsampling, would pull the two rails together and quietly shrink every
    loud passage on the strip.
    """
    a = np.asarray(vals, dtype=np.int8)
    n = len(a)
    if n <= buckets:
        return a
    edges = np.floor(np.arange(buckets, dtype=np.float64) * n / buckets).astype(np.int64)
    return reduce.reduceat(a, edges).astype(np.int8)


def _b64(a: np.ndarray) -> str:
    import base64
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.int8).tobytes()).decode()


def build_cards(out: Path, catalog: Path, buckets: int = CARD_BUCKETS) -> dict:
    """Bundle the envelopes the grid needs into one file. Returns a summary."""
    cat = json.loads(catalog.read_text())
    # Only the frames the grid actually draws: a card shows its first frame, so
    # a colour image contributes one strip and not three.
    wanted = [im["frames"][0] for im in cat["images"]]

    frames, missing = {}, []
    for fid in wanted:
        src = out / f"{fid}.json"
        if not src.exists():
            missing.append(fid)
            continue
        d = json.loads(src.read_text())
        frames[fid] = {
            "samples": d["samples"],
            "leadIn": d["leadIn"],
            "period": d["period"],
            "traces": d["traces"],
            "scale": d["scale"],
            "min": _b64(downsample(d["min"], buckets, np.minimum)),
            "max": _b64(downsample(d["max"], buckets, np.maximum)),
        }

    rec = {
        "what_this_is": (
            "Every grid strip in one file. The gallery drew these from 116 separate "
            "fetches and the tail of them went missing on a slow connection, so the "
            "grid now makes one request and the per-frame files serve the lightbox "
            "alone. Downsampled from those files by pipeline/waveform.py --cards."
        ),
        "buckets": buckets,
        "encoding": "min and max are base64 int8, full scale +-127; multiply by scale for master units",
        "frames": frames,
    }
    if missing:
        rec["missing"] = missing
    (out / CARDS).write_text(json.dumps(rec, separators=(",", ":")))
    return {"frames": len(frames), "missing": missing,
            "bytes": (out / CARDS).stat().st_size}


def check_cards(out: Path = WAVES, catalog: Path = CATALOG,
                buckets: int = CARD_BUCKETS) -> list[str]:
    """Is the shipped grid bundle what the per-frame files say it should be?

    The bundle is DERIVED data, which on this repository is the thing that
    quietly goes wrong: build.py dropped the presentation metadata, a docstring
    outlived its model, a figure outlived its decode. Each was caught by someone
    looking. A bundle regenerated from stale frames -- or not regenerated at all
    -- would show the wrong sound under the right picture, and nothing on the
    page would look broken. So it is compared byte for byte here.

    Needs no audio, so CI can run it.
    """
    if not (out / CARDS).exists():
        return [f"{CARDS} missing -- run `python -m pipeline.waveform --cards`"]
    cat = json.loads(catalog.read_text())
    have = json.loads((out / CARDS).read_text())
    frames = have.get("frames") or {}
    if have.get("buckets") != buckets:
        return [f"{CARDS} has {have.get('buckets')} buckets, this code writes {buckets}"]

    problems = []
    for im in cat["images"]:
        fid = im["frames"][0]
        if fid not in frames:
            problems.append(f"{CARDS} has no strip for {fid} (image {im.get('n')})")
            continue
        src = out / f"{fid}.json"
        if not src.exists():
            continue                       # already reported by the build
        d = json.loads(src.read_text())
        want = {"samples": d["samples"], "leadIn": d["leadIn"], "period": d["period"],
                "traces": d["traces"], "scale": d["scale"],
                "min": _b64(downsample(d["min"], buckets, np.minimum)),
                "max": _b64(downsample(d["max"], buckets, np.maximum))}
        for k, v in want.items():
            if frames[fid].get(k) != v:
                problems.append(f"{CARDS}: {fid}.{k} does not match {fid}.json")
                break
    extra = set(frames) - {im["frames"][0] for im in cat["images"]}
    if extra:
        problems.append(f"{CARDS} carries {len(extra)} strips no card draws: "
                        f"{', '.join(sorted(extra)[:5])}")
    return problems


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
    ap.add_argument("--cards", action="store_true",
                    help="only rebuild the grid bundle, from the per-frame files already here")
    ap.add_argument("--check-cards", action="store_true",
                    help="verify the shipped grid bundle against the per-frame files")
    args = ap.parse_args(argv)

    if args.check_cards:
        bad = check_cards(args.out, args.catalog)
        print("\n".join(f"  STALE: {b}" for b in bad) if bad
              else f"  {CARDS} matches the per-frame envelopes beside it")
        return 1 if bad else 0

    if args.cards:
        s = build_cards(args.out, args.catalog)
        print(f"waveform: {CARDS} has {s['frames']} strips, {s['bytes'] / 1024:.0f} KiB")
        for fid in s["missing"]:
            print(f"  MISSING {fid}.json", file=sys.stderr)
        return 1 if s["missing"] else 0

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

    # Always after the per-frame files, never conditionally: the grid bundle is
    # derived from them, so a run that rewrites one frame and leaves the bundle
    # alone would ship a strip that disagrees with the lightbox beside it.
    if written or skipped:
        s = build_cards(args.out, args.catalog)
        print(f"  {CARDS}: {s['frames']} grid strips, {s['bytes'] / 1024:.0f} KiB "
              f"(one request, replacing {s['frames']})")
        for fid in s["missing"]:
            print(f"  MISSING {fid}.json for the grid bundle", file=sys.stderr)

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
