"""Precompute the oscilloscope views: real samples, one trace at a time.

The envelope strips in waveform.py cannot show what the signal looks like. At
1200 buckets across 512 traces every bucket holds part of a sync burst, so the
envelope is nearly flat and the strip reads as a solid gold band whatever the
picture is. That was reported as "the waveforms aren't showing up anywhere",
and it was a fair report: nothing on the grid looked like a waveform.

What does look like one is the raw signal at the scale of a single trace -- the
sync burst, then the brightness of one column of the photograph as a voltage,
then the next burst. So this writes two things, both from the shipped FLACs:

  waves/scope.json      every grid card's strip: two consecutive traces from the
                        middle of the picture, as real min/max samples. One file
                        for the whole grid, for the reason cards.json is one file.

  waves/<id>.scope.png  every trace of ONE frame, one row per trace, for the
                        animation at the top of the page. A greyscale PNG,
                        because it is a 512 x 192 table of bytes and PNG is the
                        smallest thing a browser can unpack into one.

Both are pictures of the audio the browser actually fetches, measured from the
same FLACs the player decodes, not of the master.

Usage:
    python -m pipeline.scope                    # grid bundle + the hero frame
    python -m pipeline.scope --hero R043
    python -m pipeline.scope --check            # what CI runs: is it current?
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

from .waveform import CATALOG, WAVES


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Mono float samples of a shipped FLAC, via ffmpeg as raw int16 on a pipe.

    Not waveform.read_audio: ffmpeg writes WAVE_FORMAT_EXTENSIBLE headers that
    pipeline.wav does not parse, and on a machine without the `flac` CLI that
    route fails. Raw PCM has no header to disagree about. The rate is asked of
    ffprobe rather than assumed, and ffmpeg is not allowed to resample.
    """
    import subprocess
    rate = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, check=True, text=True).stdout.strip())
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le",
                          "-acodec", "pcm_s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0, rate

# Two traces, the same span the lightbox's sample strip shows, so a card and the
# strip under its opened picture look like the same thing at two sizes.
CARD_TRACES = 2
CARD_BUCKETS = 160
# Rows of the hero table. 192 points per trace is more than the animation's
# canvas is wide at desktop size divided by its trace count, and far below the
# ~800 samples a trace really has, which is the point: it is a picture of the
# trace, not a second copy of the audio.
HERO_POINTS = 192
HERO = "R043"            # the Taj Mahal, image 90
SCOPE = "scope.json"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.int8).tobytes()).decode()


def trace_slice(x: np.ndarray, lead_in: int, period: float, t: int, n: int = 1) -> np.ndarray:
    a = int(round(lead_in + t * period))
    b = int(round(lead_in + (t + n) * period))
    return x[max(0, a):min(len(x), b)]


def minmax(s: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.floor(np.arange(buckets) * len(s) / buckets).astype(np.int64)
    return np.minimum.reduceat(s, edges), np.maximum.reduceat(s, edges)


def card_entry(x: np.ndarray, frame: dict, traces: int) -> dict:
    # From the middle of the picture: the first and last columns of most of
    # these photographs are border, and a border trace is a sync burst and a
    # flat line, which is honest but tells you nothing.
    t = traces // 2
    s = trace_slice(x, frame["leadIn"], frame["periodGuess"], t, CARD_TRACES)
    lo, hi = minmax(s, CARD_BUCKETS)
    # Normalised to THIS slice, not to the frame's peak: the strip is 26px tall
    # and its job is to show the shape. The amplitude is in the lightbox.
    m = max(float(np.max(np.abs(s))), 1e-9)
    q = lambda v: np.clip(np.rint(v / m * 127), -127, 127).astype(np.int8)
    return {"trace": t, "min": _b64(q(lo)), "max": _b64(q(hi))}


def build_cards(catalog: Path = CATALOG, out: Path = WAVES) -> dict:
    cat = json.loads(catalog.read_text())
    frames = {f["id"]: f for f in cat["frames"]}
    traces = int(cat.get("tracesPerFrame", 512))
    data_dir = catalog.parent
    rec = {}
    for im in cat["images"]:
        fid = im["frames"][0]
        f = frames[fid]
        x, _ = read_audio(data_dir / (f.get("file") or f"frames/{fid}.flac"))
        rec[fid] = card_entry(x, f, traces)
    doc = {
        "what_this_is": (
            "The grid strips, as real samples: two traces from the middle of each "
            "card's picture, min/max per bucket, normalised to the slice. Written by "
            "pipeline/scope.py from the shipped FLACs."
        ),
        "traces": CARD_TRACES,
        "buckets": CARD_BUCKETS,
        "frames": rec,
    }
    (out / SCOPE).write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def build_hero(fid: str = HERO, catalog: Path = CATALOG, out: Path = WAVES) -> Path:
    from PIL import Image

    cat = json.loads(catalog.read_text())
    f = next(fr for fr in cat["frames"] if fr["id"] == fid)
    traces = int(cat.get("tracesPerFrame", 512))
    x, _ = read_audio(catalog.parent / (f.get("file") or f"frames/{fid}.flac"))
    rows = np.zeros((traces, HERO_POINTS), np.float32)
    for t in range(traces):
        s = trace_slice(x, f["leadIn"], f["periodGuess"], t)
        # Resampled, not min/max'd: one value per point keeps the trace a line,
        # which is what an oscilloscope draws.
        rows[t] = np.interp(np.linspace(0, len(s) - 1, HERO_POINTS), np.arange(len(s)), s)
    # One scale for the whole frame, so a dark column really is lower than a
    # bright one. Clipped at the 0.5th/99.5th percentiles: the sync burst
    # overshoots, and scaling to its tips would squash the picture.
    lo, hi = np.percentile(rows, [0.5, 99.5])
    q = np.clip(np.rint((rows - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
    path = out / f"{fid}.scope.png"
    Image.fromarray(q, "L").save(path, optimize=True)
    return path


def check(catalog: Path = CATALOG, out: Path = WAVES) -> list[str]:
    """Does scope.json carry a strip for every card, in the shape the page reads?

    Needs no audio, so CI can run it. It cannot re-derive the samples -- that
    needs the FLACs decoded -- but a missing card is the failure that would
    show as a blank strip, and that it can catch.
    """
    p = out / SCOPE
    if not p.exists():
        return [f"{SCOPE} missing -- run `python -m pipeline.scope`"]
    have = json.loads(p.read_text())
    cat = json.loads(catalog.read_text())
    bad = []
    for im in cat["images"]:
        fid = im["frames"][0]
        e = have.get("frames", {}).get(fid)
        if not e:
            bad.append(f"{SCOPE} has no strip for {fid} (image {im['n']})")
        elif len(base64.b64decode(e["min"])) != have.get("buckets"):
            bad.append(f"{SCOPE}: {fid} is not {have.get('buckets')} buckets")
    if not (out / f"{HERO}.scope.png").exists():
        bad.append(f"{HERO}.scope.png missing -- the page's opening animation reads it")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hero", default=HERO)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        bad = check()
        print("\n".join(f"  STALE: {b}" for b in bad) if bad else "  scope data present for every card")
        return 1 if bad else 0
    doc = build_cards()
    print(f"scope: {SCOPE} has {len(doc['frames'])} strips, "
          f"{(WAVES / SCOPE).stat().st_size / 1024:.0f} KiB")
    p = build_hero(args.hero)
    print(f"scope: {p.name}, {p.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
