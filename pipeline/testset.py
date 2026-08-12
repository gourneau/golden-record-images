"""The fixed evaluation set, and the runner that scores the current decoder.

Any change to sync.py / decode.py is judged by:

    python -m pipeline.testset            # decode the 16 frames, print scores
    python -m pipeline.testset --validate # + metric self-tests (perturbations)
    python -m pipeline.testset --save-dir /tmp/x   # also dump the PNGs scored

The set is FIXED. Do not add the frames that happen to look good after a
change; that is how a metric stops being a metric. 16 frames spanning both
channels and the whole length of the record, mixing line-art diagrams (which
the current decoder handles) and detailed photographs (which it does not),
plus the calibration circle, which additionally gets the reference ellipse
fit from quality.circle_metrics.

Decoding runs from the 384 kHz master exactly the way pipeline/build.py does
(window around Barry's seed, decimate /4, sync.recover, decode), so changes to
the sync/decode pipeline propagate here with no rebuild step.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import catalog as catalog_mod
from . import decode as decode_mod
from . import quality
from . import sync as sync_mod
from . import wav

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "data" / "master" / "384kHzStereo.wav"

MASTER_RATE = 384_000
DECIM = 4
NOMINAL_96 = sync_mod.NOMINAL_PERIOD / DECIM
PRE_384 = int(round(1.5 * sync_mod.NOMINAL_PERIOD))

# (frame_id, kind, expectation-under-the-CURRENT-decoder)
# kind: what the picture is; expect: how the current decoder does on it,
# assigned by full-resolution inspection of fresh decodes (2026-08 baseline),
# not of the stale thumbnail build. The metric is validated by agreeing with
# this column, not by defining it. Labels are FROZEN with the set.
TEST_FRAMES: list[tuple[str, str, str]] = [
    ("L000", "calibration", "good"),   # circle visible; also gets ellipse fit
    ("L002", "line-art", "good"),      # math definitions: readable, striped
    ("L020", "line-art", "bad"),       # DNA: right half has a ~90 px staircase ramp
    ("L034", "line-art", "good"),      # conception diagram: left clean
    ("L040", "photo", "bad"),          # birth: destroyed by rank-1 smear
    ("L055", "photo", "good"),         # seashore: least-damaged photo decode
    ("L075", "line-art", "bad"),       # vertebrate evolution: smeared
    ("L077", "photo", "bad"),          # dolphins: destroyed, end of left ch.
    ("R000", "photo", "bad"),          # school of fish: partial, sync losses
    ("R010", "line-art", "bad"),       # bushmen sketch: heavy streaking
    ("R022", "photo", "bad"),          # schoolroom: smeared
    ("R040", "photo", "good"),         # house interior: recognisable
    ("R056", "photo", "good"),         # X-ray: structure intact, striped
    ("R068", "line-art", "bad"),       # Newton page: left third streak-destroyed
    ("R070", "photo", "bad"),          # astronaut: smeared
    ("R077", "photo", "bad"),          # violin: streaked, very end of record
]


def decode_frame(frame: catalog_mod.Frame, master: Path = MASTER):
    """Decode one frame from the master, the same way build.py does."""
    info = wav.probe(master)
    head = PRE_384
    span = int(math.ceil((sync_mod.TRACES_PER_FRAME + 6) * sync_mod.NOMINAL_PERIOD))
    start = frame.seed_sample - head
    if start < 0:
        raise ValueError(f"{frame.id}: seed too close to file start")
    x = np.asarray(
        wav.read(info, frame.channel, start, head + span), dtype=np.float64
    )
    x96 = resample_poly(x, 1, DECIM)
    tb = sync_mod.recover(
        x96, period_guess=NOMINAL_96, n_traces=sync_mod.TRACES_PER_FRAME
    )
    cfg = decode_mod.Settings()  # no rotation: metrics run in scan geometry
    dec = decode_mod.decode(np.asarray(x96, dtype=np.float32), cfg, tb)
    return dec


COLS = [
    ("composite", "score", "{:6.1f}"),
    ("shift_rms", "shft", "{:5.2f}"),
    ("drift_span", "drift", "{:6.1f}"),
    ("step_frac", "step%", "{:5.1f}"),
    ("stair_support", "stair", "{:5.2f}"),
    ("coh_ratio", "h/v", "{:5.3f}"),
    ("parity_db", "par dB", "{:6.1f}"),
    ("accept_frac", "lock%", "{:5.1f}"),
    ("sharpness", "sharp", "{:6.4f}"),
]
PCT = {"step_frac", "accept_frac"}


def _row(fid: str, kind: str, expect: str, m: dict) -> str:
    cells = [f"{fid:5} {kind:11} {expect:4}"]
    for key, _, fmt in COLS:
        v = m.get(key, float("nan"))
        if key in PCT and np.isfinite(v):
            v = 100.0 * v
        cells.append(fmt.format(v) if np.isfinite(v) else " " * len(fmt.format(0)) + "-")
    return " ".join(cells)


def run(frames=None, save_dir: Path | None = None, validate: bool = False) -> dict:
    cat = catalog_mod.build()
    wanted = frames or [f[0] for f in TEST_FRAMES]
    meta = {fid: (kind, exp) for fid, kind, exp in TEST_FRAMES}

    header = f"{'frame':5} {'kind':11} {'exp':4} " + " ".join(
        h.rjust(len(fmt.format(0))) for _, h, fmt in COLS
    )
    print(header)
    print("-" * len(header))

    scores: dict[str, dict] = {}
    images: dict[str, np.ndarray] = {}
    for fid in wanted:
        frame = cat.by_id(fid)
        kind, expect = meta.get(fid, ("?", "?"))
        try:
            dec = decode_frame(frame)
        except Exception as exc:
            print(f"{fid:5} DECODE FAILED: {type(exc).__name__}: {exc}")
            scores[fid] = {"composite": 0.0, "error": str(exc)}
            continue
        fs = quality.frame_report(dec.image, dec.timebase, frame_id=fid)
        scores[fid] = fs.metrics
        images[fid] = dec.image
        print(_row(fid, kind, expect, fs.metrics))
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            arr = (np.clip(dec.image, 0, 1) * 255 + 0.5).astype(np.uint8)
            Image.fromarray(arr, "L").save(save_dir / f"{fid}.png")

    comps = np.array([scores[f]["composite"] for f in wanted if "error" not in scores[f]])
    good = [f for f in wanted if meta.get(f, ("", ""))[1] == "good" and "error" not in scores[f]]
    bad = [f for f in wanted if meta.get(f, ("", ""))[1] == "bad" and "error" not in scores[f]]
    g = np.array([scores[f]["composite"] for f in good]) if good else np.array([np.nan])
    b = np.array([scores[f]["composite"] for f in bad]) if bad else np.array([np.nan])

    print("-" * len(header))
    print(f"AGGREGATE mean composite {comps.mean():6.1f}   "
          f"(known-good mean {g.mean():.1f}, known-bad mean {b.mean():.1f}, "
          f"separation {g.mean() - b.mean():+.1f})")

    # --- reference metric: the calibration circle -------------------------
    if "L000" in images:
        cm = quality.circle_metrics(images["L000"])
        if np.isfinite(cm.get("axis_ratio", float("nan"))):
            flag = "" if cm["ok"] else f"   ** FLAGGED: {cm['reason']} -- treat as failing"
            print(f"CIRCLE L000: axis_ratio {cm['axis_ratio']:.4f}  "
                  f"radial_rms {cm['radial_rms']:.2f} px  "
                  f"centre offset ({cm['center_dx']:+.1f}, {cm['center_dy']:+.1f}) px  "
                  f"[{cm['inliers']} inliers, {cm['coverage_deg']:.0f} deg coverage, "
                  f"axes {cm['major']:.1f}x{cm['minor']:.1f}]{flag}")
        else:
            print(f"CIRCLE L000: FIT REJECTED ({cm['reason']}) -- treat as failing")
        scores["L000_circle"] = cm

    if validate and images:
        _validate(images, scores)

    return scores


def _validate(images: dict[str, np.ndarray], scores: dict) -> None:
    """Metric self-tests on the real decodes. The composite must fall for
    injected timing/parity damage and must NOT rise for cosmetic sharpening,
    while the sharpness column does rise -- proving the score is not
    restating sharpness."""
    print()
    print("METRIC VALIDATION (image-space perturbations; sync factors held fixed)")
    best = max(images, key=lambda f: scores[f].get("composite", 0.0))
    img = images[best]
    base = scores[best]

    def rescore(im):
        m = dict(base)
        m.update(quality.drift_metrics(im))
        m.update(quality.staircase_metrics(im))
        m.update(quality.axis_coherence(im))
        m.update(quality.parity_metrics(im))
        m["sharpness"] = quality.sharpness(im)
        m["composite"] = quality.composite_score(m)
        return m

    cases = [
        ("original", img),
        ("staircase +24px/8steps", quality.inject_staircase(img, 24.0, 8)),
        ("staircase +8px/4steps", quality.inject_staircase(img, 8.0, 4)),
        ("parity stripes 0.05", quality.inject_parity(img, 0.05)),
        ("unsharp x1.5", quality.unsharp(img, 1.5)),
    ]
    print(f"  base frame {best}")
    print(f"  {'case':24} {'composite':>9} {'sharpness':>9}")
    for name, im in cases:
        m = rescore(im)
        print(f"  {name:24} {m['composite']:9.1f} {m['sharpness']:9.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="", help="comma-separated subset")
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    frames = [s.strip() for s in args.frames.split(",") if s.strip()] or None
    run(frames, args.save_dir, args.validate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
