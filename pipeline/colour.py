"""Assemble the 20 colour images from their r/g/b separation triplets.

CHANNEL ORDER -- SETTLED
------------------------
The three frames of a triplet are, in record order, RED, GREEN, BLUE. That is
Barry's order; the b,g,r order asserted in amazing-rando's README is wrong.

Evidence (measured 2026-08-12 against this master, scripts in the session
scratchpad, results archived in data/colour/colour_report.json):

  * All 20 decoded triplets were registered against the canonical NAIC/re-lab
    source images (the actual slides photographed for the record, fetched via
    the Wayback Machine -- an independent colour ground truth, not anyone's
    record decode). Statistic per triplet:

        KEY = corr( hp(z sep1) - hp(z sep3),  hp(z refRED) - hp(z refBLUE) )

    where z is a per-image z-score (kills per-separation exposure), the 1-3
    difference cancels shared luminance, and hp is an 18 px Gaussian high-pass
    (kills AC-droop and vignette). KEY > 0 on 20 OF 20 TRIPLETS (sign test
    p ~ 1e-6), from +0.006 (a misregistered reference) to +0.37, with every
    well-registered case above +0.13.
  * The solar spectrum (image 8, L007-009) confirms it physically: the bright
    band sits ~97 px toward the canonical RED end in separation 1 and ~83 px
    toward the BLUE end in separation 3, with the X-shaped registration mark
    that the canonical slide draws in the green centre visible in separation 2.

  A WARNING about the obvious shortcut: whole-frame mean levels are NOT valid
  evidence. Measured means for the sunset triplet (image 114) are -0.009,
  -0.002, +0.000 -- the red separation is the *darkest* even though the scene
  is orange, because per-separation exposure differences of the 1977 chain
  swamp scene hue. Judging that composite by its global cast points to b,g,r;
  the reference correlation and the within-scene hue gradient (corr of
  (z1 - z3) with distance toward the sun: +0.16) both say r,g,b. This trap is
  presumably how the b,g,r claim was born.

REGISTRATION
------------
The separations are three scans taken seconds apart and land within a few
pixels of each other. Offsets measured by windowed phase correlation on this
master (display pixels, sep->green, excluding the spectrum, whose ~+-90 px
"offset" is the physical spectral band shift, not misregistration):

    red  -> green: dy mean -0.51 sd 0.91, dx mean +0.32 sd 1.05 (max |d| 2.7)
    blue -> green: dy mean -0.10 sd 0.61, dx mean -0.22 sd 1.06 (max |d| 3.4)

Random per triplet, both signs -- NOT a constant lag -- so they must be
measured per triplet, which register() does. The largest genuine offset is the
Jane Goodall triplet (image 60: red (-2.1, -1.9), blue (+0.1, -3.4)), matching
Barry's remark that it is visibly misaligned.

Usage:
    python -m pipeline.colour                  # all 20 -> data/colour/
    python -m pipeline.colour --images 12,114 --out /tmp/c
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy import ndimage

from . import catalog as catalog_mod

REPO = Path(__file__).resolve().parent.parent
WEB_DATA = REPO / "web" / "public" / "data"
OUT_DIR = REPO / "data" / "colour"

#: Colour role of a triplet's frames in record order. Empirical; see docstring.
SEPARATION_ORDER = ("r", "g", "b")


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def phase_offset(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel (dy, dx) that maps image `b` onto image `a`, plus peak height.

    Windowed phase correlation with parabolic peak refinement. The peak height
    (0..1) is the confidence: genuine triplet alignments measure ~0.1-0.6;
    below ~0.05 the answer is noise (seen only on the solar spectrum, where
    the content itself moves with wavelength).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    h, w = a.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    A = np.fft.rfft2((a - a.mean()) * win)
    B = np.fft.rfft2((b - b.mean()) * win)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    c = np.fft.irfft2(R, a.shape)
    ky, kx = np.unravel_index(int(np.argmax(c)), c.shape)

    def refine(v, k, n):
        y0, y1, y2 = v[(k - 1) % n], v[k], v[(k + 1) % n]
        d = y0 - 2 * y1 + y2
        return k + (0.5 * (y0 - y2) / d if d != 0 else 0.0)

    dy = refine(c[:, kx], ky, h)
    dx = refine(c[ky, :], kx, w)
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx), float(c[ky, kx])


def register(seps: list[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    """Align [r, g, b] separation images onto the green one.

    Returns an (H, W, 3) stack in R,G,B plane order plus the measured offsets.
    Offsets are measured per triplet because they are random, not a fixed lag.
    """
    if len(seps) != 3:
        raise ValueError(f"need 3 separations, got {len(seps)}")
    shape = seps[1].shape
    seps = [_match_shape(s, shape) for s in seps]
    planes = [None, seps[1], None]
    offsets = []
    for i in (0, 2):
        dy, dx, peak = phase_offset(seps[1], seps[i])
        planes[i] = ndimage.shift(seps[i], (dy, dx), order=1, mode="nearest")
        offsets.append(
            {"role": SEPARATION_ORDER[i], "dy": round(dy, 2), "dx": round(dx, 2),
             "peak": round(peak, 3)}
        )
    return np.stack(planes, axis=-1), offsets


def _match_shape(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample to `shape` if needed (the decoder's output height can vary)."""
    if a.shape == shape:
        return a
    return ndimage.zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)


# --------------------------------------------------------------------------
# compositing
# --------------------------------------------------------------------------


def composite(stack: np.ndarray, *, black_pct: float = 1.0, white_pct: float = 99.5,
              gamma: float = 2.2) -> np.ndarray:
    """Registered (H, W, 3) float stack -> uint8 RGB.

    Each channel is stretched on its own percentiles. That neutralises the
    chain's per-separation exposure differences, which are real and large
    (see docstring) -- but it also flattens any GLOBAL colour cast, so a scene
    that is genuinely one hue everywhere comes out under-saturated: the sunset
    (114) renders far paler than its canonical slide. Local colour is correct;
    absolute cast is not recoverable without a reference.
    """
    out = np.empty_like(stack, dtype=np.float64)
    for c in range(3):
        lo, hi = np.percentile(stack[:, :, c], [black_pct, white_pct])
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        out[:, :, c] = np.clip((stack[:, :, c] - lo) / (hi - lo), 0.0, 1.0)
    return (out ** (1.0 / gamma) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------
# decoding the three separations of one image
# --------------------------------------------------------------------------


def _decode_frame(frame: catalog_mod.Frame, meta: dict) -> np.ndarray:
    """Decode one emitted FLAC asset to an oriented float image."""
    from . import build as build_mod  # deferred: heavy imports, and the asset
    from . import decode as decode_mod  # pipeline is under active rework
    from . import sync as sync_mod

    pcm = build_mod.read_flac(WEB_DATA / meta["file"])
    xs = pcm.astype(np.float32) / 32767.0 * meta["peak"]
    tb = sync_mod.recover(xs, period_guess=meta["periodGuess"],
                          n_traces=sync_mod.TRACES_PER_FRAME + 4)
    tb = build_mod.align_timebase(tb, meta["leadIn"], sync_mod.TRACES_PER_FRAME)
    dec = decode_mod.decode(xs, decode_mod.Settings(), tb)
    return build_mod.orient(dec.image.astype(np.float64), frame.orientation)


def build_colour_image(cat: catalog_mod.Catalog, image: catalog_mod.Image,
                       frame_meta: dict[str, dict]) -> tuple[np.ndarray, list[dict]]:
    """Decode, register and stack one triplet. Returns (uint8 RGB, offsets)."""
    if not image.color:
        raise ValueError(f"image {image.n} is not a colour image")
    seps = []
    for fid in image.frames:  # record order == r, g, b (SEPARATION_ORDER)
        frame = cat.by_id(fid)
        if frame_meta.get(fid) is None:
            raise ValueError(f"{fid}: no built asset in catalog.json")
        seps.append(_decode_frame(frame, frame_meta[fid]))
    stack, offsets = register(seps)
    return composite(stack), offsets


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--catalog", type=Path, default=WEB_DATA / "catalog.json")
    ap.add_argument("--images", default="", help="comma-separated canonical numbers, e.g. 12,114")
    args = ap.parse_args(argv)

    cat = catalog_mod.build()
    doc = json.loads(args.catalog.read_text())
    frame_meta = {f["id"]: f for f in doc["frames"]}

    want = {int(s) for s in args.images.split(",") if s.strip()} or None
    targets = [im for im in cat.images if im.color and (want is None or im.n in want)]
    if want is not None and len(targets) != len(want):
        missing = sorted(want - {im.n for im in targets})
        raise SystemExit(f"not colour images (or unknown): {missing}")

    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for im in targets:
        rgb, offsets = build_colour_image(cat, im, frame_meta)
        path = args.out / f"image{im.n:03d}.png"
        PILImage.fromarray(rgb, "RGB").save(path, optimize=True)
        report.append({"n": im.n, "title": im.title, "frames": list(im.frames),
                       "offsets": offsets, "file": path.name})
        off = "  ".join(f"{o['role']}->g ({o['dy']:+.1f},{o['dx']:+.1f}) pk {o['peak']:.2f}"
                        for o in offsets)
        print(f"image {im.n:3d}  {im.frames[0]}-{im.frames[2]}  {off}  -> {path.name}")

    (args.out / "colour_report.json").write_text(json.dumps(
        {"separationOrder": list(SEPARATION_ORDER), "images": report}, indent=1))
    print(f"{len(report)} colour images -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
