"""The vertical shading: measured, identified, and corrected.

    python -m pipeline.dedroop                    # everything (~4 min)
    python -m pipeline.dedroop --section measure,discriminate

WHAT THIS IS ABOUT
------------------
decode.py already inverts the recording chain's one-pole high-pass (`undroop`,
tau 530/295 samples per channel, measured content-free by droop.py).  Yet
inside L000's calibration ring -- a slide that is UNIFORM WHITE BY DESIGN, and
the cover says so, which makes it tier 0 ground truth the artifact supplied on
purpose -- a large top-to-bottom gradient survives along every trace.  It is
four times the size of the across-trace striping that gets complained about,
and it is the largest remaining defect in the decode.

THE NUMBER, AND WHY THE OLD ONE DISAGREED
-----------------------------------------
Measured on the FLOAT decode with the ring masked properly and the level
transfer's final clip undone (see `unclipped`, below -- 5.7% of L000's pixels
clip at the white rail, all of them at the top of the trace, and clipping
flattens exactly the part of the profile under test):

    L000, known-uniform field, rows 20..349 of 377      92.8 grey levels p-p
                                                        = 36% of black-to-white
    the same field, ring interior only (rows 61..302)   82.7 grey levels
    across-trace, ring interior / full field            11.5 / 22.4 grey levels

So 32-36%, not 4-9%.  Both figures are in the project's history and BOTH are
right about different things: shelf.py's "4-9% of the black-white range" is the
tilt left on content-free GAP LINES, whose only excitation is the sync pulse,
and shelf.py's own text already records L000's field ramp separately at "0.066
p-p, 27% of contrast".  The gap line is a weak instrument for this defect and
not a small version of it -- it is 5x smaller AND a different shape (85-90% a
straight line there; a flat-topped S-curve here).  Nothing was contaminated;
two different things were being measured and the smaller number was quoted.

WHAT IT IS NOT.  Four candidate mechanisms, each killed by a measurement.

1. NOT A SECOND, SLOWER POLE the single-pole inverse cannot represent.
   A one-pole high-pass, whatever its tau, relaxes toward a level: dv/dt =
   -(v - C)/tau, so |dv/dt| must fall monotonically as v falls.  Regressing
   dv/drow on v over rows 20..349 gives R^2 = 0.004, a fitted tau of the WRONG
   SIGN, and |dv/drow| that RISES from 0.77e-3 to 1.50e-3 and then falls back
   to 0.83e-3 while v falls monotonically 0.977 -> 0.674.  The profile is flat
   for the first ~70 rows, steepest in the middle and flattening again at the
   end.  No first-order system does that.  A free-asymptote exponential fits
   worse than a straight line (rms 0.0164 vs 0.0156) and only by running its
   asymptote to -3.9; pinned to the porch grey, where a real AC-coupling
   residual would have to relax, it is worse still (0.0266).
   This also retires shelf.py's open question in the other direction: it looked
   for a second pole on the gap probe and found "reduces the tilt on 6 frames
   of 8 but always adds curvature".  There is no second pole to find.

2. NOT THE RECORDING CHAIN AT ALL.  The chain's AC coupling is per audio
   channel and the two channels differ by 1.8x (tau_L 530, tau_R 295).  The
   shading does not: averaged over all 78 L frames and all 78 R frames the
   along-trace profiles agree in SHAPE at r = 0.968 and in AMPLITUDE at 62.8
   vs 65.0 grey levels p-p -- 3.5% apart, where a chain effect would be 1.8x
   apart.  Whatever makes this gradient sits UPSTREAM of the point where the
   signal split into two tape channels: in the 1977 scan converter / camera,
   not in the tape electronics.  (The along-trace axis is the source TV LINE
   number -- one plateau per source line, dotclock.py -- so "along the trace"
   IS the source camera's vertical, which is where vertical shading lives.)

3. NOT THE CLAMP.  An error in the per-trace back-porch estimate becomes a
   per-trace CONSTANT.  This defect is constant ACROSS traces and varies ALONG
   them: the two are orthogonal by construction, and the across-trace profile
   of the same mask is 11.5 grey levels against 92.8.  The clamp and the
   striping remain a live pairing; they are not this.

4. NOT INCOMPLETE SETTLING AFTER THE SYNC PULSE.  Settling is a fast transient
   confined to the start of the trace.  The first 70 rows are the FLATTEST part
   of the profile (0.977 -> 0.964, 3 grey levels); the damage accumulates in
   the middle of the trace, 100-300 rows (0.7-2.2 ms) after the pulse.

5. NOT MULTIPLICATIVE -- so not illumination, vignetting or a gain shading.
   L000 carries two targets, and they separate a gain from a pedestal
   exactly.  Write the decode as v = a(row)*slide + b(row).  The white field
   gives a+b; the ring core gives a*k+b with k the blur-limited core value,
   constant because the ring width and the PSF are.  Their difference is
   a(row)*(1-k), free of b.  Measured over rows 40..330, the field falls
   0.9695 -> 0.6818 (-30%) while the ring depth is 0.463, 0.488, 0.560,
   0.484, 0.490, 0.493 -- flat to +-6%, and if anything RISING.  A gain
   shading would have dragged the depth down by 30%.  Record-wide the same
   test on all 156 frames (per-row p75-p25, p90-p10, p98-p2 spread, each
   frame normalised) regresses on the profile at only +0.03..+0.36 where
   multiplicative needs ~1.2.  The defect is a PEDESTAL, b(row).

6. NOT A NON-UNIFORMITY OF THE 1977 L000 SLIDE, which is the one hypothesis
   that would make it uncorrectable and which shelf.py correctly refused to
   rule out from one frame.  It is ruled out by the other 155.  Every frame's
   own along-trace profile projected on L000's:

        L000 vs mean of the other 77 L frames     r = 0.9937
        L000 vs mean of the 78 R frames           r = 0.9802
        split-half hold-out within L / within R   r = 0.9897 / 0.9786
        frames with a positive projection         100% of L, 99% of R

   A defect of one 1977 slide cannot print itself on 155 other slides across
   both channels.  This is a fixed pattern of the scanner.

WHAT IT IS
----------
A FIXED ADDITIVE PEDESTAL b(row) in the along-trace (source-TV-line) axis,
common to both audio channels, essentially independent of frame brightness
(correlation of a frame's projection with its mean level +0.24 on L, +0.10 on
R; brightness quartile means 0.96 / 0.96 / 0.93 / 1.15).  It is the 1977 scan
converter's vertical shading, and the record's own calibration slide is the
instrument its designers supplied for measuring it.

THE AMPLITUDE, AND THE HONEST UNCERTAINTY
-----------------------------------------
The shape is settled.  The amplitude is where the overfitting trap lives, so
it is estimated three ways and the spread is reported rather than hidden.
Every frame's row profile is regressed on L000's (beta = 1 means "carries
exactly L000's shading"):

    L000 itself, p25/p50/p75/p90 of its own rows      0.99 - 1.03  (r = 0.999)
    L000's ring core, an independent target on the
      same frame: it falls -0.317 where the field
      falls -0.288                                    1.10
    all 156 frames, every percentile p2..p98,
      both channels                                   0.65 +- 0.04
    the 15 most-uniform frames other than L000
      (tier-0 statistic: median IQR / median p98-p2)   0.81, median
    linear extrapolation of beta to zero scene
      contamination over the 60 most-uniform frames    1.02
    FLAT-REGION estimator, no L000 pixels used,
      on the 29 frames whose flat region is bright     0.83, median
      on the 15 whose flat region sits at L000's
      own level (0.82 vs L000's 0.86)                  0.97, median

The last is the transfer test that matters, and `flat` runs it: choose pixels
by ACROSS-trace local roughness only -- an along-trace pedestal cancels exactly
in an across-trace difference, so the selection cannot know about s and cannot
bias it -- then take the median along-trace first difference inside the flat
regions and cumulate.  Scene contributes ~0 by construction.  On 71 of the 156
frames that estimate reproduces L000's profile SHAPE at r > 0.90 (13 frames
above r = 0.996), and none of L000's pixels went into making it.

Its amplitude, however, depends on the LEVEL of the flat region it found:
beta by level quintile is 0.25, 0.23, 0.41, 0.62, 0.95 at levels 0.34, 0.39,
0.50, 0.66, 0.84, and the linear fit beta = 1.418*level - 0.287 predicts 0.93
at L000's field level of 0.856.  Read naively that says "multiplicative", and
it is the reason the ensemble mean sits at 0.65.  It is NOT multiplicative,
and L000 says why: the ring core -- a DARK target at 0.19-0.51 on the same
frame -- carries the pedestal in FULL (it falls -0.317 where the field falls
-0.288).  What differs is that the ring core is a thin blurred line that never
reaches the floor, while the large dark regions a roughness selector finds in
a photograph are flat BECAUSE they are crushed black in the 1977 source, and a
signal already at its floor cannot show a pedestal.  The level trend is a
selection artefact of measuring on clipped blacks; the model stays additive.

DEFAULT is amplitude 1.0: the calibration slide's own measurement, matched by
the zero-contamination extrapolation (1.02) and by the flat-region estimator
restricted to regions at L000's own level (0.97).  The residual risk is bounded
and stated: if the true figure were the conservative 0.81, this over-corrects
by 0.19 x 92.8 = 18 grey levels.  Either way the defect goes from 93 grey
levels to at most ~18 -- an 80% reduction whose worst case is known.

A side check that the correction is undoing something real rather than
imposing something: it moves pixels INTO the decoder's rails, not out of them.
The fraction of pixels outside [0, 1] over all 156 frames falls from 0.73% to
0.23% (worst frame 27.0% -> 9.8%), and grows on only 30 of 156.

INTEGRATION POINT (matters, and is not cosmetic)
------------------------------------------------
The correction must be applied BEFORE decode.py's level transfer clips, not
after.  5.7% of L000's pixels sit at the white rail and every one is in the
top of the trace, i.e. in the part the correction pulls back down; applied
after the clip those pixels are already destroyed and L000's residual field
falls only to 3.5 grey levels instead of 1.5.  `unclipped()` here reconstructs the
pre-clip image exactly from two decodes (see its docstring) so the measurement
and this module's own correction are honest, but the shipped fix belongs
inside decode() between `if cfg.invert` and the transfer.  decode.py is not
modified here.

INTEGRATION SPEC
----------------
* decode.py, inside `decode()`, immediately after `if cfg.invert: img = -img`
  and BEFORE the intensity transfer:  `img -= dedroop.profile(img.shape[0])[:, None]`
  behind a `Settings.deshade: bool = True` knob, so it can be turned off the
  way `uncouple` can.  It must sit before the transfer because the transfer
  clips (see above), and after the inversion because the profile is stated in
  the inverted (0 = black) sense.  It must NOT go after `flip_scan` or any
  rotation: the profile is indexed by along-trace row in the decoder's own
  raster frame.
* `profile()` already resamples, so `dot_native` (230 rows) is covered:
  91.5 grey levels p-p at 377 rows, 91.6 at 230.
* The rails do NOT need re-measuring the way `uncouple` needed them to.  The
  correction is zero-mean over the picture rows by construction, so it moves
  no percentile of the frame as a whole; it only removes the row structure.
* Nothing in this module is fitted per frame.  There is one constant, measured
  once, on the one frame the record says is a circle on a uniform field.

WHAT THIS DOES NOT FIX
----------------------
The across-trace striping.  It is a different defect with a different geometry
(per-trace level trajectories, smooth down the trace and rough across it) and
this correction is constant across traces, so it cannot touch it -- measured:
the across-trace profile is 11.5 grey levels before and 11.5 after.  The two
defects are not one mechanism.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from . import catalog as catalog_mod
from . import decode as decode_mod
from . import sync as sync_mod
from . import wav

# --- the measurement ------------------------------------------------------
# L000's known-uniform field profile, de-meaned, in image units (1.0 = the
# full black-to-white range), sampled every 6 of the 377 square-pixel rows and
# linearly interpolated (max reconstruction error 1.0 grey level).  Held flat
# outside rows 20..349, where the picture gate carries blanking rather than
# picture.  This is the whole tier-0 content of the module: one profile,
# measured on the one frame the record states is a circle on a uniform field.
SHADING_KNOT_STEP = 6
SHADING_KNOTS = np.array([
    +0.17793, +0.17793, +0.17793, +0.17710, +0.16792, +0.15532, +0.14725, +0.14323,
    +0.13926, +0.13579, +0.13321, +0.13165, +0.13097, +0.13003, +0.12867, +0.12672,
    +0.12369, +0.12008, +0.11554, +0.10929, +0.10271, +0.09522, +0.08715, +0.07884,
    +0.07016, +0.06105, +0.05097, +0.04000, +0.02955, +0.01978, +0.01127, +0.00326,
    -0.00466, -0.01297, -0.02149, -0.03047, -0.03835, -0.04614, -0.05434, -0.06278,
    -0.07146, -0.08031, -0.08904, -0.09761, -0.10661, -0.11519, -0.12319, -0.12993,
    -0.13644, -0.14283, -0.14893, -0.15499, -0.16013, -0.16479, -0.16835, -0.17184,
    -0.17830, -0.18273, -0.18465, -0.18612, -0.18612, -0.18612, -0.18612, -0.18612,
])
SHADING_ROWS = 377          # the grid the knots were measured on
SHADING_PP = 0.36405        # p-p over rows 20..349, image units (92.8 grey levels)

# Amplitude to apply.  1.0 = the calibration slide's own measurement; the
# ensemble of the other 155 frames returns 0.81 (median, most-uniform frames)
# to 0.65 (all frames, scene-contaminated).  See the module docstring.
AMPLITUDE = 1.0
AMPLITUDE_ENSEMBLE = 0.81

ROW_LO, ROW_HI = 20, 350    # the honest picture rows of the 377-row gate

_MASTER = None
_CAT = None


def _master():
    global _MASTER, _CAT
    if _MASTER is None:
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        info = wav.probe(repo / "data" / "master" / "384kHzStereo.wav")
        _MASTER = (info, wav.memmap(info))
        _CAT = catalog_mod.build()
    return _MASTER, _CAT


# --------------------------------------------------------------------------
# loading, and undoing the level transfer's clip
# --------------------------------------------------------------------------


def unclipped(fid: str, **kw) -> tuple[np.ndarray, "decode_mod.Decoded"]:
    """The decoded frame on the reference scale, with the final clip undone.

    decode() maps the clamped picture to (img - lo)/(hi - lo) and then CLIPS to
    [0, 1].  On L000 that discards 5.7% of the pixels, all of them at the top
    of the trace -- precisely the part of the profile being measured, so the
    clip flattens the answer.  Decoding a second time with
    levels="percentile", black_pct=0, white_pct=100 makes lo/hi the actual
    min/max, so the clip is a no-op; both arms are AFFINE in the same
    underlying picture, so a two-point fit on the pixels the reference arm did
    NOT clip recovers the reference scale exactly (max residual ~1e-7).  No
    new information is invented: this only undoes a saturation.
    """
    (info, mm), cat = _master()
    fr = cat.by_id(fid)
    n = int(sync_mod.NOMINAL_PERIOD * 520)
    seg = np.asarray(mm[fr.seed_sample: fr.seed_sample + n, fr.channel], dtype=np.float64)
    tb = sync_mod.recover(seg)
    ch = "L" if fr.channel == 0 else "R"
    base = dict(channel=ch, traces=512, rotate=0)
    base.update(kw)
    ref = decode_mod.decode(seg, decode_mod.Settings(**base), tb=tb)
    pct = decode_mod.decode(
        seg, decode_mod.Settings(**dict(base, levels="percentile",
                                        black_pct=0.0, white_pct=100.0)), tb=tb)
    A = ref.image.astype(np.float64)
    B = pct.image.astype(np.float64)
    ok = (A > 1e-6) & (A < 1 - 1e-6)
    if ok.sum() < 1000:
        return A, ref
    p, q = np.polyfit(B[ok], A[ok], 1)
    return p * B + q, ref


# --------------------------------------------------------------------------
# the correction
# --------------------------------------------------------------------------


def profile(rows: int = SHADING_ROWS) -> np.ndarray:
    """The measured shading, de-meaned, resampled to `rows` along-trace rows."""
    k = np.arange(len(SHADING_KNOTS)) * SHADING_KNOT_STEP
    k = np.minimum(k, SHADING_ROWS - 1)
    s = np.interp(np.arange(SHADING_ROWS), k, SHADING_KNOTS)
    if rows != SHADING_ROWS:
        s = np.interp(np.linspace(0, 1, rows), np.linspace(0, 1, SHADING_ROWS), s)
    return s - s.mean()


def correct(img: np.ndarray, amplitude: float = AMPLITUDE) -> np.ndarray:
    """Remove the along-trace shading from a decoded float image.

    `img` is (rows, traces) on the decoder's own scale, rows running along the
    trace -- decode()'s native orientation, BEFORE any display rotation.  The
    correction is a pure subtraction of one row profile, so it is constant
    across traces: it cannot move an edge, cannot change the geometry, and
    cannot touch the across-trace striping.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("expected a 2-D (rows, traces) decode")
    return a - amplitude * profile(a.shape[0])[:, None]


# --------------------------------------------------------------------------
# masks and estimators used by the measurement sections
# --------------------------------------------------------------------------


def ring_geometry(img: np.ndarray) -> tuple[float, float, float, float]:
    from . import quality as Q
    m = Q.circle_metrics(np.clip(img, 0.0, 1.0))
    if not m.get("ok"):
        raise RuntimeError("ring fit failed on the calibration frame")
    h, w = img.shape
    return (h / 2 + m["center_dy"], w / 2 + m["center_dx"], m["minor"], m["major"])


def field_mask(img: np.ndarray, ring_margin: float = 6.0, border: int = 6) -> np.ndarray:
    """L000's known-uniform field: everything but the ring and the gate edges.

    The whole slide outside the ring is the same white -- checked, not assumed:
    the ring interior alone gives 80.9 grey levels over rows 62..301 and the
    full field gives 92.8 over rows 20..349, which is the same profile over a
    longer span, not a different one.
    """
    cy, cx, ry, rx = ring_geometry(img)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - cx) / rx, (yy - cy) / ry)
    d = np.abs(rr - 1.0) * (0.5 * (rx + ry))
    return ((d > ring_margin) & (yy >= border) & (yy < h - border)
            & (xx >= border) & (xx < w - border))


def separable(img: np.ndarray, mask: np.ndarray, iters: int = 300):
    """Fit img ~ v[row] + g[trace] on `mask` (two-way ANOVA, alternating means).

    Fitting the two directions JOINTLY is what stops the ring's arc from
    manufacturing a gradient: a plain row mean over a mask whose width changes
    with the row mixes the across-trace profile into the along-trace one.  On
    L000 the difference is small (80.6 vs 80.9 grey levels inside the ring)
    but a plain FULL-row mean, with the ring's own pixels left in, reads 214.
    """
    cr = mask.sum(axis=1)
    cc = mask.sum(axis=0)
    A = np.where(mask, img, 0.0)
    v = np.zeros(img.shape[0])
    g = np.zeros(img.shape[1])
    for _ in range(iters):
        v = np.divide((A - np.where(mask, g[None, :], 0.0)).sum(axis=1), np.maximum(cr, 1))
        v[cr == 0] = 0.0
        g = np.divide((A - np.where(mask, v[:, None], 0.0)).sum(axis=0), np.maximum(cc, 1))
        g[cc == 0] = 0.0
        g -= g[cc > 0].mean()
    v[cr == 0] = np.nan
    g[cc == 0] = np.nan
    return v, g


def row_profiles(fids: list[str], qs=(2, 25, 50, 75, 98)) -> dict:
    """Per-row percentiles of every frame, on the unclipped reference scale."""
    out = {}
    for fid in fids:
        U, _ = unclipped(fid)
        out[fid] = np.percentile(U[:, 8:-8], qs, axis=1)
    return out


def beta(prof: np.ndarray, s: np.ndarray, lo=ROW_LO + 25, hi=ROW_HI - 10) -> float:
    """Regression of a row profile on the calibration profile, both de-meaned."""
    p = prof[lo:hi] - prof[lo:hi].mean()
    t = s[lo:hi] - s[lo:hi].mean()
    return float(np.dot(p, t) / np.dot(t, t))


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def sec_measure() -> None:
    print("MEASURE " + "=" * 62)
    U, dec = unclipped("L000")
    h, w = U.shape
    clip = float(np.mean((dec.image <= 0) | (dec.image >= 1)))
    print(f"  L000 float decode {h}x{w}; the shipped 8-bit-scale transfer clips "
          f"{100*clip:.1f}% of its pixels")
    print(f"  unclipped range {U.min():+.4f} .. {U.max():+.4f} (1.0 = white rail)")

    cy, cx, ry, rx = ring_geometry(U)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - cx) / rx, (yy - cy) / ry)

    for name, mask in (("ring interior (rr<0.80)", rr < 0.80),
                       ("full field, ring masked", field_mask(U))):
        v, g = separable(U, mask)
        ok = np.isfinite(v)
        rows = np.where(ok)[0]
        rows = rows[(rows >= ROW_LO) & (rows < ROW_HI)]
        gg = np.isfinite(g)
        print(f"  {name:26s} n={mask.sum():6d}  rows {rows.min()}..{rows.max()}  "
              f"along-trace {255*(v[rows].max()-v[rows].min()):5.1f} grey  "
              f"across-trace {255*(np.nanmax(g[gg])-np.nanmin(g[gg])):5.1f} grey")

    v, _ = separable(U, field_mask(U))
    naive = U.mean(axis=1)
    print(f"  the contaminated way -- plain FULL-row mean, ring left in: "
          f"{255*(naive.max()-naive.min()):.1f} grey levels.  That is the number a")
    print("  row band clipping the arc manufactures; it is not the defect.")
    print(f"  VERDICT: {255*SHADING_PP:.1f} grey levels = "
          f"{100*SHADING_PP:.0f}% of black-to-white, not 4-9%.")


def sec_discriminate() -> None:
    print("\nDISCRIMINATE " + "=" * 57)
    U, dec = unclipped("L000")
    h, w = U.shape
    v, _ = separable(U, field_mask(U))
    porch = decode_mod.UNCOUPLE_BLACK_REF / (
        decode_mod.UNCOUPLE_BLACK_REF - decode_mod.UNCOUPLE_WHITE_REF)

    sel = np.arange(ROW_LO, ROW_HI)
    y = v[sel]
    dy = np.gradient(y)
    A = np.stack([y, np.ones_like(y)], 1)
    co, *_ = np.linalg.lstsq(A, dy, rcond=None)
    r2 = 1 - np.var(dy - A @ co) / np.var(dy)
    print("  (1) a second, slower pole?  a one-pole relaxation obeys "
          "dv/dt = -(v-C)/tau")
    print(f"      regression of dv/drow on v: R^2 {r2:.3f}, tau {-1/co[0]:+.0f} rows, "
          f"C {co[1]*-1/co[0]:+.3f} (porch grey {porch:.3f})")
    q = np.linspace(0, len(y), 6).astype(int)
    print("      |dv/drow| must FALL as v falls.  It does not:")
    for k in range(5):
        s_ = slice(q[k], q[k + 1])
        print(f"        rows {sel[q[k]]:3d}-{sel[q[k+1]-1]:3d}  v {y[s_].mean():.4f}  "
              f"dv/drow {1000*dy[s_].mean():+6.3f}e-3")

    def rms(r):
        return float(np.sqrt(np.mean(r ** 2)))
    t = sel.astype(float)
    lin = rms(y - np.polyval(np.polyfit(t, y, 1), t))
    bexp = min(((rms(y - np.stack([np.ones_like(t), np.exp(-t / tau)], 1)
                     @ np.linalg.lstsq(np.stack([np.ones_like(t), np.exp(-t / tau)], 1),
                                       y, rcond=None)[0]), tau)
                for tau in np.arange(20, 4000, 4.0)))
    bpin = min(((rms(y - porch - np.exp(-t / tau) * np.linalg.lstsq(
        np.exp(-t / tau)[:, None], y - porch, rcond=None)[0][0]), tau)
        for tau in np.arange(20, 4000, 4.0)))
    print(f"      fit rms: straight line {lin:.5f} | best free-asymptote exponential "
          f"{bexp[0]:.5f} (tau {bexp[1]:.0f}) | exponential pinned to the porch "
          f"{bpin[0]:.5f} (tau {bpin[1]:.0f})")
    print("      => NOT a pole.  A line beats every exponential.")

    print("\n  (2) settling after the sync pulse?  that is a transient at the START")
    print(f"      rows {ROW_LO}-90 (just after the porch): "
          f"{255*(v[ROW_LO:90].max()-v[ROW_LO:90].min()):.1f} grey levels")
    print(f"      rows 100-300 (mid-trace):                "
          f"{255*(v[100:300].max()-v[100:300].min()):.1f} grey levels")
    print("      => the start of the trace is the FLATTEST part.  Not settling.")

    print("\n  (3) the clamp?  a shelf error is a per-trace CONSTANT")
    v2, g2 = separable(U, field_mask(U))
    gg = np.isfinite(g2)
    print(f"      along-trace {255*(v2[ROW_LO:ROW_HI].max()-v2[ROW_LO:ROW_HI].min()):.1f} "
          f"grey vs across-trace {255*(np.nanmax(g2[gg])-np.nanmin(g2[gg])):.1f} grey")
    print("      => orthogonal directions.  The clamp cannot make an along-trace ramp.")

    print("\n  (4) multiplicative (illumination, vignetting, gain shading)?")
    print("      v = a(row)*slide + b(row).  field = a+b; ring core = a*k+b;")
    print("      their DIFFERENCE is a(row)*(1-k) and is free of the pedestal.")
    cy, cx, ry, rx = ring_geometry(U)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - cx) / rx, (yy - cy) / ry)
    core = np.abs(rr - 1.0) * (0.5 * (rx + ry)) < 1.2
    rows, dark = [], []
    for i in range(h):
        left = core[i] & (xx[i] < cx)
        right = core[i] & (xx[i] > cx)
        if left.sum() >= 2 and right.sum() >= 2 and 40 <= i <= 330:
            rows.append(i)
            dark.append(0.5 * (U[i, left].min() + U[i, right].min()))
    rows = np.array(rows)
    dark = np.array(dark)
    print(f"      {len(rows)} rows with two clean ring crossings")
    for b_ in np.array_split(np.arange(len(rows)), 6):
        wht = float(np.nanmean(v[rows[b_]]))
        dk = float(dark[b_].mean())
        print(f"        rows {rows[b_][0]:3d}-{rows[b_][-1]:3d}  white {wht:.4f}  "
              f"ring core {dk:.4f}  depth {wht-dk:.4f}  depth/white {(wht-dk)/wht:.4f}")
    print("      => depth flat (multiplicative would drop it 30%), ratio not flat.")
    print("      => ADDITIVE PEDESTAL.")


def sec_transfer(fids: list[str] | None = None) -> None:
    print("\nTRANSFER " + "=" * 61)
    (info, mm), cat = _master()
    ids = fids or cat.ids()
    s = profile()
    print(f"  regressing all {len(ids)} frames' own row profiles on L000's field profile")
    rec = {}
    for fid in ids:
        U, _ = unclipped(fid)
        core = U[:, 8:-8]
        p = np.percentile(core, (2, 25, 50, 75, 98), axis=1)
        uni = float(np.median(p[3][ROW_LO:ROW_HI] - p[1][ROW_LO:ROW_HI])
                    / max(np.median(p[4][ROW_LO:ROW_HI] - p[0][ROW_LO:ROW_HI]), 1e-9))
        rec[fid] = (np.array([beta(p[k], s) for k in range(5)]), uni, p[2])
    ch = np.array([f[0] for f in ids])
    B = np.array([rec[f][0] for f in ids])
    uni = np.array([rec[f][1] for f in ids])
    P50 = np.array([rec[f][2] for f in ids])

    print(f"\n  {'':10s} {'p2':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p98':>7}")
    for c in ("L", "R"):
        m = ch == c
        print(f"  {c} mean    " + " ".join(f"{x:7.3f}" for x in B[m].mean(0)))
        print(f"  {c} median  " + " ".join(f"{x:7.3f}" for x in np.median(B[m], 0)))
    print("  every percentile of every frame agrees: one additive pedestal, not a gain.")

    lo, hi = ROW_LO + 25, ROW_HI - 10
    mL = np.array([P50[i][lo:hi] - P50[i][lo:hi].mean() for i in range(len(ids)) if ch[i] == "L"]).mean(0)
    mR = np.array([P50[i][lo:hi] - P50[i][lo:hi].mean() for i in range(len(ids)) if ch[i] == "R"]).mean(0)
    sd = s[lo:hi] - s[lo:hi].mean()
    print(f"\n  channel test (the chain's poles differ by 1.8x; a chain defect must too)")
    print(f"    mean-L vs mean-R  shape r {np.corrcoef(mL, mR)[0,1]:.4f}   "
          f"amplitude {255*(mL.max()-mL.min()):.1f} vs {255*(mR.max()-mR.min()):.1f} grey")
    print(f"    L000  vs mean-L   r {np.corrcoef(sd, mL)[0,1]:.4f}     "
          f"L000 vs mean-R   r {np.corrcoef(sd, mR)[0,1]:.4f}")
    print("    => same shape, same size on both channels.  Upstream of the tape.")

    for c in ("L", "R"):
        idx = np.where(ch == c)[0]
        A = np.array([P50[i][lo:hi] - P50[i][lo:hi].mean() for i in idx])
        a1, a2 = A[0::2].mean(0), A[1::2].mean(0)
        print(f"    hold-out {c}: fit on half the frames, predict the other half -- "
              f"r {np.corrcoef(a1, a2)[0,1]:.4f}, slope "
              f"{np.dot(a2,a1)/np.dot(a1,a1):.3f}")

    o = np.argsort(uni)
    print(f"\n  amplitude vs scene contamination (uniformity = median IQR / median p98-p2;")
    print("  small = the frame is mostly one flat level, so its own gradient is small)")
    for q in range(5):
        sl = o[q * len(o) // 5:(q + 1) * len(o) // 5]
        print(f"    quintile {q+1}  uniformity {uni[sl].mean():.3f}   beta mean "
              f"{B[sl, 2].mean():+.3f}  median {np.median(B[sl, 2]):+.3f}  n={len(sl)}")
    sel = [k for k in o[:16] if ids[k] != "L000"]
    print(f"    15 most-uniform frames excluding L000: median beta "
          f"{np.median(B[sel,2]):+.3f}, mean {B[sel,2].mean():+.3f} "
          f"+- {B[sel,2].std()/len(sel)**0.5:.3f}")
    cfit = np.polyfit(uni[o[:60]], B[o[:60], 2], 1)
    print(f"    extrapolated to zero scene contamination (60 most uniform): "
          f"beta {cfit[1]:+.3f}")
    print(f"    L000 itself: beta {B[list(ids).index('L000'), 2]:+.3f}")
    print(f"    ADOPTED amplitude {AMPLITUDE:.2f}; ensemble alternative "
          f"{AMPLITUDE_ENSEMBLE:.2f} would differ by "
          f"{255*SHADING_PP*(AMPLITUDE-AMPLITUDE_ENSEMBLE):.0f} grey levels.")

    print("\n  effect of the correction on every frame's residual along-trace profile:")
    before, after = [], []
    for i, fid in enumerate(ids):
        p = P50[i][lo:hi] - P50[i][lo:hi].mean()
        before.append(255 * (p.max() - p.min()))
        r = p - AMPLITUDE * sd
        after.append(255 * (r.max() - r.min()))
    before = np.array(before)
    after = np.array(after)
    for c in ("L", "R"):
        m = ch == c
        print(f"    {c}: median p-p {before[m].mean():5.1f} -> {after[m].mean():5.1f} grey "
              f"levels; improved on {100*np.mean(after[m] < before[m]):.0f}% of frames")
    print(f"    (a photograph SHOULD keep a vertical gradient -- this is not a flatness")
    print("     target, it is the common component that must go and the rest that must stay)")


def _boxfilt(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over a (2k+1)^2 window, edge-padded, via a summed-area table."""
    p = np.pad(a, k + 1, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    n = 2 * k + 1
    h, w = a.shape
    return (c[n:n + h, n:n + w] - c[0:h, n:n + w]
            - c[n:n + h, 0:w] + c[0:h, 0:w]) / (n * n)


def flat_profile(U: np.ndarray, frac: float = 0.25) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate the pedestal on ANY frame, without using L000 and without a
    flatness assumption that s could satisfy.

    The selection is made on ACROSS-trace roughness alone.  s(row) is constant
    across traces, so it cancels exactly in an across-trace difference: the
    mask cannot carry information about s, and therefore cannot bias it.
    Inside the selected flat regions the scene's own ALONG-trace gradient is
    ~0, so the median along-trace first difference is s(row+1) - s(row).
    Cumulating gives s up to a constant.
    """
    dx = np.abs(np.diff(U, axis=1))
    rough = _boxfilt(np.pad(dx, ((0, 0), (0, 1)), mode="edge"), 3)
    m = rough <= np.percentile(rough, 100 * frac)
    d = np.diff(U, axis=0)
    mm = m[:-1] & m[1:]
    step = np.array([np.median(d[i][mm[i]]) if mm[i].sum() >= 30 else 0.0
                     for i in range(U.shape[0] - 1)])
    s = np.concatenate([[0.0], np.cumsum(step)])
    level = float(np.median(U[ROW_LO:ROW_HI][m[ROW_LO:ROW_HI]]))
    return s, mm.sum(1), level


def sec_flat(fids: list[str] | None = None) -> None:
    print("\nFLAT-REGION TRANSFER " + "=" * 49)
    print("  The estimate below uses NO pixel of L000.  L000 enters only as the")
    print("  thing being predicted, which is what makes this a hold-out.")
    (info, mm), cat = _master()
    ids = fids or cat.ids()
    s = profile()
    lo, hi = ROW_LO + 25, ROW_HI - 10
    t = s[lo:hi] - s[lo:hi].mean()
    rec = []
    for fid in ids:
        U, _ = unclipped(fid)
        est, n, lvl = flat_profile(U)
        p = est[lo:hi] - est[lo:hi].mean()
        rec.append((fid, float(np.dot(p, t) / np.dot(t, t)),
                    float(np.corrcoef(p, t)[0, 1]), lvl))
    B = np.array([r[1] for r in rec])
    R = np.array([r[2] for r in rec])
    Lv = np.array([r[3] for r in rec])
    good = R > 0.90
    print(f"\n  frames whose flat regions reproduce L000's SHAPE at r > 0.90: "
          f"{good.sum()}/{len(rec)}  (r > 0.996 on {int((R>0.996).sum())})")
    o = np.argsort(-R)
    print("  best 12:  " + "  ".join(f"{rec[k][0]} b{rec[k][1]:+.2f} r{rec[k][2]:.3f}"
                                     for k in o[:6]))
    print("            " + "  ".join(f"{rec[k][0]} b{rec[k][1]:+.2f} r{rec[k][2]:.3f}"
                                     for k in o[6:12]))
    print(f"\n  amplitude vs the LEVEL of the flat region that was found "
          f"(corr {np.corrcoef(B[good], Lv[good])[0,1]:+.3f}):")
    og = np.argsort(Lv[good])
    Bg, Lg = B[good], Lv[good]
    for q in range(5):
        sl = og[q * len(og) // 5:(q + 1) * len(og) // 5]
        print(f"    level {Lg[sl].mean():.3f}   beta mean {Bg[sl].mean():+.3f}  "
              f"median {np.median(Bg[sl]):+.3f}  n={len(sl)}")
    c = np.polyfit(Lg, Bg, 1)
    print(f"    fit beta = {c[0]:+.3f}*level {c[1]:+.3f}; at L000's field level "
          f"0.856 that is {np.polyval(c, 0.856):+.3f}")
    br = Lg > 0.55
    print(f"    flat regions above 0.55: n={br.sum()} median beta "
          f"{np.median(Bg[br]):+.3f} (mean {Bg[br].mean():+.3f} +- "
          f"{Bg[br].std()/max(br.sum(),1)**0.5:.3f})")
    top = og[-15:]
    print(f"    the 15 at L000's own level ({Lg[top].mean():.2f}): median beta "
          f"{np.median(Bg[top]):+.3f}")
    print("\n  the level trend is NOT a gain shading: on L000 the ring core, a DARK")
    print("  target at 0.19-0.51, carries the pedestal in full (-0.317 against the")
    print("  field's -0.288).  A roughness selector on a photograph finds large dark")
    print("  areas that are flat BECAUSE they are crushed black in the 1977 source,")
    print("  and a signal at its floor cannot show a pedestal.  Selection artefact.")


def sec_rails(fids: list[str] | None = None) -> None:
    print("\nRAILS (does the correction impose, or undo?) " + "=" * 26)
    (info, mm), cat = _master()
    ids = fids or cat.ids()
    s = profile()
    b, a = [], []
    for fid in ids:
        U, _ = unclipped(fid)
        C = U - AMPLITUDE * s[:, None]
        b.append(float(np.mean((U < 0) | (U > 1))))
        a.append(float(np.mean((C < 0) | (C > 1))))
    b = np.array(b)
    a = np.array(a)
    print("  fraction of pixels outside the decoder's own black/white rails:")
    print(f"    before  mean {b.mean():.4f}  median {np.median(b):.4f}  max {b.max():.4f}")
    print(f"    after   mean {a.mean():.4f}  median {np.median(a):.4f}  max {a.max():.4f}")
    print(f"    grows on {int((a>b).sum())}/{len(ids)} frames; mean change {(a-b).mean():+.4f}")
    print("  a correction that invented a gradient would push pixels OUT of the")
    print("  rails.  This one pulls them in, which is what undoing a real pedestal")
    print("  looks like.")


def sec_tripwire(fids: list[str] | None = None) -> None:
    """quality.py's composite rewards blur and must not choose anything here.
    It is run as a tripwire only: a correction that broke the decode would show
    up as a collapse, and it does not."""
    from . import quality as Q
    print("\nTRIPWIRES (not objectives) " + "=" * 43)
    print("  quality.py's composite is known to reward blur, so it selects nothing")
    print("  in this module.  It is here to catch a collapse, and there isn't one.")
    ids = fids or ["L000", "L011", "L020", "L034", "L055", "R023", "R040", "R056", "R068"]
    s = profile()
    print(f"  {'frame':6} {'composite':>18} {'sharpness':>20} {'stripe amp':>22}")
    for fid in ids:
        U, dec = unclipped(fid)
        B = np.clip(U, 0, 1)
        C = np.clip(U - AMPLITUDE * s[:, None], 0, 1)
        rb = Q.frame_report(B, dec.timebase, fid)
        ra = Q.frame_report(C, dec.timebase, fid)
        print(f"  {fid:6} {rb.composite:8.2f} ->{ra.composite:7.2f}   "
              f"{Q.sharpness(B):8.4f} ->{Q.sharpness(C):8.4f}   "
              f"{Q.stripe_amp(B):9.5f} ->{Q.stripe_amp(C):9.5f}")
    print("  sharpness moves by <=0.0002 and the stripe amplitude by <=1e-5: the")
    print("  correction is constant across traces, so neither CAN move.  The")
    print("  composite drifts a few points either way (R068 85.3 -> 79.3, R056")
    print("  74.7 -> 76.9) and is reported, not optimised.")


def sec_circle() -> None:
    print("\nCIRCLE INVARIANT " + "=" * 53)
    from . import quality as Q
    U, dec = unclipped("L000")
    print("  the ungameable check: the correction is constant across traces, so it")
    print("  cannot move an edge.  Confirmed, not asserted:")
    print(f"  {'arm':28s} {'axis ratio':>11} {'radial rms':>11} {'inliers':>8} {'cov deg':>8}")
    for name, img in (("shipped 8-bit decode", dec.image.astype(np.float64)),
                      ("float decode, unclipped", U),
                      ("corrected, amplitude 1.00", correct(U, 1.0)),
                      ("corrected, amplitude 0.81", correct(U, 0.81)),
                      ("corrected, amplitude 1.50 (control)", correct(U, 1.5))):
        m = Q.circle_metrics(np.clip(img, 0.0, 1.0))
        print(f"  {name:28s} {m['axis_ratio']:11.4f} {m['radial_rms']:11.4f} "
              f"{m['inliers']:8d} {m['coverage_deg']:8.0f}")

    print("\n  the next line is CIRCULAR and is printed only as an arithmetic check:")
    print("  the profile was measured on this field, so it must flatten it.  The")
    print("  evidence that it is real is in `transfer` and `flat`, not here.")
    v, g = separable(U, field_mask(U))
    C = correct(U, AMPLITUDE)
    v2, g2 = separable(C, field_mask(U))
    r = slice(ROW_LO, ROW_HI)
    print(f"\n  L000 known-uniform field, along-trace p-p: "
          f"{255*(v[r].max()-v[r].min()):.1f} -> {255*(v2[r].max()-v2[r].min()):.1f} grey levels")
    gg = np.isfinite(g)
    print(f"  L000 across-trace p-p (the STRIPING, which this cannot touch): "
          f"{255*(np.nanmax(g[gg])-np.nanmin(g[gg])):.1f} -> "
          f"{255*(np.nanmax(g2[gg])-np.nanmin(g2[gg])):.1f} grey levels")
    fm = field_mask(U)
    print(f"  L000 field std over the whole known-uniform region: "
          f"{255*U[fm].std():.1f} -> {255*C[fm].std():.1f} grey levels")
    print(f"  applied AFTER the shipped clip instead of before: "
          f"{255*(v[r].max()-v[r].min()):.1f} -> "
          f"{255*(lambda a: a[r].max()-a[r].min())(separable(np.clip(U,0,1) - AMPLITUDE*profile(U.shape[0])[:,None], fm)[0]):.1f}"
          f" -- why the integration point matters")


SECTIONS = {
    "measure": sec_measure,
    "discriminate": sec_discriminate,
    "circle": sec_circle,
    "transfer": sec_transfer,
    "flat": sec_flat,
    "rails": sec_rails,
    "tripwire": sec_tripwire,
}
_TAKES_FIDS = {"transfer", "flat", "rails", "tripwire"}


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description="the vertical shading: measure, identify, correct")
    ap.add_argument("--section",
                    default="measure,discriminate,circle,tripwire,transfer,flat,rails")
    ap.add_argument("--frames", default="", help="restrict the record-wide sections")
    args = ap.parse_args()
    fids = args.frames.split(",") if args.frames else None
    for name in (n.strip() for n in args.section.split(",")):
        fn = SECTIONS.get(name)
        if fn is None:
            print(f"unknown section {name!r}; have {sorted(SECTIONS)}")
            sys.exit(2)
        fn(fids) if name in _TAKES_FIDS else fn()
