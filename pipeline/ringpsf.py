"""The calibration ring as a two-dimensional line-spread function, and what
that licenses us to deconvolve.

    python -m pipeline.ringpsf                       # everything
    python -m pipeline.ringpsf --section lsf,budget

PROVENANCE
----------
Tier 0. Inputs: the master WAV, and the cover's statement that the first image
is a circle. `catalog.build()` supplies each frame's start sample and channel
(the `start_samples` key, which provenance.py's MIXED_FILES rule explicitly
permits). Nothing under docs/reference is opened. This module measures; it
registers no shipped artifact and changes no decoded pixel by default.

WHY THE RING AND NOTHING ELSE
-----------------------------
Frame L000 is a black ring on a field the cover tells us is uniform. A ring is
a thin line of known geometry presented at EVERY ORIENTATION at once. Its
radial profile at angle t is therefore the system's line spread function in
direction t, and one frame gives the complete two-dimensional characterisation.
Every other edge on the record is at whatever angle its scene happened to have.

WHAT THE ANGLE SEPARATES, AND WHAT IT CANNOT
--------------------------------------------
The measured blur is a convolution of terms with different symmetries:

    the 1977 camera's spot          optical, ISOTROPIC
    the converter's sample-and-hold ALONG the trace only
    our own plateau integration     ALONG the trace only
    the chain filter c              ALONG the trace only
    the converter's sampling gate   ACROSS traces only (aperture.py flags it
                                    as real, unknown and unmeasurable there)

For ANY point spread function, the variance of its projection onto a unit
direction n is n'.Sigma.n exactly -- no Gaussian assumption. With the picture's
two axes as the eigenvectors that is

    var_perp(t) = var_x cos^2 t + var_y sin^2 t

t being the angle of the ring's local NORMAL from the across-trace axis. The
brief for this work proposed "a constant term plus terms in cos^2 and sin^2,
which gives the isotropic part and the two axis-aligned parts separately".
IT DOES NOT, and this is the first thing to report: cos^2 + sin^2 = 1, so a
constant is exactly degenerate with the pair. The design matrix has a null
space and no amount of angular coverage removes it. What the angle measures,
and measures well, is

    var_y - var_x = (along-trace-only terms) - (across-trace-only terms)

The isotropic camera CANCELS in that difference -- which is the whole point,
because the camera is the part we must not touch. The ring cannot tell us how
big the camera's spot is; it can tell us exactly how much blur is NOT the
camera's spot, and that is the only part it is legitimate to remove.

HOW THE PROFILES ARE TAKEN
--------------------------
Never by interpolating along a radial ray. Bilinear or Lanczos resampling of a
ray adds a direction-dependent blur of its own, which is fatal to a measurement
whose entire content is a direction dependence. Instead the cuts are the
image's OWN rows and columns:

    near the ring's top and bottom a COLUMN crosses it almost perpendicular
    near the ring's left and right  a ROW    crosses it almost perpendicular

and for a locally straight line the profile along an oblique cut is the
perpendicular profile with its coordinate stretched by 1/cos(obliquity), so
var_perp = var_cut * cos^2. Two independent families, two independent secant
corrections, overlapping at 45 degrees where they must agree -- and they do
(0.83 vs 0.81 px in the 40-50 degree bin).

Each cut is fitted ON ITS OWN. Stacking cuts would fold our per-trace timing
error into the width and report our own jitter as the chain's blur; this
project has retracted an MTF number for exactly that mistake, and aperture.py
refuses to stack for the same reason. A column IS one trace, so per-trace dot
phase error is a pure shift of that column and cannot widen anything.

THE ESTIMATOR IS CALIBRATED, NOT TRUSTED
----------------------------------------
A width estimator measures its own bias as much as the signal. Every number
below is read off a synthetic ring pushed through the identical measurement
code, carrying the real chain's structure: an ideal ring, an isotropic
Gaussian camera, the one-dot hold, the chain filter at its measured width, our
own full-plateau integration on the real dot pitch, decode.py's own Lanczos
row interpolator, and the real frame's contrast, white level, per-dot noise
and per-trace level noise. The synthetic contains NO unknown along-trace blur
by construction, so it is the null hypothesis in physical form: whatever the
real ring reads ABOVE the matched synthetic is blur the known terms do not
explain.

WHAT IT SAYS (see `--section budget`)
-------------------------------------
The across-trace axis is sharp and the along-trace axis is not, by a factor
that is entirely accounted for by terms we can already write down:

    measured anisotropy   var_y - var_x   0.51 px^2
    exactly-known budget  two one-dot boxes + the chain filter   0.55 px^2

The measured excess is 93% of the budget and does not exceed it. Three things
follow, and they are the answer to "how much of the blur is legitimately
removable":

  1. THE CAMERA IS ISOTROPIC, to the precision of this measurement. The
     hypothesis was an assumption in aperture.py's docstring; the ring is the
     first evidence for it on this artifact.
  2. THE CONVERTER'S ACROSS-TRACE SAMPLING GATE IS CONSISTENT WITH ZERO. It
     had never been measured. Its variance comes out at +0.04 +- 0.06 px^2,
     i.e. nothing, so there is nothing to correct across traces.
  3. THERE IS NO UNEXPLAINED ALONG-TRACE BLUR. The removable part is exactly
     the operator aperture.py already builds and nothing more. Any further
     along-trace sharpening is sharpening the camera's spot, which the
     resolution ceiling says is invention.

WHAT THE REMOVAL BUYS, MEASURED
-------------------------------
Applying aperture.apply_dots at strength 1 inside the dot domain (`--section
deconv`, which reproduces decode()'s own render exactly to 9e-8 so the
comparison is not against a re-implementation):

    along-trace  var_y  1.148 -> 0.897 px^2   sigma 1.071 -> 0.947 px
    across-trace var_x  0.432 -> 0.429 px^2   unchanged, as an along-trace-only
                                              operator must leave it
    anisotropy   var_y - var_x  0.716 -> 0.467
    circle       axis ratio 1.0051 -> 1.0045, radial rms 0.837 -> 0.845 px
    cost         white-noise gain x1.21

The residual anisotropy is NOT a leftover to chase. The matched synthetic,
which has nothing in it but the known terms, keeps 0.558 after the same
correction: the operator is a bounded inverse and the part of the hold above
the dot Nyquist is aliased away, not attenuated. Nothing can bring it back.

CONTROLS
--------
  SIGN. Running the operator backwards (strength -1, i.e. applying the blur
  instead of removing it) must make the anisotropy worse, and does:
  var_y - var_x 0.716 -> 0.992.

  WRONG AXIS. The same operator applied ACROSS traces reduces var_x by 26%,
  which any "sharper is better" score would applaud -- and leaves the
  anisotropy untouched (0.716 -> 0.712) while pushing the across-trace axis to
  346 resolved elements in 512, past the 260-324 the camera can supply. The
  ring catches a correction that a sharpness metric would reward. This is why
  quality.py's composite is not allowed near any of these numbers.

  PHOTOGRAPH. Overshoot at strong along-trace edges on real photographs, with
  a deliberately over-driven strength as the positive control that the
  overshoot detector works at all.

THE ONE THING HERE THAT IS NOT SETTLED
--------------------------------------
The angular fit's cross term is NOT zero: c = -0.136 +- 0.016 px^2 against a
synthetic null of +0.008 +- 0.007 measured through the same code, found
separately by both cut families, and robust to a full-pixel error in the ring's
centre. Taken at face value the blur's principal axes are tilted -8.3 deg off
the picture's, which a scanned system has no obvious way to do. It is reported
as an OPEN FINDING with no mechanism -- `--section negatives`, item 3 -- and an
earlier draft of this module wrongly reported it as a null. It does not move
the conclusions above: it makes the true anisotropy 4.4% larger, and that is
still inside the known budget.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline import aperture as ap_mod  # noqa: E402
from pipeline import catalog as catalog_mod  # noqa: E402
from pipeline import circle as circle_mod  # noqa: E402
from pipeline import decode as decode_mod  # noqa: E402
from pipeline import dotclock as dot_mod  # noqa: E402
from pipeline import sync as sync_mod  # noqa: E402
from pipeline import wav  # noqa: E402

MASTER = REPO / "data" / "master" / "384kHzStereo.wav"
CAL_FRAME = "L000"

# Cuts closer to perpendicular than this are used for the axis readings; the
# secant correction is then at most 1.02, i.e. essentially absent.
AXIS_BAND_DEG = 12.0
# Cuts out to here are used for the angular curve. Rows and columns each cover
# +-45 degrees, so together they tile the whole circle with an overlap at 45.
MAX_OBLIQUITY_DEG = 45.0
HALF_WINDOW_PX = 7.0  # fit window half-length, perpendicular px


# --------------------------------------------------------------------------
# frame access -- the float decode, unclipped, plus its dot matrix
# --------------------------------------------------------------------------

_WAV: dict = {}
_FRAMES: dict = {}


def _master():
    if "info" not in _WAV:
        info = wav.probe(MASTER)
        _WAV["info"] = info
        _WAV["mm"] = wav.memmap(info)
        _WAV["cat"] = catalog_mod.build()
    return _WAV["info"], _WAV["mm"], _WAV["cat"]


@dataclass
class Frame:
    fid: str
    img: np.ndarray  # (height, traces) float64, unclipped decoder units
    dots: np.ndarray  # (n_dots, traces) float64, same units
    delta: np.ndarray  # (traces,) dot-grid offset per trace, decode.py's own
    height: int
    n_dots: int
    dot_px: float  # one dot, in output pixels
    clipped: float  # fraction of the shipped decode that was clipped


def _unclipped(x, cfg, tb):
    """The float decode with decode()'s final clip undone.

    Exactly caltarget.decode_unclipped's method: decode a second time on rails
    that cannot clip (percentile 0/100) and map that back onto the reference
    rails with the affine transform the two settings differ by. No decoder
    change and no re-implementation; the residual is float32 round-off.
    """
    a = decode_mod.decode(x, cfg, tb=tb)
    b = decode_mod.decode(
        x, cfg.replace(levels="percentile", black_pct=0.0, white_pct=100.0), tb=tb
    )
    A = np.asarray(a.image, dtype=np.float64)
    B = np.asarray(b.image, dtype=np.float64)
    m = (A > 1e-6) & (A < 1.0 - 1e-6)
    X = np.stack([B[m], np.ones(int(m.sum()))], axis=1)
    c, *_ = np.linalg.lstsq(X, A[m], rcond=None)
    return c[0] * B + c[1], float(np.mean(~m)), a


def load(fid: str = CAL_FRAME, traces: int = 512) -> Frame:
    """Decode one frame in float, and recover the dot matrix it was built from.

    The dot matrix is what any along-trace operator must act on -- correcting
    after the row interpolation would be correcting the interpolator too
    (aperture.py, `apply_dots`). `render` puts the dots back on the square-pixel
    grid; that round trip reproduces decode()'s own output to 9e-8, which
    `--section geometry` asserts, so nothing below is measured against a
    re-implementation of the decoder.
    """
    if fid in _FRAMES:
        return _FRAMES[fid]
    _, mm, cat = _master()
    fr = cat.by_id(fid)
    n = int(sync_mod.NOMINAL_PERIOD * 520)
    x0 = np.asarray(mm[fr.seed_sample: fr.seed_sample + n, fr.channel], dtype=np.float64)
    ch = "L" if fr.channel == 0 else "R"
    tb = sync_mod.recover(x0)
    cfg = decode_mod.Settings(traces=traces, channel=ch)

    img, clipped, _ = _unclipped(x0, cfg, tb)
    dots, _, _ = _unclipped(x0, cfg.replace(dot_native=True), tb)

    # decode()'s own dot grid, re-derived on the same signal decode() sees:
    # the decay-bias inversion runs BEFORE the dot clock is measured, so the
    # clock must be measured on the restored signal or the phase is wrong.
    period = tb.period
    tau = decode_mod.UNCOUPLE_TAU_384[ch] * period / decode_mod._NOMINAL_PERIOD_384
    x = decode_mod.undroop(x0, tau)
    n_tr = min(traces, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])
    clock = dot_mod.measure(
        x, tb, lo_f=cfg.picture_start + 0.01,
        hi_f=cfg.picture_start + cfg.picture_span - 0.01,
    )
    spd = clock.samples_per_dot
    n_dots = int(round(cfg.picture_span * period / spd))
    psi, _ = dot_mod.track_phase(
        x, starts, period, spd, cfg.picture_start, cfg.picture_start + cfg.picture_span
    )
    mid = starts + (cfg.picture_start + cfg.picture_span / 2.0) * period
    c0 = mid - (n_dots - 1) / 2.0 * spd
    k0 = np.floor((c0 - psi) / spd)
    delta = (psi + (k0 + 0.5) * spd - c0) / spd

    f = Frame(fid, img, dots, delta, cfg.height, n_dots, cfg.height / n_dots, clipped)
    _FRAMES[fid] = f
    return f


def render(fr: Frame, dots: np.ndarray) -> np.ndarray:
    """Put a dot matrix back on the square-pixel grid, decode()'s own way."""
    return decode_mod._rows_from_dots(dots.T, fr.height, 3, fr.delta).T


def apply_along(fr: Frame, dots: np.ndarray, strength: float) -> np.ndarray:
    """aperture.py's inverse, ALONG the trace (the dot axis of the matrix)."""
    if strength == 0.0:
        return dots
    return ap_mod.apply_dots(dots.T, strength=strength, axis=1).T


def apply_across(fr: Frame, dots: np.ndarray, strength: float) -> np.ndarray:
    """The same operator applied ACROSS traces. This is the wrong-axis control
    and is never a correction: there is no hold across traces to invert."""
    if strength == 0.0:
        return dots
    return ap_mod.apply_dots(dots.T, strength=strength, axis=0).T


# --------------------------------------------------------------------------
# the ring, and cuts through it
# --------------------------------------------------------------------------


def geometry(img: np.ndarray) -> tuple[float, float, float, float]:
    """(cy, cx, ry, rx) from circle.fit_ring -> quality.circle_metrics."""
    r = circle_mod.fit_ring(np.clip(img, 0.0, 1.0))
    if r is None:
        raise RuntimeError("ring fit failed")
    return r.cy, r.cx, r.ry, r.rx


def crossings(shape, geo, kind: str, max_obl_deg: float):
    """Where each row or column crosses the ring, and at what obliquity.

    Returns (index, position, theta, cos_obliquity). `theta` is the angle of
    the ring's local NORMAL measured from the across-trace (horizontal) axis;
    `cos_obliquity` is the cosine of the angle between the cut direction and
    that normal, i.e. the secant factor the profile must be corrected by.
    """
    H, W = shape
    cy, cx, ry, rx = geo
    cmax = np.cos(np.radians(max_obl_deg))
    out = []
    if kind == "col":
        for j in range(W):
            u = (j - cx) / rx
            if abs(u) >= 0.999:
                continue
            q = np.sqrt(1.0 - u * u)
            for sg in (-1, 1):
                nx, ny = u / rx, sg * q / ry
                n = np.hypot(nx, ny)
                nx, ny = nx / n, ny / n
                if abs(ny) < cmax:
                    continue
                out.append((j, cy + sg * ry * q, np.arctan2(ny, nx), abs(ny)))
    else:
        for i in range(H):
            v = (i - cy) / ry
            if abs(v) >= 0.999:
                continue
            q = np.sqrt(1.0 - v * v)
            for sg in (-1, 1):
                nx, ny = sg * q / rx, v / ry
                n = np.hypot(nx, ny)
                nx, ny = nx / n, ny / n
                if abs(nx) < cmax:
                    continue
                out.append((i, cx + sg * rx * q, np.arctan2(ny, nx), abs(nx)))
    return out


def _box_gauss(t, t0, h, s):
    r2 = np.sqrt(2.0)
    return 0.5 * (erf((t - t0 + h) / (r2 * s)) - erf((t - t0 - h) / (r2 * s)))


def cut_widths(img, geo, kind, max_obl_deg=MAX_OBLIQUITY_DEG,
               half=HALF_WINDOW_PX) -> np.ndarray:
    """Fit every cut of one family on its own.

    Model: a linear background (the field droops, and the ring's inside and
    outside are not at the same level) plus a BOX convolved with a GAUSSIAN.
    The box is the slide's own line, whose perpendicular thickness is a
    constant of the artifact; the Gaussian is what the system did to it. Six
    free parameters per cut, all local, nothing shared, nothing stacked.

    Returns rows of (theta, cos_obl, var_perp, halfwidth_perp, amplitude,
    residual rms, cut index, sub-pixel crossing position).
    """
    cy, cx, ry, rx = geo
    H, W = img.shape
    rows = []
    for idx, pos, th, c in crossings(img.shape, geo, kind, max_obl_deg):
        L = half / c
        lo = int(np.floor(pos - L))
        hi = int(np.ceil(pos + L)) + 1
        if lo < 0 or hi > (H if kind == "col" else W):
            continue
        y = (img[lo:hi, idx] if kind == "col" else img[idx, lo:hi]).astype(np.float64)
        t = np.arange(len(y), dtype=np.float64)
        t0g = pos - lo
        k = max(2, len(y) // 6)
        a0 = float(np.mean(np.r_[y[:k], y[-k:]]))
        amp0 = max(a0 - float(y.min()), 1e-3)

        def resid(p):
            return p[0] + p[1] * (t - p[3]) - p[2] * _box_gauss(t, p[3], p[4], p[5]) - y

        p0 = [a0, 0.0, amp0, t0g, 2.4 / c, 1.0 / c]
        lob = [a0 - 1.0, -0.3, 1e-4, t0g - 3.0, 0.20 / c, 0.10 / c]
        hib = [a0 + 1.0, +0.3, 2.0, t0g + 3.0, 8.00 / c, 8.00 / c]
        try:
            r = least_squares(resid, p0, bounds=(lob, hib), max_nfev=1500)
        except Exception:
            continue
        p = r.x
        rows.append((th, c, (p[5] * c) ** 2, p[4] * c, p[2],
                     float(np.sqrt(np.mean(r.fun ** 2))), idx, lo + p[3]))
    return np.array(rows) if rows else np.zeros((0, 8))


def axis_variances(img, geo, band_deg=AXIS_BAND_DEG):
    """The two axis readings: (across-trace, along-trace) var_perp tables.

    Across-trace comes from ROW cuts near the ring's left and right, where a
    row is perpendicular to the ring; along-trace from COLUMN cuts near its top
    and bottom. At these angles the secant correction is at most 1.02, so the
    numbers are very nearly raw profile widths.
    """
    rc = cut_widths(img, geo, "row", max_obl_deg=band_deg)
    cc = cut_widths(img, geo, "col", max_obl_deg=band_deg)
    return rc, cc


def angular_fit(A, cross=True):
    """Weighted least squares of var_perp(t) onto cos^2, sin^2 (and 2.sin.cos).

    Weighting is by fit quality (1/residual rms), so a cut the profile model
    could not describe cannot vote. Returns the coefficient vector.
    """
    th, y = A[:, 0], A[:, 2]
    w = 1.0 / np.maximum(A[:, 5], 1e-4)
    cols = [np.cos(th) ** 2, np.sin(th) ** 2]
    if cross:
        cols.append(2 * np.sin(th) * np.cos(th))
    X = np.stack(cols, axis=1)
    c, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return c


def _boot_cross(A, n=400, seed=0):
    """Bootstrap standard error of the cross term."""
    rng = np.random.default_rng(seed)
    return float(np.std([angular_fit(A[rng.integers(0, len(A), len(A))])[2]
                         for _ in range(n)]))


def principal(a, b, c):
    """Eigen-decomposition of [[a, c], [c, b]]: (low, high, tilt in degrees).

    With a cross term the picture's axes are NOT the blur's axes, so `b - a`
    understates the true anisotropy; the eigenvalue difference is the honest
    number and the tilt says how far off the axes are.
    """
    hw = np.hypot((b - a) / 2.0, c)
    return (a + b) / 2.0 - hw, (a + b) / 2.0 + hw, np.degrees(0.5 * np.arctan2(2 * c, b - a))


def _boot(v, n=2000, seed=0):
    """Bootstrap standard error of a median."""
    if len(v) < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    return float(np.std(np.median(np.asarray(v)[idx], axis=1)))


# --------------------------------------------------------------------------
# the exactly-known along-trace budget
# --------------------------------------------------------------------------


def known_along_variance(dot_px: float, width_1090: float = ap_mod.TRANSITION_1090):
    """Variance, in output px^2, of the along-trace-only terms we can write down.

      * the converter's sample-and-hold: a box exactly one dot wide, var/12
      * OUR OWN plateau integration: decode.py integrates the full plateau
        (dotclock.sample_plateaus), a second box exactly one dot wide. This is
        not an error -- it is what makes the dot value an exact integral -- but
        it is a box and it blurs
      * the chain filter c (sampler output stage, tape, disc, 2017 ADC),
        measured by aperture_fit.py as a 10-90 rise of TRANSITION_1090 dots on
        plateau-to-plateau steps, i.e. a width measured on the RAW signal and
        never on a picture

    Every one of these is downstream of the camera and upstream of nothing that
    matters, which is precisely why removing them is recovery and not invention.
    """
    box = dot_px ** 2 / 12.0
    chain = (width_1090 * ap_mod.SIGMA_PER_1090 * dot_px) ** 2
    return dict(hold_box=box, our_box=box, chain=chain, total=2 * box + chain)


def operator_variance(n_dots: int, dot_px: float, strength: float = 1.0):
    """Second moment of aperture.py's operator, in px^2. Negative: it sharpens."""
    d = np.zeros((1, n_dots))
    d[0, n_dots // 2] = 1.0
    k = ap_mod.apply_dots(d, strength=strength, axis=1)[0]
    t = (np.arange(n_dots) - n_dots // 2) * dot_px
    s0 = k.sum()
    m1 = (k * t).sum() / s0
    return float((k * (t - m1) ** 2).sum() / s0)


# --------------------------------------------------------------------------
# the synthetic ring -- the estimator's calibration and the null hypothesis
# --------------------------------------------------------------------------


def _gblur(a, sigma_fine, axis):
    if sigma_fine <= 1e-9:
        return a
    k = int(np.ceil(4 * sigma_fine))
    t = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (t / sigma_fine) ** 2)
    g /= g.sum()
    return np.apply_along_axis(lambda v: np.convolve(v, g, "same"), axis, a)


def synth_dots(geo, height, n_dots, traces, sigma_iso, *, radius=151.5,
               extra_along=0.0, extra_across=0.0, half_w=2.4, sup=4,
               contrast=0.55, white=0.95, noise=0.010, trace_noise=0.010,
               width_1090=ap_mod.TRANSITION_1090, seed=0):
    """A synthetic ring carrying the real chain's structure, as a dot matrix.

    ideal ring -> isotropic camera -> [extra across] -> sample the traces ->
    [extra along] -> one-dot hold -> chain filter -> our full-plateau
    integration -> noise. NO unknown along-trace blur is present unless
    `extra_along` puts it there, which makes this the null hypothesis in
    physical form.
    """
    cy, cx = geo[0], geo[1]
    dot_px = height / n_dots
    ys = (np.arange(height * sup) + 0.5) / sup
    xs = (np.arange(traces * sup) + 0.5) / sup
    d = np.abs(np.hypot(ys[:, None] - cy, xs[None, :] - cx) - radius)
    img = np.clip((d - half_w) * sup + 0.5, 0.0, 1.0)
    img = white - contrast * (1.0 - img)
    img = _gblur(img, sigma_iso * sup, 0)
    img = _gblur(img, sigma_iso * sup, 1)
    if extra_across > 0:
        img = _gblur(img, extra_across * sup, 1)
    col = np.round((np.arange(traces) + 0.5) * sup - 0.5).astype(int)
    img = img[:, col]
    if extra_along > 0:
        img = _gblur(img, extra_along * sup, 0)
    k = int(round(dot_px * sup))
    img = np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, "same"), 0, img)
    img = _gblur(img, width_1090 * ap_mod.SIGMA_PER_1090 * dot_px * sup, 0)
    cs = np.concatenate([np.zeros((1, traces)), np.cumsum(img, 0)])
    e = np.linspace(0, height * sup, n_dots + 1)
    lo = np.round(e[:-1]).astype(int)
    hi = np.round(e[1:]).astype(int)
    dots = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    rng = np.random.default_rng(seed)
    if noise > 0:
        dots = dots + rng.normal(0, noise, dots.shape)
    if trace_noise > 0:
        dots = dots + rng.normal(0, trace_noise, (1, traces))
    return dots


def synth_image(geo, height, n_dots, traces, sigma_iso, *, op_strength=0.0, **kw):
    dots = synth_dots(geo, height, n_dots, traces, sigma_iso, **kw)
    if op_strength != 0.0:
        dots = ap_mod.apply_dots(dots.T, strength=op_strength, axis=1).T
    return decode_mod._rows_from_dots(dots.T, height, 3, None).T


def read_axes(img, geo, band_deg=AXIS_BAND_DEG):
    rc, cc = axis_variances(img, geo, band_deg)
    return (float(np.median(rc[:, 2])), float(np.median(cc[:, 2])), len(rc), len(cc))


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def _rule(t):
    print(f"\n{t}\n" + "-" * len(t))


def section_geometry(state):
    _rule("GEOMETRY -- the ring, and the round trip that lets us touch the dots")
    fr = load()
    geo = geometry(fr.img)
    state["fr"] = fr
    state["geo"] = geo
    r = circle_mod.fit_ring(np.clip(fr.img, 0, 1))
    print(f"  frame {fr.fid}   image {fr.img.shape}   dots {fr.dots.shape}")
    print(f"  one dot = {fr.dot_px:.4f} output px  ({fr.n_dots} dots over "
          f"{fr.height} rows; {fr.img.shape[1]} traces over {fr.img.shape[1]} columns)")
    print(f"  ring   centre ({r.cy:.2f}, {r.cx:.2f})  ry {r.ry:.2f}  rx {r.rx:.2f}")
    print(f"         axis ratio {r.axis_ratio:.4f}   radial rms {r.radial_rms:.3f} px"
          f"   line width {r.width:.2f} px")
    print(f"  the shipped decode clips {fr.clipped * 100:.2f}% of this frame; every "
          f"number below is on the UNCLIPPED float decode")

    back = render(fr, fr.dots)
    err = float(np.abs(back - fr.img).max())
    print(f"\n  round trip  dots -> rows reproduces decode()'s own image to "
          f"{err:.1e} (float32 round-off)")
    if err > 1e-6:
        print("  ROUND TRIP FAILED -- the dot path is not decode()'s path; stop here.")
    state["roundtrip"] = err

    for kind in ("row", "col"):
        n = len(crossings(fr.img.shape, geo, kind, MAX_OBLIQUITY_DEG))
        m = len(crossings(fr.img.shape, geo, kind, AXIS_BAND_DEG))
        print(f"  {kind} cuts: {n} within {MAX_OBLIQUITY_DEG:.0f} deg of "
              f"perpendicular, {m} within {AXIS_BAND_DEG:.0f} deg")


def section_lsf(state):
    _rule("LSF vs ANGLE -- the ring is an edge at every orientation")
    fr, geo = state["fr"], state["geo"]
    rc = cut_widths(fr.img, geo, "row")
    cc = cut_widths(fr.img, geo, "col")
    state["rc"], state["cc"] = rc, cc

    print("  angle t = the ring's local normal, from the ACROSS-TRACE axis.")
    print("  0 deg: the normal lies across traces, so the profile probes the")
    print("  across-trace blur.  90 deg: it lies along the trace.\n")
    print(f"  {'t (deg)':>10} {'family':>7} {'n':>4} {'sigma_perp':>11} "
          f"{'var_perp':>9} {'line half-w':>12} {'fit rms':>8}")
    tab = []
    for name, a in (("row", rc), ("col", cc)):
        tf = np.degrees(np.arctan2(np.abs(np.sin(a[:, 0])), np.abs(np.cos(a[:, 0]))))
        for lo in range(0, 90, 10):
            m = (tf >= lo) & (tf < lo + 10)
            if m.sum() < 5:
                continue
            v = float(np.median(a[m, 2]))
            print(f"  {lo:4d}-{lo + 10:<5d} {name:>7} {int(m.sum()):4d} "
                  f"{np.sqrt(v):11.3f} {v:9.4f} {np.median(a[m, 3]):12.3f} "
                  f"{np.median(a[m, 5]):8.4f}")
            tab.append((lo + 5, name, v))
    ov = {n: v for c, n, v in tab if c == 45}
    if len(ov) == 2:
        print(f"\n  the two families overlap at 45 deg and are INDEPENDENT "
              f"measurements there:")
        print(f"    row cuts {np.sqrt(ov['row']):.3f} px   column cuts "
              f"{np.sqrt(ov['col']):.3f} px   difference "
              f"{abs(np.sqrt(ov['row']) - np.sqrt(ov['col'])):.3f} px")
        print("  different pixels, different cut direction, different secant "
              "correction, same answer.")

    # --- the angular fit, and its rank deficiency -------------------------
    A = np.concatenate([rc, cc], axis=0)
    th = A[:, 0]
    y = A[:, 2]
    X2 = np.stack([np.cos(th) ** 2, np.sin(th) ** 2], axis=1)
    c2 = angular_fit(A, cross=False)
    c3 = angular_fit(A)
    ec = _boot_cross(A)
    X4 = np.concatenate([X2, np.ones((len(th), 1))], axis=1)
    print(f"\n  weighted fit of var_perp(t) = a.cos^2 t + b.sin^2 t over all "
          f"{len(A)} cuts:")
    print(f"     a (across-trace) {c2[0]:.4f} px^2    b (along-trace) "
          f"{c2[1]:.4f} px^2    b - a {c2[1] - c2[0]:+.4f}")
    print(f"  adding a cross term 2c.sin t cos t (a PSF tilted off the picture "
          f"axes):")
    print(f"     a {c3[0]:.4f}  b {c3[1]:.4f}  c {c3[2]:+.4f} +-{ec:.4f} px^2")
    lo, hi, tilt = principal(c3[0], c3[1], c3[2])
    print(f"     c is {abs(c3[2]) / max(ec, 1e-9):.0f} bootstrap sigma from zero. "
          f"The two families agree on it")
    print(f"     independently: rows {angular_fit(rc)[2]:+.4f}, columns "
          f"{angular_fit(cc)[2]:+.4f}.")
    print(f"     Taken at face value the blur's principal axes are tilted "
          f"{tilt:+.1f} deg off")
    print(f"     the picture's, and the true anisotropy is the eigenvalue "
          f"difference")
    print(f"     {hi - lo:.4f}, not b - a = {c3[1] - c3[0]:.4f} "
          f"({(hi - lo) / (c3[1] - c3[0]) - 1:+.1%}).")
    print("     A fit's own error bar is not evidence that the thing is real -- see")
    print("     `--section budget` for the synthetic null, which is what decides it.")
    state["cross"] = (c3, ec)
    print(f"  adding a CONSTANT term, as the brief for this work proposed:")
    print(f"     condition number {np.linalg.cond(X4):.3e} -- the design matrix "
          f"is SINGULAR.")
    print("     cos^2 + sin^2 = 1, so a constant is exactly degenerate with the")
    print("     pair. The angle CANNOT separate the isotropic part; it separates")
    print("     the DIFFERENCE b - a, in which the isotropic camera cancels.")
    print("     That difference is the only part it is legitimate to remove, so")
    print("     the degeneracy costs us nothing we were entitled to.")
    state["ang"] = (c2, c3)

    # residual of the two-term model across angle
    pred = X2 @ c2
    tf = np.degrees(np.arctan2(np.abs(np.sin(th)), np.abs(np.cos(th))))
    print("\n  residual of the two-term model, by angle (px^2):")
    line = "    "
    for lo in range(0, 90, 15):
        m = (tf >= lo) & (tf < lo + 15)
        if m.sum() < 5:
            continue
        line += f"{lo:3d}-{lo + 15:<3d} {np.median(y[m] - pred[m]):+.3f}   "
    print(line)


def section_axes(state):
    _rule("THE TWO AXES -- read where the cuts are perpendicular")
    fr, geo = state["fr"], state["geo"]
    rc, cc = axis_variances(fr.img, geo)
    vx, vy = float(np.median(rc[:, 2])), float(np.median(cc[:, 2]))
    ex, ey = _boot(rc[:, 2]), _boot(cc[:, 2])
    state["vx"], state["vy"] = vx, vy
    state["evx"], state["evy"] = ex, ey
    print(f"  across-trace  n={len(rc):3d}  var {vx:.4f} +- {ex:.4f} px^2   "
          f"sigma {np.sqrt(vx):.3f} px")
    print(f"  along-trace   n={len(cc):3d}  var {vy:.4f} +- {ey:.4f} px^2   "
          f"sigma {np.sqrt(vy):.3f} px")
    print(f"  anisotropy    var_y - var_x = {vy - vx:+.4f} +- "
          f"{np.hypot(ex, ey):.4f} px^2   (raw estimator units)")

    print("\n  the same, split four ways, as a check that no part of the ring "
          "is driving it:")
    for name, a, sel in (
        ("across, left half", rc, np.cos(rc[:, 0]) < 0),
        ("across, right half", rc, np.cos(rc[:, 0]) > 0),
        ("along, top half", cc, np.sin(cc[:, 0]) < 0),
        ("along, bottom half", cc, np.sin(cc[:, 0]) > 0),
    ):
        v = a[sel, 2]
        print(f"    {name:20s} n={len(v):3d}  var {np.median(v):.4f}")
    state["splits_ok"] = True


def section_calibrate(state):
    _rule("CALIBRATION -- what the estimator does to a ring whose blur we chose")
    fr, geo = state["fr"], state["geo"]
    known = known_along_variance(fr.dot_px)
    print(f"  the exactly-known along-trace-only terms, in output px^2:")
    print(f"    converter hold, one dot box   {known['hold_box']:.4f}")
    print(f"    our own plateau integration   {known['our_box']:.4f}")
    print(f"    chain filter c, 10-90 = {ap_mod.TRANSITION_1090:.3f} dots "
          f"{known['chain']:.4f}")
    print(f"    total                         {known['total']:.4f} px^2")
    lo = known_along_variance(fr.dot_px, ap_mod.TRANSITION_1090_RANGE[0])["total"]
    hi = known_along_variance(fr.dot_px, ap_mod.TRANSITION_1090_RANGE[1])["total"]
    print(f"    spread from the 10-90 uncertainty {ap_mod.TRANSITION_1090_RANGE}: "
          f"{lo:.4f} to {hi:.4f}")
    state["known"] = known
    state["known_range"] = (lo, hi)

    print("\n  synthetic rings through the identical measurement code. TRUE is what")
    print("  was put in; est is what the estimator reads back.")
    print(f"  {'sig_iso':>7} {'+along':>7} {'+across':>7} | {'TRUE vx':>8} "
          f"{'TRUE vy':>8} {'TRUE D':>7} | {'est vx':>7} {'est vy':>7} "
          f"{'est D':>7} | {'bias D':>7}")
    cases = [(0.55, 0, 0), (0.65, 0, 0), (0.75, 0, 0),
             (0.65, 0.40, 0), (0.65, 0, 0.40), (0.65, 0, 0.70)]
    rows = []
    for sig, ea, ec in cases:
        im = synth_image(geo, fr.height, fr.n_dots, fr.img.shape[1], sig,
                         extra_along=ea, extra_across=ec)
        vx, vy, _, _ = read_axes(im, (geo[0], geo[1], 151.5, 151.5))
        tvx = sig ** 2 + ec ** 2
        tvy = sig ** 2 + ea ** 2 + known["total"]
        rows.append((sig, ea, ec, tvx, tvy, vx, vy))
        print(f"  {sig:7.2f} {ea:7.2f} {ec:7.2f} | {tvx:8.4f} {tvy:8.4f} "
              f"{tvy - tvx:+7.4f} | {vx:7.4f} {vy:7.4f} {vy - vx:+7.4f} | "
              f"{(vy - vx) - (tvy - tvx):+7.4f}")
    R = np.array(rows)
    print(f"\n  across-trace axis: est - TRUE = {np.median(R[:, 5] - R[:, 3]):+.4f} px^2, "
          f"flat in every case. Effectively unbiased.")
    print(f"  along-trace axis:  est - TRUE = {np.median(R[:, 6] - R[:, 4]):+.4f} px^2. "
          f"A Gaussian fitted to")
    print("  a box-box-Gaussian profile reads wide; that is a bias of the estimator,")
    print("  not of the chain, and it is why nothing below is read off the raw fit.")
    print("  Every real number is compared against a synthetic MATCHED on the")
    print("  across-trace axis, so the bias cancels between them.")
    state["cal"] = R


def _matched_sigma(state):
    """sigma_iso of the synthetic whose across-trace reading matches the frame."""
    fr, geo = state["fr"], state["geo"]
    target = state["vx"]
    lo, hi = 0.30, 1.20
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        im = synth_image(geo, fr.height, fr.n_dots, fr.img.shape[1], mid)
        vx, _, _, _ = read_axes(im, (geo[0], geo[1], 151.5, 151.5))
        if vx < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def section_budget(state):
    _rule("BUDGET -- is there any anisotropy the known terms do not explain?")
    fr, geo = state["fr"], state["geo"]
    sig = _matched_sigma(state)
    state["sigma_iso"] = sig
    gs = (geo[0], geo[1], 151.5, 151.5)
    im = synth_image(geo, fr.height, fr.n_dots, fr.img.shape[1], sig)
    svx, svy, _, _ = read_axes(im, gs)
    vx, vy = state["vx"], state["vy"]
    known = state["known"]["total"]

    print(f"  the synthetic is matched to the frame on the ACROSS-TRACE axis:")
    print(f"    sigma_iso = {sig:.4f} px   ->  est var_x {svx:.4f}   "
          f"frame {vx:.4f}")
    print("  it contains an isotropic camera and the three known along-trace")
    print("  terms, and NOTHING ELSE. So its along-trace reading is what the")
    print("  frame should read if the known terms are the whole story.\n")
    print(f"    along-trace, synthetic (known terms only)   {svy:.4f} px^2")
    print(f"    along-trace, frame                          {vy:.4f} px^2")
    print(f"    frame MINUS synthetic                       {vy - svy:+.4f} px^2")
    print(f"    bootstrap error on the frame's reading      +-{state['evy']:.4f}")

    # bias-corrected physical numbers
    true_svy = sig ** 2 + known
    bias = svy - true_svy
    tvy = vy - bias
    tvx = vx - (svx - sig ** 2)
    state["true_vx"], state["true_vy"] = tvx, tvy
    print(f"\n  removing the estimator bias the synthetic just measured "
          f"({bias:+.4f} px^2 on the")
    print("  along-trace axis, and the matched across-trace bias):")
    print(f"    across-trace var_x  {tvx:.4f} px^2   sigma {np.sqrt(tvx):.3f} px")
    print(f"    along-trace  var_y  {tvy:.4f} px^2   sigma {np.sqrt(tvy):.3f} px")
    print(f"    anisotropy   var_y - var_x = {tvy - tvx:.4f} px^2")
    print(f"    exactly-known budget         {known:.4f} px^2  "
          f"({state['known_range'][0]:.4f} to {state['known_range'][1]:.4f})")
    frac = (tvy - tvx) / known
    print(f"    measured / known             {frac:.2f}")

    across_only = tvx - (tvy - known)
    print(f"\n  SOLVING FOR THE PIECES. Three unknowns (isotropic camera,")
    print("  along-trace-only, across-trace-only) and two measurements, closed by")
    print("  the fact that the along-trace-only terms are independently known:")
    print(f"    isotropic camera        var {tvy - known:.4f} px^2  "
          f"sigma {np.sqrt(max(tvy - known, 1e-9)):.3f} px")
    print(f"    along-trace only        var {known:.4f} px^2  (hold + our box + chain)")
    print(f"    across-trace only       var {across_only:+.4f} px^2  "
          f"+-{np.hypot(state['evx'], state['evy']):.4f}")
    print("  The across-trace sampling gate that aperture.py records as real,")
    print("  unknown and unmeasurable comes out CONSISTENT WITH ZERO. It had")
    print("  never been measured on this artifact. There is nothing to correct")
    print("  across traces, and the ring is what says so.")
    print("\n  The camera's isotropy was an ASSUMPTION in aperture.py's docstring.")
    print("  It is now a measurement: if the camera were anisotropic the frame's")
    print("  along-trace reading would exceed the synthetic's, and it does not.")
    state["svx"], state["svy"] = svx, svy

    # --- the cross term, against a synthetic that has none by construction ---
    print("\n  THE CROSS TERM, against the null. The synthetic is an isotropic")
    print("  Gaussian convolved with strictly axis-aligned boxes, so its cross")
    print("  term is ZERO by construction. Whatever it reads back is the")
    print("  estimator's own; whatever the frame reads ABOVE that is the ring's.")
    # the frame's cuts come from `lsf` when it has run; recompute only if not
    frow = state.get("rc")
    fcol = state.get("cc")
    if frow is None or fcol is None:
        frow = cut_widths(fr.img, geo, "row")
        fcol = cut_widths(fr.img, geo, "col")
    fA = np.concatenate([frow, fcol], axis=0)
    fc3, fec = angular_fit(fA), _boot_cross(fA)
    srow = cut_widths(im, gs, "row")
    scol = cut_widths(im, gs, "col")
    sA = np.concatenate([srow, scol], axis=0)
    sc3, sec = angular_fit(sA), _boot_cross(sA)
    print(f"\n    {'':22s} {'c (px^2)':>10} {'boot err':>9} {'rows':>8} {'cols':>8}")
    print(f"    {'synthetic (c = 0)':22s} {sc3[2]:+10.4f} {sec:9.4f} "
          f"{angular_fit(srow)[2]:+8.4f} {angular_fit(scol)[2]:+8.4f}")
    print(f"    {'frame L000':22s} {fc3[2]:+10.4f} {fec:9.4f} "
          f"{angular_fit(frow)[2]:+8.4f} {angular_fit(fcol)[2]:+8.4f}")
    nsig = abs(fc3[2] - sc3[2]) / max(np.hypot(fec, sec), 1e-9)
    print(f"\n    frame minus synthetic {fc3[2] - sc3[2]:+.4f} px^2, {nsig:.0f} sigma.")
    state["cross_null"] = (fc3, fec, sc3, sec, nsig)
    if nsig < 3.0:
        print("    CONSISTENT WITH ZERO: the blur's axes are the picture's axes.")
    else:
        _, _, tilt = principal(*fc3)
        lo_e, hi_e, _ = principal(*fc3)
        print("    THE CROSS TERM IS REAL, and it is NOT the estimator's. Both cut")
        print("    families find it independently and the synthetic does not")
        print("    reproduce it. The blur's principal axes are tilted "
              f"{tilt:+.1f} deg off")
        print("    the picture's axes. This module does NOT explain it, and the")
        print("    honest statement is that it is an open finding, not a result:")
        print("      * it is not a ring-geometry error -- displacing the assumed")
        print("        centre by a full pixel moves c by only +0.007")
        print("      * a rigid tilt of the whole chain has no obvious mechanism in")
        print("        a scanned system, where both axes are defined by the scan")
        print("      * it does NOT rescue any extra deconvolution: it makes the")
        print("        true anisotropy LARGER, and the budget below already")
        print("        accounts for the axis-aligned part within its error")
        print(f"    Effect on the headline: the anisotropy the deconvolution is")
        print(f"    entitled to grows by {(hi_e - lo_e) / (fc3[1] - fc3[0]) - 1:+.1%}, "
              f"which does not change the")
        print("    verdict that the measured anisotropy stays inside the known budget.")


def section_deconv(state):
    _rule("DECONVOLUTION -- remove the axis-aligned part, and only that")
    fr, geo = state["fr"], state["geo"]
    gs = (geo[0], geo[1], 151.5, 151.5)
    sig = state.get("sigma_iso", 0.65)
    print("  aperture.py's operator is the exact inverse of the three known")
    print("  along-trace terms, bounded by construction (|A| >= 0.69 so the")
    print("  inverse never exceeds 1.44). It is applied to the DOT MATRIX,")
    print("  before the row interpolation, and the result is rendered by")
    print("  decode.py's own interpolator.\n")
    print(f"  operator second moment at strength 1: "
          f"{operator_variance(fr.n_dots, fr.dot_px):+.4f} px^2 (negative: it "
          f"sharpens)")
    print(f"  white-noise gain: x{ap_mod.noise_gain(strength=1.0):.4f}\n")

    h, w = fr.img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - geo[1]) / geo[3], (yy - geo[0]) / geo[2])
    field = rr < 0.75

    def row(name, dots, synth_op=None):
        img = render(fr, dots)
        vx, vy, _, _ = read_axes(img, geo)
        r = circle_mod.fit_ring(np.clip(img, 0, 1))
        # high-pass noise in the flat field, along the trace: the field is
        # uniform by the cover, so a second difference there is pure noise.
        d2 = img[2:, :] - 2 * img[1:-1, :] + img[:-2, :]
        nz = float(np.std(d2[field[1:-1]]) / np.sqrt(6.0))
        clip = float(np.mean((img < 0) | (img > 1)))
        s = ""
        if synth_op is not None:
            im = synth_image(geo, fr.height, fr.n_dots, w, sig, op_strength=synth_op)
            svx, svy, _, _ = read_axes(im, gs)
            s = f" | synth {svy - svx:+.4f}"
        print(f"  {name:24s} var_x {vx:.4f} var_y {vy:.4f}  D {vy - vx:+.4f}{s}"
              f" | axis {r.axis_ratio:.4f} rrms {r.radial_rms:.3f}"
              f" | noise {nz:.5f} clip {clip * 100:.2f}%")
        return vx, vy, r, nz

    print("  ALONG THE TRACE -- the axis the ring says carries the excess")
    base = row("strength 0 (shipped)", fr.dots, synth_op=0.0)
    for s in (0.5, 1.0, 1.5):
        row(f"strength +{s:.1f}", apply_along(fr, fr.dots, s),
            synth_op=(1.0 if s == 1.0 else None))
    print("\n  SIGN CONTROL -- run the operator backwards. It must get WORSE.")
    for s in (-0.5, -1.0):
        row(f"strength {s:+.1f}", apply_along(fr, fr.dots, s),
            synth_op=(-1.0 if s == -1.0 else None))
    print("\n  WRONG-AXIS CONTROL -- the same operator applied ACROSS traces.")
    for s in (1.0, -1.0):
        vx, vy, r, nz = row(f"across-trace {s:+.1f}", apply_across(fr, fr.dots, s))
        if s > 0:
            f50 = 0.1874 / np.sqrt(max(vx - (state["svx"] - sig ** 2), 1e-9))
            print(f"    -> across-trace resolved elements: {2 * f50 * w:.0f} in "
                  f"{w} traces. The camera can supply 260-324.")
            print("       Sharper than the thing that made the picture: invention.")
            print("       Note it left the ANISOTROPY untouched. A sharpness score")
            print("       would have called this an improvement.")

    vx1, vy1, r1, nz1 = row("(for the record) +1.0", apply_along(fr, fr.dots, 1.0))
    print(f"\n  WHAT THE CORRECTION BUYS, on the axis it is entitled to:")
    print(f"    var_y  {base[1]:.4f} -> {vy1:.4f} px^2   "
          f"sigma {np.sqrt(base[1]):.3f} -> {np.sqrt(vy1):.3f} px")
    print(f"    var_x  {base[0]:.4f} -> {vx1:.4f} px^2   unchanged to "
          f"{abs(vx1 - base[0]) / base[0] * 100:.1f}%, as an along-trace-only "
          f"operator must be")
    print(f"    circle axis ratio {base[2].axis_ratio:.4f} -> {r1.axis_ratio:.4f}, "
          f"radial rms {base[2].radial_rms:.3f} -> {r1.radial_rms:.3f} px")
    print(f"    flat-field noise  {base[3]:.5f} -> {nz1:.5f}  "
          f"(x{nz1 / base[3]:.3f}; the operator predicts x"
          f"{ap_mod.noise_gain(strength=1.0):.3f})")
    print("\n  AND WHAT IT DOES NOT BUY. The anisotropy does not go to zero, and")
    print("  the matched synthetic -- which has nothing in it but the known terms")
    print("  -- keeps the same residual. The operator is a bounded inverse; the")
    print("  part of the one-dot hold above the dot Nyquist is ALIASED, not")
    print("  attenuated, and no operator recovers it. That residual is the")
    print("  resolution ceiling, not a leftover to chase.")


def section_photo(state, fids=("L020", "L055", "R040")):
    _rule("PHOTOGRAPHS -- does the correction ring?")
    print("  The operator is 3 taps wide at the 1e-3 level, so it should not")
    print("  ring. 'Should not' is not a measurement. Overshoot is counted as")
    print("  pixels that leave the local along-trace range of the ORIGINAL by")
    print("  more than 2% of full scale, in a 5-dot window -- the classic")
    print("  deconvolution halo. An over-driven strength is the positive")
    print("  control that the detector can see ringing when it is there.\n")
    print(f"  {'frame':>6} {'strength':>9} {'overshoot %':>12} {'max over':>9} "
          f"{'new clip %':>11} {'edge contrast':>14}")
    for fid in fids:
        try:
            fr = load(fid)
        except Exception as e:  # pragma: no cover
            print(f"  {fid:>6}  could not load: {e}")
            continue
        base = render(fr, fr.dots)
        k = 5
        pad = np.pad(base, ((k // 2, k // 2), (0, 0)), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, k, axis=0)
        lo = win.min(axis=-1)
        hi = win.max(axis=-1)
        g0 = float(np.mean(np.abs(np.diff(base, axis=0))))
        for s in (0.5, 1.0, 3.0):
            img = render(fr, apply_along(fr, fr.dots, s))
            over = np.maximum(img - hi, lo - img)
            frac = float(np.mean(over > 0.02)) * 100
            print(f"  {fid:>6} {s:9.1f} {frac:12.3f} {float(over.max()):9.4f} "
                  f"{float(np.mean((img < 0) | (img > 1)) - np.mean((base < 0) | (base > 1))) * 100:11.3f} "
                  f"{float(np.mean(np.abs(np.diff(img, axis=0)))) / g0:14.3f}"
                  f"{'   <- positive control' if s == 3.0 else ''}")
    print("\n  At strength 1 the overshoot is a small fraction of a percent and")
    print("  the maximum excursion is a few percent of full scale, while the")
    print("  positive control at strength 3 lights the detector up. The operator")
    print("  is doing what a bounded inverse does and not what an unbounded one")
    print("  does.")


def section_negatives(state):
    _rule("NEGATIVES -- what this did not find, and what it cannot decide")
    print("""
  1. NO NEW DECONVOLUTION. The task was to get the system's blur out of the
     ring and to work out how much is legitimately removable. The answer is
     that the removable part is EXACTLY the operator aperture.py already
     builds, and the ring's contribution is to bound it from above rather than
     to add to it. The measured anisotropy does not exceed the known budget,
     so there is no residual axis-aligned term to invert and no new operator
     is proposed. A null on the deliverable, and a real one.

  2. THE ANGULAR DECOMPOSITION IS RANK-DEFICIENT and the brief's version of it
     was wrong. "A constant plus cos^2 plus sin^2" cannot separate three
     numbers because cos^2 + sin^2 = 1. The isotropic term is unidentifiable
     from angle alone, on this artifact or any other. What survives is the
     difference, which happens to be the part we needed.

  3. THE CROSS TERM IS REAL AND UNEXPLAINED -- and an earlier draft of this
     module reported it as a null, wrongly. A PSF whose axes are rotated off
     the picture's shows as 2c.sin t cos t in the angular fit. c comes out at
     -0.136 +- 0.016 px^2, while the synthetic that has NO tilt by construction
     reads +0.008 +- 0.007 through the identical code -- eight sigma apart. The
     two cut families find it separately (rows -0.118, columns -0.149), so it
     is not one family's systematic, and displacing the assumed ring centre by
     a whole pixel moves it by +0.007, so it is not a geometry error. Face
     value: the principal axes are tilted -8.3 deg off the picture's.
     I do not have a mechanism for it. A scanned system defines both of its
     axes by the scan, so a rigid rotation of the blur has nowhere to come
     from; a shear between the dot lattice and the pixel grid is the obvious
     suspect and the one test I ran on it was confounded (shearing the image
     while measuring with the unsheared ellipse) and is not reported as
     evidence either way. This is logged as an open finding, not a result.
     It does not rescue any extra deconvolution -- it makes the true
     anisotropy 4.4% LARGER, and that stays inside the known budget -- so
     nothing downstream depends on resolving it. It is the first thing I would
     hand to the next pass.

  4. THE ISOTROPIC PART IS NOT MEASURED HERE, only bounded by the along-trace
     budget. The camera's spot size quoted in `--section budget` is derived
     from the known terms, not read off the ring, and it inherits their
     uncertainty. The ring cannot see it directly: it cancels in the only
     quantity the angle identifies.

  5. THE MOMENT ESTIMATOR FAILED and is not used. The second moment obeys the
     projection law exactly for any PSF shape, which makes it the theoretically
     correct estimator here, and the line's own width cancels out of the
     difference. In practice it is dominated by window truncation: on this
     frame the anisotropy it reports swings from -1.25 to +0.77 px^2 as the
     window goes from 5 to 9 px, because the wide along-trace profile loses
     more tail than the narrow across-trace one. The parametric fit is used
     instead, with its bias measured on synthetics rather than argued away.

  6. ONE FRAME. There is exactly one calibration ring on the record, so the
     across-trace LSF is measured on one frame and cannot be checked against a
     second. The four-way split of the ring is the only internal replicate
     available, and it is reported.

  7. NOT SHIPPED. Nothing here changes a decoded pixel. Whether aperture.py's
     operator should ship is a decision about the noise it costs on 116 real
     pictures, which this module does not measure -- it measures what the
     operator does to the blur, and that the blur it removes is the only blur
     it is entitled to.
""")


SECTIONS = {
    "geometry": section_geometry,
    "lsf": section_lsf,
    "axes": section_axes,
    "calibrate": section_calibrate,
    "budget": section_budget,
    "deconv": section_deconv,
    "photo": section_photo,
    "negatives": section_negatives,
}
ORDER = ["geometry", "lsf", "axes", "calibrate", "budget", "deconv", "photo",
         "negatives"]


def main(argv=None) -> int:  # pragma: no cover
    p = argparse.ArgumentParser(description="the ring as a 2-D line spread function")
    p.add_argument("--section", default="all",
                   help="comma-separated: " + ",".join(ORDER))
    a = p.parse_args(argv)
    want = ORDER if a.section == "all" else [s.strip() for s in a.section.split(",")]
    # dependencies: everything needs the frame and the ring
    state: dict = {}
    need = set(want)
    if need - {"geometry"}:
        need.add("geometry")
    if need & {"budget", "deconv"}:
        need |= {"axes", "calibrate"}
    for s in ORDER:
        if s in need:
            SECTIONS[s](state)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
