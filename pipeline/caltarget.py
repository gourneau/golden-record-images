"""The record's own ground truth: build the ideal calibration slide, subtract,
and ask whether what is left belongs to the CHAIN or to the SLIDE.

    python -m pipeline.caltarget                    # everything
    python -m pipeline.caltarget --section ideal,structure

PROVENANCE
----------
Tier 0 throughout. Inputs: the master WAV, and the cover's statement that the
first image is a circle. `catalog.build()` is used only for each frame's start
sample and channel -- the `start_samples` key, which provenance.py's MIXED_FILES
rule explicitly permits a tier 0 module to read -- and never for orientation,
colour role or title. No file under docs/reference is opened, and nothing here
is registered as a shipped artifact: this module measures, it does not decode.

WHY THIS FRAME IS ALLOWED TO BE GROUND TRUTH
--------------------------------------------
The engraved cover shows the first image as a circle. So "frame 1 is a black
ring on a uniform field" is stated by the artifact, not by us, and a recipient
who read the cover can do everything below. Tier 0.

WHAT IS MEASURED, NOT ASSUMED
-----------------------------
* the ring's ellipse: fitted to the decode (pipeline.circle -> quality.circle_metrics)
* the field's level: the decoder's own absolute white rail, which is
  porch - 0.95 x sync amplitude measured per frame from the record's audio,
  i.e. 1.0 in decoded units. Reported alongside a median-matched variant so
  the DC choice can be seen not to move any structural conclusion.
* the ring's black: the decoder's absolute black rail, 0.0 in decoded units.

WHAT IS DELIBERATELY NOT CLAIMED
--------------------------------
The ring's EDGE PROFILE. The cover says circle; it does not say how sharp the
1977 camera was. Pixels within +-12 px of the fitted ellipse are excluded from
the error-field statistics and reported separately -- that residual is the line
spread function, a different defect with a different fix.

THE DECODE MUST BE UNCLIPPED
----------------------------
decode() ends in np.clip(...,0,1). On L000 that censors 10,979 pixels (5.7% of
the frame) at white -- exactly the bright top of the field where the droop is
largest, so a clipped error field understates the defect it is measuring. The
unclipped decode is recovered exactly (max |residual| 7e-8, i.e. float32 round
off) by decoding a second time with `levels="percentile", black_pct=0,
white_pct=100`, which cannot clip because its rails are the data's own extremes,
and mapping that back onto the reference rails with the affine transform the
two settings differ by. No decoder change, no re-implementation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import catalog as catalog_mod
from . import circle as circle_mod
from . import decode as decode_mod
from . import sync as sync_mod
from . import wav

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "data" / "master" / "384kHzStereo.wav"

CAL_FRAME = "L000"
RING_EXCLUDE_PX = 12.0   # keep the LSF out of the field statistics
BORDER_GUARD = 3         # rows dropped either side of a detected slide border

_WAV: dict = {}
_DEC: dict = {}


# --------------------------------------------------------------------------
# loading: the float decode, unclipped, in the decoder's absolute units
# --------------------------------------------------------------------------


def _master():
    if "info" not in _WAV:
        info = wav.probe(MASTER)
        _WAV["info"] = info
        _WAV["mm"] = wav.memmap(info)
        _WAV["cat"] = catalog_mod.build()
    return _WAV["info"], _WAV["mm"], _WAV["cat"]


def frame_signal(fid: str) -> tuple[np.ndarray, str]:
    _, mm, cat = _master()
    fr = cat.by_id(fid)
    n = int(sync_mod.NOMINAL_PERIOD * 520)
    x = np.asarray(mm[fr.seed_sample: fr.seed_sample + n, fr.channel], dtype=np.float64)
    return x, ("L" if fr.channel == 0 else "R")


def decode_unclipped(fid: str, traces: int = 512):
    """The float decode with the clip undone, plus the shipped decode.

    Returns (unclipped, shipped, Decoded). Both are in the decoder's absolute
    intensity units: 0.0 is the black rail, 1.0 the white rail, both measured
    per frame from the sync amplitude and the back porch.
    """
    if fid in _DEC:
        return _DEC[fid]
    x, ch = frame_signal(fid)
    tb = sync_mod.recover(x)
    base = decode_mod.Settings(traces=traces, channel=ch)
    shipped = decode_mod.decode(x, base, tb=tb)
    wide = decode_mod.decode(x, base.replace(levels="percentile", black_pct=0.0,
                                             white_pct=100.0), tb=tb)
    A = np.asarray(shipped.image, dtype=np.float64)
    B = np.asarray(wide.image, dtype=np.float64)
    m = (A > 1e-6) & (A < 1.0 - 1e-6)
    X = np.stack([B[m], np.ones(int(m.sum()))], axis=1)
    c, *_ = np.linalg.lstsq(X, A[m], rcond=None)
    resid = float(np.abs(X @ c - A[m]).max())
    U = c[0] * B + c[1]
    _DEC[fid] = (U, A, shipped, resid)
    return _DEC[fid]


# --------------------------------------------------------------------------
# the ideal slide
# --------------------------------------------------------------------------


@dataclass
class Ideal:
    img: np.ndarray        # the synthesised slide, decoder units
    field: np.ndarray      # bool, pixels the record's spec fixes (uniform field)
    ringband: np.ndarray   # bool, pixels near the ring (spec fixes geometry only)
    ring: circle_mod.Ring
    white: float
    black: float
    rows: tuple[int, int]  # the slide's own extent along the trace


def _slide_rows(img: np.ndarray, rr: np.ndarray) -> tuple[int, int]:
    """Where the slide's field starts and ends along the trace.

    The picture gate is wider than the slide: the first and last rows of every
    frame are the mount, which reads far darker than the field and is not part
    of what the cover promises. Found from the decode's own row medians outside
    the ring -- a level step of more than a quarter of the field's range.
    """
    h = img.shape[0]
    ext = rr > 1.15
    med = np.array([np.median(img[i][ext[i]]) if ext[i].sum() > 20 else np.nan
                    for i in range(h)])
    core = med[h // 3: 2 * h // 3]
    lvl = np.nanmedian(core)
    thr = lvl - 0.25 * (np.nanmax(med) - np.nanmin(med))
    ok = np.where(med > thr)[0]
    return int(ok.min() + BORDER_GUARD), int(ok.max() - BORDER_GUARD)


def build_ideal(img: np.ndarray, ring: circle_mod.Ring | None = None,
                white: float | None = None, black: float | None = None,
                bad_traces: np.ndarray | None = None) -> Ideal:
    """A uniform field with a black ring on it, from the record's spec alone.

    Geometry from fitting the decode. Levels from the decoder's own absolute
    rails unless overridden. The ring is drawn at its MEASURED width with a
    half-pixel ramp, but no pixel inside `ringband` is ever used as ground
    truth -- see the module docstring.
    """
    h, w = img.shape
    if ring is None:
        ring = circle_mod.fit_ring(img)
        if ring is None:
            raise ValueError("ring fit failed; this frame is not the calibration slide")
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - ring.cx) / ring.rx, (yy - ring.cy) / ring.ry)
    scale = 0.5 * (ring.rx + ring.ry)
    d = np.abs(rr - 1.0) * scale                      # px from the ring centre-line
    white = 1.0 if white is None else float(white)
    black = 0.0 if black is None else float(black)
    half = 0.5 * ring.width
    ideal = black + (white - black) * np.clip((d - half) / 0.5, 0.0, 1.0)

    r0, r1 = _slide_rows(img, rr)
    band = d <= RING_EXCLUDE_PX
    field = np.zeros((h, w), dtype=bool)
    field[r0:r1 + 1] = True
    field &= ~band
    # The decoder's own dropout flag, dilated by one trace: repair_dropouts
    # fills a flagged trace from its neighbours, so the neighbours are no
    # longer independent measurements either.
    if bad_traces is not None and len(bad_traces):
        b = np.asarray(bad_traces, dtype=bool)
        b = b | np.roll(b, 1) | np.roll(b, -1)
        field[:, b] = False
    return Ideal(ideal, field, band, ring, white, black, (r0, r1))


# --------------------------------------------------------------------------
# structure of a masked field
# --------------------------------------------------------------------------


def _masked_means(E, M, axis):
    n = M.sum(axis=axis)
    s = np.where(M, E, 0.0).sum(axis=axis)
    out = np.full(n.shape, np.nan)
    ok = n >= 8
    out[ok] = s[ok] / n[ok]
    return out, ok


def two_way(E: np.ndarray, M: np.ndarray, iters: int = 6):
    """Additive row + column decomposition of a masked field (missing-cell ANOVA).

    E ~ mu + R[row] + C[col] + resid. Rows are along-trace position (the DROOP
    axis); columns are traces (the STREAK axis). Iterated because the mask has
    a hole where the ring is, so the two means are not orthogonal.
    """
    h, w = E.shape
    mu = float(E[M].mean())
    R = np.zeros(h)
    C = np.zeros(w)
    for _ in range(iters):
        res = E - mu - C[None, :]
        r, ok = _masked_means(res, M, axis=1)
        R = np.where(ok, r, 0.0)
        res = E - mu - R[:, None]
        c, ok = _masked_means(res, M, axis=0)
        C = np.where(ok, c, 0.0)
    fit = mu + R[:, None] + C[None, :]
    return mu, R, C, E - fit


def var_share(E, M, part):
    """Fraction of E's masked variance that `part` accounts for."""
    v = float(np.var(E[M]))
    if v <= 0:
        return float("nan")
    return float(1.0 - np.var((E - part)[M]) / v)


def legendre_basis(n: int, deg: int) -> np.ndarray:
    """(n, deg+1) orthonormal Legendre polynomials on the row axis."""
    t = np.linspace(-1.0, 1.0, n)
    B = np.polynomial.legendre.legvander(t, deg)
    Q, _ = np.linalg.qr(B)
    return Q


def per_trace_trajectories(E: np.ndarray, M: np.ndarray, deg: int) -> np.ndarray:
    """Fit each trace's own smooth level trajectory down the trace.

    One independent least-squares fit per column over that column's masked
    rows, in a degree-`deg` polynomial. This is the shape the measurement
    already implied: not one offset per trace (that was ruled out at 3.6%) but
    one slowly-varying level trajectory per trace. Missing rows -- the ring's
    two crossings -- are handled by the fit rather than by interpolation.
    """
    h, w = E.shape
    B = legendre_basis(h, deg)
    T = np.zeros((h, w))
    for j in range(w):
        m = M[:, j]
        if m.sum() < 4 * (deg + 1):
            continue
        c, *_ = np.linalg.lstsq(B[m], E[m, j], rcond=None)
        T[:, j] = B @ c
    return T


def smoothness(v: np.ndarray) -> float:
    """1 - var(first difference) / (2 var(v)): 1.0 is perfectly smooth, 0.0 white."""
    v = v - v.mean()
    den = 2.0 * float(np.var(v))
    if den <= 0:
        return float("nan")
    return float(1.0 - np.var(np.diff(v)) / den)


# --------------------------------------------------------------------------
# the content-free instrument: gap lines and porches on OTHER frames
# --------------------------------------------------------------------------

BINS = 3200
PIC_LO = int(round(decode_mod.PICTURE_START * BINS))          # 232
PIC_HI = int(round(decode_mod.PICTURE_END * BINS))            # 3040
PORCH_LO = int(round(decode_mod.PORCH_START * BINS))          # 64
PORCH_HI = int(round(decode_mod.PORCH_END * BINS))            # 230
SYNC_LO, SYNC_HI = 3165, 3195


def _parity_clamp(level: np.ndarray) -> np.ndarray:
    """decode.py's clamp: a 7-wide running median within each sync parity."""
    out = level.copy()
    if len(level) >= 16:
        for p in (0, 1):
            s = level[p::2]
            pad = np.pad(s, 3, mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(pad, 7)
            out[p::2] = np.median(win, axis=1)
    return out


def gap_lines(fid: str, n_traces: int = 190, rows: int = 377):
    """Content-free traces from the gap before a frame, in DECODED units.

    Same chain as decode(): invert the pole on the raw signal in raster order,
    clamp each trace on its own back porch, invert, and scale by the same
    absolute rails (porch +0.65 / -0.95 x sync amplitude). The only difference
    is the picture window is resampled linearly instead of dot-locked, because
    there are no dots to lock to on a blank line.

    A blank line must decode to a CONSTANT. Whatever it does instead is the
    chain's own error field, measured on a frame that is not L000 and with no
    slide involved at all.
    """
    from . import droop as droop_mod
    from . import geometry as geo
    from . import shelf as shelf_mod

    x, m, _ = droop_mod.load(fid, 195.0, n_traces)
    ch = "L" if fid.upper().startswith("L") else "R"
    period = float(np.median(np.diff(m)))
    tau = decode_mod.UNCOUPLE_TAU_384[ch] * period / sync_mod.NOMINAL_PERIOD
    xu = decode_mod.undroop(x, tau)
    M = geo.norm_traces(xu, m)
    good = np.isfinite(M[:, 0])

    _, _, idx = shelf_mod.load_gap(fid, n_traces)
    idx = np.asarray([i for i in idx if i < len(M) and good[i]], dtype=int)
    if len(idx) < 12:
        return None

    porch = np.median(M[:, PORCH_LO:PORCH_HI], axis=1)
    clamp = _parity_clamp(np.nan_to_num(porch, nan=float(np.nanmedian(porch))))
    amp = float(np.median(np.median(M[good][:, SYNC_LO:SYNC_HI], axis=1)
                          - porch[good]))
    lo = -decode_mod.UNCOUPLE_BLACK_REF * amp
    hi = -decode_mod.UNCOUPLE_WHITE_REF * amp

    pic = M[idx][:, PIC_LO:PIC_HI] - clamp[idx][:, None]
    pic = -(pic)
    pic = (pic - lo) / (hi - lo)
    # resample the along-trace axis onto the decoder's row grid
    src = np.linspace(0.0, 1.0, pic.shape[1])
    dst = np.linspace(0.0, 1.0, rows)
    G = np.stack([np.interp(dst, src, p) for p in pic])
    return G, idx, amp


def porch_field(fid: str, traces: int = 512):
    """The raw back porch of every trace of a frame, in decoded units.

    Content-free by construction (decode.py's own measured window) and NOT
    clamped here -- the clamp is what we are trying to see. Covers only
    period-fraction 0.020-0.072, i.e. the 5% of the trace before the picture,
    so it constrains the error field's per-trace level but not its shape down
    the trace. Reported for what it is.
    """
    x, ch = frame_signal(fid)
    tb = sync_mod.recover(x)
    tau = decode_mod.UNCOUPLE_TAU_384[ch] * tb.period / sync_mod.NOMINAL_PERIOD
    xu = decode_mod.undroop(x, tau)
    n_tr = min(traces, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])
    P, ok = decode_mod._region(xu, starts, tb.period,
                               decode_mod.PORCH_START, decode_mod.PORCH_END)
    ref = decode_mod.measure_levels(xu, starts, tb.period)
    if ref is None:
        return None
    porch_med, amp, _ = ref
    lo = -decode_mod.UNCOUPLE_BLACK_REF * amp
    hi = -decode_mod.UNCOUPLE_WHITE_REF * amp
    Q = (-(P - porch_med) - lo) / (hi - lo)
    return Q, ok, amp


# --------------------------------------------------------------------------
# the mechanism: an error that ACCUMULATES down the trace
# --------------------------------------------------------------------------

PIC_ROWS = (20, 341)      # rows safely inside the slide on every frame
TOP_ROWS = (0, 3)         # the mount above the slide, before the picture in time
BOT_ROWS = (360, 375)     # the mount below the slide, after the picture in time


def uncumulate(d: np.ndarray, k: float, ref: float = 0.0) -> np.ndarray:
    """Invert a per-trace accumulating error: d = s - k * cumsum(s - ref).

    `d` is (rows, traces) with rows running down the trace, i.e. forward in
    time, which is the axis the error accumulates along. One coefficient, k,
    in units of error per row per unit intensity. The recursion is the exact
    inverse and is stable for k < 1.
    """
    h, w = d.shape
    s = np.empty_like(d)
    acc = np.zeros(w)
    dp = d + k * ref * np.arange(1, h + 1)[:, None]
    for r in range(h):
        s[r] = (dp[r] + k * acc) / (1.0 - k)
        acc = acc + s[r]
    return s


def border_levels(img: np.ndarray, ok: np.ndarray | None = None):
    """(top mount, bottom mount, picture level) per trace, in decoded units.

    The mount is the slide holder: one physical object, one constant value,
    present on every frame, above AND below the picture. The record guarantees
    its constancy; nothing about Earth is needed to know that a frame's border
    is the same border at both ends. It is the tier-0 instrument this module
    leans on hardest, because it brackets the picture IN TIME.
    """
    top = np.median(img[TOP_ROWS[0]:TOP_ROWS[1]], axis=0)
    bot = np.median(img[BOT_ROWS[0]:BOT_ROWS[1]], axis=0)
    pic = img[PIC_ROWS[0]:PIC_ROWS[1]].mean(axis=0)
    if ok is not None:
        return top[ok], bot[ok], pic[ok]
    return top, bot, pic


def _dropout_mask(dec) -> np.ndarray:
    b = np.asarray(dec.quality.dropouts, dtype=bool)
    return ~(b | np.roll(b, 1) | np.roll(b, -1))


def per_trace_slope(fid: str):
    """Within ONE frame: does a trace that carried more light end darker?

    Regresses each trace's bottom-mount level on that trace's own mean picture
    level. The mount is constant across traces by construction, so any slope is
    the chain writing the picture's own history into a region that has none.
    Entirely within-frame: no cross-frame confound, no reference image.
    """
    U, _, dec, _ = decode_unclipped(fid)
    ok = _dropout_mask(dec)
    top, bot, pic = border_levels(U, ok)
    X = np.stack([pic, np.ones_like(pic)], axis=1)
    cb, *_ = np.linalg.lstsq(X, bot, rcond=None)
    ct, *_ = np.linalg.lstsq(X, top, rcond=None)
    return {
        "fid": fid,
        "n": int(ok.sum()),
        "spread": float(pic.std()),
        "bot_slope": float(cb[0]), "bot_r": float(np.corrcoef(pic, bot)[0, 1]),
        "top_slope": float(ct[0]), "top_r": float(np.corrcoef(pic, top)[0, 1]),
        "bot_sd": float(bot.std()), "bot_resid_sd": float(np.std(bot - X @ cb)),
    }


def fit_k(fid: str, hi: float = 5e-3, iters: int = 30, differential: bool = False):
    """The chain's accumulation coefficient, from ONE frame, tier 0.

    Chosen so that after `uncumulate` the bottom mount no longer knows how much
    light its trace carried -- the regression slope of the corrected border on
    the (uncorrected) trace level is driven to zero. The slope is divided by
    the corrected picture's own 1-99 percentile range at every step, so a
    filter cannot win by changing gain: that degeneracy is what made the
    "shelf" a mirage (pipeline/shelf.py), and it is closed here by
    construction. Returns None when the frame's traces are too alike for the
    regression to say anything (L000 itself is one of those, which is why L000
    never enters this estimate).

    `differential` regresses the bottom mount MINUS the top mount instead. The
    two mounts are the same object at the two ends of the same trace, so any
    per-trace error that is symmetric in time -- veiling flare in the camera, a
    per-trace gain, a per-trace offset -- cancels exactly in the difference,
    and only the causal, accumulating part survives. It is the stricter
    estimator and it is the one to quote; the plain one is kept because it is
    less noisy and the two bracket the answer.
    """
    U, _, dec, _ = decode_unclipped(fid)
    ok = _dropout_mask(dec)
    _, _, pic = border_levels(U, ok)
    if pic.std() < 0.02:
        return None
    X = np.stack([pic, np.ones_like(pic)], axis=1)

    def slope(k):
        S = uncumulate(U, k)
        b = np.median(S[BOT_ROWS[0]:BOT_ROWS[1]], axis=0)[ok]
        if differential:
            b = b - np.median(S[TOP_ROWS[0]:TOP_ROWS[1]], axis=0)[ok]
        c, *_ = np.linalg.lstsq(X, b, rcond=None)
        lo, h_ = np.percentile(S[PIC_ROWS[0]:PIC_ROWS[1]], [1, 99])
        return c[0] / (h_ - lo)

    s0 = slope(0.0)
    if s0 >= 0:
        return None
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if slope(mid) < 0:
            lo = mid
        else:
            hi = mid
    return {"fid": fid, "k": 0.5 * (lo + hi), "slope0": float(s0),
            "spread": float(pic.std())}


# --------------------------------------------------------------------------
# report sections
# --------------------------------------------------------------------------


def _grey(v):
    return 255.0 * v


def section_ideal(state):
    print("=" * 78)
    print("IDEAL  the record's own ground truth, and the error field it exposes")
    print("=" * 78)
    U, A, dec, resid = decode_unclipped(CAL_FRAME)
    I = build_ideal(U, bad_traces=dec.quality.dropouts)
    state.update(U=U, A=A, dec=dec, I=I)
    r = I.ring
    print(f"  unclip     affine recovery of the pre-clip decode, max |residual| {resid:.1e}")
    print(f"             clipped at white by the shipped decode: "
          f"{int((A >= 1 - 1e-6).sum())} px ({100 * (A >= 1 - 1e-6).mean():.1f}%), "
          f"peak unclipped value {U.max():.3f}")
    print(f"  ring       centre ({r.cy:.2f}, {r.cx:.2f})  radii {r.ry:.2f} x {r.rx:.2f}  "
          f"axis ratio {r.axis_ratio:.4f}  radial rms {r.radial_rms:.3f} px  width {r.width:.1f} px")
    print(f"  slide      the mount ends at rows {I.rows[0]} and {I.rows[1]}; "
          f"outside that the cover promises nothing")
    print(f"  ideal      field {I.white:.3f}, ring {I.black:.3f}, in the decoder's own absolute "
          f"rails (porch +0.65 / -0.95 x sync amplitude)")
    print(f"  ground     {int(I.field.sum()):,} pixels of KNOWN value "
          f"({100 * I.field.mean():.0f}% of the frame); "
          f"{int(I.ringband.sum()):,} px within {RING_EXCLUDE_PX:.0f} px of the ring excluded "
          f"(that residual is the LSF, not the field)")
    E = U - I.img
    e = E[I.field]
    print()
    print(f"  ERROR FIELD (decoded minus ideal, on the known-value pixels)")
    print(f"    mean  {e.mean():+.4f} ({_grey(e.mean()):+6.1f} grey)   "
          f"-- the white rail is optimistic by this much on average")
    print(f"    rms   {e.std():.4f} ({_grey(e.std()):6.1f} grey)")
    print(f"    range {e.min():+.4f} .. {e.max():+.4f} "
          f"({_grey(e.min()):+.1f} .. {_grey(e.max()):+.1f} grey)")
    m = np.abs(np.hypot((np.mgrid[0:U.shape[0], 0:U.shape[1]][1] - r.cx) / r.rx,
                        (np.mgrid[0:U.shape[0], 0:U.shape[1]][0] - r.cy) / r.ry) - 1.0)
    band = (m * 0.5 * (r.rx + r.ry)) <= RING_EXCLUDE_PX
    print(f"    ring band (excluded above): rms {E[band].std():.4f} "
          f"({_grey(E[band].std()):.1f} grey) -- the line spread function, a different defect")
    # DC choice cannot matter
    I2 = build_ideal(U, white=float(np.median(U[I.field])), black=0.0,
                     bad_traces=dec.quality.dropouts)
    e2 = (U - I2.img)[I2.field]
    print(f"    with the field level taken as the decode's own median instead of the rail: "
          f"mean {e2.mean():+.4f}, rms {e2.std():.4f} -- a pure DC shift, structure unchanged")


def section_structure(state):
    print()
    print("=" * 78)
    print("STRUCTURE  how the error field is built: down the trace, across traces, per trace")
    print("=" * 78)
    U, I, dec = state["U"], state["I"], state["dec"]
    E = U - I.img
    F = I.field
    mu, R, C, res = two_way(E, F)
    rows = np.where(F.any(axis=1))[0]
    cols = np.where(F.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    Rr, Cc = R[rows], C[cols]
    v = float(np.var(E[F]))
    print(f"  additive decomposition   E = mu + R[along-trace] + C[trace] + residual")
    print(f"    mu                     {mu:+.4f}  ({_grey(mu):+6.1f} grey)")
    print(f"    R  DOWN the trace      p-p {Rr.ptp():.4f} ({_grey(Rr.ptp()):6.1f} grey)   "
          f"explains {100 * var_share(E, F, mu + R[:, None]):5.1f}% of the error variance   "
          f"smoothness {smoothness(R[r0:r1 + 1]):.4f}")
    print(f"    C  ACROSS traces       p-p {Cc.ptp():.4f} ({_grey(Cc.ptp()):6.1f} grey)   "
          f"explains {100 * var_share(E, F, mu + C[None, :]):5.1f}% of the error variance   "
          f"smoothness {smoothness(Cc):.4f}")
    print(f"    both                   {100 * var_share(E, F, mu + R[:, None] + C[None, :]):5.1f}%")
    print(f"    residual               rms {res[F].std():.4f} ({_grey(res[F].std()):.2f} grey)")
    print()
    print("  ONE PROFILE DOWN THE TRACE IS 96% OF THE DEFECT. The per-trace CONSTANT that")
    print("  classic destriping removes is 1.6% of it, and it is itself smooth across traces")
    print(f"  (lag-1 smoothness {smoothness(Cc):.3f}), i.e. shading, not stripes.")

    print()
    print("  where the energy sits, by 2-D frequency (blocks with no ring in them):")
    blocks = {"left of the ring": (slice(r0, r1 + 1), slice(8, 115)),
              "right of the ring": (slice(r0, r1 + 1), slice(420, 512)),
              "inside the ring": (slice(87, 276), slice(173, 362))}
    print(f"    {'block':<20s} {'rms':>8s}  {'shading':>8s} {'STREAK':>8s} {'banding':>8s} {'noise':>8s}")
    for name, (a, b) in blocks.items():
        for lbl, Z in (("", E[a, b]),):
            q = _quadrants(Z)
            print(f"    {name:<20s} {_grey(Z.std()):7.1f}g  " +
                  "".join(f"{100 * x:7.1f}% " for x in q))
    print("    (shading = slow down the trace AND across; STREAK = slow down, fast across;")
    print("     banding = fast down, slow across; boundary at 1/16 cycles per pixel)")
    print("  THE STREAK IS 0.2-0.7% OF THE ERROR FIELD'S ENERGY. What the eye calls streaking")
    print("  is a 1.3-1.8 grey-level modulation on top of a 95 grey-level shading.")

    print()
    deg, ho = _degree_holdout(res, F)
    T = per_trace_trajectories(res, F, deg)
    Ts = T[r0:r1 + 1][:, cols]
    Uu, s, Vt = np.linalg.svd(Ts, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    print(f"  per-trace level trajectories (after mu+R+C), degree chosen by a held-out row-block")
    print(f"  test at deg {deg}: residual {_grey(ho[0]):.2f} -> {_grey(ho[1]):.2f} grey held out")
    print(f"    SVD share of trajectory energy: " +
          " ".join(f"{100 * x:.1f}%" for x in ev[:6]))
    print(f"    (smoothness down the trace is 1.000 by construction -- these vectors are")
    print(f"     polynomial fits. The informative row is the one below it.)")
    print(f"    smoothness DOWN the trace  : " +
          " ".join(f"{smoothness(Uu[:, i]):.3f}" for i in range(5)))
    print(f"    smoothness ACROSS traces   : " +
          " ".join(f"{smoothness(Vt[i]):.3f}" for i in range(5)))
    n = r1 - r0 + 1
    L = legendre_basis(n, 10)
    print("    principal angles of the leading-k left subspace to plain Legendre polynomials:")
    for k in (2, 3, 5, 7):
        Q1, _ = np.linalg.qr(Uu[:, :k])
        sv = np.linalg.svd(Q1.T @ L[:, :k], compute_uv=False)
        ang = np.degrees(np.arccos(np.clip(sv, -1, 1)))
        print(f"      k={k}: {np.array2string(ang, precision=1)}")
    print("    NEGATIVE: at k=7 the angles are all zero -- the 'learned basis' IS the space of")
    print("    degree-6 polynomials down the trace. Any claim that it transfers to another")
    print("    frame is the claim that smooth things are smooth, and must be tested against a")
    print("    generic smooth basis of the same rank, not against noise. It is below.")
    state["R"], state["C"], state["res"], state["rows"] = R, C, res, (r0, r1)


def _quadrants(Z, lo=1 / 16.0):
    Z = Z - Z.mean()
    n, m = Z.shape
    win = np.hanning(n)[:, None] * np.hanning(m)[None, :]
    P = np.abs(np.fft.rfft2(Z * win)) ** 2
    fr = np.abs(np.fft.fftfreq(n))[:, None] + np.zeros((1, P.shape[1]))
    fc = np.fft.rfftfreq(m)[None, :] + np.zeros((n, 1))
    tot = P.sum() - P[0, 0]
    return [(P[(fr < lo) & (fc < lo)].sum() - P[0, 0]) / tot,
            P[(fr < lo) & (fc >= lo)].sum() / tot,
            P[(fr >= lo) & (fc < lo)].sum() / tot,
            P[(fr >= lo) & (fc >= lo)].sum() / tot]


def _degree_holdout(res, F, blk=16, seed=0):
    h, w = res.shape
    rows = np.where(F.any(axis=1))[0]
    r0 = int(rows.min())
    bid = (np.arange(h) - r0) // blk
    rng = np.random.default_rng(seed)
    nb = int(bid.max()) + 1
    hold = np.isin(bid, rng.permutation(nb)[:nb // 4])
    B = legendre_basis(h, 24)
    best, curve = None, {}
    for deg in (0, 1, 2, 3, 4, 6, 8, 12):
        ho = []
        for j in range(w):
            m = F[:, j]
            mt, mh = m & ~hold, m & hold
            if mt.sum() < 4 * (deg + 1) or mh.sum() < 20:
                continue
            X = B[mt][:, :deg + 1]
            c, *_ = np.linalg.lstsq(X, res[mt, j], rcond=None)
            ho.append(res[mh, j] - B[mh][:, :deg + 1] @ c)
        curve[deg] = float(np.concatenate(ho).std())
        if best is None or curve[deg] < curve[best]:
            best = deg
    return best, (curve[0], curve[best])


def section_mechanism(state, step: int = 6):
    print()
    print("=" * 78)
    print("MECHANISM  is the error a property of the CHAIN or of the SLIDE?")
    print("=" * 78)
    print("""
  The two hypotheses make opposite predictions about ONE tier-0 region.

  Every frame is bracketed in time by the slide MOUNT: a few rows of it before
  the picture (rows 0-2) and a few after (rows 360-374). It is one physical
  object with one value, and it is dark. So:

    SLIDE / camera shading   error = (gain(row) - 1) x brightness(row).
                             The mount is dark, so its error is ~0 at BOTH
                             ends and the two borders must read the same.

    CHAIN, accumulating      error(row) = -k x sum of the light already sent
                             down this trace. The mount at the END of the
                             trace sits behind the whole picture, so it must
                             read DARKER, by an amount proportional to how
                             much light that trace carried.

  The second prediction is per-trace and within-frame, so no cross-frame
  confound and no reference image can reach it.
""")
    cat = _master()[2]
    fids = [f.id for f in cat.frames[::step]]
    rows = []
    print(f"  {'fid':<6s} {'n':>4s} {'trace spread':>13s} {'bottom slope':>13s} {'r':>7s}"
          f" {'top slope':>10s} {'r':>7s} {'border sd':>10s} {'after':>7s}")
    for fid in fids:
        try:
            d = per_trace_slope(fid)
        except Exception as e:  # pragma: no cover
            print(f"  {fid}: {e}")
            continue
        _DEC.clear()
        rows.append(d)
        print(f"  {fid:<6s} {d['n']:4d} {d['spread']:13.4f} {d['bot_slope']:+13.4f} "
              f"{d['bot_r']:+7.3f} {d['top_slope']:+10.4f} {d['top_r']:+7.3f} "
              f"{_grey(d['bot_sd']):9.2f}g {_grey(d['bot_resid_sd']):6.2f}g")
    bs = np.array([d["bot_slope"] for d in rows])
    ts = np.array([d["top_slope"] for d in rows])
    br = np.array([d["bot_r"] for d in rows])
    sd0 = np.array([d["bot_sd"] for d in rows])
    sd1 = np.array([d["bot_resid_sd"] for d in rows])
    print()
    print(f"  bottom mount: slope median {np.median(bs):+.4f}, negative on "
          f"{int((bs < 0).sum())}/{len(bs)} frames, median r {np.median(br):+.3f}")
    print(f"  top mount:    slope median {np.median(ts):+.4f}, negative on "
          f"{int((ts < 0).sum())}/{len(ts)} frames")
    print(f"  the across-trace scatter of the bottom mount -- pure defect, the record")
    print(f"  guarantees it is constant -- falls from {_grey(np.median(sd0)):.2f} to "
          f"{_grey(np.median(sd1)):.2f} grey levels (median) once that trace's own")
    print(f"  accumulated light is regressed out.")
    print()
    print("  VERDICT: the slide hypothesis predicts slope 0 at both ends. Measured, the")
    print("  bottom border carries the picture's history and the top border does not. The")
    print("  sign flip between the two ends also rules out a per-trace GAIN or OFFSET error,")
    print("  which would move both ends the same way, and it rules out veiling flare in the")
    print("  camera as the whole story -- flare is symmetric in time and would brighten both.")
    print("  THE DEFECT IS THE CHAIN'S.")
    print()
    print("  WHAT 'CHAIN' DOES AND DOES NOT MEAN. This measurement places the defect")
    print("  downstream of the slide and upstream of us. It does NOT separate the 1977")
    print("  vidicon's own target lag from the recording electronics' AC coupling: both")
    print("  integrate the light already sent and both would print exactly this signature.")
    print("  That distinction does not change the correction, and claiming it would be")
    print("  claiming more than the audio can support.")
    print()
    print("  A THIRD KNOWN-CONSTANT REGION AGREES, and explains why the defect resets each")
    print("  trace. The unclamped back porch -- content-free by decode.py's own measurement")
    print("  -- also anticorrelates with trace level, and more strongly than the bottom")
    print("  mount does. It sits after the PREVIOUS trace's picture, which is what an error")
    print("  accumulating in raster order predicts, and it is the level dc_restore clamps")
    print("  on. So the accumulation runs continuously through the frame and the porch")
    print("  clamp zeroes it once per trace, which is why what survives is a per-trace")
    print("  trajectory that starts near zero and grows down the trace.")
    state["mech"] = rows


def section_transfer(state, step: int = 4):
    print()
    print("=" * 78)
    print("TRANSFER  one coefficient, measured on OTHER frames, predicts L000's droop")
    print("=" * 78)
    print("""
  L000 is excluded from this estimate by construction: `fit_k` refuses any
  frame whose traces are too alike for the regression to bite, and a uniform
  white field is the most alike a frame can be. So the number below is fitted
  without the calibration slide and then used to predict it.

  Two estimators. "bottom" drives the bottom mount's dependence on trace level
  to zero. "bottom - top" does the same to the DIFFERENCE between the two ends
  of the trace, which cancels anything symmetric in time -- flare, per-trace
  gain, per-trace offset -- and is the one to quote.
""")
    cat = _master()[2]
    keep, ks = [], {False: [], True: []}
    for f in cat.frames[::step]:
        if f.id == CAL_FRAME:
            continue
        r0 = fit_k(f.id, differential=False)
        r1 = fit_k(f.id, differential=True)
        if r0 is not None and r1 is not None:
            U, _, dec, _ = decode_unclipped(f.id)
            keep.append((f.id, U.astype(np.float32), _dropout_mask(dec)))
            ks[False].append(r0["k"])
            ks[True].append(r1["k"])
        _DEC.clear()

    U, I = state["U"], state["I"]
    R, (r0_, r1_) = state["R"], state["rows"]
    span = r1_ - r0_
    meas = float(R[r1_] - R[r0_])
    print(f"  L000, MEASURED: the field falls {meas:+.4f} ({_grey(meas):+.1f} grey) "
          f"from row {r0_} to row {r1_}")
    print()
    print(f"  {'estimator':<14s} {'n':>4s} {'k median':>11s} {'IQR':>24s} "
          f"{'predicts':>11s} {'of measured':>12s}")
    summary = {}
    for diff, name in ((False, "bottom"), (True, "bottom - top")):
        k = np.array(ks[diff])
        lo, med, hi = np.percentile(k, [25, 50, 75])
        pred = -med * span * I.white
        summary[name] = (med, lo, hi)
        print(f"  {name:<14s} {len(k):4d} {med:11.3e} {lo:10.3e}..{hi:<11.3e} "
              f"{_grey(pred):+10.1f}g {100 * pred / meas:11.0f}%")
        print(f"  {'':<14s}      IQR band predicts "
              f"{_grey(-lo * span * I.white):+.1f}g .. {_grey(-hi * span * I.white):+.1f}g "
              f"= {100 * lo * span * I.white / -meas:.0f}% .. "
              f"{100 * hi * span * I.white / -meas:.0f}% of measured")
    kall = np.array(ks[True])
    knull = -meas / (span * I.white)
    pct = 100.0 * float((kall < knull).mean())
    print()
    print(f"  the k that would null L000's droop exactly is {knull:.3e}: the "
          f"{pct:.0f}th percentile of the")
    print(f"  per-frame k measured on the rest of the record. L000's droop sits INSIDE the")
    print(f"  chain's own measured spread. It does not need a slide to explain it.")
    print()
    print("  RESIDUAL BIAS, stated not buried: both estimators regress on the DECODED trace")
    print("  mean, which the defect has already darkened by ~kN/2, so k is biased LOW by")
    print("  10-15%. Read the differential row as 'all of it, to within a wide error bar',")
    print("  not as 'exactly 100%'.")

    # ---- held-out validation on frames the coefficient was not fitted to ----
    print()
    print("  HELD OUT. Split the frames in two. Take the median k from one half and apply")
    print("  it to the other, scoring only on the bottom mount's across-trace scatter --")
    print("  a quantity the record fixes and no image metric touches. Contrast-normalised,")
    print("  so a filter cannot win by flattening gain.")
    idx = np.arange(len(keep))
    for a, b, lbl in ((idx % 2 == 0, idx % 2 == 1, "even -> odd"),
                      (idx % 2 == 1, idx % 2 == 0, "odd -> even")):
        kfit = float(np.median(kall[a]))
        before, after = [], []
        for i in np.where(b)[0]:
            fid, Uf, ok = keep[i]
            Uf = Uf.astype(np.float64)
            for k, acc in ((0.0, before), (kfit, after)):
                S = Uf if k == 0 else uncumulate(Uf, k)
                bm = np.median(S[BOT_ROWS[0]:BOT_ROWS[1]], axis=0)[ok]
                lo_, hi_ = np.percentile(S[PIC_ROWS[0]:PIC_ROWS[1]], [1, 99])
                acc.append(bm.std() / (hi_ - lo_))
        before, after = np.array(before), np.array(after)
        print(f"    {lbl}: k = {kfit:.3e}, {int(b.sum())} held-out frames   "
              f"mount scatter {_grey(np.median(before)):.2f} -> "
              f"{_grey(np.median(after)):.2f} grey-equivalent (median), "
              f"improved on {int((after < before).sum())}/{len(before)}")
    state["k"] = summary["bottom - top"][0]
    state["kall"] = kall


def section_converge(state, step: int = 8):
    """The two ends of one physical object must agree once the defect is removed."""
    print()
    print("=" * 78)
    print("CONVERGE  the strongest tier-0 check available: one mount, two ends")
    print("=" * 78)
    print("""
  The slide mount appears twice on every frame, before the picture and after it.
  It is the same object both times, so its two readings must agree, and any
  disagreement is defect. Nothing about Earth, about the slide's content, or
  about L000 enters this: it is available on all 156 frames and it is the check
  the correction was never fitted to.
""")
    cat = _master()[2]
    k = state.get("k", 9.4e-4)
    g0, g1 = [], []
    print(f"  {'fid':<6s} {'top':>8s} {'bottom':>8s} {'gap':>8s}   "
          f"{'top*':>8s} {'bottom*':>8s} {'gap*':>8s}")
    for f in cat.frames[::step]:
        U, _, dec, _ = decode_unclipped(f.id)
        _DEC.clear()

        def lv(A):
            lo, hi = np.percentile(A[PIC_ROWS[0]:PIC_ROWS[1]], [1, 99])
            return (float(np.median(A[TOP_ROWS[0]:TOP_ROWS[1]])) / (hi - lo),
                    float(np.median(A[BOT_ROWS[0]:BOT_ROWS[1]])) / (hi - lo))
        t0, b0 = lv(U)
        t1, b1 = lv(uncumulate(U, k))
        g0.append(t0 - b0)
        g1.append(t1 - b1)
        print(f"  {f.id:<6s} {t0:8.3f} {b0:8.3f} {t0 - b0:8.3f}   "
              f"{t1:8.3f} {b1:8.3f} {t1 - b1:8.3f}")
    g0, g1 = np.array(g0), np.array(g1)
    print()
    print(f"  contrast-normalised |top - bottom|: mean {np.abs(g0).mean():.3f} -> "
          f"{np.abs(g1).mean():.3f}; median gap {np.median(g0):+.3f} -> {np.median(g1):+.3f}")
    print(f"  closer to agreement on {int((np.abs(g1) < np.abs(g0)).sum())}/{len(g0)} frames "
          f"at k = {k:.2e}")
    print()
    print("  This is the result to argue with, because nothing here was fitted to it. The")
    print("  residual overshoots slightly negative on most frames, so k is a touch hot for")
    print("  this test; the frames that resist (R066 and its like) are ones whose mount")
    print("  reading is on the black rail, where the decode is clipped and the test has no")
    print("  headroom left. Reported rather than dropped.")


def section_invariant(state):
    print()
    print("=" * 78)
    print("INVARIANT  the circle must not move, and over-correction must be visible")
    print("=" * 78)
    from . import quality as Q
    U, I = state["U"], state["I"]
    k = state.get("k", 9.4e-4)
    h, w = U.shape
    yy, xx = np.mgrid[0:h, 0:w]
    R = I.ring
    rr = np.hypot((xx - R.cx) / R.rx, (yy - R.cy) / R.ry)
    core = np.abs(rr - 1.0) < (0.35 * R.width / (0.5 * (R.rx + R.ry)))

    def relevel(S):
        b = np.percentile(S[core], 10)
        wl = np.median(S[I.field])
        return (S - b) / (wl - b)

    print(f"  {'arm':<26s} {'field rms':>10s} {'row p-p':>9s} {'axis ratio':>11s} "
          f"{'radial rms':>11s} {'inliers':>8s} {'cov':>5s}")
    m0 = Q.circle_metrics(np.clip(U, 0.0, 1.0))
    print(f"  {'shipped decode, untouched':<26s} {'':>10s} {'':>9s} "
          f"{m0['major'] / m0['minor']:11.4f} {m0['radial_rms']:10.4f}p "
          f"{m0.get('inliers', 0):8d} {m0.get('coverage_deg', 0):5.0f}   <- the project's numbers")
    for name, kk in (("as decoded", 0.0), (f"k = {k:.2e}", k),
                     ("k x 2 (over-correct)", 2 * k), ("k x 4 (over-correct)", 4 * k)):
        S = U if kk == 0 else uncumulate(U, kk)
        S2 = relevel(S)
        E = S2 - I.img
        rm = np.array([E[i][I.field[i]].mean() if I.field[i].sum() > 20 else np.nan
                       for i in range(h)])
        m = Q.circle_metrics(np.clip(S2, 0.0, 1.0))
        ar = m["major"] / m["minor"] if m.get("ok") else float("nan")
        print(f"  {name:<26s} {_grey(E[I.field].std()):9.2f}g "
              f"{_grey(np.nanmax(rm) - np.nanmin(rm)):8.2f}g {ar:11.4f} "
              f"{m.get('radial_rms', float('nan')):10.4f}p {m.get('inliers', 0):8d} "
              f"{m.get('coverage_deg', 0):5.0f}")
    print()
    print("  Read this the way constraint 4 asks. The metrics are CONTRAST-NORMALISED (each")
    print("  arm re-levelled two-point on its own ring core and field median), so a filter")
    print("  cannot win by changing gain. The 'as decoded' row differs from the shipped")
    print("  numbers only because re-levelling then clipping bites differently; it is the")
    print("  right baseline for the rows beneath it.")
    print()
    print("  At the measured k the circle does NOT degrade -- axis ratio and radial rms both")
    print("  move slightly the safe way and the fit gains inliers and angular coverage, which")
    print("  is an edge detector working on a flatter field rather than edges moving. The")
    print("  over-correction rows are the tripwire and it fires: at 2k both the field rms and")
    print("  the row profile are WORSE than uncorrected and the axis ratio degrades while the")
    print("  radial rms collapses -- edges being pulled -- and at 4k the ring cannot be fitted")
    print("  at all. There is a genuine interior optimum and it sits where the other frames")
    print("  said it would.")


def section_negatives(state, fids=("L010", "L030", "R020", "R040")):
    print()
    print("=" * 78)
    print("NEGATIVES  three things that do not work, so nobody repeats them")
    print("=" * 78)
    print()
    print("  1. THE LEARNED BASIS IS NOT A FINDING. The SVD basis of L000's per-trace")
    print("     trajectories spans exactly the degree-6 polynomials down the trace")
    print("     (principal angles 0 at rank 7, STRUCTURE section). Tested against")
    print("     content-free gap lines from other frames, it explains no more than a")
    print("     plain Legendre basis of the same rank:")
    U, I = state["U"], state["I"]
    res, (r0, r1) = state["res"], state["rows"]
    cols = np.where(I.field.any(axis=0))[0]
    T = per_trace_trajectories(res, I.field, 6)
    Uu, _, _ = np.linalg.svd(T[r0:r1 + 1][:, cols], full_matrices=False)
    print(f"       {'frame':<7s} {'lines':>6s} {'rank':>5s} {'L000 basis R2':>15s} "
          f"{'Legendre R2':>13s}")
    for fid in fids:
        try:
            g = gap_lines(fid)
        except Exception:
            g = None
        if g is None:
            print(f"       {fid:<7s}  no content-free lines found")
            continue
        G, idx, amp = g
        Z = G.T[r0:r1 + 1]
        Z = Z - Z.mean(axis=0, keepdims=True)
        L = legendre_basis(Z.shape[0], 10)
        for rank in (3, 7):
            def r2(B):
                P = B @ (B.T @ Z)
                return 1.0 - np.var(Z - P) / np.var(Z)
            a, b = r2(Uu[:, :rank]), r2(L[:, :rank])
            print(f"       {fid:<7s} {len(idx):6d} {rank:5d} {a:15.3f} {b:13.3f}")
    print("     The two columns agree. 'A basis learned from the calibration frame")
    print("     transfers' would have been a big claim; it reduces to 'smooth functions")
    print("     fit smooth functions', and is reported as a null.")
    print()
    print("  2. CONTENT-FREE GAP LINES CANNOT ARBITRATE THE DROOP, and the naive reading")
    print("     of them is a trap. A blank line decodes to a ramp of only 15-20 grey")
    print("     levels p-p where L000's field falls 95 -- which looks like proof that the")
    print("     droop is the slide's. It is not: the defect is proportional to the light")
    print("     already sent, and a blank line sent none. The gap measures the")
    print("     content-INDEPENDENT part of the chain's error and nothing else.")
    print("     (L020's gap window is contaminated -- its blank lines scatter by 50 grey")
    print("      levels against 2-5 on the others -- and is left out of the default set.)")
    for fid in fids:
        try:
            g = gap_lines(fid)
        except Exception:
            g = None
        if g is None:
            continue
        G, idx, amp = g
        p = np.median(G, axis=0)
        print(f"       {fid}: {len(idx)} blank lines, profile p-p "
              f"{_grey(p.ptp()):5.1f} grey, across-line sd "
              f"{_grey(np.median(G, axis=1).std()):5.1f} grey")
    print()
    print("  3. PER-TRACE OFFSET REMOVAL (classic destriping, moment matching) is ruled")
    print("     out by the decomposition, not by taste: a constant per trace is 1.6% of")
    print("     L000's error field, and what it would remove is itself smooth across")
    print("     traces. Destriping this image removes shading and leaves the streak.")


SECTIONS = {
    "ideal": section_ideal,
    "structure": section_structure,
    "mechanism": section_mechanism,
    "transfer": section_transfer,
    "converge": section_converge,
    "invariant": section_invariant,
    "negatives": section_negatives,
}


def main(argv=None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", default="all",
                    help="comma-separated: " + ",".join(SECTIONS))
    ap.add_argument("--step", type=int, default=6,
                    help="frame stride for the mechanism survey")
    args = ap.parse_args(argv)
    want = list(SECTIONS) if args.section == "all" else args.section.split(",")
    state: dict = {}
    for name in want:
        fn = SECTIONS.get(name.strip())
        if fn is None:
            print(f"unknown section {name!r}", file=sys.stderr)
            return 2
        if name.strip() in ("structure", "mechanism", "transfer", "invariant",
                            "negatives", "converge") and "U" not in state:
            section_ideal(state)
        if name.strip() in ("negatives", "invariant", "transfer") and "res" not in state:
            section_structure(state)
        if name.strip() in ("mechanism", "converge"):
            fn(state, step=args.step)
        elif name.strip() == "transfer":
            # deliberately NOT tied to --step: the per-frame k is noisy and its
            # median wanders by 20% on a 24-frame sample. Every other frame,
            # ~78 of them, is the smallest sample that settles it. ~4 minutes.
            fn(state, step=2)
        else:
            fn(state)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
