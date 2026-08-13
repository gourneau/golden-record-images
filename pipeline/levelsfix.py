"""Make the intensity rails describe the CORRECTED picture, and measure it.

    python -m pipeline.levelsfix                     # everything
    python -m pipeline.levelsfix --section verify

THE PROBLEM THIS STARTED FROM
-----------------------------
`decode._uncumulate` inverts the error that accumulates down each trace in
proportion to the light already sent (pipeline/caltarget.py measured it on the
slide mount, which appears twice on every frame and must read the same at both
ends). Applied to a FINISHED decode it is large and clean -- L000 field rms
0.14 -> 0.08, mount gap 1.478 -> 0.390 on frames the coefficient never saw.
Applied INSIDE decode() it delivered about a third of that, and L000's clipped
fraction went 5.99% -> 12.23% in exactly the bright field where the gain should
have appeared. So it shipped off.

WHAT WAS ACTUALLY WRONG: THREE THINGS, AND THE RAILS WERE THE SMALLEST
----------------------------------------------------------------------
The brief was to fix the rails. The rails were real but they were the third
term. `section_origin` takes the post-hoc result -- the one that was measured
and verified -- as the target the in-path correction has to reproduce, and asks
what stands between them:

  ORIGIN   The accumulator sums LIGHT ALREADY SENT, and light is zero at BLACK.
           The post-hoc test runs on the decoded image, whose zero IS the black
           rail. In path it ran on `pic`, which is referenced to the PORCH --
           0.406 of the way up the range. Summing a porch-referenced picture
           adds -k x 0.406 x row to every trace: -0.14 in decoded units by the
           bottom of the trace. That is the same size as the droop being
           corrected and it points the same way, so most of the correction was
           being spent undoing an artefact of its own origin. Measured on L000:
           it accounts for 0.146 of the 0.246 total shortfall at the bottom row.

  GRID     k = 9.4e-4 is per row of the 377-row decode it was fitted on. In
           path it was applied to the ~230-row DOT matrix, so the same k
           accumulated over 230 steps instead of 377 -- 61% of the intended
           total (e^(kN): 1.24 instead of 1.43). The invariant is k x rows, an
           integral of light down the trace, so k must be rescaled whenever the
           row grid changes.

  RAILS    Only then does the picture actually brighten by the full amount, and
           only then is the white-rail overflow the whole of what is left.

With origin and grid fixed, the in-path decode reproduces the validated
post-hoc image to rms 3e-4 of full scale (0.08 grey levels) -- i.e. exactly,
which is the point: the in-path correction is now the same operator that was
measured, not a cousin of it.

CORRECT THE RAILS, OR RE-MEASURE THEM?
--------------------------------------
Both were defensible before measuring, because the rails are built from the
back porch and the sync amplitude, which are part of the same signal the error
acts on. If the sync plateau carried the trace's accumulated light, `amp` would
be a contaminated ruler and would have to be corrected per frame.

MEASURED (`section_reference`): it does not. Regressing each trace's own
plateau-minus-porch on that trace's own mean picture level gives a median slope
of +0.000 x amp over 12 frames (median r +0.015, negative on 6 of 12; +0.016 on
the 6 frames with enough trace-to-trace spread to say anything) where a plateau
carrying a full trace of accumulation would give -0.216 -- more than an order of
magnitude larger and the other sign. The anchor is clean. What is stale is the
FRACTIONS +0.65 / -0.95, which were fitted to a picture that still contained
the error.

So: RE-MEASURE, by exactly the procedure that produced +0.65/-0.95 for the
undrooped picture (decode.py module docstring) -- per-frame p0.5/p99.5 of the
corrected picture in units of amp, over 20 frames spanning the record and both
channels, with the rail placed where the current rail sits in the uncorrected
distribution (2 frames of 20 keep a ~1% tail). That keeps one rule for all
three rail pairs instead of inventing a new one for this correction. It also
leaves the correction's own coefficient alone: the black rail used as the
accumulation ORIGIN stays at UNCOUPLE_BLACK_REF, because that is the zero k was
fitted against; the new rails are a display map and nothing is fitted to them.

PROVENANCE
----------
Tier 0. Inputs: the master WAV, `catalog.build()` for each frame's start sample
and channel (the `start_samples` key, which provenance.py's MIXED_FILES rule
permits a tier 0 module to read), and the cover's statement that image 1 is a
circle. No file under docs/reference is opened and nothing here is registered
as a shipped artifact: this module measures, it does not decode. The mount
geometry and the unclipping trick are imported from pipeline/caltarget.py
rather than re-implemented, so the numbers below are comparable to the ones
that motivated the change.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import caltarget as ct
from . import catalog as catalog_mod
from . import circle as circle_mod
from . import decode as dm
from . import sync as sync_mod

K = dm.UNCUMULATE_K
CAL_FRAME = "L000"
PHOTO_FRAME = "L011"     # a photograph, for the damage check

# k was fitted on cat.frames[::4] with L000 excluded (caltarget.section_transfer).
# Everything below that needs frames the coefficient never saw takes indices
# 1 mod 4, which is disjoint from that set by construction.
HELD_OUT = slice(1, None, 12)
RAIL_FRAMES = slice(None, None, 8)   # 20 frames spanning the record, both channels


def _grey(v):
    return 255.0 * v


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def frame(fid: str):
    """(signal, channel, timebase, sync amplitude) for one frame."""
    x, ch = ct.frame_signal(fid)
    tb = sync_mod.recover(x)
    n_tr = min(dm.TRACES_PER_FRAME, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])
    tau = dm.UNCOUPLE_TAU_384[ch] * tb.period / dm._NOMINAL_PERIOD_384
    ref = dm.measure_levels(dm.undroop(x, tau), starts, tb.period)
    return x, ch, tb, (None if ref is None else float(ref[1]))


def decode_pair(x, cfg, tb):
    """The shipped decode, and the same decode with the clip undone.

    caltarget.decode_unclipped's trick, generalised to arbitrary settings: the
    reference-rail decode and a percentile 0-100 decode differ by one affine
    map, which is recovered from the pixels the first one did not clip. The two
    decodes must agree on everything upstream of the rails for this to be
    exact, which is why decode() now measures the anchors in both levels modes.
    """
    A = np.asarray(dm.decode(x, cfg, tb=tb).image, dtype=np.float64)
    d = dm.decode(x, cfg.replace(levels="percentile", black_pct=0.0, white_pct=100.0), tb=tb)
    B = np.asarray(d.image, dtype=np.float64)
    m = (A > 1e-6) & (A < 1.0 - 1e-6)
    X = np.stack([B[m], np.ones(int(m.sum()))], axis=1)
    c, *_ = np.linalg.lstsq(X, A[m], rcond=None)
    return A, c[0] * B + c[1], float(np.abs(X @ c - A[m]).max())


SPAN0 = dm.UNCOUPLE_BLACK_REF - dm.UNCOUPLE_WHITE_REF      # 1.60 x amp
SPANK = dm.UNCUMULATE_BLACK_REF - dm.UNCUMULATE_WHITE_REF  # 1.53 x amp


def to_ref(U: np.ndarray, black_ref: float, white_ref: float) -> np.ndarray:
    """Put any decode into ONE common intensity scale so it can be differenced.

    Changing the rails changes the units, so an image decoded with the
    uncumulate rails cannot be subtracted from one decoded with the uncouple
    rails without first being expressed in the same units. Everything is
    referred to the k=0 rails (0 = porch + 0.65 amp, 1 = porch - 0.95 amp),
    which is the scale the published 0.14 -> 0.08 field rms is quoted in.
    """
    v = black_ref - (black_ref - white_ref) * np.asarray(U, dtype=np.float64)
    return (dm.UNCOUPLE_BLACK_REF - v) / SPAN0


def settings(ch: str, k: float) -> dm.Settings:
    return dm.Settings(traces=dm.TRACES_PER_FRAME, channel=ch, uncumulate=k)


def clip_fracs(A: np.ndarray) -> tuple[float, float]:
    return float((A >= 1.0 - 1e-6).mean()), float((A <= 1e-6).mean())


# --------------------------------------------------------------------------
# 1. is the ruler itself bent?
# --------------------------------------------------------------------------


def section_reference(step: int = 13):
    print("=" * 78)
    print("REFERENCE  is the sync-amplitude anchor contaminated by the accumulation?")
    print("=" * 78)
    print("""
  The rails are porch +/- (fraction) x amp, and amp is the short sync burst's
  plateau minus the back porch. The plateau sits AFTER the picture in time, so
  if the error that accumulates down the picture also acts on the blanking that
  follows it, then a bright trace must show a different amp from a dark one --
  by -k x n_dots = -0.216 x amp per unit of trace level. That is the difference
  between a ruler that must be CORRECTED and one that must merely be RE-READ.
""")
    cat = catalog_mod.build()
    fids = [f.id for f in cat.frames[3::step]]
    print(f"  {'fid':<6s} {'traces':>7s} {'level spread':>13s} {'amp':>8s} "
          f"{'slope':>9s} {'r':>7s}")
    rows = []
    for fid in fids:
        x, ch, tb, _ = frame(fid)
        period = tb.period
        tau = dm.UNCOUPLE_TAU_384[ch] * period / dm._NOMINAL_PERIOD_384
        xu = dm.undroop(x, tau)
        n_tr = min(dm.TRACES_PER_FRAME, len(tb.smoothed))
        starts = np.array([tb.trace_start(i) for i in range(n_tr)])
        porch, ok_p = dm._region(xu, starts, period, dm.PORCH_START, dm.PORCH_END)
        test, ok_t = dm._region(xu, starts, period, *dm._PARITY_TEST)
        plat, ok_l = dm._region(xu, starts, period, *dm._SYNC_PLATEAU)
        pic, ok_x = dm._region(xu, starts, period, dm.PICTURE_START, dm.PICTURE_END)
        ok = ok_p & ok_t & ok_l & ok_x
        idx = np.where(ok)[0]
        lvl = {p: np.median(test[idx[idx % 2 == p]]) for p in (0, 1)}
        short = 0 if lvl[0] < lvl[1] else 1
        sel = idx[idx % 2 == short]          # only the short-burst parity carries the plateau
        pm = np.median(porch[sel], axis=1)
        pl = np.median(plat[sel], axis=1)
        pv = pic[sel].mean(axis=1)
        amp = pl - pm
        X = np.stack([pv, np.ones_like(pv)], axis=1)
        c, *_ = np.linalg.lstsq(X, amp, rcond=None)
        r = float(np.corrcoef(pv, amp)[0, 1])
        rows.append((c[0], r, float(pv.std())))
        print(f"  {fid:<6s} {len(sel):7d} {pv.std():13.4f} {np.median(amp):8.4f} "
              f"{c[0]:+9.4f} {r:+7.3f}")
    S = np.array(rows)
    print()
    print(f"  slope of amp on the trace's own level: median {np.median(S[:,0]):+.4f}, "
          f"median r {np.median(S[:,1]):+.3f}, negative on {int((S[:,0]<0).sum())}/{len(S)}")
    print(f"  the same statistic on the {int((S[:,2] > 0.015).sum())} frames with enough "
          f"trace-to-trace spread to say anything (>0.015): "
          f"median {np.median(S[S[:,2]>0.015,0]):+.4f}")
    print(f"  a plateau carrying a full trace of accumulation would give {-K*230:+.4f}")
    print()
    print("  VERDICT: no contamination -- the measured slope is ~15x smaller than the")
    print("  prediction and does not hold a sign. THE ANCHOR IS CLEAN; only the rail")
    print("  FRACTIONS, which were fitted to an uncorrected picture, are stale. Re-measure")
    print("  them (section RAILS); do not 'correct' amp.")
    return S


# --------------------------------------------------------------------------
# 2. reproduce the post-hoc result in path
# --------------------------------------------------------------------------


def section_origin(fid: str = CAL_FRAME):
    print()
    print("=" * 78)
    print("ORIGIN  what stood between the in-path correction and the verified one")
    print("=" * 78)
    print("""
  The post-hoc test is the thing that was measured and verified, so it is the
  target: an in-path correction that is the same operator must reproduce its
  image. Below, each candidate in-path decode is differenced against
  caltarget.uncumulate applied to the k=0 decode. Only one of them is the same
  operator.
""")
    x, ch, tb, amp = frame(fid)
    A0, U0, resid = decode_pair(x, settings(ch, 0.0), tb)
    post = ct.uncumulate(U0, K)
    dec0 = dm.decode(x, settings(ch, 0.0), tb=tb)
    n_dots = int(dec0.diagnostics["n_dots"])
    blk = dm.UNCOUPLE_BLACK_REF * amp          # the origin decode() now uses
    S = dm.SQUARE_ROWS / n_dots                # the grid scale decode() now uses
    _orig = dm._uncumulate
    state = {"b": 0.0, "s": 1.0}

    # decode() calls _uncumulate(pic - blk, k*S) and adds blk back. To exhibit a
    # variant that uses origin b and scale s instead, the hook undoes decode's
    # choice and applies its own -- so every row below is a real decode through
    # the real path, differing only in the two constants under test.
    def hook(z, kd):
        b, s = state["b"], state["s"]
        return _orig(z + (blk - b), kd * s / S) + (b - blk)

    try:
        dm._uncumulate = hook
        print(f"  {'variant':<26s} {'rms':>8s} {'mean':>8s} {'max|.|':>8s}   "
              f"{'in grey levels':>14s}")
        keep = {}
        for name, b, s in (("as it shipped", 0.0, 1.0),
                           ("+ origin at black", blk, 1.0),
                           ("+ grid 377/%d" % n_dots, 0.0, S),
                           ("+ origin + grid (now)", blk, S)):
            state["b"], state["s"] = b, s
            _, U, _ = decode_pair(x, settings(ch, K), tb)
            # into the k=0 rail scale: the uncumulate rails are different UNITS
            U = to_ref(U, dm.UNCUMULATE_BLACK_REF, dm.UNCUMULATE_WHITE_REF)
            keep[name] = U
            d = U - post
            print(f"  {name:<26s} {d.std():8.4f} {d.mean():+8.4f} "
                  f"{np.abs(d).max():8.4f}   {_grey(d.std()):13.2f}g")
    finally:
        dm._uncumulate = _orig

    d = keep["as it shipped"] - post
    rm = d.mean(axis=1)
    print()
    print(f"  shape of the shipped-variant error, row mean: row 0 {rm[0]:+.4f}, "
          f"row {len(rm)//2} {rm[len(rm)//2]:+.4f}, row {len(rm)-1} {rm[-1]:+.4f}")
    print(f"  the origin term alone predicts -k x 0.406 x row = {-K*0.40625*(len(rm)-1):+.4f} "
          f"at the last row")
    print(f"  the grid term alone predicts a shortfall factor "
          f"e^(k*230)/e^(k*377) = {np.exp(K*230)/np.exp(K*377):.3f}")
    print()
    print("  The current decode() carries both fixes, so the last row above is what the")
    print("  real path now produces. The correction being measured from here on is the")
    print("  SAME OPERATOR the post-hoc test verified, to 0.1 grey levels.")


# --------------------------------------------------------------------------
# 3. the rails
# --------------------------------------------------------------------------


def _extremes(U: np.ndarray, black_ref: float, white_ref: float):
    """Per-frame p0.5 / p99.5 of a decode, converted back to units of amp.

    U is in the decoder's own rail units (0 = black rail, 1 = white rail), so
    the picture value in amp about the porch is black_ref - (black_ref -
    white_ref) x U. Reported the way decode.py's docstring reports them: the
    bright end (negative) and the dark end (positive).
    """
    span = black_ref - white_ref
    lo, hi = np.percentile(U, [0.5, 99.5])
    return black_ref - span * hi, black_ref - span * lo


def section_rails(step: int = 8):
    print()
    print("=" * 78)
    print("RAILS  re-measured against the corrected picture, by the same procedure")
    print("=" * 78)
    print("""
  Per-frame p0.5 / p99.5 in units of the sync amplitude, 20 frames spanning the
  record and both channels -- the measurement that produced +0.65 / -0.95 for
  the undrooped picture. The rail is then placed where the CURRENT rail sits in
  the UNCORRECTED distribution, so the same rule sets all three rail pairs: two
  frames of twenty keep a ~1% tail past each rail, and the frames that do are
  the same dark line-art frames as before. Where that leaves an interval, the
  conservative (wider) end of it is taken, so that no frame's tail GROWS: the
  black rail's interval is (+0.41, +0.49] and the midpoint +0.45 would have
  taken L032's black tail from 7.2% to 8.4%, while +0.50 takes it to 4.9%.
""")
    cat = catalog_mod.build()
    fids = [f.id for f in cat.frames[step_slice(step)]]
    print(f"  {'fid':<6s} | {'bright':>8s} {'dark':>8s} {'clip W':>7s} {'clip B':>7s} | "
          f"{'bright':>8s} {'dark':>8s} {'clip W':>7s} {'clip B':>7s}")
    print(f"  {'':6s} | {'------------ k = 0 ------------':^33s} | "
          f"{'---------- k = 9.4e-4 ----------':^33s}")
    rows = []
    for fid in fids:
        x, ch, tb, _ = frame(fid)
        A0, U0, _ = decode_pair(x, settings(ch, 0.0), tb)
        AK, UK, _ = decode_pair(x, settings(ch, K), tb)
        b0, d0 = _extremes(U0, dm.UNCOUPLE_BLACK_REF, dm.UNCOUPLE_WHITE_REF)
        bK, dK = _extremes(UK, dm.UNCUMULATE_BLACK_REF, dm.UNCUMULATE_WHITE_REF)
        w0, k0 = clip_fracs(A0)
        wK, kK = clip_fracs(AK)
        rows.append((b0, d0, bK, dK, w0, k0, wK, kK))
        print(f"  {fid:<6s} | {b0:+8.3f} {d0:+8.3f} {100*w0:6.2f}% {100*k0:6.2f}% | "
              f"{bK:+8.3f} {dK:+8.3f} {100*wK:6.2f}% {100*kK:6.2f}%")
    R = np.array(rows)
    print()
    for j, nm, rail in ((0, "bright end, k=0  ", dm.UNCOUPLE_WHITE_REF),
                        (2, "bright end, k>0  ", dm.UNCUMULATE_WHITE_REF)):
        v = np.sort(R[:, j])
        print(f"  {nm} {v.min():+.3f} .. {v.max():+.3f}   rail {rail:+.2f} passed by "
              f"{int((v < rail).sum())}/{len(v)} frames")
    for j, nm, rail in ((1, "dark end,   k=0  ", dm.UNCOUPLE_BLACK_REF),
                        (3, "dark end,   k>0  ", dm.UNCUMULATE_BLACK_REF)):
        v = np.sort(R[:, j])
        print(f"  {nm} {v.min():+.3f} .. {v.max():+.3f}   rail {rail:+.2f} passed by "
              f"{int((v > rail).sum())}/{len(v)} frames")
    print()
    print(f"  clipped at white: median {100*np.median(R[:,4]):.2f}% -> "
          f"{100*np.median(R[:,6]):.2f}%, worst {100*R[:,4].max():.2f}% -> "
          f"{100*R[:,6].max():.2f}%")
    print(f"  clipped at black: median {100*np.median(R[:,5]):.2f}% -> "
          f"{100*np.median(R[:,7]):.2f}%, worst {100*R[:,5].max():.2f}% -> "
          f"{100*R[:,7].max():.2f}%")
    span0 = dm.UNCOUPLE_BLACK_REF - dm.UNCOUPLE_WHITE_REF
    spanK = dm.UNCUMULATE_BLACK_REF - dm.UNCUMULATE_WHITE_REF
    print()
    print(f"  span {span0:.2f} -> {spanK:.2f} x amp ({100*(spanK/span0-1):+.1f}% contrast), "
          f"both rails {0.5*((dm.UNCOUPLE_BLACK_REF-dm.UNCUMULATE_BLACK_REF)+(dm.UNCOUPLE_WHITE_REF-dm.UNCUMULATE_WHITE_REF)):+.3f}"
          f" x amp toward the bright end.")
    print("  That is the signature of a droop being removed rather than contrast being")
    print("  traded: the accumulation had darkened the whole picture by about that much on")
    print("  average (k x rows / 2 x mean light), and taking it out moves both rails, not")
    print("  one. A correction that had merely stretched the range would have moved them")
    print("  apart instead.")
    return R


def step_slice(step: int) -> slice:
    return slice(None, None, step)


# --------------------------------------------------------------------------
# 4. the measurement that decides
# --------------------------------------------------------------------------


def field_stats(U: np.ndarray, dec, span: float):
    """L000's known-uniform field: residual structure, in two honest units."""
    I = ct.build_ideal(U, bad_traces=dec.quality.dropouts)
    e = (U - I.img)[I.field]
    f = U[I.field]
    return {
        "rms": float(e.std()),                    # decoder units, ideal white = the rail
        "rms_ref": float(e.std() * span / SPAN0),  # same, in the k=0 rail scale
        "mean": float(e.mean()),                  # how optimistic the white rail is
        "own_rms": float(f.std()),
        "n": int(I.field.sum()),
        "ideal": I,
    }


def mount_gap(A: np.ndarray) -> float:
    """caltarget's convergence statistic on one decode: top mount minus bottom."""
    lo, hi = np.percentile(A[ct.PIC_ROWS[0]:ct.PIC_ROWS[1]], [1, 99])
    t = float(np.median(A[ct.TOP_ROWS[0]:ct.TOP_ROWS[1]])) / (hi - lo)
    b = float(np.median(A[ct.BOT_ROWS[0]:ct.BOT_ROWS[1]])) / (hi - lo)
    return t - b


def _uncumulate_signed(pic: np.ndarray, k: float) -> np.ndarray:
    """decode._uncumulate with the sign gate removed, for the wrong-sign arm."""
    out = np.empty_like(pic)
    for i in range(pic.shape[0]):
        row, acc, o = pic[i], 0.0, out[i]
        for j in range(row.shape[0]):
            v = row[j] + k * acc
            o[j] = v
            acc += v
    return out


def section_controls(fids=("L001", "L013", "R007", "R031")):
    """The arms that must FAIL, and the one that must change nothing.

    A fourth control is not runnable from here because it needs the previous
    revision of decode.py: with cfg.uncumulate = 0 the patched decoder is
    BIT-IDENTICAL to the unpatched one (max |difference| exactly 0.0 on L000,
    L011 and R010, and on the percentile-0-100 mode decode_pair leans on).
    The levels stage changes nothing until the correction is asked for.
    """
    print()
    print("=" * 78)
    print("CONTROLS  the arms that must fail")
    print("=" * 78)
    x, ch, tb, _ = frame(CAL_FRAME)
    dec0 = dm.decode(x, settings(ch, 0.0), tb=tb)

    # --- do the new rails flatten the field by themselves? -----------------
    A0, U0, _ = decode_pair(x, settings(ch, 0.0), tb)
    F0 = field_stats(U0, dec0, SPAN0)
    Ur = (dm.UNCOUPLE_BLACK_REF - SPAN0 * U0 - dm.UNCUMULATE_BLACK_REF) / -SPANK
    Fr = field_stats(Ur, dec0, SPANK)
    print(f"\n  RAILS ALONE: the same k=0 pixels, re-expressed on the new rails")
    print(f"    field rms in the common scale   {F0['rms_ref']:.4f} -> {Fr['rms_ref']:.4f}"
          f"   (equal by construction: the rails are an affine map)")
    print(f"    clipped at white                {100*clip_fracs(A0)[0]:.2f}% -> "
          f"{100*clip_fracs(np.clip(Ur, 0.0, 1.0))[0]:.2f}%")
    print("    The rails move the CLIP and never the structure, so no field-rms gain")
    print("    below can be credited to the levels change. It buys headroom, nothing else.")

    # --- sign and dose ------------------------------------------------------
    print(f"\n  SIGN AND DOSE, every arm through the real decode path")
    print(f"  {'arm':<24s} {'L000 field rms':>15s} {'clip W %':>9s} {'axis ratio':>11s} "
          f"{'radial rms':>11s} {'mount |gap|':>12s}")
    orig = dm._uncumulate
    state = {"s": 1.0}
    try:
        for name, sgn, kk in (("k = 0", 1.0, 0.0),
                              ("k = -9.4e-4 WRONG SIGN", -1.0, K),
                              ("k = 9.4e-4", 1.0, K),
                              ("k x 2 over-correct", 1.0, 2 * K),
                              ("k x 4 over-correct", 1.0, 4 * K)):
            state["s"] = sgn
            dm._uncumulate = (lambda z, kd: _uncumulate_signed(z, state["s"] * kd)) \
                if kk > 0 else orig
            A, U, _ = decode_pair(x, settings(ch, kk), tb)
            F = field_stats(U, dec0, SPANK if kk > 0 else SPAN0)
            r = circle_mod.fit_ring(A)
            g = np.abs([mount_gap(decode_pair(*_f(fid, kk))[1]) for fid in fids]).mean()
            ar = f"{r.axis_ratio:11.4f}" if r else f"{'no fit':>11s}"
            rr = f"{r.radial_rms:11.3f}" if r else f"{'-':>11s}"
            print(f"  {name:<24s} {F['rms_ref']:15.4f} {100*clip_fracs(A)[0]:9.2f} "
                  f"{ar} {rr} {_grey(g):11.2f}g")
    finally:
        dm._uncumulate = orig
    print("""
    The wrong sign is worse than doing nothing on every instrument at once --
    the field, the mount, and the circle's axis ratio, which is the one number
    the correction is forbidden to move. Twice the dose is worse than nothing on
    the field and four times cannot be fitted at all. The optimum is interior
    and it is where 68 other frames' mounts said it would be, so k is not a free
    parameter being spent on L000.""")


def _f(fid: str, k: float):
    x, ch, tb, _ = frame(fid)
    return x, settings(ch, k), tb


def section_verify(n_held: int = 13):
    print()
    print("=" * 78)
    print("VERIFY  the whole thing, through the real decode path")
    print("=" * 78)
    cat = catalog_mod.build()

    # ---- L000: the field, the clip, the circle ---------------------------
    x, ch, tb, amp = frame(CAL_FRAME)
    span0 = dm.UNCOUPLE_BLACK_REF - dm.UNCOUPLE_WHITE_REF
    spanK = dm.UNCUMULATE_BLACK_REF - dm.UNCUMULATE_WHITE_REF
    out = {}
    for name, k, span in (("k = 0", 0.0, span0), ("k = 9.4e-4", K, spanK)):
        A, U, resid = decode_pair(x, settings(ch, k), tb)
        dec = dm.decode(x, settings(ch, k), tb=tb)
        F = field_stats(U, dec, span)
        Fc = field_stats(np.asarray(A, dtype=np.float64), dec, span)  # as shipped, clipped
        ring = circle_mod.fit_ring(A)
        w, b = clip_fracs(A)
        out[name] = (F, Fc, ring, w, b, U, A, resid)

    print("\n  L000, the calibration frame")
    print(f"  {'':<22s} {'k = 0':>12s} {'k = 9.4e-4':>12s} {'change':>10s}")

    def row(label, a, b, fmt="{:12.4f}", pct=True):
        ch_ = f"{100*(b-a)/a:+9.1f}%" if (pct and a) else f"{b-a:+10.4f}"
        print(f"  {label:<22s} {fmt.format(a):>12s} {fmt.format(b):>12s} {ch_:>10s}")

    F0, Fc0, r0, w0, b0, U0, A0, res0 = out["k = 0"]
    FK, FcK, rK, wK, bK, UK, AK, resK = out["k = 9.4e-4"]
    dec0 = dm.decode(x, settings(ch, 0.0), tb=tb)
    Fp = field_stats(ct.uncumulate(U0, K), dec0, span0)   # the post-hoc result itself
    row("field rms (unclipped)", F0["rms_ref"], FK["rms_ref"])
    row("field rms (as shipped)", Fc0["rms_ref"], FcK["rms_ref"])
    row("white rail optimism", F0["mean"], FK["mean"], pct=False)
    print(f"  {'':<22s} the post-hoc correction, on the same k=0 decode, gives field rms "
          f"{Fp['rms_ref']:.4f}")
    row("clipped at white %", 100 * w0, 100 * wK, "{:12.2f}")
    row("clipped at black %", 100 * b0, 100 * bK, "{:12.3f}", pct=False)
    print(f"  {'field pixels':<22s} {F0['n']:12d} {FK['n']:12d}")
    print(f"  {'affine unclip resid':<22s} {res0:12.1e} {resK:12.1e}")
    print()
    print(f"  {'circle axis ratio':<22s} {r0.axis_ratio:12.4f} {rK.axis_ratio:12.4f} "
          f"{rK.axis_ratio-r0.axis_ratio:+10.4f}")
    print(f"  {'circle radial rms px':<22s} {r0.radial_rms:12.3f} {rK.radial_rms:12.3f} "
          f"{rK.radial_rms-r0.radial_rms:+10.3f}")
    print(f"  {'ring width px':<22s} {r0.width:12.2f} {rK.width:12.2f}")
    print(f"  {'ring centre':<22s} {'(%.1f,%.1f)'%(r0.cy,r0.cx):>12s} "
          f"{'(%.1f,%.1f)'%(rK.cy,rK.cx):>12s}")

    # the field's own droop, the thing the correction is for
    I0 = F0["ideal"]
    for name, U, I in (("k = 0", U0, F0["ideal"]), ("k = 9.4e-4", UK, FK["ideal"])):
        M = I.field
        rmean = np.array([U[i][M[i]].mean() if M[i].sum() > 20 else np.nan
                          for i in range(U.shape[0])])
        r0_, r1_ = I.rows
        fall = float(np.nanmean(rmean[r1_-20:r1_]) - np.nanmean(rmean[r0_:r0_+20]))
        print(f"  {name:<22s} field level falls {fall:+.4f} "
              f"({_grey(fall):+6.1f} grey) from the top of the trace to the bottom")

    # ---- the mount, on frames the coefficient never saw -------------------
    print()
    print(f"  MOUNT CONVERGENCE, {n_held} frames the coefficient never saw "
          f"(indices 1 mod 4; k was fitted on 0 mod 4)")
    print(f"  {'fid':<6s} {'top':>8s} {'bottom':>8s} {'gap':>8s}   {'gap, post-hoc':>14s}"
          f"   {'gap, in path':>13s}")
    fids = [f.id for f in cat.frames[HELD_OUT]][:n_held]
    g0, g1, gp = [], [], []
    for fid in fids:
        xf, chf, tbf, _ = frame(fid)
        _, Ua, _ = decode_pair(xf, settings(chf, 0.0), tbf)
        _, Ub, _ = decode_pair(xf, settings(chf, K), tbf)

        def lv(A):
            lo, hi = np.percentile(A[ct.PIC_ROWS[0]:ct.PIC_ROWS[1]], [1, 99])
            return (float(np.median(A[ct.TOP_ROWS[0]:ct.TOP_ROWS[1]])) / (hi - lo),
                    float(np.median(A[ct.BOT_ROWS[0]:ct.BOT_ROWS[1]])) / (hi - lo))
        t0, bm0 = lv(Ua)
        t1, bm1 = lv(Ub)
        tp, bp = lv(ct.uncumulate(Ua, K))
        g0.append(t0 - bm0)
        g1.append(t1 - bm1)
        gp.append(tp - bp)
        print(f"  {fid:<6s} {t0:8.3f} {bm0:8.3f} {t0-bm0:8.3f}   {tp-bp:14.3f}"
              f"   {t1-bm1:13.3f}")
    g0, g1, gp = np.array(g0), np.array(g1), np.array(gp)
    print()
    print(f"  contrast-normalised |top - bottom|, mean: {_grey(np.abs(g0).mean()):.2f} -> "
          f"{_grey(np.abs(g1).mean()):.2f} grey-equivalent "
          f"({100*(np.abs(g1).mean()/np.abs(g0).mean()-1):+.0f}%); "
          f"the post-hoc correction gives {_grey(np.abs(gp).mean()):.2f}")
    print(f"  median gap {_grey(np.median(g0)):+.2f} -> {_grey(np.median(g1)):+.2f} "
          f"grey-equivalent; closer to agreement on "
          f"{int((np.abs(g1) < np.abs(g0)).sum())}/{len(g0)} frames")
    print(f"  in path vs post-hoc, per frame: max |difference| "
          f"{_grey(np.abs(g1-gp).max()):.3f} grey-equivalent")

    # ---- a photograph, checked for damage --------------------------------
    print()
    print(f"  A PHOTOGRAPH ({PHOTO_FRAME}) CHECKED FOR DAMAGE")
    xp, chp, tbp, _ = frame(PHOTO_FRAME)
    Ap, Up, _ = decode_pair(xp, settings(chp, 0.0), tbp)
    Bp, Vp, _ = decode_pair(xp, settings(chp, K), tbp)
    w0p, b0p = clip_fracs(Ap)
    wKp, bKp = clip_fracs(Bp)
    print(f"    clipped at white {100*w0p:.3f}% -> {100*wKp:.3f}%,  "
          f"at black {100*b0p:.3f}% -> {100*bKp:.3f}%")
    print(f"    p1/p99 of the frame {np.percentile(Ap,1):.3f}/{np.percentile(Ap,99):.3f} -> "
          f"{np.percentile(Bp,1):.3f}/{np.percentile(Bp,99):.3f}")
    d = Bp - Ap
    print(f"    change: rms {d.std():.4f} ({_grey(d.std()):.1f} grey), "
          f"mean {d.mean():+.4f}, max |.| {np.abs(d).max():.3f}")
    # structure, not level: correlation after removing each frame's own affine
    a = (Ap - Ap.mean()) / Ap.std()
    b = (Bp - Bp.mean()) / Bp.std()
    print(f"    structural correlation after removing each frame's own level and gain: "
          f"{float((a*b).mean()):.5f}")
    near = lambda A: 100 * float(((A < 1.0/255) | (A > 1 - 1.0/255)).mean())
    print(f"    pixels crushed against a rail (within 1/255 of it): "
          f"{near(Ap):.3f}% -> {near(Bp):.3f}%")
    print(f"    8-bit levels occupied: {len(np.unique((Ap*255).astype(np.uint8)))} -> "
          f"{len(np.unique((Bp*255).astype(np.uint8)))} of 256")
    dx = lambda A: float(np.abs(np.diff(A, axis=0)).mean())
    print(f"    mean |along-trace gradient| {dx(Ap):.5f} -> {dx(Bp):.5f} "
          f"(the correction has gain >= 1 along the trace, so detail cannot be lost here)")
    tb_top = float(np.median(Ap[ct.TOP_ROWS[0]:ct.TOP_ROWS[1]]))
    tb_bot = float(np.median(Ap[ct.BOT_ROWS[0]:ct.BOT_ROWS[1]]))
    tb_top2 = float(np.median(Bp[ct.TOP_ROWS[0]:ct.TOP_ROWS[1]]))
    tb_bot2 = float(np.median(Bp[ct.BOT_ROWS[0]:ct.BOT_ROWS[1]]))
    print(f"    its own mount: top {tb_top:.3f} bottom {tb_bot:.3f} (gap {tb_top-tb_bot:+.3f}) "
          f"-> top {tb_top2:.3f} bottom {tb_bot2:.3f} (gap {tb_top2-tb_bot2:+.3f})")
    return out


def write_pngs(out_dir: str):
    """L000 and the photograph, k=0 and k>0, as 8-bit PNGs for eyeballing."""
    from pathlib import Path
    try:
        from PIL import Image
    except Exception:
        print("  (PIL not available; skipping PNGs)")
        return
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for fid in (CAL_FRAME, PHOTO_FRAME):
        x, ch, tb, _ = frame(fid)
        for tag, k in (("k0", 0.0), ("kfix", K)):
            img = dm.decode(x, settings(ch, k), tb=tb).image
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
                d / f"{fid}_{tag}.png")
    print(f"  wrote PNGs to {d}")


SECTIONS = {
    "reference": lambda: section_reference(),
    "origin": lambda: section_origin(),
    "rails": lambda: section_rails(),
    "controls": lambda: section_controls(),
    "verify": lambda: section_verify(),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", default="reference,origin,rails,controls,verify")
    ap.add_argument("--pngs", default="")
    args = ap.parse_args(argv)
    for name in args.section.split(","):
        name = name.strip()
        if name not in SECTIONS:
            print(f"unknown section {name!r}; have {', '.join(SECTIONS)}", file=sys.stderr)
            return 2
        SECTIONS[name]()
    if args.pngs:
        write_pngs(args.pngs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
