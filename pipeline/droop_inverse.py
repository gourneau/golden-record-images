"""Invert the decay-bias pole that pipeline/droop.py measured.

pipeline/droop.py established (2026-08): the "embossed" look of every decoded
frame is a SINGLE-POLE AC-coupling high-pass acting on the signal in raster
time, tau_L = 530 samples at 384 kHz, tau_R = 295, different per audio
channel. This module owns the CORRECTION and its validation; droop.py owns
the measurement. decode.py is untouched: callers apply undroop() to the
signal before decoding (recover the timebase from the RAW signal and pass it
in, so the landmark convention cannot move).

THE FILTER
----------
The chain's loss is H(z) = (1 - z^-1)/(1 - a z^-1), a = exp(-1/tau). The
exact inverse is (1 - a z^-1)/(1 - z^-1): y[n] = y[n-1] + x[n] - a x[n-1],
applied in sample order, which IS raster order (along the trace and straight
across every trace boundary). Sample rate only rescales tau: at 96 kHz use
tau/4.

STABILITY, measured rather than assumed (`--section stability`). The exact
inverse has a pole at DC: a constant input offset integrates into a ramp of
offset/tau per sample, and the output carries a free constant. Anchoring
schemes compared on the real master (L000, 515 traces):
  * SUBTRACT THE WINDOW MEAN before filtering, then let the per-trace back-
    porch clamp (decode.py's dc_restore; _clamped_dots here) absorb the
    residual drift once per trace. The mean, NOT the median: this signal's
    sync pulses skew it -- L000's window median sits 5.2e-3 above its mean,
    and integrating that error blows the pre-clamp porch wander up from
    0.165 to 15.95 p-p (the `exact, median dc` row; an earlier draft of
    this module shipped exactly that bug). With the mean subtracted the
    inverted window's porch wanders 0.165 p-p over 512 traces vs 0.187
    raw: no runaway, and the clamp removes it per trace.
  * PIECEWISE-LINEAR PORCH BASELINE inside undroop() (anchor_marks=):
    subtract the interpolation of the smoothed reconstructed porch levels.
    Changes every post-clamp validation number by < 3e-4 once the mean is
    subtracted -- kept as an option for callers with no downstream clamp.
  * LEAKY POLE (rho < 1): bounded DC gain (1-a)/(1-rho) instead of an
    integrator. Measured field restoration (level -0.0848 for exact):
    -0.0842 at 32 traces of leak, -0.0827 at 8 (2.5% under), -0.0763 at 2
    (10% under, decay residual halved toward raw). Buys nothing the clamp
    does not already provide; rejected as default, kept as an option.
So: DEFAULT = exact inverse (rho = 1), mean subtracted, per-trace porch
re-anchoring downstream. The trade-off accepted: absolute DC across a whole
frame is unknowable (the chain blocked it), so each trace's level is pinned
to its porch -- which is exactly the reference decode.py already trusts.

VALIDATION (`python -m pipeline.droop_inverse`; all numbers measured on the
real master, 2026-08):
  field    L000's field loses its along-trace exponential settle: decay
           amplitude -0.234 -> -0.006 inside the ring (-101.7% -> -2.5%
           of black-white), -0.142 -> +0.028 outside; the restored field
           holds an absolute level (-0.09 = 0.59 amp bright) instead of
           relaxing to porch grey. The field's LINEAR gradient (+21% B-W
           inside) is genuine source shading, NOT droop: it is already in
           the raw decode (+19% by robust slope) and is unchanged at
           every tau 460..1000. The +0.028 outside residual is within
           droop.py's tau systematic (zero at tau ~600 vs adopted 530).
  gap      The parity-balanced mean gap profile (hum and parity response
           cancel exactly in it) drops from a decay of -13..-15% of B-W
           to -1.8..+1.1% on the four clean-gap frames (L010 L030 R010
           R020); on L055/R040 the residual is the gap's real stationary
           texture (their non-model rms is 10-20x the clean frames').
           The parity probe flattens 2.1-5.5x to 0.2-0.6% B-W on all six.
           The task brief's "constant black block before trace 0" does
           not exist -- droop.py Q4 measured the hatch marker there --
           so the gap, constant at source, is the substitute probe.
  shadow   Behind Mars (L011) the raw sky is still -4.9% of B-W dark at
           29-40 dots and recovering; undrooped it is flat to +0.3%
           (L013 Earth: -3.0% -> +1.2%). The removed component decays
           with ~the channel tau; the residual positive skirt on the
           first ~15 dots is scene (terminator/halation), shown by its
           tau-insensitivity (x0.7..x1.3 moves it ~20%).
  testset  quality.py composite 18.9 -> 19.9 mean (its factors target
           TIMING damage, which undrooping does not touch; the circle
           metric is unchanged: ratio 1.0075/1.0081, rms 0.95/0.89).
           Worse by >0.5 on 6 of 16 frames, none from new artefacts on
           inspection: the drivers are the h/v coherence factor (restored
           vertical low frequency raises v-coherence, so the ratio dips
           ~1%) and white clipping. CLIPPING IS REAL and is decode.py's
           levels calibration, not the filter: the +-0.75*amp rails were
           fitted to DROOPED pictures, whose sustained tones had decayed
           toward the porch. Undrooped, 15 frames measure p0.5/p99.5 at
           -0.94..-0.57 / +0.24..+0.62 amp: whites reach past -0.75
           (38-42% of L000/L002 pixels pin at white) while blacks never
           reach +0.75. When decode.py adopts this correction its rails
           should move to roughly white -0.95*amp, black +0.65*amp,
           re-measured properly across the record.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.signal import lfilter

from . import droop, geometry

TAU_384 = dict(droop.TAU_384)  # {"L": 530.0, "R": 295.0} samples at 384 kHz
PERIOD = geometry.NOMINAL_PERIOD
BINS = geometry.BINS
PORCH_BINS = (100.0, 225.0)  # geometry.py back-porch window


def undroop(
    x: np.ndarray,
    tau: float,
    *,
    rho: float = 1.0,
    subtract_dc: bool = True,
    anchor_marks: np.ndarray | None = None,
) -> np.ndarray:
    """Invert the one-pole high-pass: y[n] = rho*y[n-1] + x[n] - a*x[n-1].

    `tau` is in samples OF THIS SIGNAL (use tau_for(): TAU_384[ch]/4 at
    96 kHz). rho = 1.0 is the exact inverse; the output then carries a free
    DC constant and integrates any residual input offset, so it MUST be
    re-anchored downstream -- decode.py's per-trace porch clamp does this,
    or pass `anchor_marks` (per-trace landmark positions in samples) to have
    the reconstructed porch's piecewise-linear baseline subtracted here.
    rho < 1 bounds the DC gain at (1-a)/(1-rho) instead; measured to
    under-restore, see the module docstring. `subtract_dc` removes the
    window MEAN first (not the median -- see docstring) and should stay on.
    """
    a = float(np.exp(-1.0 / tau))
    x = np.asarray(x, dtype=np.float64)
    x0 = float(x.mean()) if subtract_dc else 0.0
    y = lfilter([1.0, -a], [1.0, -rho], x - x0)
    if anchor_marks is not None and len(anchor_marks) > 8:
        m = np.asarray(anchor_marks, dtype=np.float64)
        cent, lev = [], []
        for k in range(len(m) - 1):
            p = m[k + 1] - m[k]
            A = int(m[k] + PORCH_BINS[0] / BINS * p)
            B = int(m[k] + PORCH_BINS[1] / BINS * p)
            if 0 <= A < B <= len(y):
                cent.append(0.5 * (A + B))
                lev.append(float(np.median(y[A:B])))
        cent, lev = np.asarray(cent), np.asarray(lev)
        sm = lev.copy()
        if len(lev) >= 16:  # per-parity median-of-7, mirroring decode.py
            for pp in (0, 1):
                s = lev[pp::2]
                pad = np.pad(s, 3, mode="edge")
                sm[pp::2] = np.median(
                    np.lib.stride_tricks.sliding_window_view(pad, 7), axis=1)
        y = y - np.interp(np.arange(len(y), dtype=np.float64), cent, sm)
    return y


def tau_for(frame_id_or_channel, rate_divisor: float = 1.0) -> float:
    """tau in signal samples for a frame id ("L055") or channel letter,
    at 384 kHz / rate_divisor (pass 4 for the 96 kHz assets)."""
    ch = str(frame_id_or_channel)[0].upper()
    return TAU_384[ch] / rate_divisor


# --------------------------------------------------------------------------
# shared: clamped dot matrix at 384 kHz (validations run in raw signal units
# so nothing downstream can rescale them)
# --------------------------------------------------------------------------


def _porch_levels(x: np.ndarray, m: np.ndarray, k_lo: int, k_hi: int) -> np.ndarray:
    out = np.full(k_hi - k_lo, np.nan)
    for i, k in enumerate(range(k_lo, k_hi)):
        p = m[k + 1] - m[k]
        a = int(m[k] + PORCH_BINS[0] / BINS * p)
        b = int(m[k] + PORCH_BINS[1] / BINS * p)
        if 0 <= a < b <= len(x):
            out[i] = float(np.median(x[a:b]))
    return out


def _dots(x: np.ndarray, m: np.ndarray, k_lo: int, k_hi: int) -> np.ndarray:
    """(traces, dots) plateau integrals for trace slots [k_lo, k_hi)."""
    cs = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
    rows = []
    for k in range(k_lo, k_hi):
        p = m[k + 1] - m[k]
        spd = p / 262.5
        s0 = m[k] + 232.0 / BINS * p
        nd = int(2808.0 / BINS * p / spd)
        e = s0 + np.arange(nd + 1) * spd
        b = np.floor(e).astype(np.int64)
        C = cs[b] + (e - b) * x[b]
        rows.append(np.diff(C) / spd)
    n = min(len(r) for r in rows)
    return np.asarray([r[:n] for r in rows])


def _clamped_dots(x: np.ndarray, m: np.ndarray, k_lo: int, k_hi: int,
                  dehum: bool = True) -> np.ndarray:
    """Dot matrix, porch-clamped per trace (the re-anchoring that pins the
    inverse filter's free DC), optionally odd/even median-subtracted
    (removes the scan-locked hum and any parity-locked residual)."""
    D = _dots(x, m, k_lo, k_hi)
    porch = _porch_levels(x, m, k_lo, k_hi)
    porch = np.where(np.isfinite(porch), porch, np.nanmedian(porch))
    D = D - porch[:, None]
    if dehum and len(D) >= 16:
        ks = np.arange(k_lo, k_lo + len(D))
        half = 0.5 * (np.median(D[ks % 2 == 0], axis=0)
                      - np.median(D[ks % 2 == 1], axis=0))
        D[ks % 2 == 0] -= half
        D[ks % 2 == 1] += half
    return D


def _sync_amp(x: np.ndarray, m: np.ndarray, k_lo: int, k_hi: int) -> float:
    """Sync amplitude (short-burst plateau minus porch); black-white range
    is 1.5x this (decode.py's +-0.75 rails)."""
    porch = _porch_levels(x, m, k_lo, k_hi)
    test, plat = [], []
    for k in range(k_lo, k_hi):
        p = m[k + 1] - m[k]
        a, b = int(m[k] + 3090.0 / BINS * p), int(m[k] + 3140.0 / BINS * p)
        c, d = int(m[k] + 3165.0 / BINS * p), int(m[k] + 3195.0 / BINS * p)
        if d <= len(x):
            test.append(float(np.median(x[a:b])))
            plat.append(float(np.median(x[c:d])))
        else:
            test.append(np.nan)
            plat.append(np.nan)
    test, plat = np.asarray(test), np.asarray(plat)
    ks = np.arange(k_lo, k_hi)
    ok = np.isfinite(test) & np.isfinite(porch)
    lvl = {pp: np.median(test[ok & (ks % 2 == pp)]) for pp in (0, 1)}
    short = 0 if lvl[0] < lvl[1] else 1
    sel = ok & (ks % 2 == short)
    return float(np.median(plat[sel]) - np.median(porch[ok]))


def _frame_zero(m: np.ndarray, pre_traces: float) -> int:
    return int(np.argmin(np.abs(m - pre_traces * PERIOD)))


def _robust_slope(v: np.ndarray, iters: int = 4) -> float:
    """Tukey-reweighted straight-line slope of v against its index."""
    u = np.arange(len(v), dtype=np.float64)
    ok = np.isfinite(v)
    u, v = u[ok], v[ok]
    w = np.ones(len(v))
    slope = 0.0
    for _ in range(iters):
        sw = np.sqrt(w)
        A = np.column_stack([u, np.ones(len(u))])
        c, *_ = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)
        slope, icpt = float(c[0]), float(c[1])
        r = v - (icpt + slope * u)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        w = (1 - np.clip(r / (4 * s), -1, 1) ** 2) ** 2
    return slope


def _exp_lin(prof: np.ndarray, lo: int, hi: int, tau_dots: float):
    """Decompose a profile over dots [lo, hi) into A*exp(-d/tau_dots) +
    B*d + C. Returns (A, B*(hi-lo), C, residual rms): decay amplitude,
    linear tilt over the span, offset."""
    d = np.arange(lo, hi, dtype=np.float64)
    v = prof[lo:hi]
    ok = np.isfinite(v)
    d, v = d[ok], v[ok]
    X = np.column_stack([np.exp(-d / tau_dots), d, np.ones(len(d))])
    c, *_ = np.linalg.lstsq(X, v, rcond=None)
    rms = float(np.sqrt(np.mean((v - X @ c) ** 2)))
    return float(c[0]), float(c[1] * (hi - lo)), float(c[2]), rms


# --------------------------------------------------------------------------
# V1: the calibration frame's field
# --------------------------------------------------------------------------

# L000 ring in dot-matrix coordinates (geometry.py: centre trace 267 from
# frame start, bin 1589, radii 151.6 traces x 1127.8 bins; ~12.18
# samples/dot, dot 0 = bin 232).
_RING_TR, _RING_DOT = 267.0, (1589.0 - 232.0) / 12.18
_RING_RTR, _RING_RDOT = 151.6, 1127.8 / 12.18
# The frame carries dark bars across the top and bottom (real content);
# the outside-field probe stays between them.
_OUT_DOTS = (15, 200)
_IN_DOTS = (65, 158)


def _field_profiles(D: np.ndarray, k0: int):
    """(inside, outside) per-dot-row median profiles of L000's field."""
    n_tr, n_dot = D.shape
    tr = np.arange(n_tr)[:, None] - (k0 + _RING_TR)
    dt = np.arange(n_dot)[None, :] - _RING_DOT
    rr = np.sqrt((tr / _RING_RTR) ** 2 + (dt / _RING_RDOT) ** 2)
    inside = np.where(rr < 0.55, D, np.nan)
    out_mask = (rr > 1.35) & (np.arange(n_tr)[:, None] > k0 + 6) \
        & (np.arange(n_tr)[:, None] < k0 + 500)
    outside = np.where(out_mask, D, np.nan)
    with np.errstate(all="ignore"):
        pi = np.nanmedian(inside, axis=0)
        po = np.nanmedian(outside, axis=0)
        ni = np.sum(np.isfinite(inside), axis=0)
        no = np.sum(np.isfinite(outside), axis=0)
    pi[ni < 30] = np.nan
    po[no < 60] = np.nan
    return pi, po


def validate_field() -> None:
    print("V1 == L000 FIELD (raw signal units; + is darker) " + "=" * 25)
    x, m, _ = droop.load("L000", 3.0, 515)
    k0 = _frame_zero(m, 3.0)
    bw = 1.5 * _sync_amp(x, m, k0 + 4, k0 + 500)
    tau_dots = TAU_384["L"] / 12.18
    print(f"  black-white range 1.5*amp = {bw:.4f}; profile = A*exp(-d/{tau_dots:.1f} dots)"
          f" + linear;\n  the DECAY AMPLITUDE A is the droop; the linear tilt is source"
          f" shading\n  (same magnitude in the raw decode and at every tau 460..1000:"
          f" not droop)")
    for tag, sig in (("raw", x), ("undrooped", undroop(x, TAU_384["L"]))):
        D = _clamped_dots(sig, m, k0 - 2, k0 + 510)
        pi, po = _field_profiles(D, 2)
        for name, prof, (lo, hi) in (("inside ring", pi, _IN_DOTS),
                                     ("outside ring", po, _OUT_DOTS)):
            A, tilt, c, rms = _exp_lin(prof, lo, hi, tau_dots)
            lev = float(np.nanmedian(prof[lo:hi]))
            print(f"  {tag:9} {name:12}: decay A {A:+.4f} ({100 * A / bw:+5.1f}% B-W)  "
                  f"tilt {tilt:+.4f} ({100 * tilt / bw:+5.1f}%)  level {lev:+.4f}  "
                  f"fit rms {rms:.4f}")


# --------------------------------------------------------------------------
# V2: the content-free gap must map to a single constant level
# --------------------------------------------------------------------------


def validate_gap() -> None:
    print("\nV2 == GAP CONSTANCY. The task brief's 'black block before trace 0'")
    print("      does not exist (droop.py Q4: those traces are the hatch")
    print("      marker); the inter-frame gap is the constant-at-source probe.")
    print("      Two measurements per frame, both on porch-clamped dot rows:")
    print("      (a) decay A of the PARITY-BALANCED MEAN profile -- the scan-")
    print("      locked hum and the parity pulse response cancel exactly in")
    print("      it, leaving the mean sync pulse's droop tail, which must die")
    print("      by inversion; (b) rms of the even-odd parity profile about")
    print("      the fitted hum half-sine (droop.py's deterministic probe).")
    print("      'rms' beside A is non-model structure: the gap's stationary")
    print("      TEXTURE (geometry.py), genuine signal that survives on the")
    print("      two frames whose gaps carry it " + "=" * 41)
    for fid in ("L010", "L030", "R010", "R020", "L055", "R040"):
        x, m, _ = droop.load(fid, 195.0, 60)
        lo, hi = droop.gap_slots(m)
        hi = lo + 2 * ((hi - lo) // 2)  # balanced parity count
        bw = 1.5 * _sync_amp(x, m, lo, hi)
        tau_dots = TAU_384[fid[0]] / 12.18
        line = f"  {fid} ({hi - lo} traces, B-W {bw:.3f}):"
        for tag, sig in (("raw", x),
                         ("undrooped", undroop(x, TAU_384[fid[0]]))):
            D = _clamped_dots(sig, m, lo, hi, dehum=False)
            ks = np.arange(lo, lo + len(D))
            A, _, _, rms = _exp_lin(D.mean(axis=0), 3, D.shape[1] - 3, tau_dots)
            par = (np.median(D[ks % 2 == 0], axis=0)
                   - np.median(D[ks % 2 == 1], axis=0))
            d = np.arange(3, D.shape[1] - 3, dtype=np.float64)
            th = np.pi * d / 230.7
            X = np.column_stack([np.sin(th), np.cos(th), np.ones(len(d))])
            c, *_ = np.linalg.lstsq(X, par[3:-3], rcond=None)
            prms = float(np.sqrt(np.mean((par[3:-3] - X @ c) ** 2)))
            line += (f"\n    {tag:9}: mean-profile decay A {A:+.5f} "
                     f"({100 * A / bw:+6.2f}% B-W, texture rms {rms:.5f})   "
                     f"parity resid rms {prms:.5f}")
        print(line)


# --------------------------------------------------------------------------
# V3: bright objects must stop trailing shadows
# --------------------------------------------------------------------------


def _bright_edges(Du: np.ndarray, thr: float, post: int = 60,
                  min_run: int = 8) -> list[tuple[int, int]]:
    """Per trace, the LAST dot of the last >=min_run bright run in the
    UNDROOPED matrix, with a quiet dark tail after it. Edges must come from
    the corrected signal: in the raw decode a large bright object decays to
    nothing (measured on L011: raw per-trace peak brightness never exceeds
    15% of B-W although the restored disk sits at ~37%), so a raw-side
    detector cannot find the object at all."""
    bri = -Du
    edges = []
    for t in range(bri.shape[0]):
        v = bri[t]
        hot = v > thr
        edge = None
        run = 0
        for j in range(len(v)):
            run = run + 1 if hot[j] else 0
            if run >= min_run:
                edge = j
        if edge is None or edge + post + 1 >= len(v):
            continue
        if np.max(v[edge + 6: edge + post]) > thr:  # tail must stay dark
            continue
        edges.append((t, edge))
    return edges


def _edge_profile(D: np.ndarray, edges) -> np.ndarray:
    bri = -D
    rows = [bri[t, e + 1: e + 61] - np.median(bri[t, e + 40: e + 56])
            for t, e in edges]
    return np.median(np.asarray(rows), axis=0)


def validate_shadow() -> None:
    print("\nV3 == SHADOW BEHIND BRIGHT OBJECTS. Median brightness at N dots")
    print("      past the object's trailing edge, minus the same trace's level")
    print("      40-55 dots out. Edges located on the undrooped matrix (in the")
    print("      raw decode the object itself has decayed to ~nothing) and")
    print("      applied to BOTH signals. The positive residual on the first")
    print("      ~15 dots of the undrooped rows is the object's own skirt")
    print("      (terminator/halation): it is tau-INSENSITIVE (printed), so it")
    print("      is scene, not filter. The removed row is raw minus undrooped:")
    print("      the shadow the inversion took out, decaying with ~tau " + "=" * 6)
    dists = (2, 5, 10, 20, 30, 40)
    for fid, note in (("L011", "Mars limb, dark sky below"),
                      ("L013", "Earth (red sep.) on black space")):
        x, m, _ = droop.load(fid, 3.0, 515)
        k0 = _frame_zero(m, 3.0)
        bw = 1.5 * _sync_amp(x, m, k0 + 4, k0 + 500)
        tau = TAU_384[fid[0]]
        Du = _clamped_dots(undroop(x, tau), m, k0 + 4, k0 + 500)
        Dr = _clamped_dots(x, m, k0 + 4, k0 + 500)
        thr = 0.6 * float(np.percentile(-Du, 99))
        edges = _bright_edges(Du, thr)
        print(f"  {fid} ({note}; B-W {bw:.3f}, threshold {thr:.3f}, "
              f"{len(edges)} trailing edges)")
        if not edges:
            continue
        p_raw = _edge_profile(Dr, edges)
        p_und = _edge_profile(Du, edges)
        for tag, prof in (("raw", p_raw), ("undrooped", p_und),
                          ("removed", p_raw - p_und)):
            pts = "  ".join(f"@{d:2d} {prof[d - 1]:+.4f}" for d in dists)
            tail = float(np.mean(prof[28:40]))
            print(f"    {tag:9}: {pts}   tail 29-40 {tail:+.4f} "
                  f"({100 * tail / bw:+5.1f}% B-W)")
        sens = [
            _edge_profile(_clamped_dots(undroop(x, f * tau), m, k0 + 4, k0 + 500),
                          edges)[4]
            for f in (0.7, 1.3)
        ]
        print(f"    skirt tau-sensitivity @5 dots: tau x0.7 {sens[0]:+.4f}, "
              f"adopted {p_und[4]:+.4f}, tau x1.3 {sens[1]:+.4f}")


# --------------------------------------------------------------------------
# stability: exact vs leaky vs anchored inverse
# --------------------------------------------------------------------------


def validate_stability() -> None:
    """The two symptoms an unstable inverse would print: pre-clamp porch
    wander (the integrator running away) and a post-clamp within-trace ramp
    (offset error), read as extra linear tilt on the L000 outside field."""
    print("\nSTABILITY == EXACT vs LEAKY vs ANCHORED INVERSE " + "=" * 26)
    x, m, _ = droop.load("L000", 3.0, 515)
    k0 = _frame_zero(m, 3.0)
    tau_dots = TAU_384["L"] / 12.18
    cases = [
        ("raw", None, {}),
        ("exact rho=1", 1.0, {}),
        ("exact, median dc", 1.0, {"median_dc": True}),
        ("exact + anchor", 1.0, {"anchor": True}),
        ("leak 32 traces", float(np.exp(-1.0 / (32 * PERIOD))), {}),
        ("leak 8 traces", float(np.exp(-1.0 / (8 * PERIOD))), {}),
        ("leak 2 traces", float(np.exp(-1.0 / (2 * PERIOD))), {}),
    ]
    for tag, rho, kw in cases:
        if rho is None:
            sig = x
        elif kw.get("median_dc"):
            sig = undroop(x - np.median(x), TAU_384["L"], rho=rho,
                          subtract_dc=False)
        else:
            sig = undroop(x, TAU_384["L"], rho=rho,
                          anchor_marks=m if kw.get("anchor") else None)
        porch = _porch_levels(sig, m, k0 - 2, k0 + 510)
        wander = float(np.nanmax(porch) - np.nanmin(porch))
        D = _clamped_dots(sig, m, k0 - 2, k0 + 510)
        _, po = _field_profiles(D, 2)
        A, tilt, _, _ = _exp_lin(po, *_OUT_DOTS, tau_dots)
        lev = float(np.nanmedian(po[slice(*_OUT_DOTS)]))
        print(f"  {tag:16}: porch wander p-p {wander:8.4f}   post-clamp decay A "
              f"{A:+.4f}  tilt {tilt:+.4f}  field level {lev:+.4f}")


# --------------------------------------------------------------------------
# V4: the frozen test set through the real decoder
# --------------------------------------------------------------------------


def validate_testset(save_dir=None) -> None:
    import math
    from pathlib import Path

    from scipy.signal import resample_poly

    from . import catalog as catalog_mod
    from . import decode as decode_mod
    from . import quality
    from . import sync as sync_mod
    from . import wav
    from .testset import MASTER, NOMINAL_96, PRE_384, TEST_FRAMES

    print("\nV4 == FROZEN TEST SET, STOCK vs UNDROOPED DECODE (same timebase,")
    print("      recovered from the raw signal; a>b columns are stock>undrooped)")
    print(f"{'frame':5} {'exp':4} {'stock':>6} {'undrp':>6} {'delta':>6}  "
          f"{'shft':>4}>{'shft':<4} {'drift':>5}>{'drift':<5} {'par':>5}>{'par':<5} "
          f"{'h/v':>5}>{'h/v':<5} {'clip%':>5}>{'clip%':<5}")
    info = wav.probe(MASTER)
    cat = catalog_mod.build()
    rows = []
    images = {}
    for fid, kind, expect in TEST_FRAMES:
        frame = cat.by_id(fid)
        span = int(math.ceil((sync_mod.TRACES_PER_FRAME + 6) * sync_mod.NOMINAL_PERIOD))
        x = np.asarray(wav.read(info, frame.channel, frame.seed_sample - PRE_384,
                                PRE_384 + span), dtype=np.float64)
        x96 = resample_poly(x, 1, 4)
        tb = sync_mod.recover(x96, period_guess=NOMINAL_96,
                              n_traces=sync_mod.TRACES_PER_FRAME)
        cfg = decode_mod.Settings()
        d0 = decode_mod.decode(np.asarray(x96, dtype=np.float32), cfg, tb)
        y96 = undroop(x96, tau_for(fid, 4.0))
        d1 = decode_mod.decode(np.asarray(y96, dtype=np.float32), cfg, tb)
        m0 = quality.frame_report(d0.image, tb, fid).metrics
        m1 = quality.frame_report(d1.image, tb, fid).metrics
        clip0 = float(np.mean((d0.image <= 0.0) | (d0.image >= 1.0)))
        clip1 = float(np.mean((d1.image <= 0.0) | (d1.image >= 1.0)))
        rows.append((fid, expect, m0, m1))
        images[fid] = (d0.image, d1.image)
        print(f"{fid:5} {expect:4} {m0['composite']:6.1f} {m1['composite']:6.1f} "
              f"{m1['composite'] - m0['composite']:+6.1f}  "
              f"{m0['shift_rms']:4.2f}>{m1['shift_rms']:<4.2f} "
              f"{m0['drift_span']:5.1f}>{m1['drift_span']:<5.1f} "
              f"{m0['parity_db']:5.1f}>{m1['parity_db']:<5.1f} "
              f"{m0['coh_ratio']:5.3f}>{m1['coh_ratio']:<5.3f} "
              f"{100 * clip0:5.1f}>{100 * clip1:<5.1f}")
        if save_dir is not None:
            from PIL import Image
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            for suff, im in (("stock", d0.image), ("undroop", d1.image)):
                arr = (np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)
                Image.fromarray(arr, "L").save(p / f"{fid}_{suff}.png")
    c0 = np.array([r[2]["composite"] for r in rows])
    c1 = np.array([r[3]["composite"] for r in rows])
    worse = [r[0] for r in rows if r[3]["composite"] < r[2]["composite"] - 0.5]
    print(f"AGGREGATE mean composite: stock {c0.mean():.1f} -> undrooped "
          f"{c1.mean():.1f} ({c1.mean() - c0.mean():+.1f}); worse by >0.5 on "
          f"{len(worse)} frames ({', '.join(worse) or 'none'})")
    if "L000" in images:
        for tag, im in zip(("stock", "undrooped"), images["L000"]):
            cm = quality.circle_metrics(im)
            print(f"CIRCLE L000 {tag:9}: ok={cm['ok']} ratio {cm['axis_ratio']:.4f} "
                  f"rms {cm['radial_rms']:.2f} centre ({cm['center_dx']:+.1f},"
                  f"{cm['center_dy']:+.1f}) cov {cm['coverage_deg']:.0f} deg")


# --------------------------------------------------------------------------


SECTIONS = {
    "field": validate_field,
    "gap": validate_gap,
    "shadow": validate_shadow,
    "stability": validate_stability,
    "testset": validate_testset,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="invert and validate the decay-bias pole")
    ap.add_argument("--section", default="",
                    help=f"comma list of {','.join(SECTIONS)} (default: all)")
    ap.add_argument("--save-dir", default=None,
                    help="testset: also dump stock/undrooped PNG pairs here")
    args = ap.parse_args(argv)
    wanted = [s.strip() for s in args.section.split(",") if s.strip()] or list(SECTIONS)
    for name in wanted:
        if name == "testset":
            validate_testset(args.save_dir)
        else:
            SECTIONS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
