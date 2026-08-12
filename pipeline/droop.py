"""Characterise the "decay bias" (Ron Barry 2017): the distortion that makes
every decoded frame look embossed. Measurement suite -- run before inverting.

    python -m pipeline.droop            # run every measurement, print all
    python -m pipeline.droop --section step,raster,porch,preframe,plateau,invert

Every number below was measured by this module running against
data/master/384kHzStereo.wav (384 kHz). All times are 384 kHz samples unless
marked. The landmark and per-trace geometry come from pipeline/geometry.py
(find_marks, the sync trailing-edge zero crossing).

VERDICT (2026-08, this module)
------------------------------
The decay bias is a SINGLE-POLE AC-COUPLING HIGH-PASS acting on the signal in
raster time, downstream of the scan converter, with a DIFFERENT time constant
per audio channel:

    tau_LEFT  = 530 samples at 384 kHz   (stat +-50 over frames;
                systematic range 460..740 -- see the hum degeneracy below)
    tau_RIGHT = 295 samples              (stat +-10; systematic 275..430)

    i.e. fc = 384000/(2*pi*tau):  L ~115 Hz (83..133), R ~207 Hz (142..222)
    per dot (12.17 samples):      L ~43.5 dots, R ~24.2 dots
    per 96 kHz sample:            L ~132.5,     R ~73.75
    alpha = exp(-1/tau), 384 kHz: L 0.99811,    R 0.99662

Five independent measurements agree (details in each section):
  1. THE SYNC-PARITY PULSE RESPONSE (the key instrument). The sync pulse
     width alternates by parity (wide ~145 samples high, narrow ~45), so
     every second trace injects a known extra pulse area into the chain.
     The even-minus-odd median trace profile is therefore the chain's
     response to a deterministic, scene-free excitation. It decays as a
     SINGLE exponential: on content-free gap lines the exp(+locked-hum)
     fit leaves rms ~2e-4 against an amplitude of 0.03-0.06 (SNR ~150).
     A power law is 3-5x worse on the cleanest gaps and never wins; a
     second exponential improves rms <=10%. tau(L gaps) 383-604 hum-fit /
     654-737 exp-only, tau(R gaps) 284-302 / 314-333, stable over the
     whole record (L040 and R077 gaps are damaged and excluded by the
     trimmed mean). CONTROL: L000's gap carries no sync pulses, and its
     gap parity profile shows NO decay component -- no excitation, no
     response.
  2. RASTER-ORDER CONTINUATION. The differing pulse sits at the END of
     trace i (bins 3048..3155 of 3200); the response is measured in trace
     i+1 -- largest in its back porch (bins 30..230) and decaying through
     its picture. The measured porch parity offset equals A*exp(-160/tau)
     of the same fit. A per-trace mechanism cannot do this.
  3. PORCH TRANSFER. The back porch is generated content-free by the
     converter, yet its recorded level regresses on the previous trace's
     picture brightness with the negative sign and magnitude the pole
     predicts. (This also kills "vidicon lag/shading" as the cause: the
     camera cannot move a porch that is synthesised after the camera.)
  4. WITHIN-PLATEAU SLOPE. Every ~12.17-sample sample-and-hold plateau
     drifts with slope = -level/tau_eff; regressing slope on level over
     ~114k plateaus/frame gives tau_eff 551-661 (L), 273-430 (R): the SAME
     pole. No separate converter droop is needed; the residual extra
     within-plateau decay rate is bounded by |d(1/tau)| < ~4e-4/sample.
  5. INVERSE-FILTER FLATTENING. Applying y[n] = y[n-1] + x[n] - a*x[n-1]
     (the exact inverse of H(z) = (1-z^-1)/(1-a*z^-1)) to the raw signal
     in raster order flattens the gap parity profile by 5-12x, to near the
     measurement floor, at tau ~500-530 (L) / ~280-300 (R). The picture
     step-edge ensemble, which decays to ~0 by 40 dots raw, holds near 1.

STEP RESPONSE SETTLING: the chain settles within ONE trace. At the scan stop
(trace ~534) the porch steps to its new steady level between two consecutive
same-parity traces and creeps <0.001 afterwards (<0.5% of the black-white
range): there is NO second slow pole at multi-trace timescales. Slow drift
is removed by the existing per-trace porch clamp anyway; what the clamp
CANNOT remove -- and what embosses the images -- is the within-trace decay,
tau = 0.17P (L) / 0.09P (R).

THE HUM DEGENERACY (the dominant systematic). The scan-locked 60 Hz hum has
a period of exactly 2 traces, so it lives in the SAME even-minus-odd profile
as the parity response, as a half-cycle sinusoid across the trace. Fitting
exp only: tau_L ~694, tau_R ~323. Fitting exp + locked half-sine: tau_L ~530,
tau_R ~295. The inverse-filter flattening optimum and the (hum-immune)
plateau-slope regression both favour the lower values, which are adopted.

WHY THE EARLIER ATTEMPTS FAILED (measured, not guessed):
  * decode._fit_tau minimises the tilt of the mean along-trace profile;
    that profile's tilt is dominated by genuine vertical scene structure,
    so the objective is degenerate and the fit rails (tau_samples == 0.0
    on all 156 frames of data/build_report.json -- the correction is dead
    code). Fitting tau=292.6 rows on L000 failed the same way: L000 is
    also the one frame whose gap carries no pulses and whose drop-score
    landmark is picture structure (sync.py docstring), the worst possible
    calibration frame for this.
  * One global tau cannot work: the two channels genuinely differ by ~1.8x.
  * NEGATIVE RESULT kept for the record: CONTENT-based causality
    regressions of the porch on picture windows fail in every formulation
    tried, each for a measured reason. (i) A bank of 12 lagged 800-sample
    window means: windows one period apart carry nearly identical scene
    content, the design matrix is ill-conditioned, and the fitted kernel
    changes sign frame to frame. (ii) Porch vs the forward leaky mean of
    the undrooped signal (model slope -1): the scan-locked hum envelope
    survives per-parity demedianing and even a per-parity running-median
    detrend, dominates the porch variance on line-art frames, and drags
    the slope to -0.2..0 with a spurious positive acausal partial.
    (iii) 2-trace-differenced windows (which cancel parity-locked hum
    exactly): differencing induces its own MA correlation between
    neighbouring lags, and the omitted-lag bias flips coefficient signs
    (measured b_future +0.3..+0.4 on R frames). The deterministic parity
    probe replaced all three; measure_raster_order() prints formulation
    (iii)'s numbers so the failure stays on the record.
  * NEGATIVE RESULT: the picture step-edge ensemble (find flat-run edges in
    dot space, average aligned responses) is scene-contaminated: median
    response overshoots to ~1.4 at 5 dots and the up/down-edge splits
    disagree, because real scenes are not flat after an edge. It is kept
    only as a qualitative check (decay to ~0 by 40 dots raw; holds ~1 when
    inverse-filtered). Do not fit tau to it.
  * The "~20-line constant black block ending at trace 0" that this task
    description promised DOES NOT EXIST on any frame measured: traces
    -22..-1 before trace 0 are the high-contrast diagonal-hatch MARKER
    (picture-band std 0.05-0.10, vs 0.002-0.008 in the gap), exactly as
    geometry.py documented. Settling is measured from the scan-stop step
    and the stationary gap instead.

WHAT THIS PREDICTS THAT COMPETING MODELS DO NOT -- see model_summary().
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import dotclock, geometry

# Adopted constants (see module docstring for the evidence and the ranges).
TAU_384 = {"L": 530.0, "R": 295.0}  # samples at 384 kHz
TAU_STAT = {"L": 50.0, "R": 10.0}  # frame-to-frame spread, same fit
TAU_SYST = {"L": (460.0, 740.0), "R": (275.0, 430.0)}  # across methods

PERIOD = geometry.NOMINAL_PERIOD  # ~3197.4; per-trace value comes from marks
BINS = geometry.BINS  # 3200 profile bins, 1 bin ~ 0.999 samples

# Survey frames. Gap probes need ordinary gaps: L000 (no gap pulses) and
# R000 (sync losses) are excluded from the fits and L000 is kept as the
# no-excitation control.
GAP_FIDS = ["L010", "L020", "L030", "L040", "L055", "L065", "L077",
            "R010", "R020", "R030", "R040", "R056", "R065", "R077"]
FRAME_FIDS = ["L055", "L034", "R040", "R056"]

_CACHE: dict = {}


def load(fid: str, pre_traces: float, n_traces: int):
    """(signal, marks) from the master, cached. Marks via geometry.find_marks."""
    key = (fid, round(pre_traces, 3), n_traces)
    if key not in _CACHE:
        x, _ = geometry._load(fid, pre_traces=pre_traces, n_traces=n_traces)
        m, meas = geometry.find_marks(x)
        _CACHE[key] = (x, m, meas)
    return _CACHE[key]


def inverse_pole(x: np.ndarray, tau: float) -> np.ndarray:
    """Exact inverse of the one-pole high-pass H(z) = (1-z^-1)/(1-a*z^-1),
    a = exp(-1/tau): y[n] = y[n-1] + x[n] - a*x[n-1]. Applied to the RAW
    signal in raster order (sample order IS raster order). DC is a free
    constant of integration; downstream porch clamping pins it per trace."""
    a = float(np.exp(-1.0 / tau))
    x = np.asarray(x, dtype=np.float64)
    d = x.copy()
    d[1:] -= a * x[:-1]
    return np.cumsum(d)


def parity_profile(x: np.ndarray, m: np.ndarray, k_lo: int, k_hi: int) -> np.ndarray:
    """even-median minus odd-median normalized trace profile (3200 bins) over
    trace slots [k_lo, k_hi). This is the chain's response to the alternating
    sync pulse-width difference, plus the scan-locked (2-trace-period) hum."""
    M = geometry.norm_traces(x, m)
    hi = min(k_hi, len(M))
    M = M[k_lo:hi]
    ok = ~np.isnan(M[:, 0])
    M = M[ok]
    ks = np.arange(k_lo, hi)[ok]
    if (ks % 2 == 0).sum() < 8 or (ks % 2 == 1).sum() < 8:
        raise ValueError("too few traces for a parity profile")
    return np.median(M[ks % 2 == 0], axis=0) - np.median(M[ks % 2 == 1], axis=0)


def _smooth(h: np.ndarray, n: int = 15) -> np.ndarray:
    k = np.ones(n) / n
    return np.convolve(h, k, mode="same")


def fit_parity_decay(h: np.ndarray, lo: int = 30, hi: int = 3030) -> dict:
    """Fit the decay of a parity profile over bins [lo, hi).

    Models, all with a free constant:
      exp      A*exp(-b/tau)
      exp+hum  A*exp(-b/tau) + B1*sin(pi*b/3200) + B2*cos(pi*b/3200)
               (the scan-locked hum's period is exactly 2 traces, so across
               ONE trace it is half a sinusoid cycle -- see module docstring)
      exp2     A1*exp(-b/tau1) + A2*exp(-b/tau2)
      pow      A*(b+b0)^-p
    Returns per-model (rms, params). tau grids are log-spaced; the linear
    parameters are solved exactly per grid point.
    """
    hs = _smooth(h)
    bb = np.arange(lo, hi, dtype=np.float64)
    seg = hs[lo:hi]
    n = len(seg)
    th = np.pi * bb / float(BINS)
    out = {}

    def solve(X):
        c, *_ = np.linalg.lstsq(X, seg, rcond=None)
        return c, float(np.sqrt(((seg - X @ c) ** 2).sum() / n))

    taus = np.geomspace(80, 30000, 300)
    best = None
    for tau in taus:
        c, rms = solve(np.column_stack([np.exp(-bb / tau), np.ones(n)]))
        if best is None or rms < best[0]:
            best = (rms, tau, c)
    out["exp"] = {"rms": best[0], "tau": best[1], "A": best[2][0], "C": best[2][1]}

    best = None
    for tau in taus:
        X = np.column_stack([np.exp(-bb / tau), np.sin(th), np.cos(th), np.ones(n)])
        c, rms = solve(X)
        if best is None or rms < best[0]:
            best = (rms, tau, c)
    out["exp+hum"] = {"rms": best[0], "tau": best[1], "A": best[2][0],
                      "hum": float(np.hypot(best[2][1], best[2][2])), "C": best[2][3]}

    t2grid = np.geomspace(60, 30000, 60)
    best = None
    for i, t1 in enumerate(t2grid):
        E1 = np.exp(-bb / t1)
        for t2 in t2grid[i + 2:]:
            X = np.column_stack([E1, np.exp(-bb / t2), np.ones(n)])
            c, rms = solve(X)
            if best is None or rms < best[0]:
                best = (rms, (t1, t2), c)
    out["exp2"] = {"rms": best[0], "taus": best[1], "A": (best[2][0], best[2][1])}

    best = None
    for b0 in (1.0, 30.0, 100.0, 300.0):
        for p in np.linspace(0.3, 3.0, 28):
            X = np.column_stack([(bb + b0) ** (-p), np.ones(n)])
            c, rms = solve(X)
            if best is None or rms < best[0]:
                best = (rms, (b0, p), c)
    out["pow"] = {"rms": best[0], "b0_p": best[1]}
    out["tail_rms"] = float(np.std(hs[1500:3000]))
    return out


# --------------------------------------------------------------------------
# Q1: the actual step response, in the signal domain
# --------------------------------------------------------------------------


def gap_slots(m: np.ndarray) -> tuple[int, int]:
    """Trace slots [lo, hi) that are pure inter-frame gap, for a load with
    pre_traces=195: everything up to 30 slots before trace 0 (the marker is
    22 traces; 30 leaves margin)."""
    k0 = int(np.argmin(np.abs(m - 195.0 * PERIOD)))
    return 4, max(5, k0 - 30)


def measure_step_response() -> dict:
    """Q1. The chain's decay after a known excitation, measured two ways.

    PRIMARY: the sync-parity pulse response on content-free gap lines (full
    time resolution, deterministic excitation). Reports per-frame model fits.
    SECONDARY (qualitative only, scene-contaminated): median normalized
    response after real picture step edges, in dot space.
    """
    print("Q1 == STEP RESPONSE (signal domain) " + "=" * 40)
    print("primary probe: sync-parity pulse response on gap lines")
    print(f"{'frame':6} {'exp tau':>8} {'rms':>8} | {'exp+hum tau':>11} {'rms':>8} "
          f"{'A':>8} {'hum':>7} | {'exp2 rms':>8} {'pow rms':>8} {'tail':>8}")
    taus = {"L": [], "R": []}
    for fid in GAP_FIDS:
        try:
            x, m, _ = load(fid, 195.0, 60)
            lo, hi = gap_slots(m)
            h = parity_profile(x, m, lo, hi)
            r = fit_parity_decay(h)
        except Exception as exc:
            print(f"{fid:6} FAILED: {exc}")
            continue
        e, eh = r["exp"], r["exp+hum"]
        print(f"{fid:6} {e['tau']:8.0f} {e['rms']:8.5f} | {eh['tau']:11.0f} "
              f"{eh['rms']:8.5f} {eh['A']:+8.4f} {eh['hum']:7.4f} | "
              f"{r['exp2']['rms']:8.5f} {r['pow']['rms']:8.5f} {r['tail_rms']:8.5f}")
        taus[fid[0]].append(eh["tau"])
    # the no-excitation control
    try:
        x, m, _ = load("L000", 195.0, 60)
        lo, hi = gap_slots(m)
        r = fit_parity_decay(parity_profile(x, m, lo, hi))
        print(f"L000   CONTROL (gap has no sync pulses): exp A {r['exp']['A']:+.4f} "
              f"tau {r['exp']['tau']:.0f} (railed/none expected) rms {r['exp']['rms']:.5f}")
    except Exception as exc:
        print(f"L000   control failed: {exc}")
    res = {}
    for ch in "LR":
        a = np.asarray(sorted(taus[ch])[1:-1] if len(taus[ch]) > 4 else taus[ch])
        res[ch] = (float(a.mean()), float(a.std()))
        print(f"channel {ch}: tau (exp+hum, trimmed) mean {a.mean():.0f} sd {a.std():.0f} n {len(a)}")
    print("shape verdict: single exponential. Power law 3-5x worse rms on the")
    print("cleanest gaps, never better; a second exponential improves <=10%.")
    print("ONE pole per channel, high confidence on shape; tau carries the")
    print("hum-degeneracy systematic (exp-only fits run ~30-40% higher).")

    print("\nsecondary (QUALITATIVE -- scene-contaminated, do not fit): picture")
    print("step-edge ensemble, median normalized response at N dots after edge")
    for fid in ("L055", "R040", "R056"):
        x, m, _ = load(fid, 3.0, 515)
        D = _dot_matrix(x, m)
        R, S = _step_ensemble(D)
        if len(R) < 25:
            print(f"  {fid}: only {len(R)} usable edges")
            continue
        med = np.median(R, axis=0)
        y = inverse_pole(x, TAU_384[fid[0]])
        Ri, _ = _step_ensemble(_dot_matrix(y, m))
        mi = np.median(Ri, axis=0) if len(Ri) >= 25 else np.full_like(med, np.nan)
        print(f"  {fid} ({len(R):3d} edges) raw:       @2 {med[1]:+.2f} @5 {med[4]:+.2f} "
              f"@10 {med[9]:+.2f} @20 {med[19]:+.2f} @40 {med[39]:+.2f}")
        print(f"  {fid} {'':11} undrooped: @2 {mi[1]:+.2f} @5 {mi[4]:+.2f} "
              f"@10 {mi[9]:+.2f} @20 {mi[19]:+.2f} @40 {mi[39]:+.2f}")
    print("  raw response decays toward 0 by ~40 dots; inverse-filtered holds")
    print("  near 1 (L055's overshoot suggests its own tau sits at the upper end")
    print("  of the L systematic band -- its frame-body exp-only fit gives ~670).")
    return res


def _dot_matrix(x: np.ndarray, m: np.ndarray, n_max: int = 512) -> np.ndarray:
    """(traces, dots) plateau-integral matrix on the predicted dot pitch."""
    cs = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
    rows = []
    for k in range(4, min(len(m) - 1, n_max)):
        p = m[k + 1] - m[k]
        spd = p / dotclock.DOTS_PER_TRACE
        s0 = m[k] + 232.0 / BINS * p
        nd = int(2808.0 / BINS * p / spd)
        e = s0 + np.arange(nd + 1) * spd
        b = np.floor(e).astype(np.int64)
        C = cs[b] + (e - b) * x[b]
        rows.append(np.diff(C) / spd)
    n = min(len(r) for r in rows)
    return np.asarray([r[:n] for r in rows])


def _step_ensemble(D: np.ndarray, min_step: float = 0.03, pre: int = 6,
                   post: int = 45, flat: float = 0.006):
    """Aligned, normalized post-edge responses at flat-run picture edges."""
    resp, steps = [], []
    for tr in range(D.shape[0]):
        d = D[tr]
        dd = np.abs(np.diff(d))
        for k in range(pre + 1, len(d) - post - 2):
            st = d[k + 1] - d[k]
            if abs(st) < min_step or np.std(d[k - pre:k + 1]) > flat:
                continue
            if np.max(dd[k + 1:k + post]) > 0.6 * abs(st) and np.max(dd[k + 1:k + post]) > min_step:
                continue
            resp.append((d[k + 1:k + 1 + post] - d[k - 3:k + 1].mean()) / st)
            steps.append(st)
    return np.asarray(resp), np.asarray(steps)


# --------------------------------------------------------------------------
# Q2: along-trace, across-trace, or raster order?
# --------------------------------------------------------------------------


def measure_raster_order() -> None:
    """Q2. Is the mechanism a temporal LTI filter crossing trace boundaries
    in raster order? Deterministic probes say yes:

    (a) The parity pulse response: the differing pulse area sits at the END
        of trace i (bins 3048..3155); the response is read in trace i+1,
        largest in its back porch and decaying through its picture. The
        measured porch parity offset must equal A*exp(-160/tau) of the
        decay fitted further along the SAME profile.
    (b) L000 control: its gap has no pulse-width alternation, and shows no
        decay response (printed under Q1).
    (c) The scan-stop step (printed under Q4): when the excitation changes
        at trace ~534, the porch of the FOLLOWING gap traces lands on a new
        level -- the response crosses the boundary and settles within one
        trace.

    Also prints the best content-based causality regression tried
    (2-trace-differenced windows, which cancel the parity-locked hum
    exactly) as the documented NEGATIVE result: scene autocorrelation plus
    differencing-induced MA structure make its coefficients uninterpretable
    (see module docstring). Do not build on it.
    """
    print("\nQ2 == RASTER ORDER " + "=" * 55)
    print("(a) response continues across the trace boundary (gap lines):")
    for fid in ("L030", "L055", "R020", "R040"):
        x, m, _ = load(fid, 195.0, 60)
        lo, hi = gap_slots(m)
        h = parity_profile(x, m, lo, hi)
        r = fit_parity_decay(h)["exp+hum"]
        hs = _smooth(h, 25)
        porch_meas = float(np.median(hs[100:225]))
        porch_pred = r["A"] * np.exp(-160.0 / r["tau"])
        print(f"  {fid}: pulse at end of trace i -> porch of trace i+1 moves "
              f"{porch_meas:+.4f}; A*exp(-160/tau) predicts {porch_pred:+.4f}")
    print("  start-of-next-trace level tracks end-of-previous-trace excitation,")
    print("  with the amplitude the within-trace decay fit predicts (3-20%).")
    print("  (b) no-excitation control: see the L000 line under Q1.")
    print("  (c) boundary-crossing at the scan stop: see Q4.")

    print("(d) NEGATIVE RESULT, kept honest: content-based causality regression")
    print("    (2-trace-differenced porch vs prev-trace-end and own-picture means):")
    W = 800
    for fid in FRAME_FIDS:
        x, m, _ = load(fid, 3.0, 515)
        n = min(len(m) - 1, 512)
        porch = np.full(n, np.nan)
        past = np.full(n, np.nan)
        fut = np.full(n, np.nan)
        for k in range(6, n):
            p = m[k + 1] - m[k]
            pa, pb = int(m[k] + 100 / 3200 * p), int(m[k] + 225 / 3200 * p)
            pe = int(m[k - 1] + 3040 / 3200 * p)  # prev picture end
            fs = int(m[k] + 232 / 3200 * p)
            if pe - W < 0 or fs + W > len(x):
                continue
            porch[k] = float(np.median(x[pa:pb]))
            past[k] = float(x[pe - W:pe].mean())
            fut[k] = float(x[fs:fs + W].mean())
        idx = np.arange(8, n)
        dP = porch[idx] - porch[idx - 2]
        dPa = past[idx] - past[idx - 2]
        dF = fut[idx] - fut[idx - 2]
        ok = np.isfinite(dP) & np.isfinite(dPa) & np.isfinite(dF)
        dP, dPa, dF = dP[ok], dPa[ok], dF[ok]
        med = np.median(dP)
        mad = 1.4826 * np.median(np.abs(dP - med)) + 1e-12
        k = np.abs(dP - med) < 8 * mad
        dP, dPa, dF = dP[k], dPa[k], dF[k]
        X = np.column_stack([dPa, dF])
        c, *_ = np.linalg.lstsq(X, dP, rcond=None)
        res = dP - X @ c
        cov = float((res ** 2).sum() / (len(dP) - 2)) * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        tau = TAU_384[fid[0]]
        b_pred = -(tau / W) * (1 - np.exp(-W / tau)) * np.exp(-322.0 / tau)
        print(f"    {fid}: b_past {c[0]:+.3f}+-{se[0]:.3f} (pole would predict "
              f"{b_pred:+.3f})  b_future {c[1]:+.3f}+-{se[1]:.3f} (causal: 0)")
    print("    coefficients disagree with ANY causal kernel because the")
    print("    predictors are not independent probes: scene autocorrelation +")
    print("    differencing-induced MA structure. The deterministic probes")
    print("    (a)-(c) are the raster-order evidence; this one is recorded so")
    print("    nobody re-invents it.")


# --------------------------------------------------------------------------
# Q3: how much does the blanking shelf move?
# --------------------------------------------------------------------------


def measure_porch_transfer() -> None:
    """Q3. Porch level vs preceding-trace mean picture brightness.

    The porch is constant at the source; any correlation of its recorded
    level with preceding content is pure distortion. NOTE ON UNITS: the
    regressor (the RECORDED previous-trace mean) is itself the pole's
    output, i.e. high-passed -- slow scene brightness is heavily attenuated
    in it while the porch deviation (-1 x the leaky mean of the source)
    carries the slow scene brightness at nearly full amplitude. The slope
    is therefore expected to EXCEED 1 in magnitude; the physical statement
    is its SIGN plus the correlation, and the model's absolute prediction:
    the pole blocks DC completely, so a sustained bright field displaces
    the following porch by the full brightness deviation, reached with the
    channel tau (95% inside ~3*tau = half a trace on L)."""
    print("\nQ3 == PORCH (BLANKING SHELF) TRANSFER " + "=" * 36)
    for fid in FRAME_FIDS:
        x, m, _ = load(fid, 3.0, 515)
        n = min(len(m) - 1, 512)
        porch, pm, par = [], [], []
        for k in range(6, n):
            p = m[k + 1] - m[k]
            pa, pb = int(m[k] + 100 / 3200 * p), int(m[k] + 225 / 3200 * p)
            a, b = int(m[k - 1] + 232 / 3200 * p), int(m[k - 1] + 3040 / 3200 * p)
            porch.append(float(np.median(x[pa:pb])))
            pm.append(float(x[a:b].mean()))
            par.append(k % 2)
        porch, pm, par = np.asarray(porch), np.asarray(pm), np.asarray(par)
        for pp in (0, 1):
            porch[par == pp] -= np.median(porch[par == pp])
        pm -= pm.mean()
        med = np.median(porch)
        mad = 1.4826 * np.median(np.abs(porch - med)) + 1e-12
        k = np.abs(porch - med) < 8 * mad
        porch, pm = porch[k], pm[k]
        b = float((pm * porch).sum() / (pm * pm).sum())
        r = float((pm * porch).sum() / np.sqrt((pm * pm).sum() * (porch * porch).sum()))
        se = float(np.sqrt(((porch - b * pm) ** 2).sum() / (len(pm) - 2) / (pm * pm).sum()))
        prng = float(porch.max() - porch.min())
        print(f"  {fid}: dPorch/d(recorded prev-trace mean) {b:+.2f} +- {se:.2f}  "
              f"r {r:+.2f}  porch wander p-p {prng:.4f} "
              f"({100 * prng / 0.24:.0f}% of the black-white range)")
    print("  negative on all four frames: brighter preceding content pushes the")
    print("  recorded porch down -- Barry's shadow/anti-shadow, measured on the")
    print("  one signal that is constant at the source. |slope|>1 is expected")
    print("  (see docstring: the regressor is high-passed, the porch is not).")


# --------------------------------------------------------------------------
# Q4: the region before trace 0, and settling with no picture content
# --------------------------------------------------------------------------


def measure_preframe() -> None:
    """Q4. What actually precedes trace 0, and how the chain settles when
    the excitation changes (scan stop), measured content-free."""
    print("\nQ4 == PRE-FRAME REGION AND SETTLING " + "=" * 38)
    print("is there a constant black block before trace 0?  (per-trace picture-band std)")
    for fid in ("L002", "L034", "L055", "R010", "R040", "R056"):
        x, m, _ = load(fid, 40.0, 40)
        k0 = int(np.argmin(np.abs(m - 40.0 * PERIOD)))
        gap_std, mark_std = [], []
        for k in range(max(0, k0 - 40), k0):
            p = m[k + 1] - m[k]
            a, b = int(m[k] + 232 / 3200 * p), int(m[k] + 3040 / 3200 * p)
            if a < 0 or b > len(x):
                continue
            (mark_std if k >= k0 - 22 else gap_std).append(float(np.std(x[a:b])))
        print(f"  {fid}: gap std {np.median(gap_std):.4f}  traces -22..-1 std "
              f"{np.median(mark_std):.4f} (min {np.min(mark_std):.4f})"
              f"  -> {'HATCH MARKER, not constant' if np.median(mark_std) > 0.02 else 'quiet'}")
    print("  NO constant black block exists; the 20-odd lines before trace 0 are")
    print("  the diagonal-hatch marker (geometry.py). Settling is measured below.")

    print("settling at the scan stop (~trace 534; porch level, same-parity steps):")
    for fid in ("L055", "R040"):
        x, m, _ = load(fid, 5.0, 620)
        k0 = int(np.argmin(np.abs(m - 5.0 * PERIOD)))
        porch = {}
        for k in range(k0 + 520, min(k0 + 580, len(m) - 1)):
            p = m[k + 1] - m[k]
            pa, pb = int(m[k] + 100 / 3200 * p), int(m[k] + 225 / 3200 * p)
            porch[k - k0] = float(np.median(x[pa:pb]))
        pre = np.median([v for t, v in porch.items() if 521 <= t <= 533 and t % 2 == 1])
        step_tr = 535
        post0 = porch.get(step_tr, np.nan)
        post = np.median([v for t, v in porch.items() if 541 <= t <= 579 and t % 2 == 1])
        creep = abs(post0 - post)
        print(f"  {fid}: porch (odd parity) {pre:+.4f} before stop -> {post0:+.4f} at "
              f"trace {step_tr} -> {post:+.4f} settled; step completes in <1 trace, "
              f"creep after {creep:.4f} ({100 * creep / 0.24:.1f}% of B-W range)")
    print("  no slow second pole at multi-trace timescales: consistent with")
    print("  tau ~0.1-0.2 trace, which finishes settling inside one period.")


# --------------------------------------------------------------------------
# Q5: separate the mechanisms -- converter S&H droop vs chain AC coupling
# --------------------------------------------------------------------------


def measure_plateau_droop() -> None:
    """Q5. Within-plateau slope vs plateau level. Under pure chain AC
    coupling, slope = -level/tau with the SAME tau as the long decay; a
    separate converter sample-and-hold droop would add to the rate."""
    print("\nQ5 == WITHIN-PLATEAU SLOPE vs LONG-TIMESCALE DECAY " + "=" * 23)

    class _Shim:
        def __init__(self, m, n):
            self.m, self.period, self.n_traces = m, float(np.median(np.diff(m))), n

        def trace_start(self, i):
            return float(self.m[i])

    for fid in FRAME_FIDS:
        x, m, _ = load(fid, 3.0, 515)
        n_tr = min(len(m) - 1, 512)
        tb = _Shim(m, n_tr)
        clock = dotclock.measure(x, tb, lo_f=232 / 3200 + 0.01, hi_f=3040 / 3200 - 0.01)
        spd = clock.samples_per_dot
        psi, _ = dotclock.track_phase(x, m[:n_tr], tb.period, spd, 232 / 3200, 3040 / 3200)
        slopes, means, steps = [], [], []
        for i in range(4, n_tr):
            a = m[i] + 232 / 3200 * tb.period
            b = m[i] + 3040 / 3200 * tb.period
            k0 = int(np.ceil((a - psi[i]) / spd))
            k1 = int(np.floor((b - psi[i]) / spd)) - 1
            prev = None
            for k in range(k0, k1):
                e0 = psi[i] + k * spd
                i0, i1 = int(np.ceil(e0 + 3.0)), int(np.floor(e0 + spd - 2.0))
                if i1 - i0 < 4:
                    prev = None
                    continue
                seg = x[i0:i1 + 1]
                t = np.arange(len(seg), dtype=np.float64)
                t -= t.mean()
                sl = float((t * (seg - seg.mean())).sum() / (t * t).sum())
                mu = float(seg.mean())
                if prev is not None:
                    slopes.append(sl)
                    means.append(mu)
                    steps.append(mu - prev)
                prev = mu
        S, M, T = (np.asarray(v) for v in (slopes, means, steps))
        ok = (np.abs(S) < np.percentile(np.abs(S), 99)) & (np.abs(M) < 0.5)
        S, M, T = S[ok], M[ok], T[ok]
        X = np.column_stack([M, T, np.ones(len(M))])
        c, *_ = np.linalg.lstsq(X, S, rcond=None)
        r = S - X @ c
        cov = float((r ** 2).sum() / (len(S) - 3)) * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        tau_eff = -1.0 / c[0] if c[0] < 0 else float("inf")
        tau_ch = TAU_384[fid[0]]
        extra = abs(c[0]) - 1.0 / tau_ch
        print(f"  {fid}: n {len(S)} plateaus  slope/level {c[0]:+.6f}+-{se[0]:.6f} "
              f"per sample -> tau_eff {tau_eff:5.0f} (chain tau {tau_ch:.0f}; "
              f"excess rate {extra:+.1e}/sample)   edge-settle term {c[1]:+.4f} "
              f"(the ~2-sample channel rise, not droop)"
              + ("   [dot lock weak]" if clock.strength < 1.8 else ""))
    print("  the within-plateau decay rate IS the chain pole. Any separate S&H")
    print("  droop is bounded by |excess| < ~4e-4/sample (<0.5%/dot).")


# --------------------------------------------------------------------------
# validation: the inverse filter against the content-free probe
# --------------------------------------------------------------------------


def measure_inversion() -> None:
    """Apply inverse_pole at candidate taus; the gap parity profile must
    flatten to the hum/noise floor at the correct tau."""
    print("\nVALIDATION == INVERSE FILTER FLATTENS THE CONTENT-FREE PROBE " + "=" * 13)

    def dehum_rms(h):
        hs = _smooth(h, 25)
        bb = np.arange(30, 3030, dtype=np.float64)
        th = np.pi * bb / float(BINS)
        X = np.column_stack([np.sin(th), np.cos(th), np.ones(len(bb))])
        seg = hs[30:3030]
        c, *_ = np.linalg.lstsq(X, seg, rcond=None)
        return float(np.sqrt(((seg - X @ c) ** 2).mean()))

    for fid in ("L030", "L055", "R020", "R040"):
        x, m, _ = load(fid, 195.0, 60)
        lo, hi = gap_slots(m)
        raw = dehum_rms(parity_profile(x, m, lo, hi))
        line = f"  {fid}: raw {raw:.5f}"
        cands = (450, 500, 530, 600, 700) if fid[0] == "L" else (260, 280, 295, 320, 360)
        best = (np.inf, 0)
        for tau in cands:
            v = dehum_rms(parity_profile(inverse_pole(x, tau), m, lo, hi))
            line += f"  tau{tau} {v:.5f}"
            if v < best[0]:
                best = (v, tau)
        print(line + f"   -> best tau {best[1]} ({raw / best[0]:.1f}x flatter)")
    print("  the adopted taus sit at/near the flattening optimum on both channels.")


# --------------------------------------------------------------------------
# Q6: the model
# --------------------------------------------------------------------------


def model_summary(step_res: dict | None = None) -> None:
    print("\nQ6 == BEST PHYSICAL MODEL " + "=" * 48)
    for ch in "LR":
        tau = TAU_384[ch]
        lo, hi = TAU_SYST[ch]
        print(f"  channel {ch}: one-pole AC-coupling high-pass, "
              f"tau = {tau:.0f} samples at 384 kHz (+-{TAU_STAT[ch]:.0f} stat, "
              f"{lo:.0f}..{hi:.0f} syst) = {tau / 12.17:.1f} dots = "
              f"fc {384000 / (2 * np.pi * tau):.0f} Hz;  alpha_384 = "
              f"{np.exp(-1.0 / tau):.5f}, alpha_96 = {np.exp(-4.0 / tau):.5f}, "
              f"alpha_dot = {np.exp(-12.17 / tau):.4f}")
    print("""  acting on the signal in RASTER TIME (along the trace, continuing
  across every trace boundary), downstream of the scan converter --
  i.e. tape/cutting/playback electronics, NOT the camera and NOT the
  converter's sample-and-hold. The two channels genuinely differ (~1.8x),
  so tau must be per-channel; one global constant is wrong by construction.

  what this model predicts that competitors do not:
    * temporal one-pole: start-of-trace level depends on end-of-previous-
      trace excitation with the SAME tau as the within-trace decay --
      measured (Q2a: next-trace porch offset = A*exp(-160/tau) to 3-20%).
      A per-trace effect (converter reset, fixed line shading) predicts no
      cross-boundary dependence: refuted.
    * downstream of the camera: the back porch is synthesised by the
      converter, content-free; vidicon lag/shading cannot move it, yet it
      moves against preceding picture brightness on every frame (Q3), and
      the decay appears in the GAP where no camera picture exists at all:
      vidicon origin refuted (for THIS artefact; camera softness is
      separate, see decode.py PSF).
    * single pole: the decay is one exponential to <=10% (Q1); RIAA-type
      multi-corner EQ error would leave comparable second components and
      identical response on both channels: refuted.
    * chain (not converter S&H) droop: within-plateau slope carries the
      same tau (Q5); a converter droop would decay WITHIN plateaus and
      reset at each transition, decoupled from the trace-scale decay.
    * no slow second pole: scan-stop settling completes within one trace
      (Q4); the per-trace porch clamp already removes anything slower.

  consequences for correction (decode.py, future work -- NOT changed here):
    invert with y[n] = y[n-1] + x[n] - a*x[n-1] on the RAW signal in raster
    order at full rate, a = exp(-1/tau_channel), BEFORE dot integration and
    porch clamping; the porch clamp then pins the integrator's free DC
    per trace. Do not fit tau per frame from picture statistics -- that is
    the degenerate objective that railed decode._fit_tau; use the per-
    channel constants above (or refit them per frame from the gap parity
    probe, which is deterministic).""")
    if step_res:
        for ch, (mu, sd) in step_res.items():
            print(f"  (this run re-measured channel {ch}: tau {mu:.0f} +- {sd:.0f})")


# --------------------------------------------------------------------------


SECTIONS = {
    "step": measure_step_response,
    "raster": measure_raster_order,
    "porch": measure_porch_transfer,
    "preframe": measure_preframe,
    "plateau": measure_plateau_droop,
    "invert": measure_inversion,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="measure the decay-bias distortion")
    ap.add_argument("--section", default="",
                    help=f"comma list of {','.join(SECTIONS)} (default: all)")
    args = ap.parse_args(argv)
    wanted = [s.strip() for s in args.section.split(",") if s.strip()] or list(SECTIONS)
    step_res = None
    for name in wanted:
        if name == "step":
            step_res = SECTIONS[name]()
        else:
            SECTIONS[name]()
    model_summary(step_res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
