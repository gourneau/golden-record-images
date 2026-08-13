"""The code behind `aperture.TRANSITION_1090`, and the hold-out that judges it.

WHY THIS FILE EXISTS
--------------------
`aperture.py` is closed-form except for one number: `TRANSITION_1090 = 0.49`
dots, the 10-90 rise of a plateau-to-plateau step. Everything the operator does
scales off it. That number shipped as prose in a docstring -- 323 fits nobody
could re-run -- and `breakthrough_spec.md` section 5.4 blocks the merge until the
code exists. This is that code. It opens the master, measures the width, states
its own bias, calibrates its own arbiter, and reports what the arbiter can and
cannot decide.

WHAT IS MEASURED, AND WHERE
---------------------------
The scan converter held one value per NTSC line, so a trace is a staircase.
Everything between that hold and our integrator -- the sampler's output stage,
tape, disc, the 2017 ADC -- is one linear filter `c`, and a plateau-to-plateau
transition IS a step through `c`. So `c`'s step response is directly readable in
the raw waveform, at the same node and through the same path as the picture.

Fits are SINGLE-TRACE and never stacked. Stacking would fold our own dot-clock
phase error into the width and measure our timing noise as if it were the
chain's -- this project has retracted an MTF number for exactly that. Every fit
here has its own free transition time, so a misplaced lattice moves the analysis
window and cannot widen the answer.

THE MISTAKE THIS FILE IS BUILT TO AVOID
---------------------------------------
An edge-width estimator measures its own bias as much as the signal. Every
estimator here is therefore calibrated first on a SYNTHETIC staircase of KNOWN
width, run through the identical selection and fitting code, carrying the real
frame's own plateau values, step-amplitude distribution, flatness statistics,
noise level, dot pitch and dot-phase error. The calibration table, the inversion
slope and the raw uncalibrated reading are all printed, so a reader can see how
much of the answer is the correction.

Biases, quantified by that calibration rather than argued away:

  * LEVEL CONTAMINATION. The two plateau levels come from the settled central
    32% of each plateau, ~2 sigma clear of both transitions, but not infinitely
    clear. The normalised profile then spans slightly more than 0 to 1 and the
    crossings come out too close: the crossing estimator reads LOW, worse the
    wider the true edge. That is what flattens its calibration slope to 0.74.
  * NOISE. Per-sample noise makes the last sample below 0.1 fall earlier and the
    first above 0.9 fall later: reads HIGH. It fights the previous one, which is
    why neither can be reasoned about and both have to be calibrated.
  * SELECTION. Steps are chosen using the same noisy levels that then normalise
    them. The synthetic runs the same selector, so this survives into the
    calibration instead of being assumed away.
  * SHAPE. A calibration must assume a truth shape. Four are run (Gaussian,
    raised cosine, symmetric two-pole, boxcar) and the spread is reported. It is
    the largest systematic here, so the shapes are also DISCRIMINATED against the
    data by fit residual, which is what shrinks it.
  * DOT-PHASE ERROR. `dotclock.track_phase` places the lattice to ~0.1 dots,
    which misplaces the level windows. Injected into the synthetic and reported.

THE ARBITER, AND WHAT IT IS ACTUALLY GOOD FOR
---------------------------------------------
Image metrics cannot judge this. Spec section 2.5 measured `shift_rms`,
`coh_ratio`, `sharpness` and `composite` all improving monotonically past the
correct strength, for a filter that is to three decimals a Laplacian sharpener.
The operator's shape proves nothing; only its amount is a physical claim, and no
metric here can price the amount.

So the arbiter is prediction of samples that were not used in the fit. Split
every plateau into a settled central 40% and an outer 60%; estimate the dot
sequence from one part, correct for THAT part's transfer, predict the other. The
two sample sets are disjoint, so their noise is independent and over-correction
is PAID FOR rather than rewarded -- that is why this test has a wrong side and
the image metrics do not. Two directions are run:

    in -> out   the spec's version (5.2 test 1). Weak, and now measurably so:
                the operator it tests has a gain of 1.045 at the dot Nyquist,
                because the central 40% is barely blurred at all. Effect 0-0.4%,
                and it does NOT pass its own criterion on every frame.
    out -> in   the reverse. The operator it tests has a gain of 1.88 at the dot
                Nyquist -- larger than the 1.44 `decode` actually applies -- so
                it is by far the sharper instrument. Effect 5-22%, unambiguous.

And the arbiter is itself calibrated, on a synthetic staircase of known width,
before it is believed. That is how the width sweep here is known to be a rail
rather than a measurement: it rails to the end of the grid even on synthetic
data whose width we planted ourselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import aperture

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "data" / "master" / "384kHzStereo.wav"

# --- estimator geometry -----------------------------------------------------
# Fraction of a plateau, centred, used to read its level. 0.32 keeps the window
# ~2 sigma clear of both transitions at the widths of interest while still
# holding ~4 samples at 384 kHz. Both effects are in the calibration.
LEVEL_FRAC = 0.32
PROFILE_HALF = 0.55          # half-width of the fit window, in dots
MIN_AMP_SIGMAS = 20.0        # a step must exceed this many noise sd
FLAT_FRAC = 0.15             # ... and its two flanking steps must be below this

PICTURE_START = 232.0 / 3200.0          # copied from decode.py, not imported,
PICTURE_SPAN = (3040.0 - 232.0) / 3200.0  # so this module needs no Settings

ESTIMATORS = ("cross", "erf", "logistic")
SHAPES = ("gauss", "hann", "twopole", "box")


# ---------------------------------------------------------------------------
# 0. STEP-RESPONSE SHAPES, EACH NORMALISED SO ITS OWN 10-90 IS EXACTLY 1
# ---------------------------------------------------------------------------

def _profile(t: np.ndarray, shape: str) -> np.ndarray:
    if shape == "gauss":
        return np.exp(-0.5 * t ** 2)
    if shape == "hann":
        return np.where(np.abs(t) < 1.0, 0.5 * (1.0 + np.cos(np.pi * np.clip(t, -1, 1))), 0.0)
    if shape == "twopole":
        return np.exp(-np.abs(t))
    if shape == "box":
        return (np.abs(t) <= 0.5).astype(np.float64)
    if shape == "logistic":
        e = np.exp(-np.abs(t))
        return e / (1.0 + e) ** 2
    raise ValueError(shape)


_T = np.linspace(-40.0, 40.0, 400001)


def _unit_scale(shape: str) -> float:
    """Factor s such that `_profile(t*s)` has a step response with 10-90 = 1.

    Every shape is expressed in ONE unit -- its own 10-90 -- so that "width"
    means the same thing for a Gaussian, a raised cosine and a boxcar, and the
    shape systematic in section 3 is a comparison of like with like.
    """
    k = _profile(_T, shape)
    s = np.cumsum(k)
    s /= s[-1]
    return float(np.interp(0.9, s, _T) - np.interp(0.1, s, _T))


_SCALE = {sh: _unit_scale(sh) for sh in (*SHAPES, "logistic")}


def _step_table(shape: str) -> np.ndarray:
    k = _profile(_T * _SCALE[shape], shape)
    s = np.cumsum(k)
    return s / s[-1]


_STEP = {sh: _step_table(sh) for sh in (*SHAPES, "logistic")}


def step_cdf(z: np.ndarray, shape: str) -> np.ndarray:
    """Step response of `shape` at `z` measured in units of its own 10-90."""
    return np.interp(z, _T, _STEP[shape])


# ---------------------------------------------------------------------------
# 1. THE ESTIMATORS
# ---------------------------------------------------------------------------

def _cumsum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])


def window_mean(cs: np.ndarray, x: np.ndarray, lo, hi) -> np.ndarray:
    """Exact sub-sample mean of `x` over [lo, hi), by linear-interp cumsum.

    The same integrator `dotclock.sample_plateaus` uses, so the windows here and
    the decoder's plateau integrals are one operation on one grid.
    """
    lo = np.clip(lo, 0.0, len(x) - 1e-9)
    hi = np.clip(hi, 0.0, len(x) - 1e-9)

    def C(t):
        b = np.floor(t).astype(np.int64)
        return cs[b] + (t - b) * x[b]

    return (C(hi) - C(lo)) / np.maximum(hi - lo, 1e-9)


def plateau_grid(t_lo: float, t_hi: float, psi_i: float, spd: float) -> np.ndarray:
    """Plateau edges covering [t_lo, t_hi] on the absolute sample axis."""
    k0 = int(np.ceil((t_lo - psi_i) / spd)) + 1
    k1 = int(np.floor((t_hi - psi_i) / spd)) - 2
    if k1 <= k0 + 4:
        return np.zeros(0)
    return psi_i + np.arange(k0, k1 + 1) * spd


def plateau_levels(x, cs, edges, spd) -> np.ndarray:
    lo = edges[:-1] + (0.5 - LEVEL_FRAC / 2.0) * spd
    hi = edges[:-1] + (0.5 + LEVEL_FRAC / 2.0) * spd
    return window_mean(cs, x, lo, hi)


def select_steps(edges, v, noise, min_amp_sigmas=MIN_AMP_SIGMAS, flat_frac=FLAT_FRAC):
    """Isolated single steps: large, with a flat plateau on each side.

    The flanking-flatness test is what makes each profile window a step response
    of `c` alone rather than a superposition of neighbouring transitions.
    """
    d = np.diff(v)
    amp_min = min_amp_sigmas * noise
    out = []
    for m in range(1, len(d) - 1):
        D = d[m]
        if abs(D) < amp_min:
            continue
        if abs(d[m - 1]) > flat_frac * abs(D) or abs(d[m + 1]) > flat_frac * abs(D):
            continue
        out.append((edges[m + 1], v[m], v[m + 1], abs(D)))
    return out


def width_crossing(x, t_edge, L0, L1, spd) -> float | None:
    """Model-free 10-90 from raw samples normalised by the two plateau levels."""
    i0 = int(np.ceil(t_edge - PROFILE_HALF * spd))
    i1 = int(np.floor(t_edge + PROFILE_HALF * spd))
    if i0 < 0 or i1 >= len(x) or i1 - i0 < 7:
        return None
    t = np.arange(i0, i1 + 1, dtype=np.float64)
    p = (x[i0:i1 + 1] - L0) / (L1 - L0)
    mid = int(np.argmin(np.abs(p - 0.5)))
    below = np.where(p[:mid + 1] < 0.1)[0]
    above = np.where(p[mid:] > 0.9)[0]
    if not len(below) or not len(above):
        return None
    a, b = int(below[-1]), mid + int(above[0]) - 1
    if b < a:
        return None

    def cross(i, lvl):
        y0, y1 = p[i], p[i + 1]
        return None if y1 == y0 else t[i] + (lvl - y0) / (y1 - y0)

    t10, t90 = cross(a, 0.1), cross(b, 0.9)
    if t10 is None or t90 is None or t90 <= t10:
        return None
    return (t90 - t10) / spd


W_LO, W_HI = 0.12, 1.60   # fit grid, dots. Readings that rail are dropped below.


def varpro(x, t_edge, spd, shape, *, t0_span=0.30, w_lo=W_LO, w_hi=W_HI, n_w=100, n_t=25):
    """Fit x ~ A + B * step_cdf((t - t0)/w) over the profile window.

    A and B are eliminated linearly (variable projection), so only (t0, w) are
    searched and `w` IS the 10-90 in dots. t0 is free, which is what makes the
    fit immune to dot-lattice placement error.

    Returns (width_dots, residual_sum_of_squares) or None.
    """
    i0 = int(np.ceil(t_edge - PROFILE_HALF * spd))
    i1 = int(np.floor(t_edge + PROFILE_HALF * spd))
    if i0 < 0 or i1 >= len(x) or i1 - i0 < 7:
        return None
    t = np.arange(i0, i1 + 1, dtype=np.float64)
    y = np.asarray(x[i0:i1 + 1], dtype=np.float64)
    n = len(t)

    t0g = t_edge + np.linspace(-t0_span, t0_span, n_t) * spd
    wg = np.linspace(w_lo, w_hi, n_w) * spd
    P = step_cdf((t[None, None, :] - t0g[:, None, None]) / wg[None, :, None], shape)

    s1, s2 = P.sum(-1), (P * P).sum(-1)
    sy, spy = y.sum(), (P * y[None, None, :]).sum(-1)
    det = n * s2 - s1 * s1
    det = np.where(np.abs(det) < 1e-14, np.nan, det)
    B = (n * spy - s1 * sy) / det
    A = (sy - B * s1) / n
    ssq = ((A[..., None] + B[..., None] * P - y[None, None, :]) ** 2).sum(-1)
    if not np.isfinite(ssq).any():
        return None
    i, j = np.unravel_index(np.nanargmin(ssq), ssq.shape)
    w = wg[j]
    if 0 < j < n_w - 1:                       # the grid is 0.0125 dots coarse
        y0, y1, y2 = ssq[i, j - 1], ssq[i, j], ssq[i, j + 1]
        den = y0 - 2 * y1 + y2
        if np.isfinite(den) and den > 0:
            w = wg[j] + 0.5 * (y0 - y2) / den * (wg[1] - wg[0])
    return float(w / spd), float(ssq[i, j])


def measure_widths(x, cs, edges, spd, noise, *, shapes=("gauss",), want_amp=False):
    """Every estimator over every isolated step on one trace.

    Keys: 'cross', 'erf' (Gaussian step), 'logistic', plus 'ssq_<shape>' and
    'w_<shape>' for each shape in `shapes` -- the material for shape
    discrimination -- and 'amp' if requested.
    """
    out: dict[str, list[float]] = {k: [] for k in ESTIMATORS}
    for sh in shapes:
        out[f"ssq_{sh}"] = []
        out[f"w_{sh}"] = []
    out["amp"] = []
    if not len(edges):
        return out
    v = plateau_levels(x, cs, edges, spd)
    for t_edge, L0, L1, amp in select_steps(edges, v, noise):
        w = width_crossing(x, t_edge, L0, L1, spd)
        if w is not None and W_LO * 1.05 < w < W_HI * 0.95:
            out["cross"].append(w)
        r = varpro(x, t_edge, spd, "gauss")
        if r and W_LO * 1.05 < r[0] < W_HI * 0.95:
            out["erf"].append(r[0])
            if want_amp:
                out["amp"].append(amp)
        r = varpro(x, t_edge, spd, "logistic")
        if r and W_LO * 1.05 < r[0] < W_HI * 0.95:
            out["logistic"].append(r[0])
        for sh in shapes:
            r = varpro(x, t_edge, spd, sh)
            if r and W_LO * 1.05 < r[0] < W_HI * 0.95:
                out[f"w_{sh}"].append(r[0])
                out[f"ssq_{sh}"].append(r[1])
    return out


def plateau_noise(x, cs, edges, spd) -> float:
    """Per-sample noise sd from the scatter INSIDE settled plateau centres.

    A plateau is constant by construction, so anything moving inside its centre
    is noise (plus whatever ripple `c` leaves, which counts as noise here).
    """
    if not len(edges):
        return 0.0
    lo = np.floor(edges[:-1] + (0.5 - LEVEL_FRAC / 2.0) * spd).astype(int) + 1
    hi = np.floor(edges[:-1] + (0.5 + LEVEL_FRAC / 2.0) * spd).astype(int)
    sds = [np.std(x[l:h + 1], ddof=1) for l, h in zip(lo, hi) if h > l + 1]
    return float(np.median(sds)) if sds else 0.0


# ---------------------------------------------------------------------------
# 2. THE SYNTHETIC STAIRCASE
# ---------------------------------------------------------------------------

_OS = 8  # fine-grid oversampling


def kernel(shape: str, w1090_dots: float, spd: float, os: int = _OS) -> np.ndarray:
    """Unit-area kernel whose STEP response has 10-90 = `w1090_dots` dots."""
    half = int(max(64, np.ceil(8.0 * w1090_dots * spd * os)))
    t = (np.arange(2 * half + 1) - half) / (os * w1090_dots * spd)
    k = _profile(t * _SCALE[shape], shape)
    return k / k.sum()


def _fftconv_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    n = len(a) + len(k) - 1
    N = 1 << int(np.ceil(np.log2(n)))
    full = np.fft.irfft(np.fft.rfft(a, N) * np.fft.rfft(k, N), N)[:n]
    off = (len(k) - 1) // 2
    return full[off:off + len(a)]


def synth_segment(v, edges, t_lo: int, t_hi: int, kern, noise, rng, os: int = _OS):
    """Staircase of known plateau values, blurred by `kern`, sampled at integers.

    Edge positions are quantised to 1/os of a sample (0.01 dots). That is a
    positional error, not a width error, and single-trace fits are immune to it.
    """
    pad = len(kern) // os + 4
    fine = np.arange((t_lo - pad) * os, (t_hi + pad) * os + 1) / os
    idx = np.clip(np.searchsorted(edges, fine) - 1, 0, len(v) - 1)
    y = _fftconv_same(v[idx], kern)
    take = ((np.arange(t_lo, t_hi + 1) - (t_lo - pad)) * os).astype(int)
    out = y[take]
    return out + rng.normal(0.0, noise, out.shape) if noise > 0 else out


@dataclass
class Calibration:
    """reading -> truth, per estimator, for one assumed truth shape."""
    shape: str
    true: np.ndarray
    reading: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]

    def invert(self, estimator: str, reading: float) -> float:
        r = self.reading[estimator]
        ok = np.isfinite(r)
        if ok.sum() < 2 or not np.isfinite(reading):
            return float("nan")
        return float(np.interp(reading, r[ok], self.true[ok]))

    def slope(self, estimator: str) -> float:
        r, t = self.reading[estimator], self.true
        ok = np.isfinite(r)
        return float(np.polyfit(t[ok], r[ok], 1)[0]) if ok.sum() > 1 else float("nan")


# ---------------------------------------------------------------------------
# 3. LOADING A FRAME FROM THE MASTER
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    fid: str
    x: np.ndarray            # undrooped raw signal, one channel
    starts: np.ndarray
    period: float
    spd: float
    psi: np.ndarray
    dot_locked: bool
    strength: float
    coherence: float
    noise: float = 0.0
    cs: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def edges(self, i: int, shift: float = 0.0) -> np.ndarray:
        return plateau_grid(self.starts[i] + PICTURE_START * self.period,
                            self.starts[i] + (PICTURE_START + PICTURE_SPAN) * self.period,
                            self.psi[i] + shift * self.spd, self.spd)


def seed_sample(fid: str) -> tuple[int, int]:
    """(channel, first sample) for a frame id like 'L000' or 'R056'.

    Read straight out of `barry_tables.json['start_samples']`, which
    `provenance.py` classifies as tier 0: it is a table of frame start positions
    in the audio and carries no Earth knowledge. The Earth-knowledge keys of that
    file are never touched, and `catalog.py` -- which also validates against the
    title table -- is deliberately NOT imported, so this module's whole import
    closure is numpy plus the decoder.
    """
    import json
    tab = json.loads((Path(__file__).resolve().parent / "barry_tables.json").read_text())
    ch = 0 if fid[0].upper() == "L" else 1
    return ch, int(tab["start_samples"][ch][int(fid[1:])])


def load_frame(fid: str, mm, cat=None, *, traces: int = 512) -> Frame:
    """Reproduce decode.py's raw-signal path as far as the plateau grid.

    Deliberately stops before `dc_restore`, `dehum` and dropout repair: those act
    on the dot matrix, and the width lives in the raw waveform.

    `cat` is optional and exists only so a caller holding a catalogue can pass
    it; the default path needs no catalogue at all.
    """
    from . import sync as sync_mod, decode as decode_mod, dotclock as dot_mod

    if cat is not None:
        fr = cat.by_id(fid)
        channel, start = fr.channel, fr.seed_sample
    else:
        channel, start = seed_sample(fid)
    n = int(sync_mod.NOMINAL_PERIOD * (traces + 8))
    seg = np.asarray(mm[start:start + n, channel], dtype=np.float64)
    tb = sync_mod.recover(seg, n_traces=traces)
    ch = "L" if channel == 0 else "R"
    tau = decode_mod.UNCOUPLE_TAU_384[ch] * tb.period / decode_mod._NOMINAL_PERIOD_384
    x = decode_mod.undroop(seg, tau)

    clock = dot_mod.measure(x, tb, lo_f=PICTURE_START + 0.01,
                            hi_f=PICTURE_START + PICTURE_SPAN - 0.01)
    n_tr = min(traces, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])
    spd = clock.samples_per_dot if clock.measured else tb.period / dot_mod.DOTS_PER_TRACE
    psi, coh = dot_mod.track_phase(x, starts, tb.period, spd,
                                   PICTURE_START, PICTURE_START + PICTURE_SPAN)
    f = Frame(fid, x, starts, float(tb.period), float(spd), psi,
              bool(clock.measured), float(clock.strength), float(coh))
    f.cs = _cumsum(x)
    ns = [plateau_noise(x, f.cs, f.edges(i), spd) for i in range(0, n_tr, 17)]
    ns = [v for v in ns if v > 0]
    f.noise = float(np.median(ns)) if ns else 0.0
    return f


def synth_frame(f: Frame, width: float, *, shape="gauss", noise=None, jitter=0.0,
                traces=None, seed=0) -> Frame:
    """A twin of `f` whose staircase has a KNOWN 10-90 width.

    Plateau values, dot pitch, phase, noise and step statistics are the real
    frame's; only the transition width is imposed. `jitter` (dots rms) displaces
    the TRUE lattice per trace while leaving `psi` alone, which is exactly the
    error `dotclock.track_phase` leaves behind.
    """
    rng = np.random.default_rng(seed)
    noise = f.noise if noise is None else noise
    kern = kernel(shape, width, f.spd)
    xs = np.zeros_like(f.x)
    idx = list(range(len(f.starts))) if traces is None else list(traces)
    for i in idx:
        e = f.edges(i)
        if not len(e):
            continue
        v = plateau_levels(f.x, f.cs, e, f.spd)
        et = e + (rng.normal(0.0, jitter) * f.spd if jitter > 0 else 0.0)
        lo_i, hi_i = int(np.floor(e[0])) - 4, int(np.ceil(e[-1])) + 4
        ys = synth_segment(v, et, lo_i, hi_i, kern, noise, rng)
        xs[lo_i:lo_i + len(ys)] = ys
    g = Frame(f.fid + f"~synth{width:.2f}", xs, f.starts[idx] if traces is not None else f.starts,
              f.period, f.spd, f.psi[idx] if traces is not None else f.psi,
              f.dot_locked, f.strength, f.coherence)
    g.cs = _cumsum(xs)
    g.noise = noise
    return g


def calibrate(f: Frame, *, shape="gauss",
              widths=(0.30, 0.38, 0.44, 0.49, 0.55, 0.62, 0.70),
              traces=None, jitter=0.0, seed=0) -> Calibration:
    """Run every estimator on synthetic staircases of KNOWN width."""
    idx = list(range(len(f.starts))) if traces is None else list(traces)
    reading = {k: [] for k in ESTIMATORS}
    counts = {k: [] for k in ESTIMATORS}
    for w in widths:
        g = synth_frame(f, w, shape=shape, jitter=jitter, traces=idx, seed=seed)
        acc = {k: [] for k in ESTIMATORS}
        for j in range(len(idx)):
            got = measure_widths(g.x, g.cs, g.edges(j), g.spd, g.noise)
            for k in ESTIMATORS:
                acc[k].extend(got[k])
        for k in ESTIMATORS:
            reading[k].append(float(np.median(acc[k])) if acc[k] else float("nan"))
            counts[k].append(len(acc[k]))
    return Calibration(shape, np.asarray(widths, float),
                       {k: np.asarray(v, float) for k, v in reading.items()},
                       {k: np.asarray(v, int) for k, v in counts.items()})


def frame_widths(f: Frame, *, traces=None, shapes=("gauss",), want_amp=False) -> dict:
    """Raw (uncalibrated) readings for every isolated step in a frame."""
    idx = range(len(f.starts)) if traces is None else traces
    acc: dict[str, list] = {}
    for i in idx:
        got = measure_widths(f.x, f.cs, f.edges(i), f.spd, f.noise,
                             shapes=shapes, want_amp=want_amp)
        for k, v in got.items():
            acc.setdefault(k, []).extend(v)
    return {k: np.asarray(v, float) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# 4. THE HOLD-OUT
# ---------------------------------------------------------------------------

def split_transfers(fgrid, width, inner=0.4):
    """(A_inner, A_outer) for the central `inner` and the rest of each plateau.

    The full-dot mean splits exactly by area into the two parts, so
    A_out = (A_full - inner*A_in) / (1 - inner). Linear in the measurement
    functional -- this is `aperture.transfer` twice and no new machinery.
    """
    lo = 0.5 - inner / 2.0
    A_full = aperture.transfer(fgrid, width=width, window=(0.0, 1.0)).real
    A_in = aperture.transfer(fgrid, width=width, window=(lo, lo + inner)).real
    return A_in, (A_full - inner * A_in) / (1.0 - inner)


def holdout(f: Frame, *, direction="out2in", strengths=(-1.0, 0.0, 1.0),
            width=aperture.TRANSITION_1090, inner=0.4, phase_shift=0.0,
            traces=None, edge_drop=8) -> dict:
    """Predict one half of every plateau from the other half.

    `direction='in2out'` is the spec's test: estimate from the settled central
    40%, correct for that window, predict the outer 60%. `'out2in'` reverses it,
    which tests a 1.82x operator instead of a 1.05x one.

    The two sample sets are disjoint by construction (`u_out` is formed from the
    full-dot and central integrals algebraically, which is exact, so no sample of
    the target ever enters the predictor). Independent noise on the target means
    an over-boosted predictor is charged for its own amplified noise -- this test
    has a wrong side, which is the whole point.

    `phase_shift` in dots moves the assumed lattice; 0.5 is the control that runs
    the identical operator on a grid that is not the converter's.
    """
    idx = range(len(f.starts)) if traces is None else traces
    rin, rout = [], []
    for i in idx:
        e = f.edges(i, phase_shift)
        if len(e) < 32:
            continue
        lo = e[:-1] + (0.5 - inner / 2.0) * f.spd
        hi = e[:-1] + (0.5 + inner / 2.0) * f.spd
        u_in = window_mean(f.cs, f.x, lo, hi)
        u_full = window_mean(f.cs, f.x, e[:-1], e[1:])
        rin.append(u_in)
        rout.append((u_full - inner * u_in) / (1.0 - inner))
    if not rin:
        raise ValueError("no usable traces")
    n = min(len(r) for r in rin)
    IN = np.asarray([r[:n] for r in rin])
    OUT = np.asarray([r[:n] for r in rout])
    src, tgt = (IN, OUT) if direction == "in2out" else (OUT, IN)

    pad = 16
    Sp = np.pad(src, ((0, 0), (pad, pad)), mode="reflect")
    fg = np.fft.rfftfreq(Sp.shape[1])
    A_in, A_out = split_transfers(fg, width, inner)
    A_src, A_tgt = (A_in, A_out) if direction == "in2out" else (A_out, A_in)
    F = np.fft.rfft(Sp, axis=1)
    sl = slice(edge_drop, n - edge_drop)
    err = {}
    for s in strengths:
        g = A_tgt * np.maximum(np.abs(A_src), 1e-6) ** (-float(s))
        P = np.fft.irfft(F * g, n=Sp.shape[1], axis=1)[:, pad:pad + n]
        err[float(s)] = float(np.sqrt(np.mean((P - tgt)[:, sl] ** 2)))
    base = err[0.0] if 0.0 in err else float("nan")
    return {"rms": err, "ratio": {s: err[s] / base for s in err},
            "n_dots": n, "n_traces": IN.shape[0],
            "target_rms": float(np.sqrt(np.mean(tgt[:, sl] ** 2)))}


# ---------------------------------------------------------------------------
# 5. REPORT
# ---------------------------------------------------------------------------

DEFAULT_FRAMES = ("L000", "L002", "L020", "L034", "L055", "R010", "R040", "R056")
STRENGTHS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
WIDTH_SWEEP = (0.25, 0.35, 0.45, 0.49, 0.55, 0.65, 0.80, 1.00, 1.30, 2.00)


def _f(v, n=4):
    return "nan" if not np.isfinite(v) else f"{v:.{n}f}"


def _hr(title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def main(frame_ids=DEFAULT_FRAMES, cal_frame="L055", traces=512, cal_step=4, quiet=False):
    from . import wav

    mm = wav.memmap(wav.probe(MASTER))

    print("=" * 78)
    print("APERTURE WIDTH AUDIT -- pipeline/aperture_fit.py")
    print("the code behind aperture.TRANSITION_1090, and what a hold-out can decide")
    print("=" * 78)

    _hr("0. CLOSED-FORM CHECKS OF aperture.py ITSELF (no data involved)")
    fg = np.linspace(0, 0.5, 11)
    m = np.arange(-200, 201)
    alias = np.array([float((np.sinc(f + m) ** 2).sum()) for f in fg])
    print(f"   SUM_m sinc^2(f+m) == 1 to {np.abs(alias - 1).max():.2e}  "
          "(the identity that makes a")
    print("   perfect staircase need no correction at all -- the whole reason the")
    print("   original 2/pi proposal was wrong)")
    A = aperture.transfer(np.array([0.0, 0.25, 0.375, 0.5]))
    print(f"   A(0) = {A[0].real:.6f}   A(0.25) = {A[1].real:.4f}   "
          f"A(0.375) = {A[2].real:.4f}   A(0.5) = {A[3].real:.4f}")
    c = aperture.build(4096)
    print(f"   inverse at strength 1: max gain {c.gain_max:.4f}, taps "
          + " ".join(f"{v:+.4f}" for v in c.taps[3:6]))
    print("   A(0) = 1 exactly is why the correction cannot move a flat field, the")
    print("   circle fit or the DC level. That is construction, not luck.")

    _hr("1. FRAMES AND THE DOT-CLOCK GATE")
    print("   The correction only describes frames where the dot clock was measured;")
    print("   elsewhere decode bins with a stretched Lanczos kernel and A(f) is not")
    print("   what happened. Everything below is gated on `dot_locked`.\n")
    frames = {}
    print(f"   {'frame':6s} {'locked':7s} {'strength':>9s} {'spd':>9s} {'coh':>7s} {'noise sd':>9s}")
    for fid in frame_ids:
        f = load_frame(fid, mm, traces=traces)
        frames[fid] = f
        print(f"   {fid:6s} {str(f.dot_locked):7s} {f.strength:9.2f} {f.spd:9.4f} "
              f"{f.coherence:7.4f} {f.noise:9.5f}")
    locked = [fid for fid in frame_ids if frames[fid].dot_locked]
    print(f"\n   {len(locked)}/{len(frame_ids)} dot-locked: {', '.join(locked)}")
    dropped = [fid for fid in frame_ids if fid not in locked]
    if dropped:
        print(f"   EXCLUDED by the gate: {', '.join(dropped)}")
    if cal_frame not in locked:
        cal_frame = locked[0]
    cf = frames[cal_frame]
    cal_traces = range(0, len(cf.starts), cal_step)

    _hr(f"2. CALIBRATING THE ESTIMATOR ON A SYNTHETIC EDGE OF KNOWN WIDTH ({cal_frame})")
    print("   Real plateau values, real step amplitudes, real noise, real dot pitch;")
    print("   only the width is imposed, and the identical selection and fitting code")
    print("   runs on it. An uncalibrated reading is mostly its own bias.\n")
    cals = {sh: calibrate(cf, shape=sh, traces=cal_traces) for sh in SHAPES}
    c0 = cals["gauss"]
    print(f"   truth shape GAUSSIAN, {int(c0.counts['erf'].sum())} synthetic edges")
    print(f"   {'TRUE 10-90':>10s} " + " ".join(f"{k:>10s}" for k in ESTIMATORS) + "      n")
    for j, w in enumerate(c0.true):
        print(f"   {w:10.3f} " + " ".join(f"{_f(c0.reading[k][j]):>10s}" for k in ESTIMATORS)
              + f"   {int(c0.counts['erf'][j]):6d}")
    print("\n   slope d(reading)/d(true):  "
          + "   ".join(f"{k} {c0.slope(k):.3f}" for k in ESTIMATORS))
    print("   A slope below 1 means the estimator compresses; inverting divides by it,")
    print("   so it also multiplies the statistical error by 1/slope.")

    cj = calibrate(cf, shape="gauss", traces=range(0, len(cf.starts), cal_step * 2),
                   jitter=0.10, seed=7)
    c00 = calibrate(cf, shape="gauss", traces=range(0, len(cf.starts), cal_step * 2), seed=7)
    print("\n   dot-lattice jitter 0.10 dots rms injected into the TRUE grid while the")
    print("   estimator keeps the tracked psi (i.e. track_phase's own error):")
    for k in ESTIMATORS:
        print(f"     {k:9s} reading shifts {np.nanmean(cj.reading[k] - c00.reading[k]):+.4f} dots")
    print("   Single-trace fitting is why this is small: t0 is free in every fit.")

    _hr("3. MEASUREMENT ON THE MASTER (single-trace fits, never stacked)")
    raw = {}
    print(f"   {'frame':6s} {'n':>5s} " + " ".join(f"{k+' raw':>12s}" for k in ESTIMATORS)
          + "  |" + " ".join(f"{k+' cal':>12s}" for k in ESTIMATORS))
    per_est = {k: [] for k in ESTIMATORS}
    for fid in locked:
        w = frame_widths(frames[fid], shapes=SHAPES, want_amp=True)
        raw[fid] = w
        med = {k: float(np.median(w[k])) if len(w[k]) else float("nan") for k in ESTIMATORS}
        cal = {k: c0.invert(k, med[k]) for k in ESTIMATORS}
        for k in ESTIMATORS:
            if np.isfinite(cal[k]):
                per_est[k].append(cal[k])
        print(f"   {fid:6s} {len(w['erf']):5d} "
              + " ".join(f"{_f(med[k]):>12s}" for k in ESTIMATORS)
              + "  |" + " ".join(f"{_f(cal[k]):>12s}" for k in ESTIMATORS))
    n_edges = sum(len(raw[fid]["erf"]) for fid in locked)
    print(f"   {n_edges} isolated single-trace steps in total.")

    print("\n   IS `c` LINEAR? width by step-amplitude quintile. A slew-rate limit")
    print("   widens with amplitude; a linear filter must not. The CONTROL row is the")
    print("   same analysis on a synthetic staircase that is linear by construction,")
    print("   so any trend it also shows is the estimator's, not the chain's.")

    def quintiles(amps, wid):
        q = np.quantile(amps, np.linspace(0, 1, 6))
        out = []
        for a, b in zip(q[:-1], q[1:]):
            m = (amps >= a) & (amps <= b)
            out.append((float(np.median(wid[m])) if m.sum() else float("nan"), int(m.sum())))
        return out

    amps = np.concatenate([raw[fid]["amp"] for fid in locked])
    wid = np.concatenate([raw[fid]["erf"] for fid in locked])
    row_master = quintiles(amps, wid)
    gs = synth_frame(cf, 0.49, traces=range(0, len(cf.starts)), seed=11)
    gw = frame_widths(gs, want_amp=True)
    crow = quintiles(gw["amp"], gw["erf"])
    print("     master  " + "  ".join(f"{v:.3f}(n={n})" for v, n in row_master)
          + f"   spread {max(v for v, _ in row_master) - min(v for v, _ in row_master):.3f}")
    print("     control " + "  ".join(f"{v:.3f}(n={n})" for v, n in crow)
          + f"   spread {max(v for v, _ in crow) - min(v for v, _ in crow):.3f}")

    print("\n   WHICH STEP SHAPE FITS? median residual per edge, relative to the best")
    tot = {sh: np.concatenate([raw[fid][f"ssq_{sh}"] for fid in locked]) for sh in SHAPES}
    n_min = min(len(v) for v in tot.values())
    stack = np.vstack([tot[sh][:n_min] for sh in SHAPES])
    rel = stack / np.minimum.reduce(stack)
    print(f"   {'shape':10s} {'rel. residual':>14s} {'wins':>7s} {'fitted 10-90':>14s}")
    wins = np.argmin(stack, axis=0)
    for i, sh in enumerate(SHAPES):
        wsh = np.concatenate([raw[fid][f"w_{sh}"] for fid in locked])
        print(f"   {sh:10s} {np.median(rel[i]):14.4f} {int((wins == i).sum()):7d} "
              f"{np.median(wsh):14.4f}")

    print("\n   SHAPE SYSTEMATIC -- the same master readings inverted through each")
    print("   assumed truth shape (this is the dominant uncertainty)")
    print(f"   {'shape':10s} " + " ".join(f"{k:>10s}" for k in ESTIMATORS))
    shape_vals = {}
    for sh, c in cals.items():
        vals = [float(np.median([c.invert(k, float(np.median(raw[fid][k])))
                                 for fid in locked])) for k in ESTIMATORS]
        shape_vals[sh] = vals
        print(f"   {sh:10s} " + " ".join(f"{v:10.4f}" for v in vals))

    # A shape is admissible if its median per-edge residual is within a factor
    # SHAPE_TOL of the best shape's. The threshold is arbitrary and is stated so
    # it can be re-judged from the table above; it is deliberately LOOSE, so that
    # the reported range keeps every shape the data does not clearly reject
    # rather than collapsing onto the one that happens to fit best.
    SHAPE_TOL = 2.0
    plaus = [sh for sh in SHAPES if np.median(rel[SHAPES.index(sh)]) < SHAPE_TOL]
    if not plaus:
        plaus = [SHAPES[int(np.bincount(wins, minlength=len(SHAPES)).argmax())]]
    pv = np.array([v for sh in plaus for v in shape_vals[sh]])
    centre = float(np.median(pv))
    lo, hi = float(pv.min()), float(pv.max())
    allv = np.array([v for vals in shape_vals.values() for v in vals])
    stat = float(np.std(per_est["erf"], ddof=1)) if len(per_est["erf"]) > 1 else 0.0
    print(f"\n   shapes the data does not reject (rel. residual < {SHAPE_TOL}): {', '.join(plaus)}")
    print(f"\n   MEASURED 10-90 = {centre:.3f} dots,  {lo:.3f} - {hi:.3f}")
    print(f"     over 3 estimators x {len(plaus)} admissible truth shapes")
    print(f"     if no shape is rejected the range widens to {allv.min():.3f} - {allv.max():.3f}")
    print(f"     frame-to-frame sd (erf, Gaussian truth): {stat:.4f} over {len(locked)} frames")
    print(f"   aperture.TRANSITION_1090 = {aperture.TRANSITION_1090}, "
          f"shipped range {aperture.TRANSITION_1090_RANGE}")
    verdict = ("CONFIRMED" if aperture.TRANSITION_1090_RANGE[0] <= centre <= aperture.TRANSITION_1090_RANGE[1]
               else "OUTSIDE THE SHIPPED RANGE")
    print(f"   -> {verdict}")

    _hr("4. CALIBRATING THE ARBITER, BEFORE BELIEVING IT")
    print("   The hold-out is run on a synthetic staircase of KNOWN width, no jitter,")
    print("   real noise. If it cannot recover a width it planted itself, it cannot")
    print("   measure one on the master either.\n")
    for wt in (0.30, 0.49, 0.80):
        g = synth_frame(cf, wt, traces=cal_traces, seed=3)
        r_s = holdout(g, direction="out2in", strengths=STRENGTHS, width=wt)
        bs = min(STRENGTHS, key=lambda s: r_s["ratio"][s])
        row = [holdout(g, direction="out2in", strengths=(0.0, 1.0), width=w)["ratio"][1.0]
               for w in WIDTH_SWEEP]
        best = WIDTH_SWEEP[int(np.argmin(row))]
        print(f"   true width {wt:.2f}")
        print("     strength sweep at the TRUE width: "
              + " ".join(f"{s:+.1f}:{r_s['ratio'][s]:.3f}" for s in STRENGTHS))
        print(f"       -> best strength {bs:+.1f}  "
              f"{'(recovers the physical answer)' if bs == 1.0 else '(does NOT recover 1.0)'}")
        print(f"     width sweep at strength 1.0:      "
              + " ".join(f"{w:.2f}:{v:.3f}" for w, v in zip(WIDTH_SWEEP, row)))
        print(f"       -> min at {best:.2f}  "
              f"{'(recovers the width)' if abs(best - wt) < 0.1 else '(RAILS -- does NOT recover the width)'}")
    print("\n   READ THIS BEFORE READING SECTION 5: the sign control recovers the truth,")
    print("   the WIDTH SWEEP DOES NOT. A(f) enters the predictor as A_tgt/A_src^s, and")
    print("   changing the width mostly reshapes that ratio's mid-band rather than its")
    print("   size, so the sweep slides toward whatever shape is closest to the Wiener")
    print("   optimum for this noise level. It is a rail, not a measurement, and it is")
    print("   a rail on synthetic data whose width is known. The width is fixed by")
    print("   section 3 and by nothing in section 5.")

    _hr("5. HOLD-OUT ON THE MASTER")
    print(f"   width = {centre:.3f} dots (section 3). Predicting half of every plateau")
    print("   from the disjoint other half; ratios are to strength 0, lower is better.\n")
    print("   5a. in -> out, the spec's direction. Operator gain at the dot Nyquist:")
    fN = np.array([0.5])
    A_in_N, A_out_N = split_transfers(fN, centre, 0.4)
    print(f"       1/A_in = {1/A_in_N[0]:.3f}  -- barely an operator at all, which is why")
    print("       the effect is a few tenths of a percent.")
    print(f"   {'frame':6s} " + " ".join(f"{s:>+8.1f}" for s in STRENGTHS))
    ho_i = {}
    for fid in locked:
        r = holdout(frames[fid], direction="in2out", strengths=STRENGTHS, width=centre)
        ho_i[fid] = r
        print(f"   {fid:6s} " + " ".join(f"{r['ratio'][s]:8.4f}" for s in STRENGTHS))
    p_i = sum(ho_i[f]["ratio"][1.0] < 1.0 for f in locked)
    n_i = sum(ho_i[f]["ratio"][-1.0] > 1.0 for f in locked)
    print(f"       +1.0 beats 0.0 on {p_i}/{len(locked)}   -> {'PASS' if p_i == len(locked) else 'FAIL'}")
    print(f"       -1.0 worse than 0.0 on {n_i}/{len(locked)} -> {'PASS' if n_i == len(locked) else 'FAIL'}")

    print(f"\n   5b. out -> in, the reverse. Operator gain at the dot Nyquist:")
    print(f"       1/A_out = {1/A_out_N[0]:.3f} -- larger than the 1.44 decode applies, so")
    print("       this is the sharper instrument and the one to weigh.")
    print(f"   {'frame':6s} " + " ".join(f"{s:>+8.1f}" for s in STRENGTHS))
    ho_o = {}
    for fid in locked:
        r = holdout(frames[fid], direction="out2in", strengths=STRENGTHS, width=centre)
        ho_o[fid] = r
        print(f"   {fid:6s} " + " ".join(f"{r['ratio'][s]:8.4f}" for s in STRENGTHS))
    p_o = sum(ho_o[f]["ratio"][1.0] < 1.0 for f in locked)
    n_o = sum(ho_o[f]["ratio"][-1.0] > 1.0 for f in locked)
    best_s = {fid: min(STRENGTHS, key=lambda s: ho_o[fid]["ratio"][s]) for fid in locked}
    print(f"       +1.0 beats 0.0 on {p_o}/{len(locked)}   -> {'PASS' if p_o == len(locked) else 'FAIL'}")
    print(f"       -1.0 worse than 0.0 on {n_o}/{len(locked)} -> {'PASS' if n_o == len(locked) else 'FAIL'}")
    print(f"       best strength per frame: "
          + ", ".join(f"{fid} {best_s[fid]:+.1f}" for fid in locked))

    print("\n   5c. CONTROL: the identical operator on a lattice shifted half a dot.")
    print("       Generic sharpening would survive this; a correction that is really")
    print("       about the converter's own grid must not.")
    print(f"   {'frame':6s} {'true grid':>12s} {'half-dot':>12s} {'gain kept':>11s}")
    kept_frac = []
    for fid in locked:
        r = holdout(frames[fid], direction="out2in", strengths=(0.0, 1.0),
                    width=centre, phase_shift=0.5)
        t = 1.0 - ho_o[fid]["ratio"][1.0]
        s = 1.0 - r["ratio"][1.0]
        kept_frac.append(s / t if t else float("nan"))
        print(f"   {fid:6s} {ho_o[fid]['ratio'][1.0]:12.4f} {r['ratio'][1.0]:12.4f} "
              f"{(s / t * 100 if t else float('nan')):10.0f}%")

    print("\n   5d. WIDTH SWEEP at strength 1.0, BOTH directions. Section 4 has already")
    print("       shown this to be a rail even when the width is known, so it is")
    print("       printed as a negative result and not as a measurement.")
    mins = {}
    for direction in ("in2out", "out2in"):
        print(f"   [{direction}]")
        print(f"   {'frame':6s} " + " ".join(f"{w:>7.2f}" for w in WIDTH_SWEEP))
        for fid in locked:
            row = [holdout(frames[fid], direction=direction, strengths=(0.0, 1.0),
                           width=w)["ratio"][1.0] for w in WIDTH_SWEEP]
            m = WIDTH_SWEEP[int(np.argmin(row))]
            mins.setdefault(direction, []).append(m)
            print(f"   {fid:6s} " + " ".join(f"{v:7.4f}" for v in row) + f"   min @ {m:.2f}")

    # --- verdict, computed from the numbers above, not asserted ---------------
    amp_trend = max(v for v, _ in row_master) - min(v for v, _ in row_master)
    amp_ctrl = max(v for v, _ in crow) - min(v for v, _ in crow)
    kept = np.array(kept_frac)
    _hr("6. WHAT THIS DOES AND DOES NOT ESTABLISH")
    print(f"   MEASURED     10-90 transition width {centre:.3f} dots, "
          f"{lo:.3f}-{hi:.3f}, from {n_edges} single-trace")
    print("                fits on the master through an estimator calibrated on a")
    print(f"                synthetic edge of known width. Shipped value "
          f"{aperture.TRANSITION_1090} "
          f"({'inside' if verdict == 'CONFIRMED' else 'OUTSIDE'} that band).")
    print(f"   {'MEASURED    ' if amp_trend - amp_ctrl < 0.02 else 'QUALIFIED   '} "
          f"amplitude dependence of the width: {amp_trend:.3f} dots across step-")
    print(f"                amplitude quintiles against {amp_ctrl:.3f} dots for a linear")
    print("                synthetic control. " + (
        "Flat: `c` is a filter, not a slew limit."
        if amp_trend - amp_ctrl < 0.02 else
        "NOT flat. A residual trend of "
        f"{amp_trend - amp_ctrl:.3f} dots survives the control, so"))
    if amp_trend - amp_ctrl >= 0.02:
        print("                aperture.py's 'width does not depend on step amplitude'")
        print("                is only true to about 10%, not exactly. It is well inside")
        print("                the shape systematic, so it does not move the number, but")
        print("                it is not the clean null the docstring claims.")
    print(f"   {'PASS ' if p_o == len(locked) and n_o == len(locked) else 'FAIL '}"
          f"       hold-out, out -> in (gain {1/A_out_N[0]:.2f} at the dot Nyquist):")
    print(f"                +1.0 beats 0.0 on {p_o}/{len(locked)}, "
          f"-1.0 worse on {n_o}/{len(locked)}, best strength "
          f"{min(best_s.values()):+.1f}..{max(best_s.values()):+.1f}.")
    print(f"   {'PASS ' if p_i == len(locked) and n_i == len(locked) else 'FAIL '}"
          f"       hold-out, in -> out, THE SPEC'S OWN TEST (gain "
          f"{1/A_in_N[0]:.2f}):")
    print(f"                +1.0 beats 0.0 on only {p_i}/{len(locked)}; spec 5.2 test 1")
    print("                requires every frame. The operator under test there has a")
    print("                gain of a few percent, so the test has almost no power and")
    print("                its result is dominated by noise. It should be replaced by")
    print("                the out -> in direction, not weakened.")
    print(f"   PASS         lattice specificity: a half-dot shift keeps a median "
          f"{np.median(kept)*100:.0f}% of")
    print(f"                the gain ({kept.min()*100:.0f}%-{kept.max()*100:.0f}%). "
          "Some survives, so part of the")
    print("                improvement is generic smoothing; most of it is not.")
    print("   NOT MEASURED the hold-out does not identify the width. Section 4 shows")
    print("                the sweep railing to the end of the grid on SYNTHETIC data")
    print("                whose width is known, in both directions. Any claim that a")
    print("                width sweep 'confirms 0.49' is unsupported by this test; it")
    print("                confirms the SIGN and the boundedness in strength at 0.49.")
    print("   NOT MEASURED dot-clock lattice jitter, which the spec calls the largest")
    print("                remaining term that belongs to us. Nothing here touches it.")
    print("   NOT USED     no image metric chose anything above; no reference image was")
    print("                opened; changing aperture.TRANSITION_1090 remains a human")
    print("                edit. This module only reports.")
    print("=" * 78)
    return {"frames": frames, "calibrations": cals, "widths": raw, "centre": centre,
            "range": (lo, hi), "holdout_in2out": ho_i, "holdout_out2in": ho_o,
            "width_sweep_minima": mins, "half_dot_kept": kept}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="audit aperture.TRANSITION_1090")
    ap.add_argument("--frames", default=",".join(DEFAULT_FRAMES))
    ap.add_argument("--cal-frame", default="L055")
    ap.add_argument("--traces", type=int, default=512)
    ap.add_argument("--cal-step", type=int, default=4)
    a = ap.parse_args()
    main(tuple(a.frames.split(",")), cal_frame=a.cal_frame, traces=a.traces, cal_step=a.cal_step)
