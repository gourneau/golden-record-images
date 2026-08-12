"""Recover the dot clock: the sample rate of the 1977 scan converter itself.

THE POINT
---------
Colorado Video's Series-262 scan converter sampled the source television
picture ONCE PER SCAN LINE and held each value -- patent US4802008 quotes the
spec. So a trace is a staircase of ~262.5 discrete plateaus (one NTSC field,
262.5 lines), not a ramp. That is a sampled-data signal, and it should be
decoded like one: find the dot clock, then integrate each dot over exactly its
own plateau. Binning on any other grid mixes adjacent dots.

CONFIRMED IN THE DATA
---------------------
The picture-band spectrum, averaged over ~440 traces with the smooth
background divided out, shows a narrow line at the predicted rate on 16 of 22
frames sampled across the record: dots/trace mean 262.519, sd 0.043, vs the
predicted 262.500. The line only appears when the picture has enough
dot-to-dot contrast to excite it (R013 gave nothing above 1.41x background),
so the clock is measured where visible and predicted from the trace period
where not; `measured` says which.

This also explains the "-12 samples on even traces" Ron Barry hardcoded in
2017: twelve 384 kHz samples is one dot -- one NTSC line period at the 2x tape
speed. It was the converter's own clock showing through.

PHASE: MEASURED PER BLOCK, NOT GLOBALLY (2026-08 rebuild)
---------------------------------------------------------
Plateau transitions were located by folding |dx| of the picture band at the
dot period. Measured on L000/L020/L055/R040:

  * relative to each trace's own start the transition phase ALTERNATES
    ~0.36-0.45 dots with trace parity -- the half-dot slip that 262.5
    dots/trace forces. An absolute-axis grid (position mod spd) reproduces
    this alternation by construction, so that is the grid used;
  * the absolute phase DRIFTS 0.6-1.4 dots across a frame (tape wow plus
    residual clock error), so one global phase is wrong for most of the
    frame. It is therefore measured in overlapping ~64-trace blocks and
    circularly smoothed; block estimates at 96 kHz and 384 kHz agree to
    ~0.05 dots, so the tracker resolves real drift with ~0.1-dot noise;
  * the phase estimator of the first version of this module (sum of
    row-relative Fourier coefficients) was WRONG: the half-dot alternation
    makes consecutive rows' coefficients cancel, so its phase was noise.
    Retired, not kept as a fallback.

Verified by decode phase sweep on L020/L000 (see decode.py): along-trace step
contrast of the sampled dots peaks at the tracked phase and dips at the
tracked phase + half a dot, exactly as plateau integration predicts.

NOTE: the phase enters only modulo one dot. The picture CONTENT sits on a
uniform per-trace grid (adjacent columns align to ~0.2 samples at 96 kHz --
see sync.py), so the dot grid controls only where integration windows fall,
never which scene row a dot is assigned to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# One NTSC field is 262.5 lines, and the converter took one sample per line.
DOTS_PER_TRACE = 262.5

# Spectral excess (peak/background) below which the clock line is declared
# invisible on this frame and the predicted rate is used instead.
MIN_STRENGTH = 1.8


@dataclass
class DotClock:
    samples_per_dot: float
    dots_per_trace: float
    strength: float  # spectral excess over background, 1.0 means nothing there
    measured: bool  # False if we fell back to the predicted rate
    # Per-trace absolute plateau-edge offset in samples: near trace i, plateau
    # k spans [phase[i] + k*spd, phase[i] + (k+1)*spd). Meaningful mod spd.
    phase: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # 0..1: coherence of the |dx| fold the phase came from. ~0.01-0.05 on real
    # frames (the transitions are a small part of |dx|); 0 when not tracked.
    coherence: float = 0.0


def _picture_stack(x: np.ndarray, tb, lo_f: float, hi_f: float, n_traces: int) -> np.ndarray:
    lo = int(lo_f * tb.period)
    hi = int(hi_f * tb.period)
    width = hi - lo
    rows = []
    for i in range(10, min(n_traces, tb.n_traces) - 10):
        a = int(tb.trace_start(i)) + lo
        if a < 0 or a + width > len(x):
            continue
        rows.append(x[a : a + width])
    if not rows:
        raise ValueError("no usable traces for dot-clock measurement")
    R = np.asarray(rows, dtype=np.float64)
    return R - R.mean(axis=1, keepdims=True)


def measure(
    x: np.ndarray,
    tb,
    *,
    lo_f: float,
    hi_f: float,
    n_traces: int = 460,
    search: float = 0.03,
) -> DotClock:
    """Find the dot clock rate in one frame's picture band.

    Works in units of the signal's own samples throughout (rate-agnostic).
    Phase is NOT measured here -- call track_phase for that.
    """
    predicted_spd = tb.period / DOTS_PER_TRACE

    R = _picture_stack(x, tb, lo_f, hi_f, n_traces)
    n = R.shape[1]
    w = np.hanning(n)
    P = np.mean(np.abs(np.fft.rfft(R * w, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n)  # cycles per sample

    # Divide out a smooth background so a narrow clock line stands out against
    # the picture's own broadband spectrum.
    bg = np.convolve(P, np.ones(41) / 41, mode="same")
    excess = P / np.maximum(bg, 1e-30)

    predicted_f = 1.0 / predicted_spd
    band = (f > predicted_f * (1 - search)) & (f < predicted_f * (1 + search))
    if not band.any():
        return DotClock(predicted_spd, DOTS_PER_TRACE, 1.0, False)

    fb, eb = f[band], excess[band]
    k = int(np.argmax(eb))
    strength = float(eb[k])

    if strength < MIN_STRENGTH:  # nothing convincing; trust the trace period
        return DotClock(predicted_spd, DOTS_PER_TRACE, strength, False)

    # Parabolic refinement of the spectral peak.
    peak = fb[k]
    if 0 < k < len(eb) - 1:
        y0, y1, y2 = eb[k - 1], eb[k], eb[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            peak = fb[k] + 0.5 * (y0 - y2) / den * (fb[1] - fb[0])

    spd = 1.0 / peak
    return DotClock(float(spd), float(tb.period / spd), strength, True)


def track_phase(
    x: np.ndarray,
    starts: np.ndarray,
    period: float,
    spd: float,
    lo_f: float,
    hi_f: float,
    block: int = 64,
    hop: int = 32,
) -> tuple[np.ndarray, float]:
    """Per-trace absolute plateau-transition phase, in samples.

    Sample-and-hold means the signal only changes at plateau boundaries, so
    |dx| concentrates there (content edges included -- they too can only occur
    on a boundary). Fold |dx| at the dot period on the ABSOLUTE sample axis
    per block of traces: the circular first moment's angle is the transition
    phase, its magnitude the evidence. Blocks are then circularly smoothed
    (complex moving average -- no unwrapping to get wrong) and interpolated to
    every trace.

    Returns (phase[n_traces] in samples, coherence 0..1). Phase is the
    position of a plateau EDGE mod spd on the absolute axis.
    """
    d = np.abs(np.diff(np.asarray(x, dtype=np.float64)))
    n = len(starts)
    centres, zs, tots = [], [], []
    for b0 in range(0, max(n - block, 0) + 1, hop):
        z = 0.0 + 0.0j
        tot = 0.0
        for i in range(b0, min(b0 + block, n)):
            a = int(starts[i] + lo_f * period)
            b = int(starts[i] + hi_f * period)
            if a < 0 or b > len(d) or b <= a:
                continue
            t = np.arange(a, b) + 0.5  # |dx| sample sits between t and t+1
            seg = d[a:b]
            z += np.sum(seg * np.exp(2j * np.pi * t / spd))
            tot += float(np.sum(seg))
        centres.append(b0 + block / 2.0)
        zs.append(z)
        tots.append(tot)
    if not zs:
        return np.zeros(n), 0.0
    zs = np.asarray(zs)
    tots = np.asarray(tots)
    coherence = float(np.abs(zs.sum()) / max(tots.sum(), 1e-12))

    # Circular smoothing: average the complex moments over +-1 neighbouring
    # blocks (~128 traces). Wider would bias against the measured 0.6-1.4 dot
    # per-frame drift; narrower leaves ~0.2-dot block noise in.
    sm = np.empty_like(zs)
    for i in range(len(zs)):
        lo = max(0, i - 1)
        sm[i] = zs[lo : i + 2].sum()

    ph_dots = (np.angle(sm) / (2 * np.pi)) % 1.0  # transition phase, dots
    # Interpolate to every trace on the complex circle (no branch cuts).
    zc = np.exp(2j * np.pi * ph_dots)
    tr = np.arange(n, dtype=np.float64)
    re = np.interp(tr, centres, zc.real)
    im = np.interp(tr, centres, zc.imag)
    phase = (np.angle(re + 1j * im) / (2 * np.pi)) % 1.0
    return phase * spd, coherence


def sample_plateaus(
    x: np.ndarray,
    mid: float,
    n_out: int,
    spd: float,
    psi: float,
    cs: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate `n_out` plateaus of one trace, centred on `mid`.

    Output row r's nominal centre is mid + (r - (n_out-1)/2)*spd on the
    trace's UNIFORM grid; the value is the exact integral of the plateau
    [psi + k*spd, psi + (k+1)*spd) that CONTAINS that centre. Snapping row
    centres (which step by exactly spd) to plateaus guarantees consecutive,
    distinct plateaus -- no duplicated or skipped scene line, and no
    knife-edge where a parity phase flip could shift every row index by one.
    The at-most-half-dot offset between nominal centre and plateau centre is
    the converter's own half-dot sampling alternation, i.e. physically real.

    Integration is exact to sub-sample via a linearly interpolated cumulative
    sum (`cs` = concat([0], cumsum(x)), passable in to amortise).
    """
    if cs is None:
        cs = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
    c0 = mid - (n_out - 1) / 2.0 * spd
    k0 = np.floor((c0 - psi) / spd)
    edges = psi + (k0 + np.arange(n_out + 1)) * spd
    edges = np.clip(edges, 0.0, len(x) - 1e-9)

    base = np.floor(edges).astype(np.int64)
    frac = edges - base
    C = cs[base] + frac * x[base]  # integral of x over [0, edge)
    return np.diff(C) / spd
