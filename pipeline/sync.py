"""Timebase recovery: find where every trace begins, to a fraction of a sample.

v2 -- rebuilt on measurements of the 384 kHz master. The v1 template-fold
approach lives in sync_v1_backup.py; it is structurally wrong for this signal
because THE SYNC PULSE WIDTH ALTERNATES WITH TRACE PARITY (wide pulse: high
plateau for the last ~140 samples at 384 kHz before the fall; narrow pulse:
high only for the last ~40), so a single folded template correlates early on
one parity and late on the other -- and worse, v1's blanking detector latched a
different pulse feature on different frames: the measured spread of its anchor
across frames was 38 to 104 samples at 96 kHz, i.e. the picture window moved by
up to 17 pixels from frame to frame.

What both parities share is the TRAILING FALLING EDGE: from the high plateau
(~0.10-0.19) through a deep trough within ~8 samples at 384 kHz. Its
half-amplitude downward crossing measured the most stable landmark on this
master (spacing std 5.5-6.2 samples at 384 kHz, vs ~100 for the peak Barry's
decoder used). We lock onto that crossing:

  1. Period first, by FFT autocorrelation over a ~50-line baseline
     (estimate_period, unchanged from v1). This cannot be skipped: it was
     tried during the v2 rebuild, and with the nominal period 0.7 samples/line
     wrong the phase vote smeared by ~360 samples and half the frames lost
     lock.
  2. A bipolar "drop score" over the whole signal: mean of a short window
     before each sample minus mean of a short window after. Picture content
     does not produce the plateau-then-trough swing, so the score is large
     only on sync falls. A phase vote (fold the score at the period over all
     traces) finds the frame phase without any threshold.
  3. Predict-and-correct with coasting: search a narrow window around each
     predicted landmark, take the drop-score argmax, refine to the sub-sample
     half-amplitude crossing. Half-amplitude, not zero: the record's
     AC-coupling offset means the raw signal need not cross zero at all -- an
     absolute-zero test measurably loses half the traces on some frames
     because the trough after the narrow pulse is shallow. A validity gate
     (score above 0.45 x the frame median) rejects traces where picture or
     dropouts swamp the sync; those coast on the smoothed prediction and are
     flagged, never invented.
  4. Robust straight-line fit for period/phase + Savitzky-Golay smoothing of
     the residual for wow and flutter, iterated with a shrinking window.
  5. Parity: the crossing alternates a further ~1.6 samples at 96 kHz between
     even and odd traces. Surveyed over all 156 frames: 148 measure
     -1.62 +/- 0.03, TWO measure +1.6 (R015, R025 -- the sign flip prior work
     saw; a global rule would misplace their traces by 1.7 px), and six
     left-channel frames (L000/L004/L005/L021/L037/L075) measure ~0.0 with no
     mid-frame flip (checked by running median), i.e. their pulses genuinely
     do not alternate. So the offset MUST be measured per frame, never
     hardcoded. 1.6 samples at 96 kHz is half of one scan-converter dot
     (period/262.5): this is Barry's "+/-12 samples at 384 kHz on even
     traces" fudge -- one dot -- measured instead of guessed. The offset is
     measured from the fit residuals and REMOVED from the reported trace
     starts, because the picture itself sits on a uniform grid: removing it
     lowers the image-domain odd/even misalignment energy (parity_db 7.4 ->
     3.5 on L055, 6.7 -> 4.7 on R040), and a direct high-passed NCC between
     adjacent picture columns then measures the residual alternation at only
     -0.19 +/- 0.04 samples across frames spanning the whole record.

Measured per-landmark precision, second-difference method (immune to genuine
smooth flutter): 0.41-0.70 samples RMS at 96 kHz (~1.7-2.8 at 384 kHz, ~0.24
px) on frames spanning the record, photographs included. A parity-split
template correlator measured the same (0.39-0.70), so the simpler half-cross
is used. v1 measured 0.5-2.5 on line art and 2-6 on photographs.

The trace-start convention is the half-amplitude downward crossing of the sync
falling edge ("the crossing"). Relative to it, measured on sub-sample-aligned
folds of eight frames spanning both channels and the whole record (96 kHz
samples, period ~799.35): trough at +1..+3, picture from about +6 to about
+716, content-free shelf from ~+717, next pulse's high plateau ending at the
next crossing one period later. decode.py's window constants are expressed in
this convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Measured from the master by autocorrelation over a 50-line baseline. The
# record cover specifies 8.34 ms/trace (11845632 hydrogen hyperfine periods);
# this digitisation runs ~0.16% fast and the period DRIFTS monotonically
# across the record (~3197.2 down to ~3193.6 at 384 kHz), so this is only ever
# a starting guess; the real value is fitted per frame.
NOMINAL_PERIOD = 3197.4
TRACES_PER_FRAME = 512

# Detector geometry scales with the period so recovery behaves identically at
# 96 kHz (period ~799.35) and 384 kHz (period ~3197.4).
_GAP_FRAC = 0.00375  # half-gap of the bipolar drop windows (3 samples at 96 kHz)
_AVG_FRAC = 0.010  # length of each drop window (8 samples at 96 kHz)
_SMOOTH_N = 3  # boxcar on the signal for the sub-sample crossing
_GATE = 0.45  # validity gate: drop score vs frame median
_SG_WINDOW = 31  # Savitzky-Golay window (traces) for the wow/flutter trend
_TPL_BACK_FRAC = 0.115  # template extent before the crossing (~92 samples at 96k)
_TPL_FWD_FRAC = 0.031  # and after (~25 samples)


@dataclass
class Timebase:
    """Recovered line timing for one frame.

    `positions` holds the measured landmark of every trace where the validity
    gate passed, and the smoothed prediction where it did not; `located` says
    which is which. `smoothed` is the parity-corrected uniform-grid trajectory
    that trace_start() serves.
    """

    phase: float  # sub-sample position of trace 0, from the straight-line fit
    period: float  # samples per trace, fitted
    positions: np.ndarray  # (n,) measured landmark, or prediction where coasted
    smoothed: np.ndarray  # (n,) trend with measurement noise + parity removed
    residuals: np.ndarray  # (n,) positions minus fit, i.e. wow and flutter
    template: np.ndarray  # mean sync-region fold (for PSF measurement etc.)
    template_origin: int  # index within `template` of the trace start
    n_traces: int
    blank_len: float = 0.0  # measured blanking extent, in samples
    located: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    parity_offset: float = 0.0  # even-minus-odd landmark alternation, samples
    lock_quality: float = 0.0  # located fraction x noise figure, 0..1

    @property
    def jitter_rms(self) -> float:
        """RMS departure from constant line rate, in samples (located traces)."""
        r = self.residuals[self.located] if self.located.any() else self.residuals
        return float(np.sqrt(np.mean(r**2))) if len(r) else 0.0

    @property
    def measurement_noise(self) -> float:
        """RMS of what smoothing removed, over traces actually measured."""
        if not self.located.any():
            return float("inf")
        d = (self.positions - self._parity_signed() - self.smoothed)[self.located]
        return float(np.sqrt(np.mean(d**2)))

    def _parity_signed(self) -> np.ndarray:
        """Per-trace landmark offset implied by the measured parity alternation."""
        n = len(self.positions)
        s = np.where(np.arange(n) % 2 == 0, 0.5, -0.5)
        return s * self.parity_offset

    def trace_confidence(self) -> np.ndarray:
        """Per-trace 0..1: 0 where coasted, else how well the landmark agreed
        with the smooth trend the rest of the frame voted for."""
        dev = np.abs(self.positions - self._parity_signed() - self.smoothed)
        good = dev[self.located]
        scale = 1.4826 * np.median(np.abs(good - np.median(good))) + 1e-9 if len(good) else 1.0
        conf = np.clip(1.0 - dev / (8.0 * scale), 0.0, 1.0)
        conf[~self.located] = 0.0
        return conf

    def trace_start(self, i: int) -> float:
        """Where trace `i` begins (the parity-corrected uniform grid)."""
        if 0 <= i < len(self.smoothed):
            return float(self.smoothed[i])
        return self.phase + i * self.period


def _savgol(y: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing with edge handling by polynomial extension."""
    n = len(y)
    window = min(window | 1, n if n % 2 else n - 1)
    if window < order + 2:
        return y.copy()
    half = window // 2
    out = np.empty(n)
    t = np.arange(window, dtype=np.float64) - half
    vander = np.vander(t, order + 1)
    pinv = np.linalg.pinv(vander)
    for i in range(n):
        lo = min(max(0, i - half), n - window)
        seg = y[lo : lo + window]
        coef = pinv @ seg
        out[i] = np.polyval(coef, i - lo - half)
    return out


def estimate_period(x: np.ndarray, guess: float, tolerance: float = 0.02) -> float:
    """Measure the line period by autocorrelation, before any landmark work.

    This has to happen first. Both the phase vote and the landmark search
    predict positions from the period; an error of only 0.12 samples per trace
    accumulates to 60 samples over a 512-trace frame -- enough to walk the
    search clean off the sync and lock onto picture content instead, which
    decodes as a diagonal staircase sweeping across the image.

    Autocorrelation has no such failure mode: successive traces of a
    photograph resemble each other, so the correlation peaks at the line
    period regardless of how far that is from our guess. Measuring across a
    ~50-trace baseline then divides the timing error by 50.
    """
    n = len(x)
    lag = int(guess * 50)
    span = min(n, max(int(guess * 80), lag * 2))
    if span < lag + 16:
        return guess
    seg = np.asarray(x[:span], dtype=np.float64)
    seg = seg - seg.mean()
    # Autocorrelation via FFT; direct correlation is O(n^2) and hangs.
    nfft = 1 << int(np.ceil(np.log2(2 * span)))
    F = np.fft.rfft(seg, nfft)
    ac = np.fft.irfft(F * np.conj(F), nfft)[:span]
    if ac[0] <= 0:
        return guess
    ac = ac / ac[0]

    lo = int(lag * (1 - tolerance))
    hi = min(len(ac) - 2, int(lag * (1 + tolerance)))
    if hi <= lo:
        return guess
    k = lo + int(np.argmax(ac[lo:hi]))
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    den = y0 - 2 * y1 + y2
    if den != 0:
        k = k + 0.5 * (y0 - y2) / den
    return float(k) / 50.0


def _drop_score(x: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bipolar edge score: mean-before minus mean-after at every sample.

    High only where a sustained high level is followed by a sustained low one
    within a few samples -- the sync falling edge. Returns (score, left mean,
    right mean); the levels feed the half-amplitude crossing.
    """
    gap = max(2, int(round(_GAP_FRAC * period)))
    k = max(4, int(round(_AVG_FRAC * period)))
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(n)
    lo = np.clip(idx - gap - k, 0, n)
    hi = np.clip(idx - gap, 0, n)
    left = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    lo2 = np.clip(idx + 1 + gap, 0, n)
    hi2 = np.clip(idx + 1 + gap + k, 0, n)
    right = (c[hi2] - c[lo2]) / np.maximum(hi2 - lo2, 1)
    return left - right, left, right


def _phase_vote(drop: np.ndarray, period: float, n_traces: int) -> int:
    """Fold the drop score at the period; the sync phase wins the vote.

    Threshold-free and dropout-proof: the sync edge stacks coherently across
    every trace while picture features drift, so the sync column dominates
    even on frames whose picture also carries strong horizontal edges.
    """
    W = int(period)
    v = np.zeros(W)
    for k in range(n_traces):
        o = int(round(k * period))
        seg = drop[o : o + W]
        if len(seg) == W:
            v += seg
    return int(np.argmax(v))


def _half_cross(xs: np.ndarray, j: int, mid: float, radius: int) -> float | None:
    """Sub-sample position where xs crosses `mid` downward, nearest to j."""
    for d in range(radius + 1):
        for jj in (j + d, j - d) if d else (j,):
            if jj < 1 or jj + 1 >= len(xs):
                continue
            if xs[jj] >= mid > xs[jj + 1]:
                a, b = xs[jj], xs[jj + 1]
                return jj + (a - mid) / (a - b)
    return None


def _robust_line(k: np.ndarray, p: np.ndarray, iters: int = 5) -> tuple[float, float]:
    """IRLS straight-line fit (Tukey biweight); returns (slope, intercept)."""
    A = np.vstack([k, np.ones_like(k)]).T
    w = np.ones(len(k))
    slope, intercept = 0.0, 0.0
    for _ in range(iters):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], p * sw, rcond=None)
        slope, intercept = float(coef[0]), float(coef[1])
        r = p - (intercept + slope * k)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.clip(r / (4.0 * s), -1, 1)
        w = (1 - u**2) ** 2
    return slope, intercept


def _fold_template(
    x: np.ndarray, pos: np.ndarray, located: np.ndarray, period: float
) -> tuple[np.ndarray, int]:
    """Mean sync-region waveform, sub-sample aligned on the located crossings."""
    back = int(round(_TPL_BACK_FRAC * period))
    fwd = int(round(_TPL_FWD_FRAC * period))
    grid = np.arange(-back, fwd + 1)
    acc = np.zeros(len(grid))
    used = 0
    for p in pos[located]:
        idx = p + grid
        if idx[0] < 0 or idx[-1] + 1 >= len(x):
            continue
        base = np.floor(idx).astype(np.int64)
        fr = idx - base
        acc += x[base] * (1 - fr) + x[base + 1] * fr
        used += 1
    if used == 0:
        return np.zeros(len(grid)), back
    return acc / used, back


def recover(
    x: np.ndarray,
    *,
    period_guess: float = NOMINAL_PERIOD,
    n_traces: int = TRACES_PER_FRAME,
    search: int = 0,  # initial half-window; 0 means 4% of a period
    passes: int = 3,
    smooth_window: int = _SG_WINDOW,
    remove_parity: bool = True,
) -> Timebase:
    """Recover the timebase of one frame from its signal.

    `x` must start at or slightly before the frame's first trace. Works at any
    sample rate; pass the matching `period_guess`.
    """
    x = np.asarray(x, dtype=np.float64)
    period = estimate_period(x, period_guess)

    drop, left, right = _drop_score(x, period)
    # 3-sample boxcar: suppresses single-sample noise on the crossing without
    # moving a ~2-sample edge.
    kernel = np.ones(_SMOOTH_N) / _SMOOTH_N
    xs = np.convolve(np.pad(x, _SMOOTH_N // 2, mode="edge"), kernel, mode="valid")[: len(x)]

    phi = float(_phase_vote(drop, period, n_traces))
    pred = phi + np.arange(n_traces) * period
    W = max(6, int(round(0.04 * period))) if search == 0 else search
    cross_rad = max(3, int(round(0.006 * period)))
    edge_margin = max(16, int(round(0.02 * period)))

    pos = np.full(n_traces, np.nan)
    score = np.zeros(n_traces)
    gate = np.zeros(n_traces, dtype=bool)
    smoothed = pred.copy()
    signed = np.zeros(n_traces)
    alt = 0.0
    slope, intercept = period, phi

    for _ in range(passes):
        pos[:] = np.nan
        score[:] = 0.0
        for k in range(n_traces):
            c = int(round(pred[k]))
            a, b = c - W, c + W + 1
            if a < edge_margin or b + edge_margin > len(x):
                continue
            j = a + int(np.argmax(drop[a:b]))
            p = _half_cross(xs, j, 0.5 * (left[j] + right[j]), cross_rad)
            if p is None:
                continue
            pos[k] = p
            score[k] = drop[j]

        valid = np.isfinite(pos)
        if valid.sum() < 32:
            raise ValueError(f"only {int(valid.sum())} traces located; signal too short?")
        med = np.median(score[valid])
        gate = valid & (score > _GATE * med)
        if gate.sum() < 32:
            raise ValueError(f"only {int(gate.sum())} traces passed the sync gate")

        kv = np.where(gate)[0].astype(np.float64)
        slope, intercept = _robust_line(kv, pos[gate])
        line = intercept + np.arange(n_traces) * slope
        res_g = pos[gate] - line[gate]

        # Parity alternation of the landmark, measured per frame.
        par = kv.astype(int) % 2
        if (par == 0).any() and (par == 1).any():
            alt = float(res_g[par == 0].mean() - res_g[par == 1].mean())
        else:
            alt = 0.0
        signed = np.where(np.arange(n_traces) % 2 == 0, 0.5, -0.5) * alt

        # Smooth trend of the parity-decoupled residual = wow and flutter.
        res_d = pos[gate] - signed[gate] - line[gate]
        r_all = np.interp(np.arange(n_traces), kv, res_d)
        s = 1.4826 * np.median(np.abs(res_d - np.median(res_d))) + 1e-9
        r_all = np.clip(r_all, np.median(res_d) - 6 * s, np.median(res_d) + 6 * s)
        smoothed = line + _savgol(r_all, smooth_window)
        pred = smoothed + signed  # predict the landmark, parity included
        W = max(4, int(round(0.0075 * period)))

    period = slope
    # Coasted traces carry the prediction, flagged via `located`.
    coasted = ~gate
    pos[coasted] = pred[coasted]

    noise = pos[gate] - signed[gate] - smoothed[gate]
    noise_rms = float(np.sqrt(np.mean(noise**2))) if gate.any() else float("inf")
    lock = float(gate.mean()) / (1.0 + noise_rms / (0.002 * period))

    template, origin = _fold_template(x, pos, gate, period)

    return Timebase(
        phase=intercept,
        period=period,
        positions=pos,
        smoothed=smoothed if remove_parity else smoothed + signed,
        residuals=pos - (intercept + np.arange(n_traces) * period),
        template=template,
        template_origin=origin,
        n_traces=int(n_traces),
        blank_len=0.104 * period,  # shelf start to crossing, measured fraction
        located=gate.copy(),
        parity_offset=alt,
        lock_quality=lock,
    )
