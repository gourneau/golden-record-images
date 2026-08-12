"""Timebase recovery: find where every trace begins, to a fraction of a sample.

Barry's 2017 decoder picked the maximum sample in a 190-sample window, then the
minimum after it. That works, but the sync region has several features and
peak-picking latches onto whichever one happens to be tallest on a given line --
which is why his traces came out alternately 3100 and 3300 samples apart and why
he needed a hardcoded `+/-12 samples on even traces, changing at trace 164` fudge.

We do it differently:

  1. Fold several hundred lines together at the nominal period. Picture content
     averages away; everything locked to the line clock survives. That gives a
     high-SNR template of the *whole* sync region.
  2. Cross-correlate each line against that template and parabolically
     interpolate the correlation peak -> sub-sample position, using every sync
     feature at once instead of one fragile extremum.
  3. Fit position-vs-line-index with iteratively reweighted least squares. The
     slope is the true line period, the intercept the frame phase, and the
     residuals are the master tape's wow and flutter.

Two passes: the first template is built on the nominal period, then rebuilt on
the fitted one so it is not smeared by a wrong guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Measured from the master by autocorrelation over a 50-line baseline. The
# record cover specifies 8.34 ms/trace (11845632 hydrogen hyperfine periods);
# this digitisation runs ~0.16% fast, so we only ever use this as a starting
# guess and fit the real value per frame.
NOMINAL_PERIOD = 3197.4
TRACES_PER_FRAME = 512


@dataclass
class Timebase:
    """Recovered line timing for one frame."""

    phase: float  # sub-sample position of trace 0, from the straight-line fit
    period: float  # samples per trace, fitted
    positions: np.ndarray  # (n,) measured sub-sample position of each trace
    smoothed: np.ndarray  # (n,) measured positions with measurement noise removed
    residuals: np.ndarray  # (n,) measured minus fitted, i.e. wow and flutter
    template: np.ndarray  # the matched filter used
    template_origin: int  # index within `template` treated as the trace start
    n_traces: int
    blank_len: float = 0.0  # measured blanking burst length, in samples

    @property
    def jitter_rms(self) -> float:
        """RMS departure from constant line rate, in samples."""
        return float(np.sqrt(np.mean(self.residuals**2)))

    @property
    def measurement_noise(self) -> float:
        """RMS of what smoothing removed -- our own uncertainty, not the tape's."""
        return float(np.sqrt(np.mean((self.positions - self.smoothed) ** 2)))

    def trace_start(self, i: int) -> float:
        """Where trace `i` begins.

        Uses the *smoothed measured* position rather than the straight-line fit,
        so genuine wow and flutter in the master is corrected rather than
        averaged into a slant. Falls back to the fit outside the measured range.
        """
        if 0 <= i < len(self.smoothed):
            return float(self.smoothed[i])
        return self.phase + i * self.period


def _savgol(y: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing with edge handling by polynomial extension.

    The timing drift is smooth (lag-1 autocorrelation ~0.9) while our
    per-line measurement noise is not, so this separates the two.
    """
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
    """Measure the line period by autocorrelation, before any template work.

    This has to happen first. The template search only looks +/- `search`
    samples around where it predicts a trace should be, and that prediction
    comes from the period. An error of only 0.12 samples per trace accumulates
    to 60 samples over a 512-trace frame -- enough to walk the search window
    clean off the sync burst and start locking onto picture content instead,
    which decodes as a diagonal staircase sweeping across the image.

    Autocorrelation has no such failure mode: successive traces of a photograph
    resemble each other, so the correlation peaks at the line period regardless
    of how far that is from our guess. Measuring across a ~50-trace baseline
    then divides the timing error by 50.
    """
    n = len(x)
    lag = int(guess * 50)
    span = min(n, max(int(guess * 80), lag * 2))
    if span < lag + 16:
        return guess
    seg = np.asarray(x[:span], dtype=np.float64)
    seg = seg - seg.mean()
    # Autocorrelation via FFT. np.correlate(mode="full") is direct O(n^2), which
    # on a quarter-million samples is ~65 billion operations -- it hangs rather
    # than returns, and would be equally fatal in the browser port.
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


def fold(x: np.ndarray, period: float, n_lines: int, width: int) -> np.ndarray:
    """Average `n_lines` lines aligned on the nominal period."""
    acc = np.zeros(width)
    used = 0
    for i in range(n_lines):
        o = int(round(i * period))
        seg = x[o : o + width]
        if len(seg) < width:
            break
        acc += seg
        used += 1
    if used == 0:
        raise ValueError("no complete lines available to fold")
    return acc / used


def find_blanking(folded: np.ndarray) -> tuple[int, int]:
    """Locate the sync/blanking burst within a folded trace.

    This is the only stretch of a trace that carries no picture, which makes it
    the only honest thing to build a matched filter from. It shows up in the
    fold as a strong, sustained positive excursion.

    Getting this wrong is not a subtle error. An earlier version of this file
    anchored on the deep negative notch and cut the template from `notch +/-
    120` samples -- but the notch sits *inside* the picture, 2563 samples before
    the burst, so the "matched filter" was correlating against averaged picture
    content. It held up on the line-art diagrams, whose columns all resemble
    each other, and collapsed on the detailed photographs, which is exactly the
    pattern of failure we saw across the frame set.
    """
    n = len(folded)
    # Smooth over ~2% of a trace so picture detail cannot form a false peak.
    k = max(3, (n // 50) | 1)
    pad = np.pad(folded, k // 2, mode="wrap")
    sm = np.convolve(pad, np.ones(k) / k, mode="valid")[:n]

    thr = sm.mean() + 0.5 * (sm.max() - sm.mean())
    on = sm > thr
    # Longest contiguous run, treating the trace as circular.
    best_len, best_lo, cur = 0, 0, None
    for i in range(2 * n):
        if on[i % n] and cur is None:
            cur = i
        elif not on[i % n] and cur is not None:
            if i - cur > best_len:
                best_len, best_lo = i - cur, cur
            cur = None
            if i >= n:
                break
    if best_len == 0 or best_len > n // 2:
        # No clear burst; fall back to the strongest single point.
        p = int(np.argmax(sm))
        return p, min(n // 8, 400)
    return best_lo % n, best_len


def _template_from_fold(
    folded: np.ndarray, margin: int, tail: int
) -> tuple[np.ndarray, int]:
    """Cut the blanking burst out of a folded trace, plus a little either side.

    The returned origin is the index within the template of the burst's leading
    edge, which we adopt as the trace start. That convention makes the picture
    a single contiguous run from the end of blanking to the start of the next,
    with no wrap.
    """
    start, length = find_blanking(folded)
    idx = np.arange(start - margin, start + length + tail) % len(folded)
    return folded[idx].copy(), margin


def _refine_peak(c: np.ndarray, k: int) -> float:
    """Parabolic sub-sample refinement of a correlation peak at index k."""
    if k <= 0 or k >= len(c) - 1:
        return float(k)
    y0, y1, y2 = c[k - 1], c[k], c[k + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(k)
    return k + 0.5 * (y0 - y2) / denom


def _correlate_line(
    x: np.ndarray, centre: float, template: np.ndarray, search: int
) -> float | None:
    """Sub-sample offset of the template near `centre`, or None if out of range."""
    lo = int(round(centre)) - search
    hi = lo + len(template) + 2 * search
    if lo < 0 or hi > len(x):
        return None
    seg = x[lo:hi]
    # Correlate on the derivative: it emphasises the sync edges and is immune to
    # the slow AC-coupling droop that otherwise dominates the raw signal.
    c = np.correlate(np.diff(seg), np.diff(template), mode="valid")
    k = int(np.argmax(c))
    return lo + _refine_peak(c, k)


def _irls_fit(idx: np.ndarray, pos: np.ndarray, iters: int = 6) -> tuple[float, float, np.ndarray]:
    """Robust straight-line fit; returns (intercept, slope, residuals)."""
    w = np.ones_like(pos)
    intercept = slope = 0.0
    for _ in range(iters):
        sw = w.sum()
        mx = (w * idx).sum() / sw
        my = (w * pos).sum() / sw
        sxx = (w * (idx - mx) ** 2).sum()
        sxy = (w * (idx - mx) * (pos - my)).sum()
        slope = sxy / sxx
        intercept = my - slope * mx
        r = pos - (intercept + slope * idx)
        # Tukey biweight, scaled by a robust sigma estimate.
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.clip(r / (4.685 * s), -1, 1)
        w = (1 - u**2) ** 2
    return float(intercept), float(slope), pos - (intercept + slope * idx)


def recover(
    x: np.ndarray,
    *,
    period_guess: float = NOMINAL_PERIOD,
    n_traces: int = TRACES_PER_FRAME,
    search: int = 60,
    margin: int = 140,
    tail: int = 140,
    passes: int = 3,
    smooth_window: int = 15,
) -> Timebase:
    """Recover the timebase of one frame from its signal.

    `x` must start at or slightly before the frame's first trace.
    """
    # Measure the period before any template work; see estimate_period.
    period = estimate_period(x, period_guess)
    template = np.zeros(0)
    origin = 0
    positions = np.zeros(0)
    residuals = np.zeros(0)
    phase = 0.0

    for p in range(passes):
        width = int(round(period)) + 1
        n_fold = min(n_traces, max(1, (len(x) - width) // int(round(period))))
        folded = fold(x, period, n_fold, width)
        template, origin = _template_from_fold(folded, margin, tail)

        # Trace 0 is the first blanking burst that leaves room for the whole
        # template plus the search window on either side.
        burst, blank_len = find_blanking(folded)
        first = float(burst) + period
        raw_idx, raw_pos = [], []
        for i in range(n_traces):
            centre = first + i * period - origin
            off = _correlate_line(x, centre, template, search)
            if off is None:
                continue
            raw_idx.append(i)
            raw_pos.append(off + origin)  # convert back to trace-start position

        if len(raw_idx) < 32:
            raise ValueError(f"only {len(raw_idx)} traces located; signal too short?")

        idx = np.asarray(raw_idx, dtype=np.float64)
        pos = np.asarray(raw_pos, dtype=np.float64)
        phase, period, residuals = _irls_fit(idx, pos)
        positions = pos
        if p == passes - 1:
            smoothed = _savgol(pos, smooth_window)
            return Timebase(
                blank_len=float(blank_len),
                phase=phase,
                period=period,
                positions=pos,
                smoothed=smoothed,
                residuals=residuals,
                template=template,
                template_origin=origin,
                n_traces=len(idx),
            )

    raise AssertionError("unreachable")
