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


def _template_from_fold(folded: np.ndarray, pre: int, post: int) -> tuple[np.ndarray, int]:
    """Cut the sync region out of a folded line.

    The trace start is the deep negative notch that follows the sync plateau's
    falling edge. We centre the template on it and keep `pre` samples before
    (covering the plateau and edge) and `post` after.
    """
    notch = int(np.argmin(folded))
    idx = (np.arange(notch - pre, notch + post)) % len(folded)
    return folded[idx].copy(), pre


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
    pre: int = 120,
    post: int = 40,
    passes: int = 2,
    smooth_window: int = 15,
) -> Timebase:
    """Recover the timebase of one frame from its signal.

    `x` must start at or slightly before the frame's first trace.
    """
    period = period_guess
    template = np.zeros(0)
    origin = 0
    positions = np.zeros(0)
    residuals = np.zeros(0)
    phase = 0.0

    for p in range(passes):
        width = int(round(period)) + 1
        n_fold = min(n_traces, max(1, (len(x) - width) // int(round(period))))
        folded = fold(x, period, n_fold, width)
        template, origin = _template_from_fold(folded, pre, post)

        # First guess at where trace 0 sits: the notch nearest the start of x.
        # `pre` samples of template precede the notch, so the first line we can
        # correlate is the one whose template window fits inside x.
        first = float(np.argmin(folded)) + period
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
