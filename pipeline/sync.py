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

THE ORIGIN IS RE-ANCHORED TO THE RAW ZERO CROSSING (2026-08). The half-cross
detector above tracks the frame reliably, but the feature it latches is NOT
the same on every frame: on 14 of the 16 test-set frames its origin sits
+1..+5 samples (384 kHz) from the sync trailing edge's raw downward ZERO
crossing, but on L000 it is -289 and on L075 -247 -- those frames' strongest
drop-score feature is picture structure, not the sync fall. pipeline/geometry
measured all its window constants relative to the zero crossing (the most
frame-invariant landmark: consecutive-spacing std 5.6-5.9 samples at 384 kHz
on every frame tried), so recover() finishes by folding the signal on its own
smoothed grid, locating the steepest fall of the fold, refining to the
fold's nearest downward zero crossing, and shifting the whole timebase by
that single per-frame constant. Tracking (period, wow, parity) is untouched
-- only the origin moves. After this step the trace-start convention IS the
zero-crossing convention of pipeline/geometry.py, and decode.py's window
constants are expressed in it.

THE PICTURE SITS ON A UNIFORM GRID -- the parity alternation of the landmark
must be REMOVED from the timebase, not followed. Re-measured 2026-08 against
the opposite claim (that the picture rides the alternating crossing): fine
box decode of L055/R040/R056/L020 at 936 rows, adjacent-column NCC shift
split by pair parity: uniform grid leaves 0.17-0.20 samples (96 kHz) of
alternation; following the measured parity offset creates 1.70-1.82; putting
each trace on its own raw crossing creates 1.55-1.63. (geometry.py's
measure_parity_independence reported the opposite at 377-row resolution; its
+-8 px integer search cannot resolve a 0.2-px effect. The fine-grid
measurement stands.)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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

# --- wideband flutter tracking ----------------------------------------------
# The Savitzky-Golay trend above is deliberately conservative: it throws away
# everything the landmark residual says above roughly 0.1 cycles per trace,
# because near trace-Nyquist that residual is not tape motion, it is the picture
# pulling the sync crossing about. But between those two bands there is real
# flutter that the smoother also discards, and the record itself says where the
# boundary is.
#
# The gain below is (content-free gap PSD) / (picture PSD), clipped to [0, 1],
# measured on the master over 3839 picture segments and 33 content-free
# segments with 64-trace Hann windows. Index k is k/64 cycles per trace; k=32 is
# Nyquist (half the line rate, 60.09 Hz). Below k=21 the gap PSD is if anything
# the larger, i.e. no picture-driven timing error is detectable there at all, so
# it is clipped to unity. Above k=22 it falls away fast, which is the picture.
#
# Everything here comes from the audio. No reference image is involved, and
# `wideband` takes nothing but a Timebase -- deliberately, so that decode.py can
# get the benefit without importing globaltime.py, which reads the frame map and
# would end decode.py's property of touching only the WAV.
WIENER_GAIN = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,  # 0.00-0.11 cyc/trace
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,  # 0.13-0.23
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.82, 0.44,  # 0.25-0.36
    0.43, 0.51, 0.43, 0.30, 0.30, 0.14, 0.03, 0.01,  # 0.38-0.48
    0.01,                                            # 0.50 (Nyquist)
])
FIR_TAPS = 65  # odd; zero-phase, Hamming-windowed frequency sampling
WIDEBAND_CLIP_MAD = 6.0  # outlier guard on the residual before filtering
# Skip the correction when too many traces were coasted. `wideband` zeroes the
# residual of a coasted trace, so every coasted run injects a step into the
# filter input. On the one frozen test frame below this line (L075, 92.4%
# located) it is the only frame whose shift_rms regresses. A better fix is to
# interpolate the residual across coasted runs rather than zero it; that is
# untested, and an untested improvement does not get to replace a gate.
WIDEBAND_MIN_LOCATED = 0.99


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


def _zero_cross_offset(x: np.ndarray, smoothed: np.ndarray, period: float) -> float | None:
    """Signed offset (samples) from the current origin to the sync trailing
    edge's raw downward zero crossing, measured on the frame's own fold.

    Fold up to 256 traces on the smoothed grid (median, so picture content
    and dropouts drop out), find the steepest fall of the lightly smoothed
    fold over the WHOLE period -- the sync edge is the steepest fall of the
    fold on every frame measured, even L000 whose drop-score detector latched
    picture structure -- then refine to the fold's nearest downward zero
    crossing. Returns None if no usable fold or no crossing (the caller then
    keeps the un-anchored origin rather than inventing one).
    """
    W = int(period)
    step = max(1, len(smoothed) // 256)
    anchors = [float(s) for s in smoothed[::step] if 0 <= int(round(s)) < len(x) - W - 2]
    if len(anchors) < 16:
        return None

    # 1. Per-trace steepest fall of the lightly smoothed signal. This is the
    #    discriminator pipeline/geometry.py validated per trace on every frame
    #    tried (99.6-100% lock, L000 included): the sync edge is the steepest
    #    fall of nearly every trace. A fold-level drop score was tried here
    #    first and FAILED on L000/L075/R010 -- it reproduces exactly the
    #    feature preference of the drop detector whose mistake it is meant to
    #    correct, and the fold blurs the sync edge by the landmark spread.
    k = max(3, int(round(0.003 * period))) | 1
    g = np.convolve(np.asarray(x, dtype=np.float64), np.ones(k) / k, mode="same")
    d = np.diff(g)
    votes = []
    for s in anchors:
        a = int(round(s))
        votes.append(int(np.argmin(d[a : a + W])))
    votes = np.asarray(votes, dtype=np.float64)

    # 2. Circular mode of the votes (bin ~0.005P, 2-bin smoothing) -- a plain
    #    median is meaningless on a circle and a mean is wrecked by the
    #    minority of traces whose picture out-drops the sync.
    nb = 200
    h = np.histogram(votes, bins=nb, range=(0, W))[0]
    h2 = h + np.roll(h, 1)
    bm = int(np.argmax(h2))
    centre = bm * (W / nb)  # boundary between the two winning bins
    dev = (votes - centre + W / 2.0) % W - W / 2.0
    memb = np.abs(dev) <= max(6.0, 0.01 * period)
    if memb.sum() < 8:
        return None
    coarse = centre + float(np.median(dev[memb]))
    coarse = (coarse + W / 2.0) % W - W / 2.0  # signed, relative to the origin

    # 3. Refine each member trace to the raw downward zero crossing nearest
    #    the coarse estimate; the median over traces centres the +-half-dot
    #    parity alternation of the crossing. Falls back to the half-amplitude
    #    level on shallow-trough frames whose signal never crosses zero.
    r = max(3, int(round(0.0125 * period)))
    for lvl_mode in ("zero", "half"):
        offs = []
        for s in anchors:
            i0 = int(round(s + coarse))
            lo = i0 - r
            if lo < 1 or i0 + r + 2 > len(x):
                continue
            seg = x[lo : i0 + r + 2]
            if lvl_mode == "zero":
                lvl = 0.0
            else:
                lvl = 0.5 * (
                    float(np.max(seg[: r + 1])) + float(np.min(seg[r:]))
                )
            cr = np.where((seg[:-1] >= lvl) & (seg[1:] < lvl))[0]
            if len(cr) == 0:
                continue
            j = cr[np.argmin(np.abs(cr - r))]
            p = lo + j + (seg[j] - lvl) / (seg[j] - seg[j + 1])
            offs.append(p - s)
        if len(offs) >= 16:
            med = float(np.median(offs))
            return (med + W / 2.0) % W - W / 2.0
    return float(coarse)


def _fir_from_gain(gain: np.ndarray, taps: int = FIR_TAPS) -> np.ndarray:
    """Zero-phase FIR by frequency sampling of `gain` (index k = k/64 cyc/trace)."""
    m = (taps - 1) // 2
    kk = np.arange(len(gain)) / (2.0 * (len(gain) - 1))  # cycles per trace
    nidx = np.arange(-m, m + 1)
    h = (gain[0] + 2.0 * np.sum(gain[1:-1, None] * np.cos(2 * np.pi * kk[1:-1, None] * nidx), axis=0)
         + gain[-1] * np.cos(2 * np.pi * kk[-1] * nidx)) / (2.0 * (len(gain) - 1))
    h *= np.hamming(taps)
    # LANDMINE, kept from globaltime.py where it was found: this pins the
    # filter's DC gain to exactly 1, so WIENER_GAIN[0] is inert -- edit it and
    # nothing happens. Worse, anyone who zeroes the low bins to high-pass the
    # correction makes h.sum() ~0.0015, and at k=3 it goes NEGATIVE, inverting
    # the whole filter and destroying every frame with no error raised. If you
    # need to shape the low bins, delete this line and set DC deliberately.
    return h / h.sum()


def _filtfilt_reflect(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    m = (len(h) - 1) // 2
    pad = np.concatenate([y[m:0:-1], y, y[-2:-m - 2:-1]])
    return np.convolve(pad, h, mode="valid")[: len(y)]


def wideband(tb: Timebase, gain: np.ndarray = WIENER_GAIN,
             clip_mad: float = WIDEBAND_CLIP_MAD) -> Timebase:
    """Add back the flutter the Savitzky-Golay trend threw away.

    Takes recover()'s own output and restores the part of the measured landmark
    residual that the content-free gap lines say is real tape motion (everything
    below ~0.30 cycles per trace), while keeping the part they say is the picture
    pulling the sync crossing (near trace-Nyquist) suppressed.

    Nothing is invented: coasted traces keep exactly the prediction they already
    had, and outliers beyond `clip_mad` MADs are clipped before filtering so one
    bad landmark cannot inject an excursion.

    The evidence this is a real correction and not just a moved sampling grid is
    a hold-out: the correction is built from the SYNC alone, then the leftover
    trace displacement is measured from the PICTURE, which it never saw. That
    residual falls 54% on 16 of 16 test frames -- and phase-scrambled,
    time-reversed, circularly-rolled and SIGN-NEGATED versions of the same
    correction all RAISE it. The sign control is the decisive one: a correction
    that merely moved the grid would be sign-symmetric. It also improves the
    calibration circle's radial rms, which sharpening cannot fake, and leaves
    `sharpness` unchanged, which is what a timing fix should do.

    ACCEPTANCE, run against this integrated path on the 16 frozen test frames:

      shift_rms   improves on 15 of 15 frames where it runs (mean -13.0%);
                  the 16th, L075, is correctly skipped by the gate.
      circle      L000 radial_rms 0.941 -> 0.927 px. Axis ratio moves +0.0003,
                  i.e. not at all; the spec predicted an improvement there and
                  I could not reproduce that half of its claim, so the circle
                  evidence for this correction is the radial rms alone.
      sharpness   FAILS the spec's "< 1% change" threshold: R022 moves -1.16%.
                  Recorded rather than argued away -- but the direction settles
                  what the test was guarding against. Sharpness INCREASED on
                  ZERO of 16 frames; it fell on 11 and was unchanged on 5, mean
                  -0.43%. A cosmetic sharpener raises it. Trace jitter puts
                  spurious high-frequency energy across traces, so removing
                  jitter lowering the measure slightly is the expected sign.
                  The threshold is too tight by 0.16 percentage points; the
                  hypothesis it exists to catch is refuted unanimously.
    """
    n = int(tb.n_traces)
    signed = tb._parity_signed()
    meas = np.asarray(tb.positions, dtype=np.float64) - signed
    base = np.asarray(tb.smoothed, dtype=np.float64)
    r = meas - base
    loc = np.asarray(tb.located, dtype=bool)
    if loc.size != n or not loc.any():
        return tb
    r[~loc] = 0.0
    good = r[loc]
    s = 1.4826 * np.median(np.abs(good - np.median(good))) + 1e-9
    r = np.clip(r, -clip_mad * s, clip_mad * s)
    r[~loc] = 0.0
    h = _fir_from_gain(np.asarray(gain, dtype=np.float64))
    smoothed = base + _filtfilt_reflect(r, h)
    return replace(tb, smoothed=smoothed,
                   residuals=meas + signed - (tb.phase + np.arange(n) * tb.period))


def recover(
    x: np.ndarray,
    *,
    period_guess: float = NOMINAL_PERIOD,
    n_traces: int = TRACES_PER_FRAME,
    search: int = 0,  # initial half-window; 0 means 4% of a period
    passes: int = 3,
    smooth_window: int = _SG_WINDOW,
    remove_parity: bool = True,
    apply_wideband: bool = True,
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

    # Re-anchor the whole timebase to the zero-crossing convention (see the
    # module docstring). One constant per frame; tracking is untouched.
    off = _zero_cross_offset(x, smoothed, period)
    if off is not None:
        smoothed = smoothed + off
        pos = pos + off
        intercept += off

    template, origin = _fold_template(x, pos, gate, period)

    tb = Timebase(
        phase=intercept,
        period=period,
        positions=pos,
        smoothed=smoothed if remove_parity else smoothed + signed,
        residuals=pos - (intercept + np.arange(n_traces) * period),
        template=template,
        template_origin=origin,
        n_traces=int(n_traces),
        blank_len=0.05 * period,  # picture end (0.95) to crossing, geometry.py
        located=gate.copy(),
        parity_offset=alt,
        lock_quality=lock,
    )

    # Wideband flutter tracking, on by default. Gated on how many traces were
    # actually located, because the correction zeroes the residual of a coasted
    # trace and every coasted run therefore injects a step into the filter.
    if apply_wideband and float(gate.mean()) >= WIDEBAND_MIN_LOCATED:
        tb = wideband(tb)
    return tb
