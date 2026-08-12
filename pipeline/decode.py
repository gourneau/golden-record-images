"""Turn one frame of Voyager image audio into a picture.

Reference implementation. All geometry below comes from pipeline/geometry.py,
which measured it on the 384 kHz master; the numbers are quoted here, never
re-derived here.

WHAT THE SIGNAL LOOKS LIKE
--------------------------
One trace is one vertical column of the image. The trace-start convention is
the RAW DOWNWARD ZERO CROSSING of the sync trailing edge -- the most
frame-invariant landmark on this master (consecutive-spacing std 5.6-5.9
samples at 384 kHz on every frame tried) -- and sync.recover() re-anchors its
timebase to it, so the constants here can be stated as fractions of the local
line period after that crossing (geometry.py's 3200-bin grid, 1 bin =
period/3200):

       0.0        zero crossing; undershoot minimum just after
   0.031..0.070   BACK PORCH (bins 100..225): the only content-free DC
                  reference present on both sync-pulse parities
      0.0725      PICTURE START (bin 232)
      0.9500      PICTURE END (bin 3040): the front-porch dip sits at
                  3044-3063 on every parity-median profile
   0.9525..1.0    blanking: front porch, then the sync burst (its leading
                  edge alternates with trace parity -- the US3950607 field
                  code -- its falling edge is shared), to the next crossing

The picture is ONE CONTIGUOUS RUN of 87.75% of the period and DOES NOT WRAP
the landmark. It is a NEGATIVE: more signal means darker.

Within the picture the trace is not a continuous sweep but a staircase of
sample-and-hold plateaus, one per source TV line (~231 carry picture inside
the gate) -- see pipeline/dotclock.py. The decoder therefore samples each
trace by integrating each plateau on the measured dot clock (the matched
filter), falling back to Lanczos binning at the predicted dot pitch only when
the clock line is invisible on a frame.

GEOMETRY OF THE OUTPUT
----------------------
The calibration circle fixes the isotropy at 7.440 bins of trace-time per
trace spacing, so a square-pixel render of the full gate is 512 traces x 377
rows (aspect 1.357). An exactly-4:3 frame is the first ~503 traces; the
cover's "512" is the scan's trace count, not the width of the 4:3 area. The
default output is 377 rows x 512 traces, square pixels; `dot_native` gives
the raw ~231 dot rows instead.

INTENSITY
---------
Levels are anchored to the signal's own references, not to percentiles: each
trace is clamped on its back porch, and black/white sit at porch +/- 0.75 x
the sync amplitude (short-burst plateau minus porch). On L000 -- binary by
design -- the measured references are porch - 0.745 x amp (white) and porch +
0.741 x amp (black), symmetric to 0.5%; across 16 frames spanning the record
picture p1/p99 stay within these rails to ~1% tails. The transfer is LINEAR:
the record carries no gray-step target, so transfer curvature cannot be
measured from the signal; any gamma or cosine shaping is taste, not
restoration (gamma stays available as a display knob, default 1.0).

WHAT IS WRONG WITH THE RECORDING, AND WHAT WE DO ABOUT IT
---------------------------------------------------------
  60 Hz amplitude   Genuine scan-locked hum, 5-30% of picture RMS (frequency
  hum               tracks the drifting line rate). Odd/even median
                    subtraction removes it exactly.
  parity timing     The sync WIDTH alternates by design; the landmark's
                    crossing alternates with it. The PICTURE sits on a
                    uniform grid (re-measured 2026-08: uniform grid leaves
                    0.17-0.20 samples of odd/even alternation at 96 kHz;
                    following the crossing creates 1.6-1.8), so the parity
                    term is estimated and DISCARDED by sync.recover().
  wow and flutter   Every trace located independently; picture sampled on
                    the smoothed per-trace timing.
  AC-coupling droop Each trace is clamped on its back porch. The optional
                    along-trace pole inversion (`uncouple`) is OFF by
                    default: its time-constant fit rails at its search
                    bounds on most frames and destroyed R040 (tau=19 dots);
                    when enabled, railed fits are now rejected.
  dropouts, clicks  Flagged per trace, repaired where possible, reported.

Anything we cannot fix, we score. `Decoded.quality` carries a per-trace
confidence so the interface can show which parts of a picture to trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import dotclock as dot_mod
from . import sync as sync_mod

# --- geometry, measured (pipeline/geometry.py; fractions of the local line
# period, relative to the sync trailing edge's downward zero crossing) -------
PICTURE_START = 232.0 / 3200.0  # 0.0725
PICTURE_END = 3040.0 / 3200.0  # 0.9500
PICTURE_SPAN = PICTURE_END - PICTURE_START  # 0.8775
PORCH_START = 100.0 / 3200.0  # back porch: the per-trace DC reference
PORCH_END = 225.0 / 3200.0
# Sync-amplitude reference windows (for the intensity transfer): the window
# that separates the two pulse parities, and the short burst's high plateau.
_PARITY_TEST = (3090.0 / 3200.0, 3140.0 / 3200.0)
_SYNC_PLATEAU = (3165.0 / 3200.0, 3195.0 / 3200.0)

ISOTROPY = 7.440 / 3200.0  # periods of trace-time per trace spacing
TRACES_PER_FRAME = 512  # the scan's trace count (converter runs to ~535)
SQUARE_ROWS = 377  # square-pixel rows for the full gate (2808/7.440)
ASPECT_43_TRACES = 503  # an exactly-4:3 frame is the first ~503 +- 4 traces

BLACK_REF = +0.75  # black/white sit at porch +/- 0.75 x sync amplitude
WHITE_REF = -0.75  # (L000, binary by design: +0.741 / -0.745)


@dataclass
class Settings:
    """Every knob the decoder exposes. The web UI drives the same set."""

    height: int = SQUARE_ROWS  # output rows; square pixels at 512 traces
    traces: int = TRACES_PER_FRAME

    picture_start: float = PICTURE_START
    picture_span: float = PICTURE_SPAN

    resample: str = "lanczos3"  # fallback binning: peak | box | area | lanczos3
    lanczos_a: int = 3

    # Sample on the scan converter's own dot clock (dotclock.py): integrate
    # each sample-and-hold plateau exactly, on the measured rate and the
    # block-tracked phase. This is the PRIMARY path; `resample` binning at
    # the predicted dot pitch is the fallback when the clock line is
    # invisible on a frame (dot_clock_strength < 1.8).
    dot_lock: bool = True
    # Output the raw dot rows (~231) instead of resampling them to `height`.
    # The dots are the honest resolution; `height` is the square-pixel render.
    dot_native: bool = False

    dehum: bool = True  # remove the scan-locked 60 Hz fixed pattern
    dc_restore: bool = True  # clamp each trace on its back porch
    # Invert the chain's high-pass along the trace. OFF by default: the tau
    # fit rails at its search bounds on most frames (it destroyed R040 with
    # tau=19 dots before the fit was gated). Enable only with a plausible
    # explicit tau, or accept that a railed fit now falls back to no-op.
    uncouple: bool = False
    tau: float = 0.0  # dots; 0 means fit it (gated)

    deconv: float = 0.0
    deconv_noise: float = 0.02

    despike: float = 0.0  # Hampel threshold in MADs, 0 disables
    repair_dropouts: bool = True
    destripe: bool = False

    invert: bool = True
    # Intensity transfer. "reference": black/white at porch +/- 0.75 x sync
    # amplitude, measured per frame (the record's own absolute anchors).
    # "percentile": the old content-dependent stretch, kept as fallback.
    levels: str = "reference"
    black_pct: float = 1.0
    white_pct: float = 99.0
    gamma: float = 1.0  # display knob; the record's transfer is unmeasurable
    transfer: str = "linear"  # linear | barry (Barry's cosine, taste)

    rotate: int = 0
    flip_scan: bool = False

    def replace(self, **kw) -> "Settings":
        return replace(self, **kw)


@dataclass
class Quality:
    """What we know about how much to trust this decode."""

    sync_confidence: np.ndarray  # per trace, 0..1
    dropouts: np.ndarray  # per trace, bool
    jitter_rms: float  # samples
    measurement_noise: float  # samples
    hum_amplitude: float  # signal units, 0 if not measured
    porch_drift: float
    clipped_fraction: float
    noise_rms: float
    snr_db: float

    @property
    def score(self) -> float:
        """One number, 0..1, for sorting and for the gallery badge."""
        ok = 1.0 - float(self.dropouts.mean())
        j = 1.0 / (1.0 + self.jitter_rms / 8.0)
        s = min(1.0, max(0.0, self.snr_db / 24.0))
        return float(0.4 * ok + 0.3 * j + 0.3 * s)


@dataclass
class Decoded:
    image: np.ndarray  # (height, traces) float32 in [0, 1]
    confidence: np.ndarray  # (traces,) float32 in [0, 1]
    timebase: sync_mod.Timebase
    quality: Quality
    tau: float
    levels: tuple[float, float]  # (black, white) in raw signal units
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def _sample_trace(
    x: np.ndarray, start: float, span: float, n_out: int, mode: str, a: int
) -> np.ndarray:
    """Reduce one trace's picture span to `n_out` pixel values (fallback path).

    `start` is fractional: the timebase locates traces to a fraction of a
    sample, and rounding here would throw away the precision we just spent the
    effort to measure.
    """
    if mode == "lanczos3":
        scale = span / n_out
        # Each output pixel integrates `scale` input samples, so when
        # downsampling the kernel must widen to match or we alias.
        stretch = max(1.0, scale)
        taps = int(np.ceil(2 * a * stretch)) + 2
        centres = start + (np.arange(n_out) + 0.5) * scale
        base = np.floor(centres).astype(np.int64) - taps // 2 + 1
        idx = base[:, None] + np.arange(taps)[None, :]
        t = (centres[:, None] - idx) / stretch
        w = np.sinc(t) * np.sinc(t / a)
        w[np.abs(t) >= a] = 0.0
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        np.clip(idx, 0, len(x) - 1, out=idx)
        return (x[idx] * w).sum(axis=1)

    edges = start + np.arange(n_out + 1) * (span / n_out)
    lo = np.floor(edges[:-1]).astype(np.int64)
    hi = np.maximum(np.ceil(edges[1:]).astype(np.int64), lo + 1)
    np.clip(lo, 0, len(x) - 1, out=lo)
    np.clip(hi, 1, len(x), out=hi)

    if mode == "peak":  # Barry's first pass
        return np.array([x[lo[k] : hi[k]].max() for k in range(n_out)])
    if mode in ("box", "area"):  # Barry's final method
        cs = np.concatenate([[0.0], np.cumsum(x)])
        return (cs[hi] - cs[lo]) / (hi - lo)
    raise ValueError(f"unknown resample mode {mode!r}")


def _rows_from_dots(
    pic: np.ndarray, height: int, a: int = 3, delta: np.ndarray | None = None
) -> np.ndarray:
    """Lanczos-interpolate dot rows (traces, n_dots) to (traces, height).

    Upsampling only (377 rows from ~231 dots), so the kernel is not
    stretched. `delta` (dots, one per trace) is where trace i's plateau
    centres ACTUALLY sit relative to the nominal row centres: the converter's
    sampling grid alternates ~half a dot with trace parity and drifts across
    the frame (dotclock.py), so plateau-snapped values are honest samples at
    known offset positions. Interpolating each trace at its own offset places
    them back on the common uniform grid the scene sits on -- without it the
    alternation prints a half-dot comb into every sharp horizontal edge
    (measured: shift_rms 0.2 -> 0.7-1.2 px on L002/L055/L034).
    """
    n_tr, n = pic.shape
    if delta is None:
        delta = np.zeros(n_tr)
    u = (np.arange(height) + 0.5) * (n / height) - 0.5  # dot-index coordinates
    ui = u[None, :] - delta[:, None]  # (traces, height): value r sits at r+delta
    base = np.floor(ui).astype(np.int64) - a + 1
    idx = base[:, :, None] + np.arange(2 * a)[None, None, :]
    t = ui[:, :, None] - idx
    w = np.sinc(t) * np.sinc(t / a)
    w[np.abs(t) >= a] = 0.0
    w /= np.maximum(w.sum(axis=2, keepdims=True), 1e-12)
    np.clip(idx, 0, n - 1, out=idx)
    tr = np.arange(n_tr)[:, None, None]
    return (pic[tr, idx] * w).sum(axis=2)


# --------------------------------------------------------------------------
# defect measurement and repair
# --------------------------------------------------------------------------


def _region(x, starts, period, lo_f, hi_f):
    """Stack a fractional sub-region of every trace into a rectangle."""
    lo = int(round(lo_f * period))
    hi = int(round(hi_f * period))
    w = hi - lo
    out = np.zeros((len(starts), w))
    ok = np.ones(len(starts), dtype=bool)
    for i, s in enumerate(starts):
        a = int(round(s)) + lo
        if a < 0 or a + w > len(x):
            ok[i] = False
            continue
        out[i] = x[a : a + w]
    return out, ok


def measure_levels(
    x: np.ndarray, starts: np.ndarray, period: float
) -> tuple[float, float, float] | None:
    """Per-frame absolute intensity anchors from the signal's own references.

    Returns (porch, amp, ok...) -> (porch level, sync amplitude) as
    (porch_med, amp) with amp = short-burst plateau minus porch, or None when
    the measurement is implausible (caller falls back to percentiles).
    Black = porch + 0.75*amp, white = porch - 0.75*amp (geometry.py, measured
    on L000 which is binary by design, verified across 16 frames).
    """
    porch, ok_p = _region(x, starts, period, PORCH_START, PORCH_END)
    test, ok_t = _region(x, starts, period, *_PARITY_TEST)
    plat, ok_l = _region(x, starts, period, *_SYNC_PLATEAU)
    ok = ok_p & ok_t & ok_l
    if ok.sum() < 16:
        return None
    porch_med = float(np.median(porch[ok]))
    idx = np.where(ok)[0]
    lvl_test = {p: np.median(test[idx[idx % 2 == p]]) for p in (0, 1)}
    short = 0 if lvl_test[0] < lvl_test[1] else 1
    amp = float(np.median(plat[idx[idx % 2 == short]])) - porch_med
    if not (0.02 < amp < 1.0):
        return None
    return porch_med, amp, float(ok.mean())


def measure_hum(pic: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate the parity-locked fixed pattern.

    Averaging all odd traces gives (mean picture column + hum); the same for
    even traces. Picture content is uncorrelated with trace parity, so it
    cancels in the difference and what is left is the hum profile. Returns the
    profile to subtract from odd traces (negate it for even) and its amplitude.
    """
    if len(pic) < 8:
        return np.zeros(pic.shape[1]), 0.0
    odd = np.median(pic[1::2], axis=0)
    even = np.median(pic[0::2], axis=0)
    half = 0.5 * (odd - even)
    return half, float(np.sqrt(np.mean(half**2)))


# NOTE(negative result, 2026-08, sync-v2 rebuild): a per-trace hum-envelope
# tracker (project each trace onto the odd/even profile, smooth the
# alternating-sign envelope over ~63 traces, subtract envelope x profile) was
# tried here to chase residual odd/even striping. Measured on the 16-frame
# test set it LOST to this plain global median subtraction (mean composite
# 9.7 vs 12.7): the envelope estimate carries ~12% content noise which leaves
# an alternating residual on every frame, while the striping it targeted
# turned out to be resolved DOT-PHASE structure (the plateau grid alternates
# half a dot per trace), which no amplitude correction can remove -- only
# dot-locked sampling can.


def _hampel(x: np.ndarray, k: int, n_sigmas: float) -> np.ndarray:
    """Replace impulsive outliers (clicks, tape ticks) with the local median.

    Applied to the oversampled signal, before pixel binning: a click is a few
    samples wide there but smears across a whole pixel once binned.
    """
    if n_sigmas <= 0:
        return x
    pad = np.pad(x, k, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1)
    med = np.median(win, axis=1)
    mad = 1.4826 * np.median(np.abs(win - med[:, None]), axis=1)
    bad = np.abs(x - med) > n_sigmas * np.maximum(mad, 1e-9)
    out = x.copy()
    out[bad] = med[bad]
    return out


def _fit_tau(pic: np.ndarray) -> float:
    """Fit the chain's high-pass time constant along the trace, GATED.

    After clamping, the mean profile along a trace should have no systematic
    ramp -- picture content averages out across 512 traces, but the pole's step
    response does not. We pick the tau whose inverse filter flattens it.

    The raw minimiser rails at its search bounds on most real frames (the
    mean profile's ramp is dominated by real vertical brightness structure,
    not the pole), and a railed low tau is catastrophic: tau=19 dots on R040
    turned the inverse filter into a huge high-boost and destroyed the frame.
    A fit within 10% of either search bound is therefore rejected (returns 0,
    i.e. no correction). This is also why `Settings.uncouple` now defaults to
    False.
    """
    y = pic.mean(axis=0)
    n = len(y)
    if n < 64:
        return 0.0
    y = y - y.mean()
    t = np.linspace(-1, 1, n)
    lo_b, hi_b = 0.05 * n, 20.0 * n
    best, best_tau = np.inf, 0.0
    for tau in np.geomspace(lo_b, hi_b, 64):
        alpha = float(np.exp(-1.0 / tau))
        c = np.empty(n)
        acc = 0.0
        for i in range(n):
            acc = alpha * acc + y[i]
            c[i] = acc
        c = c - c.mean()
        sd = c.std()
        if sd < 1e-12:
            continue
        score = abs(float(np.polyfit(t, c / sd, 1)[0]))
        if score < best:
            best, best_tau = score, tau
    if best_tau <= lo_b * 1.1 or best_tau >= hi_b * 0.9:
        return 0.0  # railed: the profile's ramp is content, not the pole
    return best_tau


def _uncouple_rows(pic: np.ndarray, tau: float) -> np.ndarray:
    """Invert one pole of high-pass ALONG each trace.

    The chain's loss is H(z) = (1 - z^-1) / (1 - a*z^-1), so the correction is
    its reciprocal, y[n] = y[n-1] + x[n] - a*x[n-1]. Note that as a -> 1 this
    telescopes to the identity, which is the behaviour we want when the fit says
    no correction is needed. A plain leaky integrator, 1/(1 - a*z^-1), is a
    low-pass and would smear the frame instead of flattening it.

    `pic` is (traces, samples-along-trace); we integrate along axis 1, the scan
    direction, because that is the direction time ran in the recording.
    """
    if tau <= 0:
        return pic
    alpha = float(np.exp(-1.0 / tau))
    out = np.empty_like(pic)
    prev_in = np.zeros(pic.shape[0])
    prev_out = np.zeros(pic.shape[0])
    for j in range(pic.shape[1]):
        cur = pic[:, j]
        prev_out = prev_out + cur - alpha * prev_in
        out[:, j] = prev_out
        prev_in = cur
    return out


def _wiener_rows(img: np.ndarray, psf: np.ndarray, strength: float, noise: float) -> np.ndarray:
    """Wiener deconvolution along the trace, using the measured system PSF."""
    if strength <= 0 or psf is None or len(psf) < 3:
        return img
    n = img.shape[0]
    k = np.zeros(n)
    m = min(len(psf), n)
    k[:m] = psf[:m]
    k = np.roll(k, -(m // 2))
    s = k.sum()
    if abs(s) > 1e-12:
        k /= s
    K = np.fft.rfft(k)
    G = np.conj(K) / (np.abs(K) ** 2 + noise)
    G = 1.0 + strength * (G * K - 1.0) / np.maximum(np.abs(K), 1e-6)
    return np.fft.irfft(np.fft.rfft(img, axis=0) * G[:, None], n=n, axis=0)


def measure_psf(tb: sync_mod.Timebase) -> np.ndarray:
    """System impulse response from the sync edge.

    KNOWN LIMITATION: this extracts the recording channel's edge response,
    which is already sharp (10-90 rise 1.75-2.37 samples at 384 kHz);
    deconvolving it is nearly a no-op. The resolution limit is the 1977
    CAMERA chain (~15-sample ESF), which does not appear on the sync edge and
    is not measured here yet.
    """
    t = tb.template
    if len(t) < 16:
        return np.array([1.0])
    d = -np.diff(t)
    k = int(np.argmax(np.abs(d)))
    psf = d[max(0, k - 12) : k + 13].astype(np.float64)
    if psf.sum() <= 0:
        return np.array([1.0])
    return psf / psf.sum()


# --------------------------------------------------------------------------


def decode(x: np.ndarray, cfg: Settings, tb: sync_mod.Timebase | None = None) -> Decoded:
    """Decode one frame. `x` starts at or slightly before the first trace."""
    if tb is None:
        tb = sync_mod.recover(x, n_traces=cfg.traces)
    period = tb.period

    clipped = float(np.mean(np.abs(x) >= 0.999 * np.max(np.abs(x)))) if len(x) else 0.0
    if cfg.despike > 0:
        x = _hampel(x, 6, cfg.despike)

    n_tr = min(cfg.traces, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])

    # --- defect measurement on the content-free back porch ------------------
    porch, porch_ok = _region(x, starts, period, PORCH_START, PORCH_END)
    porch_level = np.median(porch, axis=1)
    med = np.median(porch_level)
    mad = 1.4826 * np.median(np.abs(porch_level - med)) + 1e-9
    dropouts = (np.abs(porch_level - med) > 6 * mad) | (~porch_ok)

    # The CLAMP uses a per-parity running median of the porch (7 same-parity
    # traces): the back porch sits right after the sync pulse, whose width
    # alternates with parity, so AC recovery makes the raw porch level itself
    # alternate and jitter -- clamping on it per trace transfers that into the
    # picture (measured: parity_db +6.2 vs +1.9 unclamped on R068). Smoothing
    # keeps the slow AC drift the clamp exists to remove and leaves the
    # constant alternating residual to the hum profile, which absorbs it.
    # Dropout DETECTION above stays on the raw per-trace level on purpose.
    clamp_level = porch_level.copy()
    if len(porch_level) >= 16:
        for p in (0, 1):
            s = porch_level[p::2]
            pad = np.pad(s, 3, mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(pad, 7)
            clamp_level[p::2] = np.median(win, axis=1)

    # sync confidence: how sharply this trace's timing was determined, relative
    # to the smooth trend the rest of the frame agrees on.
    if hasattr(tb, "trace_confidence") and getattr(tb, "located", np.zeros(0)).size:
        conf = tb.trace_confidence()[:n_tr].copy()
    else:
        dev = np.abs(tb.positions[:n_tr] - tb.smoothed[:n_tr]) if len(tb.positions) >= n_tr else None
        if dev is None:
            conf = np.ones(n_tr)
        else:
            scale = 1.4826 * np.median(np.abs(dev - np.median(dev))) + 1e-9
            conf = np.clip(1.0 - dev / (6.0 * scale), 0.0, 1.0)
    conf[dropouts] = 0.0

    # --- absolute intensity anchors, from the raw signal --------------------
    ref = measure_levels(x, starts, period) if cfg.levels == "reference" else None

    # --- sample the picture: dot-locked primary, binning fallback -----------
    span = cfg.picture_span * period
    predicted_spd = period / dot_mod.DOTS_PER_TRACE

    clock = None
    if cfg.dot_lock:
        try:
            clock = dot_mod.measure(
                x, tb,
                lo_f=cfg.picture_start + 0.01,
                hi_f=cfg.picture_start + cfg.picture_span - 0.01,
            )
        except ValueError:
            clock = None

    dot_locked = bool(clock is not None and clock.measured)
    spd = clock.samples_per_dot if dot_locked else predicted_spd
    n_dots = int(round(span / spd))

    if dot_locked:
        psi, coherence = dot_mod.track_phase(
            x, starts, period, spd,
            cfg.picture_start, cfg.picture_start + cfg.picture_span,
        )
        clock.phase, clock.coherence = psi, coherence
        cs = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
        pic = np.empty((n_tr, n_dots))
        delta = np.empty(n_tr)  # actual minus nominal row position, in dots
        for i in range(n_tr):
            mid = starts[i] + (cfg.picture_start + cfg.picture_span / 2.0) * period
            pic[i] = dot_mod.sample_plateaus(x, mid, n_dots, spd, psi[i], cs=cs)
            c0 = mid - (n_dots - 1) / 2.0 * spd
            k0 = np.floor((c0 - psi[i]) / spd)
            delta[i] = (psi[i] + (k0 + 0.5) * spd - c0) / spd
    else:
        coherence = 0.0
        delta = None
        pic = np.empty((n_tr, n_dots))
        for i in range(n_tr):
            s0 = starts[i] + cfg.picture_start * period
            pic[i] = _sample_trace(x, s0, span, n_dots, cfg.resample, cfg.lanczos_a)

    if cfg.dc_restore:
        pic -= clamp_level[:, None]

    hum_amp = 0.0
    if cfg.dehum:
        profile, hum_amp = measure_hum(pic)
        pic[1::2] -= profile
        pic[0::2] += profile

    if cfg.repair_dropouts and dropouts.any() and (~dropouts).sum() > 4:
        good = np.where(~dropouts)[0]
        for i in np.where(dropouts)[0]:
            j = good[np.argsort(np.abs(good - i))[:2]]
            pic[i] = pic[j].mean(axis=0)

    tau = cfg.tau
    if cfg.uncouple:
        if tau <= 0:
            tau = _fit_tau(pic)
        pic = _uncouple_rows(pic, tau)

    if not cfg.dot_native:
        pic = _rows_from_dots(pic, cfg.height, cfg.lanczos_a, delta)

    img = pic.T  # (height, traces): along-trace becomes vertical

    if cfg.deconv > 0:
        img = _wiener_rows(img, measure_psf(tb), cfg.deconv, cfg.deconv_noise)

    if cfg.destripe:
        col = img.mean(axis=0)
        smooth = np.convolve(np.pad(col, 8, mode="edge"), np.ones(17) / 17, mode="valid")
        img = img - (col - smooth)[None, :]

    if cfg.invert:
        img = -img

    # --- noise, from adjacent traces in the flattest part of the picture ----
    d = np.diff(img, axis=1)
    w = max(8, img.shape[0] // 12)
    var = np.array([d[k : k + w].var() for k in range(0, max(1, d.shape[0] - w), w)])
    noise = float(np.sqrt(max(var.min(), 1e-20) / 2)) if len(var) else 0.0
    sig = float(img.std())
    snr_db = 20 * np.log10(sig / noise) if noise > 0 else 0.0

    # --- intensity transfer -------------------------------------------------
    # After porch clamping and inversion, img = porch - signal, so black
    # (signal = porch + 0.75*amp) sits at -0.75*amp and white at +0.75*amp.
    sgn = -1.0 if cfg.invert else 1.0
    if ref is not None:
        porch_med, amp, _ = ref
        lo = sgn * BLACK_REF * amp  # black end of img's range
        hi = sgn * WHITE_REF * amp  # white end
        levels = (porch_med + BLACK_REF * amp, porch_med + WHITE_REF * amp)
    else:
        lo = float(np.percentile(img, cfg.black_pct))
        hi = float(np.percentile(img, cfg.white_pct))
        levels = (float(med + sgn * lo), float(med + sgn * hi))  # signal units
    if abs(hi - lo) < 1e-9:
        hi = lo + 1e-9
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)

    if cfg.transfer == "barry":
        img = 0.5 - 0.5 * np.cos(img * np.pi)
    if cfg.gamma != 1.0:
        img = img ** (1.0 / cfg.gamma)
    if cfg.flip_scan:
        img = img[:, ::-1]
        conf = conf[::-1]

    quality = Quality(
        sync_confidence=conf.astype(np.float32),
        dropouts=dropouts,
        jitter_rms=tb.jitter_rms,
        measurement_noise=tb.measurement_noise,
        hum_amplitude=hum_amp,
        porch_drift=float(porch_level.max() - porch_level.min()),
        clipped_fraction=clipped,
        noise_rms=noise,
        snr_db=snr_db,
    )
    return Decoded(
        image=img.astype(np.float32),
        confidence=conf.astype(np.float32),
        timebase=tb,
        quality=quality,
        tau=tau,
        levels=levels,
        diagnostics={
            "period": period,
            "dots_per_trace": (clock.dots_per_trace if clock else None),
            "samples_per_dot": spd,
            "dot_clock_strength": (clock.strength if clock else None),
            "dot_locked": dot_locked,
            "dot_phase_coherence": coherence,
            "n_dots": int(n_dots),
            "n_traces": n_tr,
            "dropouts": int(dropouts.sum()),
            "hum_amplitude": hum_amp,
            "snr_db": snr_db,
            "tau_dots": tau,
            "levels_mode": "reference" if ref is not None else "percentile",
            "black_level": levels[0],
            "white_level": levels[1],
            "sync_amplitude": (ref[1] if ref is not None else None),
        },
    )
