"""Turn one frame of Voyager image audio into a picture.

Reference implementation. web/src/dsp mirrors it step for step and is held to
pixel parity by pipeline/test_parity.py.

WHAT THE SIGNAL LOOKS LIKE
--------------------------
Everything below was measured from the 384 kHz master, not assumed. Positions
are given in samples after the detected trace start, for a trace period of
3197.4 samples (the record cover specifies 8.34 ms; this digitisation runs
~0.16% fast, so the period is fitted per frame and never hardcoded).

       0            trace start: half-amplitude crossing of the sync falling edge
      +4 ..   +12   trough, then recovery
     +24 .. +2864   picture -- one vertical column of the image, ~2840 samples
   +2868 .. +3197   blanking: content-free shelf, then the next sync pulse
                    (high for the last ~140 samples on wide-pulse traces, ~40
                    on narrow ones), ending at the next crossing

The picture does NOT wrap across the trace start. An earlier revision anchored
on a notch inside the picture and concluded it did; that "wrap" was an artefact
of the wrong landmark, confirmed by the calibration circle sitting +105 px off
centre until the window was re-derived against the crossing.

The picture is a NEGATIVE: more signal means darker.

WHAT IS WRONG WITH THE RECORDING, AND WHAT WE DO ABOUT IT
---------------------------------------------------------
This is one digitisation of a 1977 analog chain, so the defects are the ones
analog tape and disc always have. Each is measured and either corrected or
reported:

  60 Hz amplitude   A genuine mains-frequency hum in the picture, 5-30% of picture RMS.
  hum               It is scan-locked rather than grid-locked: its frequency tracks the
                    drifting line rate (60.036 -> 60.173 Hz across the record), because the
                    1977 scan start was itself mains-synchronised. Odd/even median
                    subtraction removes it exactly.
  parity-locked     Separately, the sync burst's LEADING edge alternates ~100 samples
  sync alternation  between traces. This is by design: Colorado Video's converter coded the
                    source field parity into the blanking width (US3950607). It is a TIMING
                    effect, not amplitude, so one matched template cannot fit both parities.
                    The picture itself does not alternate -- adjacent picture columns agree
                    to ~0.12 samples on a purely linear clock -- so this parity component is
                    estimated and DISCARDED from the timebase, never tracked.
                    NOTE: an earlier docstring conflated these two and quoted "~70% of
                    picture amplitude". That figure was measured inside the sync burst,
                    which was leaking into the picture window through a geometry bug.
  wow and flutter   Tape speed wanders. Every trace is located independently and
                    the picture is resampled onto corrected timing.
  AC-coupling droop Nothing in the chain passed DC, so brightness decays across
                    each trace and bright areas trail shadows. Each trace is
                    clamped on its back porch and the pole is inverted.
  dropouts, clicks  Flagged per trace, repaired where possible, reported where
                    not, and never silently smoothed over.

Anything we cannot fix, we score. `Decoded.quality` carries a per-trace
confidence so the interface can show which parts of a picture to trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import dotclock as dot_mod
from . import sync as sync_mod

# --- geometry, measured (see pipeline/README.md for the derivation) ---------
# Fractions of a trace period, so they hold at any sample rate.
#
# These are expressed against the trace start that sync.recover (v2) returns,
# which is the HALF-AMPLITUDE DOWNWARD CROSSING OF THE SYNC FALLING EDGE --
# the one landmark shared by both sync-pulse parities (spacing std 5.5-6.2
# samples at 384 kHz vs ~100 for the peak). Earlier revisions anchored on a
# notch inside the picture and then on the burst's leading edge; the leading
# edge ALTERNATES ~100 samples with trace parity and v1's detector latched a
# different feature on different frames (measured spread 38-104 samples at 96
# kHz, i.e. the window moved by up to 17 px frame to frame). Any value here
# must be re-derived if the landmark moves again.
#
# Measured on sub-sample-aligned folds of eight frames spanning the record
# (values below in 384 kHz samples after the crossing): trough +4..+12,
# picture from ~+24 to ~+2864, content-free shelf ~+2868..+3040, next pulse
# high plateau until the next crossing at +period. The picture does NOT wrap.
# PICTURE_SPAN was then calibrated on the L000 circle (axis_ratio -> 1).
_P = 3197.4
BLANK_START = 2868.0 / _P       # shelf start; blanking runs to the next crossing
BLANK_END = 1.0
PICTURE_START = 24.0 / _P       # first clean sample after the trough recovery
PICTURE_SPAN = 2848.0 / _P      # to the shelf. It does NOT wrap.
# The content-free shelf inside blanking: across-trace spread here is a third
# of picture level on every measured frame, which is how it was located.
PORCH_START = 2890.0 / _P
PORCH_END = 3020.0 / _P

TRACES_PER_FRAME = 512
FRAME_ASPECT = 4.0 / 3.0  # from the cover; 512 traces wide => 384 px tall


@dataclass
class Settings:
    """Every knob the decoder exposes. The web UI drives the same set."""

    height: int = 384
    traces: int = TRACES_PER_FRAME

    picture_start: float = PICTURE_START
    picture_span: float = PICTURE_SPAN

    resample: str = "lanczos3"  # peak | box | area | lanczos3 | dots
    lanczos_a: int = 3

    # Sample on the scan converter's own dot clock rather than an arbitrary
    # bin grid. The trace is 262.5 sample-and-hold plateaus (one NTSC field,
    # one sample per line), so integrating each dot over its own plateau is the
    # matched filter; any other grid mixes adjacent dots. Falls back to
    # `resample` when the clock cannot be measured on a given frame.
    dot_lock: bool = True
    # Output height. 0 means "use the true dot count", which is the honest
    # resolution: ~234 dots across the active window. Larger values upscale
    # from the dot samples rather than pretending to resolve more.
    dot_native: bool = True

    dehum: bool = True  # remove the parity-locked 60 Hz fixed pattern
    dc_restore: bool = True  # clamp each trace on its back porch
    uncouple: bool = True  # invert the chain's high-pass along the trace
    tau: float = 0.0  # samples; 0 means fit it

    deconv: float = 0.0
    deconv_noise: float = 0.02

    despike: float = 0.0  # Hampel threshold in MADs, 0 disables
    repair_dropouts: bool = True
    destripe: bool = False

    invert: bool = True
    black_pct: float = 1.0
    white_pct: float = 99.0
    gamma: float = 1.0
    transfer: str = "linear"  # linear | barry

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
    levels: tuple[float, float]
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def _sample_trace(
    x: np.ndarray, start: float, span: float, n_out: int, mode: str, a: int
) -> np.ndarray:
    """Reduce one trace's picture span to `n_out` pixel values.

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
# turned out to be resolved DOT-PHASE structure (262.5 dots/trace => the dot
# grid alternates half a dot per trace), which no amplitude correction can
# remove -- only dot-locked sampling can.


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


def _fit_tau(pic: np.ndarray, period: float) -> float:
    """Fit the chain's high-pass time constant along the trace.

    After clamping, the mean profile along a trace should have no systematic
    ramp -- picture content averages out across 512 traces, but the pole's step
    response does not. We pick the tau whose inverse filter flattens it.
    """
    y = pic.mean(axis=0)
    n = len(y)
    if n < 64:
        return 0.0
    y = y - y.mean()
    t = np.linspace(-1, 1, n)
    best, best_tau = np.inf, 0.0
    for tau in np.geomspace(0.05 * n, 20.0 * n, 64):
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

    The sync plateau's falling edge is a step that travelled through the whole
    chain, so its derivative is the system's impulse response. Note this is only
    as sharp as our timebase: if the traces are not aligned to a fraction of a
    sample before folding, this measures our own jitter instead.
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

    # --- defect measurement on the content-free blanking interval ----------
    porch, porch_ok = _region(x, starts, period, PORCH_START, PORCH_END)
    porch_level = np.median(porch, axis=1)
    med = np.median(porch_level)
    mad = 1.4826 * np.median(np.abs(porch_level - med)) + 1e-9
    dropouts = (np.abs(porch_level - med) > 6 * mad) | (~porch_ok)

    # sync confidence: how sharply this trace's timing was determined, relative
    # to the smooth trend the rest of the frame agrees on. sync v2 computes
    # this itself (its positions carry a deliberate parity offset that must
    # not read as deviation, and coasted traces must score 0).
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

    # --- sample the picture ------------------------------------------------
    span = cfg.picture_span * period

    clock = None
    if cfg.dot_lock:
        try:
            # Restrict the spectral search to the active picture band under the
            # current landmark convention; the defaults predate the v2 landmark.
            clock = dot_mod.measure(
                x, tb, lo_f=cfg.picture_start + 0.01, hi_f=cfg.picture_start + cfg.picture_span - 0.01
            )
        except ValueError:
            clock = None

    if clock is not None and clock.measured:
        n_dots = dot_mod.active_dots(span, clock)
        pic = np.empty((n_tr, n_dots))
        for i in range(n_tr):
            s0 = starts[i] + cfg.picture_start * period
            pic[i] = dot_mod.sample_dots(x, s0, span, clock, n_dots)
    else:
        n_dots = cfg.height
        pic = np.empty((n_tr, cfg.height))
        for i in range(n_tr):
            s0 = starts[i] + cfg.picture_start * period
            pic[i] = _sample_trace(x, s0, span, cfg.height, cfg.resample, cfg.lanczos_a)

    if cfg.dc_restore:
        pic -= porch_level[:, None]

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
            tau = _fit_tau(pic, period)
        pic = _uncouple_rows(pic, tau)

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

    lo = float(np.percentile(img, cfg.black_pct))
    hi = float(np.percentile(img, cfg.white_pct))
    if hi - lo < 1e-9:
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
        levels=(lo, hi),
        diagnostics={
            "period": period,
            "dots_per_trace": (clock.dots_per_trace if clock else None),
            "samples_per_dot": (clock.samples_per_dot if clock else None),
            "dot_clock_strength": (clock.strength if clock else None),
            "dot_locked": bool(clock and clock.measured),
            "n_dots": int(pic.shape[1]),
            "n_traces": n_tr,
            "dropouts": int(dropouts.sum()),
            "hum_amplitude": hum_amp,
            "snr_db": snr_db,
            "tau_traces": tau,
        },
    )
