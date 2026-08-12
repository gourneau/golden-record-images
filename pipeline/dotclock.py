"""Recover the dot clock: the sample rate of the 1977 scan converter itself.

THE POINT
---------
Every decoder written for this record, including earlier versions of this one,
treats a trace as a continuous brightness sweep to be chopped into N equal
bins. It is not. Colorado Video's Series-262 scan converter sampled the source
television picture ONCE PER SCAN LINE and held each value -- patent US4802008
quotes the spec, and Murmurs of Earth's "eight seconds each" agrees. So a trace
is a staircase of ~262.5 discrete plateaus (one NTSC field, 262.5 lines), not a
ramp.

That is a sampled-data signal, and it should be decoded like one: find the dot
clock, then integrate each dot over exactly its own plateau. Binning on any
other grid mixes adjacent dots together and throws away real resolution, and
choosing an output height unrelated to the dot count either invents detail or
discards it.

CONFIRMED IN THE DATA
---------------------
Predicted dot rate for 262.5 dots per 3197.4-sample trace is 31.53 kHz. The
picture-band spectrum, averaged over 440 traces with the smooth background
divided out, shows a narrow line at:

    L000 (calibration circle)  31.522 kHz  ->  262.47 dots/trace  (4.55x bg)
    L020 (DNA diagram)         31.545 kHz  ->  262.51 dots/trace  (3.70x bg)

against a prediction of 262.500. Not every frame shows it -- R013 gave nothing
above 1.41x -- because the line only appears when the picture has enough
dot-to-dot contrast to excite it. So the clock is measured where it is visible
and predicted from the trace period where it is not; `confidence` says which.

This also explains the "-12 samples on even traces" that Ron Barry hardcoded in
2017 and described as a brainless heuristic. Twelve samples is one dot: one
NTSC line period at the 2x tape speed. It was never a fudge factor -- it was the
scan converter's own clock showing through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# One NTSC field is 262.5 lines, and the converter took one sample per line.
DOTS_PER_TRACE = 262.5
# Vertical blanking costs the top ~19-20 lines of a field, so not every dot
# carries picture. Refined by measurement in `active_dots`.
BLANKED_DOTS = 19.5


@dataclass
class DotClock:
    samples_per_dot: float
    dots_per_trace: float
    phase: float  # sub-sample offset of the first dot centre from trace start
    strength: float  # spectral excess over background, 1.0 means nothing there
    measured: bool  # False if we fell back to the predicted rate

    @property
    def confidence(self) -> float:
        return float(np.clip((self.strength - 1.0) / 2.5, 0.0, 1.0))


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
    sample_rate: float = 384000.0,
    lo_f: float = 0.11,
    hi_f: float = 0.99,
    n_traces: int = 460,
    search: float = 0.03,
) -> DotClock:
    """Find the dot clock in one frame's picture band."""
    predicted_spd = tb.period / DOTS_PER_TRACE
    predicted_hz = sample_rate / predicted_spd

    R = _picture_stack(x, tb, lo_f, hi_f, n_traces)
    n = R.shape[1]
    w = np.hanning(n)
    P = np.mean(np.abs(np.fft.rfft(R * w, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # Divide out a smooth background so a narrow clock line stands out against
    # the picture's own broadband spectrum.
    bg = np.convolve(P, np.ones(41) / 41, mode="same")
    excess = P / np.maximum(bg, 1e-30)

    band = (f > predicted_hz * (1 - search)) & (f < predicted_hz * (1 + search))
    if not band.any():
        return DotClock(predicted_spd, DOTS_PER_TRACE, 0.0, 1.0, False)

    fb, eb = f[band], excess[band]
    k = int(np.argmax(eb))
    strength = float(eb[k])

    if strength < 1.8:  # nothing convincing; trust the trace period instead
        return DotClock(predicted_spd, DOTS_PER_TRACE, 0.0, strength, False)

    # Parabolic refinement of the spectral peak.
    peak = fb[k]
    if 0 < k < len(eb) - 1:
        y0, y1, y2 = eb[k - 1], eb[k], eb[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            peak = fb[k] + 0.5 * (y0 - y2) / den * (fb[1] - fb[0])

    spd = sample_rate / peak
    # Phase of that component, so we can integrate each dot over its own
    # plateau rather than straddling two.
    kk = int(np.argmin(np.abs(f - peak)))
    Fk = np.fft.rfft(R * w, axis=1)[:, kk]
    phase_rad = float(np.angle(Fk.sum()))
    phase = (-phase_rad / (2 * np.pi)) * spd + lo_f * tb.period

    return DotClock(
        samples_per_dot=float(spd),
        dots_per_trace=float(tb.period / spd),
        phase=float(phase % spd),
        strength=strength,
        measured=True,
    )


def active_dots(picture_span_samples: float, clock: DotClock) -> int:
    """How many dots actually carry picture in the active window."""
    return int(round(picture_span_samples / clock.samples_per_dot))


def sample_dots(
    x: np.ndarray, start: float, span: float, clock: DotClock, n_out: int
) -> np.ndarray:
    """Integrate each dot over its own plateau.

    This is the matched filter for a sample-and-hold signal: the value was held
    constant across the dot, so averaging over exactly that interval maximises
    signal-to-noise and, unlike an arbitrary bin grid, never mixes two dots.
    """
    spd = clock.samples_per_dot
    # Align the output grid to the measured dot phase.
    first = start + ((clock.phase - start) % spd)
    centres = first + (np.arange(n_out) + 0.5) * (span / n_out)
    half = spd * 0.5
    lo = np.maximum(np.floor(centres - half).astype(np.int64), 0)
    hi = np.minimum(np.ceil(centres + half).astype(np.int64), len(x))
    hi = np.maximum(hi, lo + 1)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return (cs[hi] - cs[lo]) / (hi - lo)
