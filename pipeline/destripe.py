"""Remove the per-trace streaking, using the calibration frame as the yardstick.

    python -m pipeline.destripe                 # the full control report (~90 s)
    python -m pipeline.destripe --frames 24     # wider transfer survey

WHAT THE DEFECT IS
------------------
A decoded frame is (height, traces): a COLUMN is one trace of the scan, and the
streaks are vertical. Measured on L000's ring interior, which the record's own
cover tells us is a uniform white field:

    84 grey levels of smooth shading from the top of the trace to the bottom
                                             <- the droop; NOT touched here
    residual after removing that smooth 2-D shading:  sd 1.87 grey
        of which a per-trace CONSTANT explains 23%   <- classic destriping,
                                                        moment matching and
                                                        one-offset-per-column,
                                                        is ruled out by this
    the rest is a per-trace slowly-varying LEVEL TRAJECTORY.

That last line is the whole design brief. The residual's 2-D power spectrum
says the rest, and one of its two answers corrected a wrong guess of mine:

    DOWN the trace it is steeply red -- power density falls 3.9e-4 -> 2.8e-4 ->
    1.1e-4 -> 5.3e-5 -> 4.3e-5 -> 3.1e-5 over |f_y| = 0-0.01, 0.01-0.02,
    0.02-0.04, 0.04-0.08, 0.08-0.16, 0.16-0.25 cycles/px. Smooth down the
    trace, as advertised.

    ACROSS traces it is nearly WHITE. Power at |f_x| > 0.35 is within a factor
    1.1-1.5 of power at f_x = 0.03-0.13 at every f_y. Each trace really is
    independent of its neighbours. An earlier reading of mine that the streak
    "peaks at 8-32 traces wide" was an artefact of measuring it on the column
    MEAN, which is itself a low-f_y projection; on the 2-D spectrum it is flat.

So the scene is smooth in BOTH directions; the streak is smooth DOWN the trace
and rough ACROSS traces. Those are different regions of the 2-D Fourier plane,
which is the classical separation used for stripe removal in tomography and
satellite imagery.

WHAT WAS TAKEN FROM THE LITERATURE
----------------------------------
* Munch, Trtik, Marone, Stampanoni, "Stripe and ring artifact removal with
  combined wavelet-Fourier filtering", Opt. Express 17(10):8567 (2009). The
  idea that stripes occupy the wavelet detail band that is high-pass ACROSS
  the stripes and low-pass ALONG them, and that a Gaussian notch at low |f|
  along the stripe direction takes them out. Implemented here (`munch_stripe`)
  and MEASURED AGAINST the shipped filter -- it loses, see NEGATIVES.
* Bouali & Ladjal, "Toward optimal destriping of MODIS data using a
  unidirectional variational model", IEEE TGRS 49(8):2924 (2011), and the
  unidirectional-variational family generally: the insight that the correction
  must be constrained along the stripe direction only, and that a quadratic
  penalty on the across-stripe gradient eats real edges, so the estimator needs
  a robust term. Here that robust term is an amplitude limiter calibrated on
  the record itself, not a TV solve -- see THE LIMITER.
* Wiener/adaptive spectral weighting, as used in destriping literature that
  weights the notch by an estimated stripe-to-signal power ratio rather than
  cutting at a hand-set frequency.

WHAT IS NEW HERE, AND WHY
-------------------------
The hard question is WHERE THE CUT GOES. Too aggressive and a lighthouse tower
or the ring's own vertical tangent is a "stripe"; too timid and the streaks
stay. A hand-set (f_y, f_x) cut cannot answer it, because the answer depends on
the frame: a picture of the sky can take a hard cut, a picture with a mast in
it cannot.

The record hands us the instrument to remove the guess. L000's interior is
uniform BY DESIGN, so everything in it is streak, and its 2-D power spectrum
P_streak(f_y, f_x) is a calibrated, tier-0 measurement of the defect. For any
frame we then compute its OWN local power spectrum P_frame and weight

    W(f) = P_streak(f) / max(P_frame(f), P_streak(f))

so the correction is full strength exactly where the frame carries no more
power than the calibrated streak, and shrinks itself wherever the frame carries
more -- which is what a real vertical edge looks like. The cut is not chosen;
it is measured, per frame and per frequency. Measured against the best hand-set
notch at equal flat-field improvement it removes 3x more streak per unit of
damage to real vertical structure (median ratio 8.0 against 2.5 over twelve
frames, both measured by the same transpose control; see TRANSFER below).

P_frame is computed on OVERLAPPING BLOCKS of 128 traces rather than the whole
frame. A single strong feature -- the ring, the slide mount's vertical edge --
otherwise inflates the global spectrum and suppresses the correction over the
entire picture. Blocks are full height, because a streak is coherent down the
whole trace and windowing that axis would destroy the thing being estimated.

L000 fixes the streak's SHAPE but not its amplitude everywhere: block by block
across L000 itself the amplitude runs 0.7x to 7.2x the frame average. So the
level is read off each block, in the one corner of the plane where the streak
dominates and photographs do not (`block_level`). The calibration frame
supplies the shape; every frame supplies its own scale.

THE LIMITER, AND THE CONTROL THAT FORCED IT
-------------------------------------------
A purely linear filter DEGRADES THE CIRCLE. The hand-set notch at f_y0 = 0.012,
f_x0 = 0.02 takes the ring's radial rms from 0.837 to 0.962 px, ten times the
metric's own scatter, and its |stripe| estimate reaches 0.49 -- half of full
scale. The cause is not subtle once looked at: the frame's dark top and bottom
borders have a ragged, per-trace-jittering edge, which is exactly the signature
the filter is tuned for, and it smears that edge over 80 rows.

So the stripe field is passed through a soft limiter, tanh at c times its own
robust scale, c = 3. That is not a taste parameter: the streak on L000 is a
roughly Gaussian field of about 1.3 grey levels rms, and 3 sigma is the point
beyond which 0.3% of a Gaussian lies -- i.e. the limiter clips only what a
streak of the calibrated amplitude never produces. The measured cost of moving
it, on L000's flat field and on L055's damage ratio:

    c = 2   flat-field streak variance -25%   L055 removed/damage  8.4
    c = 3   flat-field streak variance -28%   L055 removed/damage  7.8
    c = 5   flat-field streak variance -32%   L055 removed/damage  6.8
    none    flat-field streak variance -47%   L055 removed/damage  0.9

That last row is the whole argument. Without the limiter the calibration frame
improves nearly twice as much and the filter is, on a real photograph, removing
as much picture as streak. A destriper tuned on the flat field alone would have
shipped that.

WHAT IT COSTS, MEASURED
-----------------------
On L000's ring interior the streak band falls 1.291 -> 1.097 grey rms, a 28%
cut in streak variance, and applying the same correction with the sign
reversed makes it 55% worse -- so the thing being removed is the thing that is
there. The circle does not move: axis ratio 1.00509 -> 1.00500, radial rms
0.8373 -> 0.8412 px, both inside the fitter's own +/-0.0005 / +/-0.008 scatter,
and the ring's contrast changes by +0.05%.

What it does cost is a ringing halo. This is a Fourier filter, so a strong
curved edge rings: |stripe| averages 1.08 grey within 3 px of the ring against
0.68 grey in the open field, decaying back to background by about 30 px. The
limiter caps it at 2.4 grey. A spatially-localised formulation -- a proper
undecimated wavelet-domain or split-Bregman TV solve -- would localise that
better, and is the obvious next thing to try.

On the twelve non-calibration frames surveyed it removes a median 8.0x more
than the transpose control's estimate of scene damage. On L055 the 2000 pixels
carrying the strongest tall-thin vertical structure -- the lighthouse tower and
its neighbours -- carry 30.7 grey of it and lose 0.75, which is 2.5%.

WHAT THIS DOES NOT DO
---------------------
It does not touch the DROOP. f_x = 0 -- the component common to every trace --
is explicitly zeroed in the calibrated spectrum, because a level trajectory
shared by all traces is indistinguishable from a scene that is dark at the
bottom, and on 115 of the 116 pictures it IS the scene. The 84-grey top-to-
bottom shading on L000 is common mode and survives this filter untouched. That
is a deliberate refusal, not an oversight: droop_inverse.py and droop.py are
where that problem lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# the calibration window: L000's ring interior
#
# The ring on the shipped 377x512 float decode sits at centre (181.2, 267.5)
# with radii (151.2, 152.0). This is the largest axis-aligned rectangle inside
# 0.85 of that ellipse, i.e. entirely within the uniform white field with
# margin to the ring. It is a constant rather than a fit so that the shipped
# calibration cannot drift with the ring fitter; `pipeline.circle.fit_ring`
# reproduces the geometry it was derived from.
# --------------------------------------------------------------------------
INTERIOR = (slice(90, 272), slice(176, 358))

# Detrend orders used to strip the smooth 2-D shading before the streak is
# measured. Order 5 down the trace, because the droop is a strong smooth
# function of row; order 1 ACROSS traces, because the field is uniform there by
# design and a higher order would eat the broad streaks we are trying to
# measure (order 5 across traces loses 10% of the streak band).
DETREND_Y, DETREND_X = 5, 1

BLOCK_TRACES = 128     # local spectrum window, traces
PSD_SMOOTH = 2.0       # bins, smoothing of both periodograms
LIMIT_MAD = 3.0        # soft limiter, in robust scales of the stripe field
GAIN = 1.0             # 1.0 = the streak spectrum exactly as calibrated
AUTO_LEVEL = True      # read the streak's AMPLITUDE off each block; see block_level


# --------------------------------------------------------------------------
# flat-field measurement
# --------------------------------------------------------------------------


def detrend(P: np.ndarray, oy: int = DETREND_Y, ox: int = DETREND_X) -> np.ndarray:
    """Remove the smooth 2-D shading, leaving the defect."""
    h, w = P.shape
    Y, X = np.mgrid[0:h, 0:w]
    Y = (Y - h / 2) / h
    X = (X - w / 2) / w
    T = [Y ** i * X ** j for i in range(oy + 1) for j in range(ox + 1)]
    A = np.stack([t.ravel() for t in T], axis=1)
    c, *_ = np.linalg.lstsq(A, P.ravel(), rcond=None)
    return (P.ravel() - A @ c).reshape(h, w)


def _smooth_y(R: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Gaussian low-pass down the trace, by FFT so there is no scipy dependency."""
    n = R.shape[0]
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(R, axis=0) * np.exp(-2 * (np.pi * sigma * f) ** 2)[:, None],
                        n=n, axis=0)


def flat_metrics(P: np.ndarray) -> dict:
    """Everything a known-uniform patch can say, in grey levels of 255.

    `band` is the headline: the residual after the smooth shading is removed
    AND after low-passing down the trace, which suppresses white pixel noise by
    5x and leaves the per-trace level trajectory. `noise` is the along-trace
    pixel noise, the floor `band` cannot go below.
    """
    R = detrend(P)
    noise = 255 * np.diff(R, axis=0).std() / np.sqrt(2)
    return {
        "sd": 255 * R.std(),
        "band": 255 * _smooth_y(R).std(),
        "noise": noise,
        "band_floor": noise * np.sqrt(1.0 / (2 * 8.0 * np.sqrt(np.pi))),
        "col_sd": 255 * R.mean(axis=0).std(),
    }


# --------------------------------------------------------------------------
# the calibrated streak spectrum
# --------------------------------------------------------------------------


@dataclass
class StreakPSD:
    """The defect's own 2-D power spectrum, measured on known-uniform pixels."""

    P: np.ndarray          # (h, w) power, fftshift-free (numpy fftfreq order)
    fy: np.ndarray
    fx: np.ndarray
    noise_floor: float
    source: str = ""
    stats: dict = field(default_factory=dict)

    @property
    def sd_grey(self) -> float:
        """The streak's rms in grey levels, noise removed."""
        return 255 * float(np.sqrt(self.P.sum() / self.P.size))


def _periodogram(R: np.ndarray, smooth: float = PSD_SMOOTH) -> np.ndarray:
    h, w = R.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    U = float((win ** 2).mean())
    P = np.abs(np.fft.fft2((R - R.mean()) * win)) ** 2 / (h * w * U)
    return _blur_wrap(P, smooth)


def _blur_wrap(P: np.ndarray, sigma: float) -> np.ndarray:
    """Circular Gaussian blur in the frequency plane (frequency wraps)."""
    if sigma <= 0:
        return P
    out = P
    for ax, n in enumerate(P.shape):
        f = np.fft.fftfreq(n)
        k = np.exp(-2 * (np.pi * sigma * f) ** 2)
        shape = [1] * P.ndim
        shape[ax] = n
        out = np.fft.ifft(np.fft.fft(out, axis=ax) * k.reshape(shape), axis=ax).real
    return np.clip(out, 0.0, None)


def noise_transfer(n_rows: int, n_dots: int = 231, lanczos_a: int = 3,
                   traces: int = 512, seed: int = 0) -> np.ndarray:
    """|K(f_y)|^2: how the decoder's own resampler colours per-dot white noise.

    The decoded picture is not white in f_y and assuming it is throws the noise
    floor away by a factor of 40. Each trace carries ~231 sample-and-hold dots
    and `decode._rows_from_dots` Lanczos-resamples them to `n_rows`, so the
    noise is flat to f_y ~ 0.2 cycles/px and then rolls off into the dot grid's
    own Nyquist at 231/(2*377) = 0.306. This pushes white noise through the
    decoder's actual resampler and measures the shape, rather than modelling
    it: it is a property of the shipped decoder, so it is tier 0.
    """
    from . import decode as decode_mod

    rng = np.random.default_rng(seed)
    rows = decode_mod._rows_from_dots(
        rng.normal(0.0, 1.0, (traces, n_dots)), n_rows, lanczos_a, None)
    P = _periodogram(rows.T, 0.0).mean(axis=1)
    f = np.abs(np.fft.fftfreq(n_rows))
    return P / float(P[f < 0.05].mean())


def calibrate(patch: np.ndarray, smooth: float = PSD_SMOOTH,
              source: str = "L000 ring interior", n_dots: int = 231) -> StreakPSD:
    """Measure the streak spectrum on a patch the record says is uniform.

    Everything in a uniform patch is defect, but not all of it is STREAK: the
    decoder's pixel noise is in there too. Removing noise would be denoising,
    which blurs and which the composite metric would happily reward, so it is
    subtracted rather than exploited. The noise model has exactly one free
    parameter -- its level -- because its SHAPE is measured from the decoder's
    resampler (`noise_transfer`). The level is fitted where the streak is
    negligible: |f_y| in 0.15..0.28 and |f_x| > 0.35, i.e. structure that
    alternates every three rows and every three traces, which no level
    trajectory can produce.

    f_x = 0 is zeroed afterwards: a trajectory common to every trace is the
    droop, and is not ours to remove.
    """
    R = detrend(patch)
    P = _periodogram(R, smooth)
    h, w = R.shape
    fy, fx = np.fft.fftfreq(h), np.fft.fftfreq(w)
    K = noise_transfer(h, n_dots)
    fit = (np.abs(fy)[:, None] > 0.15) & (np.abs(fy)[:, None] < 0.28) \
        & (np.abs(fx)[None, :] > 0.35)
    lvl = float(np.median((P / np.maximum(K[:, None], 1e-9))[fit]))
    P = np.clip(P - lvl * K[:, None], 0.0, None)
    P[:, 0] = 0.0
    stats = flat_metrics(patch)
    stats["noise_model_grey"] = 255 * float(np.sqrt(lvl * K.mean()))
    return StreakPSD(P, fy, fx, lvl, source, stats)


def _lift(psd: StreakPSD, H: int, W: int) -> np.ndarray:
    """Nearest-|f| lift of the calibrated spectrum onto an (H, W) FFT grid."""
    gy, gx = np.fft.fftfreq(H), np.fft.fftfreq(W)
    iy = np.abs(np.abs(psd.fy)[None, :] - np.abs(gy)[:, None]).argmin(axis=1)
    ix = np.abs(np.abs(psd.fx)[None, :] - np.abs(gx)[:, None]).argmin(axis=1)
    return psd.P[np.ix_(iy, ix)]


# --------------------------------------------------------------------------
# the filter
# --------------------------------------------------------------------------


def _soft_limit(S: np.ndarray, c: float) -> np.ndarray:
    """Bound the stripe field at c robust scales. See THE LIMITER above."""
    if c <= 0 or not np.isfinite(c) or c > 100:
        return S
    A = c * 1.4826 * float(np.median(np.abs(S - np.median(S))))
    return A * np.tanh(S / A) if A > 0 else S


def block_level(Pf: np.ndarray, Ps: np.ndarray, H: int, W: int) -> float:
    """How much streak THIS block carries, in units of the calibrated spectrum.

    L000 fixes the streak's SHAPE, but not its level everywhere: measured
    block by block across L000 itself the level runs 0.7 to 7.2, so a single
    global amplitude under-corrects the streaky blocks and over-corrects the
    quiet ones. The level is therefore read off each block in the corner of the
    plane where the streak dominates and a photograph does not: |f_y| < 0.05
    and |f_x| > 0.25, i.e. slow down the trace and alternating every two to
    four traces. The streak is close to WHITE across traces (its power at
    |f_x| > 0.35 is within 1.4x of its power at f_x = 0.03-0.13), so that
    corner is a fair sample of it, while scene structure that fine and that
    tall is rare.
    """
    fy = np.abs(np.fft.fftfreq(H))[:, None]
    fx = np.abs(np.fft.fftfreq(W))[None, :]
    m = (fy < 0.05) & (fx > 0.25) & (Ps > 0)
    if not m.any():
        return 1.0
    return max(float(np.median((Pf / np.maximum(Ps, 1e-300))[m])), 0.0)


def _block_stripe(block: np.ndarray, psd: StreakPSD, gain: float,
                  auto: bool) -> np.ndarray:
    h = block.shape[0]
    py = h // 2
    B = np.pad(block, ((py, py), (0, 0)), mode="reflect")   # kill the y wrap
    H, W = B.shape
    Ps = _lift(psd, H, W)
    Pf = _periodogram(B)
    a = gain * (block_level(Pf, Ps, H, W) if auto else 1.0)
    Ps = Ps * a
    Wf = np.nan_to_num(Ps / np.maximum(Pf, Ps + 1e-300))
    return np.fft.ifft2(np.fft.fft2(B) * Wf).real[py:py + h]


def stripe_field(img: np.ndarray, psd: StreakPSD, *, block: int = BLOCK_TRACES,
                 gain: float = GAIN, limit: float = LIMIT_MAD,
                 auto: bool = AUTO_LEVEL) -> np.ndarray:
    """The streak this frame carries, as a field to subtract."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    if block >= w:
        return _soft_limit(_block_stripe(img, psd, gain, auto), limit)
    step = max(block // 2, 1)
    starts = list(range(0, w - block + 1, step))
    if starts[-1] + block < w:
        starts.append(w - block)
    acc = np.zeros((h, w))
    wsum = np.zeros(w)
    win = np.hanning(block + 2)[1:-1]
    for s in starts:
        acc[:, s:s + block] += _block_stripe(img[:, s:s + block], psd, gain, auto) * win
        wsum[s:s + block] += win
    return _soft_limit(acc / np.maximum(wsum, 1e-9)[None, :], limit)


def apply(img: np.ndarray, psd: StreakPSD, *, block: int = BLOCK_TRACES,
          gain: float = GAIN, limit: float = LIMIT_MAD,
          auto: bool = AUTO_LEVEL, sign: float = 1.0) -> np.ndarray:
    """Destripe. `sign=-1` is the control that must make the flat field worse."""
    return np.asarray(img, dtype=np.float64) - sign * stripe_field(
        img, psd, block=block, gain=gain, limit=limit, auto=auto)


# --------------------------------------------------------------------------
# the damage control
# --------------------------------------------------------------------------


def anisotropy(img: np.ndarray) -> float:
    """How much more structure this frame has across traces than down them.

    Note first what does NOT work: any mask symmetric in (f_y, f_x) gives
    exactly 1, because the transposed frame's spectrum is the transpose of the
    original's. So this compares an ASYMMETRIC band -- energy at |f_y| > 0.25
    against the same frame transposed. Measured on real frames it comes out
    0.43-0.59, i.e. a decoded frame carries about half as much fine structure
    down the trace as across it, partly real scene anisotropy and partly the
    dot-to-row resampler's roll-off.

    It is REPORTED and not applied. Correcting the transpose control by it
    would roughly double every ratio below in this module's favour, and the two
    axes are not physically interchangeable enough to justify that. The
    uncorrected transpose control is the conservative one, so that is the one
    that is used.
    """
    h, w = img.shape
    P = np.pad(img, ((h // 2, h // 2), (w // 2, w // 2)), mode="reflect")
    H, W = P.shape
    m = (np.abs(np.fft.fftfreq(H))[:, None] > 0.25) & \
        (np.abs(np.fft.fftfreq(W))[None, :] > 0.02)
    e = float(np.sqrt((np.abs(np.fft.fft2(P)) ** 2 * m).sum())) / (H * W)
    Pt = np.pad(img.T, ((w // 2, w // 2), (h // 2, h // 2)), mode="reflect")
    Ht, Wt = Pt.shape
    mt = (np.abs(np.fft.fftfreq(Ht))[:, None] > 0.25) & \
         (np.abs(np.fft.fftfreq(Wt))[None, :] > 0.02)
    et = float(np.sqrt((np.abs(np.fft.fft2(Pt)) ** 2 * mt).sum())) / (Ht * Wt)
    return e / max(et, 1e-12)


def damage(img: np.ndarray, psd: StreakPSD, **kw) -> tuple[float, float]:
    """(removed, scene-damage bound), both in grey levels.

    The frame's streaks run down the trace. TRANSPOSE it and every scene
    structure survives with its statistics intact while the streaks rotate out
    of the filter's target region -- so whatever the filter takes off the
    transposed frame is scene, not streak. The ratio removed/damage is the
    number that says whether a setting is a destriper or a vandal. It is
    reported uncorrected; see `anisotropy` for why that is the conservative
    choice and how much is being given away by it.
    """
    removed = 255 * float(stripe_field(img, psd, **kw).std())
    d_t = 255 * float(stripe_field(img.T, psd, **kw).std())
    return removed, d_t


# --------------------------------------------------------------------------
# the alternatives that lost, kept so the claim is falsifiable
# --------------------------------------------------------------------------


def offset_stripe(img: np.ndarray, span: int = 17) -> np.ndarray:
    """Classic destriping: one constant per trace, high-passed across traces.

    This is what decode.Settings(destripe=True) does and what moment matching
    amounts to. On L000's interior a per-trace constant explains 17% of the
    defect, so this cannot work, and the measurement says so.
    """
    col = img.mean(axis=0)
    pad = np.pad(col, span // 2, mode="edge")
    smooth = np.convolve(pad, np.ones(span) / span, mode="valid")
    return np.broadcast_to((col - smooth)[None, :], img.shape).copy()


def munch_stripe(img: np.ndarray, level: int = 4, sigma: float = 4.0,
                 wavelet: str = "db8") -> np.ndarray:
    """Munch et al. 2009 wavelet-FFT destriping, for comparison.

    Measured on this signal: the vertical-detail band cV is the one that holds
    stripes running down the trace (verified on a synthetic stripe field, not
    assumed from the pywt naming). Needs pywt.
    """
    import pywt

    h, w = img.shape
    py, px = h // 2, w // 2
    P = np.pad(img, ((py, py), (px, px)), mode="reflect")
    co = pywt.wavedec2(P, wavelet, level=level, mode="symmetric")
    out: list = [np.zeros_like(co[0])]
    for k in range(1, len(co)):
        cH, cV, cD = co[k]
        F = np.fft.fft(cV, axis=0)
        f = np.fft.fftfreq(F.shape[0]) * F.shape[0]
        damp = 1.0 - np.exp(-(f ** 2) / (2.0 * sigma ** 2))
        low = cV - np.fft.ifft(F * damp[:, None], axis=0).real
        out.append((np.zeros_like(cH), low, np.zeros_like(cD)))
    S = pywt.waverec2(out, wavelet, mode="symmetric")[:P.shape[0], :P.shape[1]]
    return S[py:py + h, px:px + w]


def notch_stripe(img: np.ndarray, fy0: float = 0.016, fx0: float = 0.08,
                 limit: float = LIMIT_MAD) -> np.ndarray:
    """The hand-set unidirectional Fourier notch: low |f_y| times high |f_x|.

    The honest baseline the calibrated-spectrum filter has to beat. Its cut
    frequencies are the thing this module exists to stop guessing.
    """
    h, w = img.shape
    py, px = h // 2, w // 2
    P = np.pad(img, ((py, py), (px, px)), mode="reflect")
    H, W = P.shape
    fy = np.abs(np.fft.fftfreq(H))[:, None]
    fx = np.abs(np.fft.rfftfreq(W))[None, :]
    M = np.exp(-0.5 * (fy / fy0) ** 2) * (1.0 - np.exp(-0.5 * (fx / fx0) ** 2))
    S = np.fft.irfft2(np.fft.rfft2(P) * M, s=P.shape)
    return _soft_limit(S[py:py + h, px:px + w], limit)


# --------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json
    import sys
    from pathlib import Path

    REPO = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO))

    from pipeline import circle as circle_mod
    from pipeline import decode as decode_mod
    from pipeline import sync as sync_mod
    from pipeline import wav

    ap = argparse.ArgumentParser(description="destripe control report")
    ap.add_argument("--frames", type=int, default=12,
                    help="how many frames in the transfer survey")
    args = ap.parse_args()

    T = json.loads((REPO / "pipeline" / "barry_tables.json").read_text())
    START = T["start_samples"]          # audio measurement; no Earth knowledge
    info = wav.probe(REPO / "data" / "master" / "384kHzStereo.wav")
    mm = wav.memmap(info)
    N = int(sync_mod.NOMINAL_PERIOD * 520)

    def load(fid: str) -> np.ndarray:
        ch = 0 if fid[0].upper() == "L" else 1
        s0 = START[ch][int(fid[1:])]
        seg = np.asarray(mm[s0:s0 + N, ch], dtype=np.float64)
        tb = sync_mod.recover(seg)
        cfg = decode_mod.Settings(traces=512, rotate=0, channel="LR"[ch])
        return decode_mod.decode(seg, cfg, tb=tb).image.astype(np.float64)

    def ring(im):
        r = circle_mod.fit_ring(im)
        return (r.axis_ratio, r.radial_rms) if r else (float("nan"), float("nan"))

    img0 = load("L000")
    psd = calibrate(img0[INTERIOR])
    base = flat_metrics(img0[INTERIOR])
    ar0, rms0 = ring(img0)

    print(__doc__.split("WHAT THE DEFECT IS")[0].strip())
    print()
    print("=" * 78)
    print("1. THE CALIBRATION")
    print("=" * 78)
    print(f"  window {INTERIOR[0].start}:{INTERIOR[0].stop} x "
          f"{INTERIOR[1].start}:{INTERIOR[1].stop} of L000, "
          f"{(INTERIOR[0].stop-INTERIOR[0].start)*(INTERIOR[1].stop-INTERIOR[1].start)} px "
          "the record says are uniform white")
    print(f"  flat field before : sd {base['sd']:.3f}  streak band {base['band']:.3f}  "
          f"pixel noise {base['noise']:.3f}  (band floor {base['band_floor']:.3f}) grey")
    print(f"  per-trace CONSTANT explains "
          f"{100*(base['col_sd']/base['sd'])**2:.0f}% of the defect variance "
          "-- classic one-offset destriping is ruled out")
    print(f"  decoder pixel-noise model fitted at {psd.noise_floor:.2e} "
          f"= {psd.stats['noise_model_grey']:.3f} grey rms, and SUBTRACTED "
          "(this is a destriper, not a denoiser)")
    print(f"  calibrated streak spectrum integrates to {psd.sd_grey:.3f} grey rms")

    print()
    print("=" * 78)
    print("2. THE FLAT FIELD, AND THE SIGN CONTROL")
    print("=" * 78)
    out = apply(img0, psd)
    wrong = apply(img0, psd, sign=-1.0)
    m1, m2 = flat_metrics(out[INTERIOR]), flat_metrics(wrong[INTERIOR])
    for lab, m in (("before", base), ("destriped", m1), ("WRONG SIGN", m2)):
        print(f"  {lab:>11}  sd {m['sd']:.3f}  streak band {m['band']:.3f} grey"
              f"   (band variance {100*(m['band']/base['band'])**2-100:+.0f}%)")
    ok_sign = m2["band"] > base["band"] > m1["band"]
    print(f"  sign control: applying it backwards makes the flat field "
          f"{'WORSE, as it must' if ok_sign else 'NOT worse -- FAILED'}")
    print()
    print("  the limiter is the one knob that matters; here is what moving it costs,")
    print("  on the flat field AND on a real photograph at the same time:")
    f55 = load("L055")
    for c in (2.0, 3.0, 5.0, 1e9):
        mlim = flat_metrics(apply(img0, psd, limit=c)[INTERIOR])
        rem = 255 * stripe_field(f55, psd, limit=c).std()
        dmg = 255 * stripe_field(f55.T, psd, limit=c).std()
        tag = f"{c:.0f}" if c < 100 else "none"
        print(f"      c = {tag:>4}   flat-field streak variance "
              f"{100*(mlim['band']/base['band'])**2-100:+.0f}%   "
              f"L055 removed/damage {rem/max(dmg,1e-9):5.1f}")

    print()
    print("  a second uniform region, never used for tuning "
          "(rows 20:340, traces 424:507, outside the ring):")
    SIDE = (slice(20, 340), slice(424, 507))
    b2, a2 = flat_metrics(img0[SIDE]), flat_metrics(out[SIDE])
    print(f"      before sd {b2['sd']:.3f} band {b2['band']:.3f}  ->  "
          f"after sd {a2['sd']:.3f} band {a2['band']:.3f} grey")

    print()
    print("=" * 78)
    print("3. THE HOLD-OUT: calibrate on half the traces, measure on the other half")
    print("=" * 78)
    ys, xs = INTERIOR
    mid = (xs.start + xs.stop) // 2
    for a, b, lab in ((slice(xs.start, mid), slice(mid, xs.stop), "left->right"),
                      (slice(mid, xs.stop), slice(xs.start, mid), "right->left")):
        p = calibrate(img0[ys, a])
        o = apply(img0, p)
        mb, ma = flat_metrics(img0[ys, b]), flat_metrics(o[ys, b])
        print(f"  {lab}: held-out traces {b.start}:{b.stop}   band "
              f"{mb['band']:.3f} -> {ma['band']:.3f} grey "
              f"({100*(ma['band']/mb['band'])**2-100:+.0f}% variance)")

    print()
    print("=" * 78)
    print("4. THE CIRCLE -- the invariant that must not move")
    print("=" * 78)
    print(f"  before      axis ratio {ar0:.5f}   radial rms {rms0:.4f} px")
    for lab, im in (("destriped", out),
                    ("hand-set notch", img0 - notch_stripe(img0)),
                    ("notch, NO limiter", img0 - notch_stripe(img0, limit=1e9)),
                    ("Munch wavelet-FFT", img0 - munch_stripe(img0))):
        a, r = ring(im)
        print(f"  {lab:<18} axis ratio {a:.5f}   radial rms {r:.4f} px")
    print("  metric's own scatter, from repeated fits with noise added: "
          "+/-0.0005 on the ratio, +/-0.008 px on the rms")
    print()
    print("  the geometry survives, but a Fourier filter still rings at a strong")
    print("  curved edge, and the honest thing is to measure the halo rather than")
    print("  hide behind a metric that cannot see it:")
    S0 = stripe_field(img0, psd)
    rg = circle_mod.fit_ring(img0)
    yy, xx = np.mgrid[0:img0.shape[0], 0:img0.shape[1]]
    rr = np.hypot((xx - rg.cx) / rg.rx, (yy - rg.cy) / rg.ry)
    dpx = np.abs(rr - 1.0) * 0.5 * (rg.rx + rg.ry)
    inner = (yy > 20) & (yy < 340)
    for lo, hi in ((0, 3), (3, 8), (8, 16), (16, 32), (32, 64)):
        m = (dpx >= lo) & (dpx < hi) & inner
        print(f"      {lo:2d}-{hi:2d} px from the ring: |stripe| mean "
              f"{255*np.abs(S0[m]).mean():.3f}  p99 {255*np.percentile(np.abs(S0[m]),99):.3f} grey")
    m = (dpx > 64) & inner
    print(f"      far from the ring   : |stripe| mean "
          f"{255*np.abs(S0[m]).mean():.3f}  p99 {255*np.percentile(np.abs(S0[m]),99):.3f} grey")

    def _depth(im):
        pr, _ = circle_mod._radial_profile(im, rg.cy, rg.cx, rg.ry, rg.rx)
        ok = np.isfinite(pr)
        return float(pr[ok].max() - pr[ok].min())
    print(f"      the ring's own contrast: {_depth(img0):.4f} -> {_depth(out):.4f}"
          f" ({100*(_depth(out)/_depth(img0)-1):+.2f}%)")

    print()
    print("=" * 78)
    print("5. TRANSFER: does it survive frames that are not the calibration frame?")
    print("=" * 78)
    print("   removed = rms of what is taken off; damage = the same filter run on the")
    print("   TRANSPOSED frame -- scene intact, streaks rotated out of the target band.")
    print("   Uncorrected, so it is a conservative bound (see `anisotropy`). The last")
    print("   column is the hand-set notch, run through the identical control.")
    print()
    fids = [f"L{i:03d}" for i in (55, 11, 20, 38, 2, 30, 62, 70)] + \
           [f"R{i:03d}" for i in (20, 40, 5, 60)]
    fids = fids[:args.frames]
    print(f"  {'frame':>6} {'removed':>9} {'damage':>8} {'ratio':>7}   "
          f"{'notch removed':>14} {'damage':>8} {'ratio':>7}")
    rows = []
    for fid in fids:
        f = load(fid)
        rem, dmg = damage(f, psd)
        nr = 255 * notch_stripe(f).std()
        nd = 255 * notch_stripe(f.T).std()
        rows.append((rem / max(dmg, 1e-9), nr / max(nd, 1e-9), anisotropy(f)))
        print(f"  {fid:>6} {rem:9.2f} {dmg:8.2f} {rem/max(dmg,1e-9):7.2f}   "
              f"{nr:14.2f} {nd:8.2f} {nr/max(nd,1e-9):7.2f}")
    w = np.array(rows)
    print(f"  {'median':>6} {'':9} {'':8} {np.median(w[:,0]):7.2f}   "
          f"{'':14} {'':8} {np.median(w[:,1]):7.2f}")
    print(f"  frame anisotropy (reported, NOT applied): median {np.median(w[:,2]):.2f} "
          "-- applying it would roughly double both columns")

    print()
    print("=" * 78)
    print("5b. WHAT IT COSTS ON REAL VERTICAL STRUCTURE")
    print("=" * 78)
    print("   The hardest single object on the record for a destriper: a bright")
    print("   vertical tower a few traces wide, standing tall down the trace.")
    for fid in ("L055", "L011"):
        f = load(fid)
        S = stripe_field(f, psd)
        # the frame's own strongest tall-thin vertical features: columns whose
        # across-trace second difference is large and coherent down the trace.
        v = f - 0.5 * (np.roll(f, 3, axis=1) + np.roll(f, -3, axis=1))
        v[:, :4] = v[:, -4:] = 0.0
        score = _smooth_y(v, 12.0)
        k = np.unravel_index(np.argsort(np.abs(score).ravel())[-2000:], score.shape)
        amp = 255 * float(np.abs(score[k]).mean())
        cost = 255 * float(np.abs(S[k]).mean())
        rest = 255 * float(np.abs(S).mean())
        print(f"   {fid}: the 2000 pixels with the strongest tall-thin vertical")
        print(f"        structure carry {amp:6.2f} grey of it; the filter takes "
              f"{cost:.3f} grey off them")
        print(f"        ({100*cost/amp:.1f}% of the structure), against "
              f"{rest:.3f} grey off an average pixel.")

    print()
    print("=" * 78)
    print("6. NEGATIVES")
    print("=" * 78)
    o_off = img0 - offset_stripe(img0)
    m_off = flat_metrics(o_off[INTERIOR])
    print(f"  one constant per trace (decode's own destripe=True): band "
          f"{base['band']:.3f} -> {m_off['band']:.3f} grey. The defect is a"
          " trajectory, not an offset.")
    o_m = img0 - munch_stripe(img0)
    m_m = flat_metrics(o_m[INTERIOR])
    a_m, r_m = ring(o_m)
    print(f"  Munch wavelet-FFT (level 4, sigma 4): band {m_m['band']:.3f} grey but"
          f" radial rms {r_m:.4f} px -- it moves the ring.")
    print(f"  the linear filter with no limiter: |stripe| reaches "
          f"{np.abs(notch_stripe(img0, limit=1e9)).max():.3f} of full scale at the"
          " frame borders.")
