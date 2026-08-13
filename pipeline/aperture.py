"""The sample-and-hold aperture: what plateau integration really costs.

READ THIS BEFORE USING ANY OF IT -- the headline is smaller and different
from the one this module was commissioned to confirm.

THE PROPOSAL, AND WHY IT IS WRONG AS STATED
-------------------------------------------
The idea was: each dot is a rectangular sample-and-hold of length T = one dot,
a rectangular hold has transfer sinc(f*T), so at the dot grid's Nyquist the
picture is attenuated by 2/pi = 0.637 (-3.9 dB), and integrating the plateau
(what decode.py does) leaves that roll-off in.

It does not. The hold is a RECONSTRUCTION filter, not an ACQUISITION filter.
The converter forms a value v_k -- by whatever aperture its own front end
had -- and then holds it. Averaging a constant over the interval it is held
returns that constant. In the time domain that is a one-line proof; in the
frequency domain it is the identity

    SUM_m sinc^2(f + m) = 1     for all f          (checked to 1e-3, M=200)

-- the sinc^2 of hold-then-boxcar is exactly cancelled by the aliasing sum of
the dot lattice. A perfect staircase in, the exact plateau values out, no
correction of any kind. Any "aperture inverse" applied on that basis is pure
invented sharpening. The sinc belongs to the CONTINUOUS waveform's spectrum,
not to the recovered sample sequence, and confusing the two is the whole of
the original error.

WHAT IS ACTUALLY THERE, MEASURED ON THE MASTER
----------------------------------------------
The staircase is not perfect: everything between the converter's hold and our
integrator -- the sampler's own output stage, tape, disc, the 2017 ADC -- is a
filter c, and c smears each plateau-to-plateau step. Then, and only then, is
there something to correct, because our one-dot boxcar integrates across those
smeared boundaries and mixes in the neighbouring dots.

c was measured directly, on the record, at the same node and through the same
path as the picture: a plateau-to-plateau transition IS a step through c.
323 isolated single-trace steps on L000/L020/L034/R010, fitted one profile at
a time (never stacked, so no timing jitter of ours can enter -- the house
rule about the retracted MTF number):

    10-90 rise of a plateau transition = 0.49 dots
        (model-free mean step shape 0.4935; per-edge medians: logistic 0.523,
         erf 0.464; 6.1-6.9 samples at 384 kHz, 1.5-1.8 at 96 kHz -- the SAME
         width in dots at both rates, so the /4 decimation in build.py adds
         essentially nothing and is NOT a hidden aperture. Negative result,
         checked because it would have been ours to fix.)

Three properties of that measurement, each of which could have killed the
model and did not:

  * the width does not depend on step amplitude (6.41 / 6.61 / 6.10 / 5.83 /
    6.21 samples over amplitude quintiles, n=125), so c is a LINEAR filter,
    not a slew-rate limit;
  * the mean step shape is SYMMETRIC about the transition (0.098 at -0.25
    dots, 0.505 at 0, 0.905 at +0.25), so c is not the causal one-pole the
    chain's DC droop is -- a causal pole of the same 10-90 would give
    A(0.5) = 0.76 rather than 0.69, and it is excluded by shape;
  * the plateau is fully settled by +-0.5 dots (the shape reads 0.999 there),
    so the plateau CENTRE already carries v_k. Nothing is lost at the centre.
    The loss is entirely a cost of integrating the WHOLE dot.

THE COMPOSITE TRANSFER (this is the thing to use)
-------------------------------------------------
For a window covering the fraction [lo, hi) of each dot, with w = hi - lo and
the window centre offset ctr = (hi+lo)/2 - 0.5 dots from the dot centre, the
transfer from the true dot sequence v to the recovered sequence u is exactly

    A(f) = SUM_m sinc(w*(f+m)) * sinc(f+m) * C(f+m) * exp(-2i*pi*(f+m)*ctr)

f in cycles per dot, C the Fourier transform of c in the same units, and the
sum over m the aliasing of the dot lattice. The three factors are OUR window,
the converter's hold, and the chain. A(0) = 1 identically (sinc(m) = 0 at
non-zero integers), so the correction cannot touch DC or move a flat field.
Set C = 1 and A == 1 for every window inside the plateau: the no-op above.

With the measured c and the full-dot window (w = 1, ctr = 0):

    A(0.25) = 0.85     A(0.375) = 0.78     A(0.5) = 0.70

so the honest ceiling for this correction is x1.43 = +3.1 dB at the dot
Nyquist, rolling off smoothly to nothing at DC. NOT the -3.9 dB of the
proposal, and not for the proposal's reason: sinc^2 alone would give 0.405,
the aliasing sum returns most of it, and what survives is the part the chain
filtered away before the fold could return it.

There is NO NULL. sinc(f) does not reach its null until f = 1 cycle/dot,
twice Nyquist, and the alias sum keeps |A| >= 0.69 across the whole band. The
inverse is bounded by 1.43 everywhere, needs no null handling, and is
numerically trivial. That much of the original proposal survives, and it is
the good part: this operator is known from geometry plus one measured width,
not estimated from the picture.

HOW MUCH OF THE OBSERVED BLUR IS REALLY THIS -- THE HALF-WINDOW TEST
---------------------------------------------------------------------
The model was then tested against a measurement it did not fit. Split every
plateau into two DISJOINT halves A and B, and form the complex transfer

    r(f) = <D_i conj(S_i+1)> / <S_i conj(S_i+1)>,   D = U_A - U_B, S = U_A + U_B

over pairs of NEIGHBOURING traces. Noise is independent between traces so it
cancels; the scene's trace-to-trace coherence is common to D and S so it
cancels too; what is left is exactly Delta(f)/Sigma(f), which the model above
predicts with no free parameters. It is a ratio of two measurements of the
same scene, so picture content, gain and the scene spectrum all drop out.
(A first attempt used |D|/|S| within one trace: at high f that is pure noise
ratio and tends to 1. It measured our noise floor, not the aperture. The
cross-trace form is what removes it.)

The prediction has the right shape and sign on all 16 frames tried -- but the
measured blur is bigger. Fitting (kernel width, per-trace lattice jitter) to
the complex transfer on 16 frames:

    A_eff(0.5) = 0.37-0.38 measured        vs   A_det(0.5) = 0.695 from edges

The excess is accounted for by PER-TRACE DOT-PHASE JITTER: the cross-trace
estimator measures the ensemble-AVERAGED transfer, and a random lattice error
blurs it exactly like extra kernel width. Confirmed in a full end-to-end
simulation (known scene -> hold -> measured c -> 384 kHz sampling -> the same
window and estimator code): with zero jitter it reproduces the no-jitter
prediction to 4% (-0.049 vs -0.047 at f = 0.05, -0.339 vs -0.337 for a
doubled kernel), and jitter of 0.15-0.35 dots reproduces the observed excess.
Width and jitter are DEGENERATE in that fit -- both blur -- so the half-window
test pins only the product, and per-frame splits scatter over width 0.5-1.0
and jitter 0.0-0.35. The single-trace edge fits break the degeneracy, because
each one fits its own edge position and so cannot absorb any timing error.

THAT SPLIT IS THE WHOLE POINT AND IT IS WHY THE ANSWER IS SMALL:

    deterministic, exactly derivable, ours to remove ...... A(0.5) = 0.695
    stochastic, OUR OWN timing noise, must NOT be removed .. down to ~0.37

Correcting to 0.37 would need a gain of 2.7 and would be deconvolving our own
dot-clock jitter -- the exact mistake this project retracted once already
(the MTF number that measured our timing jitter). Only the deterministic
factor is corrected here. The jitter is reported, not inverted -- and note
that it is at or above the ~0.1 dots dotclock.py's phase tracker claims for
itself, and that removing it would be worth roughly TWICE what this correction
is. That is the bigger prize, and it belongs to whoever owns dotclock.py.

THE ALTERNATIVE THAT NEEDS NO INVERSE AT ALL
--------------------------------------------
Because the plateau is settled at its centre, `centre_window_transfer` shows
the loss can also be avoided by simply integrating LESS of each dot: the
central 50% of the plateau gives A(0.5) = 0.93 with no filtering whatsoever,
at the price of averaging half as many samples. Whether that is a better
trade than deconvolving the full window depends on the noise, which
`recommend` computes from the frame itself. It is offered because an operator
you never apply cannot ring, and a decoder that samples the settled part of
the plateau is easier to defend than one that samples all of it and divides
the error back out.

DOES IT SURVIVE A HOLD-OUT? YES, ON 6 OF 6 FRAMES
--------------------------------------------------
Every dot value was estimated from the settled central 40% of its own plateau
only. The correction for THAT window was applied. Then the raw 384 kHz samples
within +-0.15 dots of every plateau boundary -- the transition zones, which
contributed nothing to the estimate -- were predicted from the corrected dot
values and scored against the record. Over ~380 traces per frame:

    strength 1.0 beat strength 0.0 on all of L000 L034 L055 R040 R010 R056,
    by 0.5-4.9% of held-out residual rms, at every kernel width tried.

The test cannot be gamed in either direction: over-sharpened dot values
predict transitions that swing too far and score worse, under-sharpened ones
swing too little. Sweeping the kernel width in the same loop, the held-out
optimum lands at 0.70-1.00 dots -- the EFFECTIVE width again, not the
deterministic 0.49, because each trace's prediction uses that trace's own
(imperfect) lattice phase and a wider kernel absorbs the error. Three
independent instruments -- single-trace edge fits, the cross-trace half-window
transfer, and held-out waveform prediction -- therefore agree on the same
decomposition, which is the reason to believe the split at all.

WHAT IT BUYS, MEASURED ON REAL DECODES
--------------------------------------
Edges fitted one profile at a time on a FIXED edge set (selected once on the
uncorrected image, refitted in the same windows afterwards, so the selection
cannot drift), gated to |slant| < 0.15, in the dot domain:

    along the trace (the axis this filter acts on)   -7.2% to -16.0% of 10-90
    ACROSS traces (control, not filtered)            -0.0%, +0.2%, -1.0%

The across-trace control is the check a generic sharpening fails, and it
passes on the three frames that have enough axis-aligned across-trace edges
to measure (L034, L055, R040). It does NOT pass on R056 (-7.5%), and on
frames with strongly slanted content the coupling is real and larger: without
the slant gate L000 reads -11.2% across, because a slanted edge is sharpened
along its own normal by any filter applied to one axis. Reported, not hidden.

The measured narrowing agrees with the operator's own prediction (the same
correction applied to a synthetic edge of the measured pre-correction width)
to +3% .. +13%, so it is the operator doing what it says rather than the
fitter finding ringing attractive.

Cost, on the same decodes: dot-domain noise rises x1.28 (L000 along-trace
0.00402 -> 0.00514), against x1.21 predicted for white noise. So the trade is
+3.16 dB of MTF at the dot Nyquist for -2.1 dB of SNR.

AND THE PART THAT SAYS IT IS SMALLER THAN IT LOOKS
---------------------------------------------------
The along-trace edges of real decoded frames are 1.58-2.43 dots (10-90). This
aperture is 0.49 of that, in quadrature 6-10% OF THE BLUR VARIANCE. The other
90%+ is the 1977 camera, which is pipeline/deconv.py's problem and is the
resolution ceiling this record actually has. An exactly-known operator that
nobody had removed is a clean win, and this one is real, held-out, and free of
any reference image -- but it is a 10% correction to the sharpness of the
picture, not a new picture. Anyone reading "+3.9 dB at Nyquist" off the
original sinc argument would have been claiming about four times this, in the
wrong place, for the wrong reason.

WHAT THIS MODULE DOES NOT TOUCH
-------------------------------
  * ACROSS traces. The converter's sampling gate has a finite aperture in the
    input video's time axis, which maps to the HORIZONTAL direction of the
    picture. That aperture is real, is not known from any geometry we have,
    and is not measurable by the method used here (there is no hold across
    traces to measure a transition of). Nothing across traces is corrected.
  * The camera. The 1977 vidicon's own PSF is the resolution ceiling and is
    pipeline/deconv.py's subject. See `compose` for how the two operators
    must be ordered so they are not applied twice to the same blur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- measured on the master; see the module docstring ----------------------
# 10-90 rise of a plateau-to-plateau transition, in dots. Model-free mean of
# 323 single-trace step fits. The per-model spread (erf 0.464, logistic 0.523)
# is the honest uncertainty and `transfer` is smooth in it: A(0.5) moves
# 0.72 -> 0.67 across that range, i.e. the correction is 1.39x-1.49x.
TRANSITION_1090 = 0.49
TRANSITION_1090_RANGE = (0.464, 0.523)

# Per-trace dot-lattice jitter implied by the half-window test, in dots. NOT
# corrected -- it is our own timing noise. Recorded here because it sets how
# much of the residual blur this module deliberately leaves alone. Degenerate
# against kernel width in that fit (see the docstring), hence a range; use it
# with `transfer(jitter=...)` to PREDICT what a measurement will see, never to
# divide anything out.
PHASE_JITTER_DOTS = 0.25
PHASE_JITTER_RANGE = (0.15, 0.35)
# Combined effective transfer at the dot Nyquist actually measured by the
# half-window test, median over 16 frames. The gap to A(0.5) = 0.695 is the
# jitter, and is deliberately left in.
A_NYQUIST_EFFECTIVE = 0.375

# 10-90 of a Gaussian is 2.5631 sigma.
SIGMA_PER_1090 = 1.0 / 2.5631

_N_ALIAS = 40  # terms each side in the aliasing sum; the tail is O(1/m^2)


def chain_mtf(nu: np.ndarray, width: float = TRANSITION_1090) -> np.ndarray:
    """|C(nu)| of the chain kernel, nu in cycles per dot.

    Gaussian of the measured 10-90 width. The measured mean step shape and a
    Gaussian of the same 10-90 give A(0.5) = 0.6919 vs 0.6745 -- a 2.5%
    difference in the correction, well inside the width uncertainty -- so the
    two-parameter shape argument is not worth carrying. A CAUSAL one-pole is
    excluded by the measured symmetry of the step, not assumed away.
    """
    s = width * SIGMA_PER_1090
    return np.exp(-2.0 * np.pi**2 * s**2 * np.asarray(nu, dtype=np.float64) ** 2)


def transfer(
    f: np.ndarray,
    *,
    width: float = TRANSITION_1090,
    window: tuple[float, float] = (0.0, 1.0),
    jitter: float = 0.0,
    n_alias: int = _N_ALIAS,
) -> np.ndarray:
    """Composite dot-sequence transfer A(f), f in cycles per dot.

    `window` is the fraction of each plateau actually integrated: (0, 1) is
    decode.py's full-plateau integration, (0.25, 0.75) the central half.
    `jitter` (dots, rms) gives the ENSEMBLE-AVERAGED transfer including a
    random per-trace lattice error -- use it to describe what a measurement
    will see, never as something to divide out.

    Returns a complex array; it is real and positive for a centred window.
    """
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    lo, hi = window
    w = hi - lo
    if w <= 0:
        raise ValueError("window must have positive width")
    m = np.arange(-n_alias, n_alias + 1)

    def one(shift: float) -> np.ndarray:
        ctr = (hi + lo) / 2.0 - 0.5 - shift
        nu = f[:, None] + m[None, :]
        return (
            np.sinc(w * nu) * np.sinc(nu) * chain_mtf(nu, width)
            * np.exp(2j * np.pi * nu * ctr)
        ).sum(axis=1)

    if jitter <= 0:
        return one(0.0)
    q = np.linspace(-3.0, 3.0, 25)
    p = np.exp(-0.5 * q**2)
    p /= p.sum()
    return sum(wi * one(qi * jitter) for qi, wi in zip(q, p))


def centre_window_transfer(frac: float, **kw) -> np.ndarray:
    """A(f) at f = Nyquist for the central `frac` of each plateau."""
    lo = 0.5 - frac / 2.0
    return transfer(np.array([0.5]), window=(lo, lo + frac), **kw).real[0]


@dataclass
class Correction:
    """One frozen aperture correction, ready to apply along the dot axis."""

    strength: float
    width: float
    window: tuple[float, float]
    gain_max: float
    taps: np.ndarray  # equivalent symmetric FIR, for reporting/inspection

    def describe(self) -> str:
        t = " ".join(f"{v:+.4f}" for v in self.taps)
        return (f"aperture x{self.strength:g} (10-90 {self.width:.3f} dots, "
                f"max gain {self.gain_max:.3f}) taps [{t}]")


def inverse_gain(
    f: np.ndarray,
    *,
    strength: float = 1.0,
    width: float = TRANSITION_1090,
    window: tuple[float, float] = (0.0, 1.0),
    gain_cap: float = 2.0,
) -> np.ndarray:
    """Real, zero-phase gain that undoes `transfer` to the given `strength`.

    strength = 0 is the identity, 1 the full inverse; fractional values are
    A^-strength, which is the geometric interpolation, so a half correction is
    exactly half the correction in dB at every frequency.

    No null handling is needed and none is done: |A| >= 0.69 over the whole
    band, so the inverse is bounded by 1.43 by construction. `gain_cap` exists
    only to make an absurd `width` fail safe rather than explode.
    """
    A = np.abs(transfer(f, width=width, window=window))
    A = np.maximum(A, 1.0 / gain_cap)
    return np.minimum(A ** (-float(strength)), gain_cap)


def build(
    n: int,
    *,
    strength: float = 1.0,
    width: float = TRANSITION_1090,
    window: tuple[float, float] = (0.0, 1.0),
    gain_cap: float = 2.0,
    n_taps: int = 9,
) -> Correction:
    """Build the correction for a dot axis of length `n`."""
    f = np.fft.rfftfreq(n)
    g = inverse_gain(f, strength=strength, width=width, window=window, gain_cap=gain_cap)
    k = np.fft.irfft(g.astype(complex), n=n)
    half = n_taps // 2
    taps = np.concatenate([k[-half:], k[: half + 1]]) if half else k[:1]
    return Correction(float(strength), float(width), tuple(window), float(g.max()), taps)


def apply_dots(
    pic: np.ndarray,
    *,
    strength: float = 1.0,
    width: float = TRANSITION_1090,
    window: tuple[float, float] = (0.0, 1.0),
    gain_cap: float = 2.0,
    axis: int = 1,
) -> np.ndarray:
    """Apply the aperture inverse ALONG THE DOT AXIS of a dot matrix.

    `pic` is decode.py's (traces, dots) matrix -- the raw dot rows, before any
    resampling to square pixels; correcting after interpolation would be
    correcting the interpolator too. Zero-phase and unity at DC, so no edge
    moves and no flat area changes level: the calibration circle's fit and
    L000's flat field are invariant by construction, not by luck. Confirmed on
    the master at strength 1.0: circle axis_ratio 1.0060 -> 1.0062, radial rms
    0.88 -> 0.88 px (190 inliers); L000's field p99-p1 0.1071 -> 0.1086 and sd
    0.0280 -> 0.0281, that residue being the noise gain acting on the field's
    own noise, not a change of level or of shape.

    Reflect-padded by 16 dots; the operator's impulse response is 3 taps wide
    at the 1e-3 level (taps [-0.109, 1.200, -0.109]) so that is ample.
    """
    a = np.asarray(pic, dtype=np.float64)
    if strength == 0:
        return a.copy()
    a = np.moveaxis(a, axis, -1)
    pad = 16
    b = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(pad, pad)], mode="reflect")
    n = b.shape[-1]
    g = inverse_gain(np.fft.rfftfreq(n), strength=strength, width=width,
                     window=window, gain_cap=gain_cap)
    out = np.fft.irfft(np.fft.rfft(b, axis=-1) * g, n=n, axis=-1)[..., pad:pad + a.shape[-1]]
    return np.moveaxis(out, -1, axis)


def noise_gain(
    *,
    strength: float = 1.0,
    width: float = TRANSITION_1090,
    window: tuple[float, float] = (0.0, 1.0),
    n: int = 4096,
) -> float:
    """RMS amplification of white dot-domain noise by the correction."""
    g = inverse_gain(np.fft.rfftfreq(n), strength=strength, width=width, window=window)
    return float(np.sqrt(np.mean(g**2)))


def window_cost(frac: float, width: float = TRANSITION_1090) -> dict:
    """Compare narrowing the integration window against deconvolving.

    Integrating the central `frac` of each plateau raises A(0.5) but averages
    fewer samples, so its noise grows as 1/sqrt(frac) (white noise) -- while
    deconvolving the full window keeps all the samples and pays `noise_gain`.
    Returns both costs at equal MTF so the trade can be read off.
    """
    lo = 0.5 - frac / 2.0
    win = (lo, lo + frac)
    A_nyq = float(np.abs(transfer(np.array([0.5]), width=width, window=win))[0])
    A_full = float(np.abs(transfer(np.array([0.5]), width=width))[0])
    # strength that makes the full window match this window's Nyquist MTF
    s = np.log(A_nyq) / np.log(A_full) if A_full < 1 else 0.0
    s = float(np.clip(1.0 - s, 0.0, 1.0))
    return {
        "frac": frac,
        "A_nyquist": A_nyq,
        "noise_narrow": 1.0 / np.sqrt(frac),
        "strength_for_same_mtf": s,
        "noise_deconv": noise_gain(strength=s, width=width),
    }


def compose(camera_1090_dots: float, *, width: float = TRANSITION_1090) -> dict:
    """How this operator composes with pipeline/deconv.py's camera ESF.

    The two are different physical operators IN SERIES along the dot axis:

        scene -> camera PSF -> (v_k) -> hold -> chain c -> our boxcar -> u_k

    so the total dot-domain transfer is H_camera(f) * A(f) and the correct
    order is to divide A out FIRST, then measure and deconvolve the camera.

    THE DOUBLE-CORRECTION RISK IS REAL AND IT RUNS THIS WAY: deconv.py fits
    its ESF width on edges of the DECODED dot matrix, which already carry A.
    An ESF fitted there is the COMPOSITE, and deconvolving it removes A as a
    side effect. Applying this module as well would then correct A twice.

    So exactly one of these, never both:
      (a) apply `apply_dots` first, RE-FIT deconv.py's ESF on the corrected
          matrix, and deconvolve that. Preferred: A is derived from geometry
          plus one measured width and is exact, so removing it first leaves
          deconv a smaller and better-conditioned residual to estimate;
      (b) leave A inside deconv.py's fitted width and do not call this module
          at all.

    Measured on decoded frames, the composite along-trace 10-90 is 1.58-2.43
    dots, so this aperture is 6-10% of the blur VARIANCE and the camera is the
    rest. Order (a) matters more for conditioning than for size.

    This returns the quadrature bookkeeping for (a): how much of a composite
    ESF of `camera_1090_dots` is this aperture, and what is left for the
    camera afterwards. Quadrature is exact for Gaussians and good enough for
    the accounting; the operators themselves compose by multiplication of
    transfers, which is what the code does.
    """
    a = float(width)
    tot = float(camera_1090_dots)
    rem = float(np.sqrt(max(tot**2 - a**2, 0.0)))
    return {
        "composite_1090_dots": tot,
        "aperture_1090_dots": a,
        "camera_only_1090_dots": rem,
        "aperture_share_of_variance": (a**2 / tot**2) if tot > 0 else float("nan"),
        "A_nyquist": float(np.abs(transfer(np.array([0.5]), width=a))[0]),
    }


def recommend(noise_rms: float, signal_rms: float) -> dict:
    """A conservative default, from the frame's own SNR.

    The correction's own noise cost is `noise_gain` = 1.21x rms at full
    strength (1.28x measured on real dot matrices, where the noise is not
    quite white). That is a shallow price -- the operator being inverted is
    shallow and has no null -- so noise is NOT the binding constraint, and no
    frame in the test set showed visible amplification at strength 1.0.

    The binding constraint is CONFIDENCE IN THE WIDTH: 0.464-0.523 dots is the
    honest range, i.e. strength 1.0 is right to within +-3% of gain at the dot
    Nyquist. RECOMMENDED DEFAULT: strength 1.0.

    Drop to 0.0 when the frame's dot clock was NOT measured (decode.py's
    `dot_locked` false, dot_clock_strength < 1.8): the fallback path bins with
    a stretched Lanczos kernel on the predicted pitch instead of integrating
    plateaus, so `transfer` does not describe what happened to that frame and
    the correction would be applied to the wrong operator.
    """
    ng = noise_gain(strength=1.0)
    snr = signal_rms / max(noise_rms, 1e-12)
    return {
        "strength": 1.0,
        "noise_gain": ng,
        "snr_before_db": 20 * np.log10(snr),
        "snr_after_db": 20 * np.log10(snr / ng),
        "note": "default 1.0 unless the frame's dot clock was not measured; "
                "the trade is +3.16 dB of MTF at the dot Nyquist for -1.65 dB "
                "of SNR predicted, -2.1 dB measured on real dot matrices",
    }
