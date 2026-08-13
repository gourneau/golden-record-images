"""Denoise from the calibration circle alone: measure the chain's noise, then
filter optimally. No training scenes, no repeats, no corpus of any kind.

THE QUESTION. The denoiser that ships (n2n.py) learns from the twenty colour
triplets -- three independent scans of one scene. That is tier 1 and it works
(+7.73 dB in the chroma-free high band, low band better on 15/19). But it only
has training data for 20 of the 116 images; on the other 96 single-scanned
frames it is extrapolating. Could the same value come from the CALIBRATION
CIRCLE instead, which calibrates the CHAIN and therefore works identically on
all 156 frames?

WHAT THE CIRCLE GIVES. Frame L000 is a black ring on a field that is uniform
white BY DESIGN, and the record's own cover says the first image is a circle,
so this is tier 0 ground truth the artifact supplies deliberately. Inside the
ring the true image is a constant. Everything seen there beyond a smooth
shading term IS the chain's noise, observable directly over ~32,000 pixels of
unclipped interior. That is the one place on the record where the noise can be
measured at EVERY spatial frequency, including the low-frequency corner where a
photograph can never separate noise from picture.

  measured here, on the float decode (377x512 square-pixel render, fuse.py's
  ordinary chain -- the same render the arbiter scores):
    82.5 grey levels of droop down the trace (rows), the dominant defect
    after the shading model below, residual noise sd 1.81 grey levels
    lag-1 correlation +0.833 DOWN the trace, +0.084 ACROSS traces
      -> the noise is smooth along a trace and nearly white across traces,
         exactly the anisotropy the streaks look like

HOW SHADING WAS SEPARATED FROM NOISE, because that choice is where this route
can quietly go wrong. Shading is defined as THE PART THAT IS COMMON TO THE
TRACES: for each row of the render (each position along the trace) the model
removes a mean and a linear ramp across traces, and nothing else. Nothing is
smoothed along the trace, so the droop -- which is a per-trace time response,
identical trace to trace -- is captured exactly at full resolution and the
noise's own slow structure DOWN the trace survives untouched. The alternative
(smoothing each trace along its own length) would have eaten precisely the
signal we are trying to measure.

  What this costs, stated exactly: it zeroes the fx = 0 column of the noise
  spectrum and attenuates the two or three bins next to it. For noise
  independent across traces the bias elsewhere is 1/144 = 0.7%.
  Cross-check: a completely different shading model (2-D polynomial, degree 6
  along the trace x degree 2 across) gives residual sd 1.805 vs 1.806 grey
  levels -- the measurement does not depend on the choice.

  THE HONEST LIMIT, and it is not small: the measured spectrum RISES 8.4 dB
  above its plateau at the lowest across-trace frequencies (|fx| < 0.02
  cyc/px). That rise cannot be told apart from a genuine slow illumination
  gradient across the calibration slide, which is not noise at all. Treating
  it as noise would license a Wiener filter to attack real low-frequency
  structure in every photograph. So the default template is CAPPED flat below
  |fx| = 0.09; the uncapped variant is built and scored too, and the
  difference is reported rather than chosen.

WHAT THE SPECTRUM LOOKS LIKE (dB relative to its own plateau), and it is not
isotropic in any of its three features:
  along the trace (fy): +11.2 dB below fy = 0.02 -- the slow per-trace level
    trajectory, i.e. the streaks -- then a flat plateau 0.05-0.24, then a
    cliff past 0.26 which is the decoder resampling ~230 dots onto 377 rows
  across traces (fx), and it depends on fy, so the spectrum is NOT separable:
    for fy > 0.05 it is flat to within 1 dB from 0.02 to 0.5 -- white across
    traces; for fy < 0.04 it carries +13.0 dB below fx = 0.02 (the ambiguous
    corner above) and +5.8 dB at fx = 0.5, which is trace PARITY -- the
    odd/even alternation decode.py's clamp docstring already names. Parity is
    a per-trace offset, so it is constant DOWN the trace, which is exactly why
    it appears only at low fy and why one separable A(fy)B(fx) misses it
    (that model recovers 22% of the parity power; the two-regime model used
    here recovers 50%).
  the model is scored where the truth is known: over the arbiter's high band
    the two-regime model accounts for 1.02 of the measured noise power, over
    its low band 0.79 (the shortfall is the capped corner), over fy 0.05-0.25
    0.97 and past fy 0.25 1.11.

DOES IT TRANSFER? The measurement is only useful if the noise is a property of
the CHAIN, not of L000. Two independent checks, on 14 frames spanning both
channels and both ends of the record:

  1. PICTURE DOMAIN. In any frame the region |fx| > 0.35 cyc/px is past the
     1977 camera's across-trace resolution (we sample 1.4-1.9x above its
     Nyquist), so what lives there is noise. The circle proves that reading is
     right: in L000's ring interior, where the truth is known to be constant,
     that band's spectrum matches the same band read off the whole frame to
     within ~1.5 dB. Across the 14 frames the noise-band shape agrees with the
     circle to an rms of 1.93 dB (worst 2.71) over fy in 0.01-0.35. SHAPE
     TRANSFERS.
  2. SIGNAL DOMAIN. The back porch (frac 0.020-0.072 of every trace) is
     content-free by construction -- decode.py already clamps on it -- so it
     is a noise-only strip on every frame, 512 traces x 42 samples, before any
     resampling. Converted to grey by each frame's own sync anchors: within-
     porch scatter 8.96 +/- 1.68 grey levels (5.4-11.9), post-clamp per-trace
     streak 0.77 +/- 0.31 grey (0.28-1.50). L000 sits inside both ranges
     (11.91 and 1.06): the calibration frame is an ORDINARY frame of this
     record, which is the premise the whole route rests on.

  WHAT DOES NOT TRANSFER, and it decides the design: the AMPLITUDE. The
  low-frequency streak boost ranges from +3.2 dB (L025) to +19.7 dB (R077)
  frame to frame, and the cliff depth past fy = 0.40 varies by 20 dB with how
  well the dot clock locked. A single fixed noise level measured on L000 and
  applied everywhere would be wrong by up to 16 dB. So the filter takes from
  the circle what only the circle can give -- the shape ACROSS traces down to
  low fx, the parity spike, and the proof that the |fx|>0.35 band is noise --
  and reads the along-trace profile and the absolute level from each frame's
  own noise-only band. Still tier 0: nothing but the frame itself and L000.

THE FILTER. With a measured noise power N(f) and a signal power estimated from
the frame itself, S(f) = max(P_obs(f) - N(f), 0), the Wiener gain
G = S/(S+N) is the provably optimal LINEAR estimator for that noise. P_obs is
the Hann-windowed periodogram smoothed in frequency (sigma 3 bins, ~200 dof).
The filter is applied by reflect-padding 48 px, multiplying in the Fourier
domain and cropping back, so the frame's non-periodicity does not wrap.

THE ARBITER IS n2n.py's, UNMODIFIED. `n2n.evaluate_triplet` is called with a
duck-typed module that applies this filter instead of the CNN, so the crops,
the tone maps, the bands, the blur control and the noise algebra are literally
the same code; the numbers line up with n2n's table by construction. There is
no training, so there are no folds: every one of the 19 triplets is scored, and
each is as unseen as the next.

MEASURED RESULT (2026-08, data/master/384kHzStereo.wav, all 19 triplets, one
run, nothing tuned on the arbiter). The n2n column is that module's own
published table on the identical arbiter; the id / blur / avg rows reproduce
n2n.py's to five decimals, which is the check that the two routes really are
being scored by the same code.

  held-out full-band MSE predicting the withheld separation, mean over 19
  scenes:                                       CIRCLE-WIENER      n2n
      identity, the raw plane                        0.03410     0.03410
      blur sigma = 1 (n2n's a-priori control)        0.03928     0.03928
      the denoiser, single plane                     0.03283     0.03217
        beats identity                                 19/19       19/19
        beats the sigma = 1 blur                       17/19       19/19
      2-plane mean (the fuse.py estimator)           0.02546     0.02546
      denoise then mean                              0.02515     0.02524
        beats the plain mean                           17/19       14/19
  low band 0.02-0.35 cyc/px, where the picture lives:
      identity 0.02967   circle-wiener 0.02929   blur s=1 0.03819
      beats identity                                   17/19       15/19
      worst scene                              img 12, +1.10%   img 12, +6.2%
  high band 0.40-0.71 cyc/px (chroma-free), noise algebra:
      per-plane noise N                             1.184e-3    1.18e-3
      residual after the denoiser                   4.210e-4    2.40e-4
      gain, mean of per-scene dB                      +4.97      +7.73
      gain, dB of the means                           +4.49      +6.92
      blur sigma = 1 control                          +7.48 / +5.60 (see below)
  chroma retention 0.99 (0.92-1.07); n2n 0.93 (0.78-1.16).

  Note on the dB convention, because the two numbers differ by 1.9 dB and the
  comparison turns on it: n2n.py's docstring quotes the network as +7.73 dB
  (the mean of the per-scene dB, which is what its own code prints) and the
  blur control as +5.60 dB (the dB of the mean residuals). Computed the same
  way, the sigma = 1 blur reaches +7.48 dB mean-of-dB against the network's
  +7.73 -- i.e. on that statistic the blur is nearly level with the network,
  which is exactly n2n's own stated point that hi-band dB is cheap. Both
  conventions are given above for this route so nothing rests on the choice.

  THE MANDATORY CONTROL, AT MATCHED STRENGTH, AND IT IS THE MAIN NEGATIVE.
  The full Gaussian sweep on the same 19 scenes (mean over scenes):
      sigma   full band   low band   hi residual   low beats identity
      0.30     0.03400     0.02961     1.141e-3        19/19
      0.35     0.03370     0.02943     1.009e-3        19/19
      0.40     0.03323     0.02919     7.920e-4        19/19
      0.45     0.03283     0.02906     5.694e-4        16/19
      0.50     0.03268     0.02916     4.052e-4        14/19   <- matched
      0.55     0.03278     0.02949     3.098e-4        10/19
      1.00     0.03928     0.03819     3.261e-4         2/19   <- n2n's control
      circle-wiener        0.03283     0.02929     4.210e-4    17/19
  Sigma = 0.50 suppresses the high band as hard as this filter does (4.05e-4
  vs 4.21e-4) and is then not worse in the low band -- it is marginally
  BETTER: full 0.03268 vs 0.03283, low 0.02916 vs 0.02929, and it wins the low
  band against this filter on 10 of 19 scenes. So on this arbiter THE
  CIRCLE-CALIBRATED WIENER FILTER IS INDISTINGUISHABLE FROM A WELL-CHOSEN MILD
  GAUSSIAN BLUR. The measured anisotropy -- the whole reason to go to the
  circle -- buys nothing the arbiter can see.
    Two things survive that, and only two. (a) Every one of those sigmas was
    chosen BY SWEEPING AGAINST THE ARBITER, so they are oracle controls; the
    honest a-priori sigma = 1, which is n2n.py's, lands at 0.03928, far worse
    than doing nothing. The Wiener filter picks its own strength per frame
    from that frame's spectrum and the circle's, with nothing tuned, and lands
    on the oracle blur's optimum. That is a real property -- but of automatic
    gain selection, not of knowing the noise's shape. (b) Noise2Noise beats
    the whole sweep (0.03217 against the oracle's best 0.03268); this filter
    does not.
    A caution about the arbiter itself that this sweep exposes: mild blur is
    REWARDED here -- sigma 0.30-0.40 beats identity on 19/19 scenes in the low
    band. The target is a different colour channel, so some of what a blur
    removes is unshared chroma detail and registration residual, not signal.
    n2n.py's "blur is worse than identity on 17/19" is true at ITS sigma = 1
    and false at sigma <= 0.45. Low-band win counts should be read with that
    in mind, for both routes.

  AGAINST NOISE2NOISE, plainly: the circle route recovers about two thirds of
  the network's full-band gain over the raw plane (0.00127 of 0.00193), less
  than two thirds of its high-band dB, and it does NOT reach the network. It
  is better than the network in three narrower places -- low-band wins 17/19
  vs 15/19, worst-case low-band cost +1.1% vs +6.2%, chroma retention 0.99 vs
  0.93 -- all of which say the same thing: a linear filter with a measured
  noise model is gentler and safer, and weaker.

WHAT THE ROUTE DOES DELIVER, and it is the reason to keep it:

  * IT COVERS ALL 156 FRAMES. Nothing here is fitted to a scene, so the 96
    single-scanned frames get exactly the same treatment as the 20 colour
    ones. The n2n route has training data for 20 images and extrapolates on
    the rest; this one does not extrapolate anywhere. Run on mono frames the
    noise comes out at 0.15-0.88% of frame power with mean Wiener gains
    0.24-0.36 -- ordinary numbers, no degeneracies.
  * THE CIRCLE'S GEOMETRY IS NOT DAMAGED (constraint 4): filtering L000 moves
    the fitted axis ratio 1.0050 -> 1.0049 and the radial rms 0.863 -> 0.847
    px. Both improve slightly; neither degrades.
  * IT IS COMPOSABLE. Filter-then-average beats plain 2-plane averaging on
    17/19 scenes (0.02515 vs 0.02546), slightly ahead of n2n's 14/19 and
    0.02524, so where repeats DO exist the two routes stack rather than
    compete.
  * NO DESATURATION: chroma retention 0.99 mean, 0.92 worst. The full-band
    win is not bought by stripping channel identity.

NEGATIVES, per project policy, none dropped:

  * THE HEADLINE ONE: at matched high-band strength a plain Gaussian equals
    this filter (above). The arbiter cannot see the anisotropy.
  * The low band gets WORSE on 2 of 19 scenes (images 12 and 13, both about
    +1.1%). Small, but "no structure destroyed" is not a claim this supports.
  * One third of the measured residual power in the ring interior sits in the
    (fy < 0.05, fx < 0.09) corner, and it is NOT identifiable as noise: a
    quadratic across traces absorbs a third of it, degree five absorbs half,
    and neither touches the streak plateau or the parity spike at all. The
    template therefore refuses to claim it. Running the filter with the corner
    claimed as noise instead changes the result by 0.00004 in full-band MSE
    (0.03279 vs 0.03283) -- so the ambiguity turns out not to matter here, but
    it would matter for any method that leaned on the low-frequency noise
    power, and it is the place where this route could quietly go wrong.
  * Amplitude is read from each frame's own |fx| > 0.35 band, which is
    noise-only for the CAMERA but not necessarily for the decoder: a frame
    with strong aliasing or sharp black/white edges puts some real energy
    there and gets slightly over-filtered. Measured on L000, where the truth
    is known: the whole-frame read exceeds the ring-interior read by up to
    ~1.5 dB in mid fy. No correction is applied, because any correction would
    have to be fitted to scenes.
  * The filter is GLOBAL and stationary -- one gain per frequency for the
    whole frame. Real pictures are not stationary, and that, not the noise
    model, is the most likely reason it lands level with a mild blur. A local
    (block or guided) Wiener using the same measured N is the obvious next
    step and is not tried here; it would need its strength chosen somehow, and
    the only clean way to choose it is another blind rule, not this arbiter.
  * The circle is a single frame, so the noise SHAPE rests on one realisation;
    the transfer check bounds the disagreement at ~2 dB rms but cannot rule
    out a systematic shared error in the shading model.

PROVENANCE: TIER 0 (the record itself). Inputs are the master WAV, the
calibration frame's known constancy (stated on the record's own cover), and
each frame's own spectrum. The catalog is used only to locate frames. No Earth
photograph, no pretrained network, no reference image, no external corpus. The
colour triplets are used ONLY by the arbiter, to score -- never to build the
filter -- and the arbiter is n2n.py's, called unmodified.

Usage:
    python -m pipeline.calnoise --measure    # circle spectrum + transfer only
    python -m pipeline.calnoise --report     # the whole thing, incl. arbiter
    python -m pipeline.calnoise --report --matched-blur   # + the blur sweep
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from . import catalog as catalog_mod
from . import fuse as fuse_mod
from . import n2n as n2n_mod

REPO = Path(__file__).resolve().parent.parent

#: The calibration frame. The record's cover states image 1 is a circle.
CAL_FRAME = "L000"

#: Ring interior rectangle used for the noise measurement, in the 377x512
#: render. Chosen once, geometrically: the largest axis-aligned rectangle that
#: stays inside 0.95 of the fitted ring radius AND clear of the top rows where
#: the white field clips at 1.0 (the droop starts the trace above white).
RECT = np.s_[86:306, 197:341]

#: Across-trace band with no camera signal: we sample 1.4-1.9x above the 1977
#: camera's Nyquist, so |fx| beyond this is noise. Verified on the circle.
NOISE_FX = 0.35
#: Along-trace band used to read the across-trace shape (avoids the streak
#: corner and the resampler cliff).
SHAPE_FY = (0.05, 0.24)
#: Below this the measured across-trace rise cannot be told from a real
#: illumination gradient on the slide; the template is held flat there.
FX_CAP = 0.09

#: Frequency-domain smoothing of the per-frame periodogram, in bins.
PSD_SMOOTH = 3.0
#: Reflect padding before the Fourier multiply.
PAD = 48


# --------------------------------------------------------------------------
# 1. measure the noise from the circle
# --------------------------------------------------------------------------


def decode_cal(fid: str = CAL_FRAME) -> np.ndarray:
    """The calibration frame on the ordinary chain, float, 377x512."""
    cat = catalog_mod.build()
    return fuse_mod.decode_frame(cat.by_id(fid))


def shading_model(patch: np.ndarray, degree: int = 1) -> np.ndarray:
    """Per-row mean + degree-`degree` polynomial ACROSS traces: what traces share.

    See the module docstring. Nothing is smoothed along the trace, so the
    droop is removed exactly and the noise's slow structure down the trace is
    left entirely alone.
    """
    nx = patch.shape[1]
    x = np.arange(nx) - (nx - 1) / 2.0
    x = x / np.abs(x).max()
    A = np.stack([x ** k for k in range(degree + 1)], 1)
    return (A @ np.linalg.lstsq(A, patch.T, rcond=None)[0]).T


def shading_ladder(patch: np.ndarray, verbose: bool = True) -> dict:
    """Which parts of the "noise" are really unmodelled shading?

    Raising the across-trace polynomial degree can only remove structure that
    is SMOOTH ACROSS TRACES. Run the ladder and watch which regions of the
    spectrum move. The answer decides the one genuinely ambiguous choice in
    this module (see FX_CAP).
    """
    out = {}
    for deg in (1, 2, 3, 5):
        n = patch - shading_model(patch, deg)
        P = periodogram(n)
        ny, nx = n.shape
        fy, fx = np.fft.fftfreq(ny), np.fft.fftfreq(nx)
        corner = (np.abs(fy) < 0.05)[:, None] & (np.abs(fx) < FX_CAP)[None, :]
        mid = ((np.abs(fy) >= 0.05) & (np.abs(fy) < 0.25))[:, None] \
            & (np.abs(fx) > FX_CAP)[None, :]
        par = np.zeros_like(P, dtype=bool)
        par[np.abs(fy) < 0.05, nx // 2] = True
        out[deg] = {"var": float(n.var()), "corner": float(P[corner].sum()),
                    "mid": float(P[mid].sum()), "parity": float(P[par].sum()),
                    "corner_frac": float(P[corner].sum() / P.sum())}
        if verbose:
            print(f"      degree {deg}: residual sd {255 * n.std():.3f} grey   "
                  f"low corner {1e5 * out[deg]['corner']:.3f}e-5 "
                  f"({100 * out[deg]['corner_frac']:.1f}% of it)   "
                  f"streak plateau {1e5 * out[deg]['mid']:.3f}e-5   "
                  f"parity {1e5 * out[deg]['parity']:.4f}e-5")
    return out


def periodogram(a: np.ndarray) -> np.ndarray:
    """Hann-windowed power spectrum, normalised so its sum is the variance."""
    h, w = a.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    F = np.fft.fft2((a - a.mean()) * win)
    return np.abs(F) ** 2 / (a.size ** 2) / (win ** 2).mean()


def _bin_profile(P: np.ndarray, f_sel: np.ndarray, f_axis: np.ndarray,
                 axis: int, nb: int = 24) -> np.ndarray:
    """Mean of P over the selected slab, binned in |f| along `axis`."""
    prof = P[:, f_sel].mean(1) if axis == 0 else P[f_sel, :].mean(0)
    edges = np.linspace(0.0, 0.5, nb + 1)
    idx = np.clip(np.digitize(np.abs(f_axis), edges) - 1, 0, nb - 1)
    tot = np.zeros(nb)
    cnt = np.zeros(nb)
    np.add.at(tot, idx, prof)
    np.add.at(cnt, idx, 1.0)
    return tot / np.maximum(cnt, 1.0)


#: The across-trace shape is read in TWO along-trace regimes, because the
#: circle shows the noise is not separable: the trace-PARITY component (an
#: alternating per-trace offset, +5.4 dB at fx = 0.5) is constant down the
#: trace, so it lives only at low fy, and a single B(fx) averaged over all fy
#: dilutes it to nothing. Split at fy = 0.05 with a smooth blend.
FY_SPLIT = (0.04, 0.07)


@dataclass
class CircleNoise:
    fy_prof: np.ndarray      # along-trace shape, 24 bins of |fy|
    fx_lo: np.ndarray        # across-trace shape for |fy| < FY_SPLIT
    fx_hi: np.ndarray        # across-trace shape for |fy| > FY_SPLIT
    fx_lo_raw: np.ndarray    # the same, without the low-fx cap
    var: float               # residual variance in decode units (0..1 grey)
    sd_grey: float
    lag1_along: float
    lag1_across: float
    sep_db: float            # rms dB of the separable model's residual
    cover: dict              # circle-domain coverage of the fitted model
    n_px: int


def measure_circle(img: np.ndarray | None = None, verbose: bool = True
                   ) -> CircleNoise:
    """The whole tier-0 measurement: noise field, its 2-D spectrum, its shape."""
    if img is None:
        img = decode_cal()
    patch = img[RECT]
    noise = patch - shading_model(patch)
    P = periodogram(noise)
    ny, nx = noise.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)

    ctr = np.linspace(0, 0.5, 25)[:-1] + 0.5 / 48
    nsel = np.abs(fx) > NOISE_FX
    fy_prof = _bin_profile(P, nsel, fy, 0)

    def _bx(sel_y):
        b = _bin_profile(P, sel_y, fx, 1)
        # Normalised on the FLAT part of the shape, 0.09 <= |fx| < 0.35, not on
        # the noise-only band: in the low-fy regime the noise-only band sits on
        # top of the parity spike, and normalising there would push the whole
        # plateau down by 2 dB. The per-frame read is corrected for that
        # instead (see _noise_model).
        return b / b[(ctr >= FX_CAP) & (ctr < NOISE_FX)].mean()

    fx_lo_raw = _bx(np.abs(fy) < FY_SPLIT[0])
    fx_hi = _bx((np.abs(fy) >= SHAPE_FY[0]) & (np.abs(fy) < SHAPE_FY[1]))
    fx_lo = fx_lo_raw.copy()
    fx_lo[ctr < FX_CAP] = 1.0

    # separability: how well does a SINGLE A(fy) x B(fx) reproduce the 2-D
    # spectrum? A chi-square-2 periodogram has an intrinsic 5.57 dB rms in log
    # even for a perfect model, so that floor is subtracted before quoting.
    Ay = np.interp(np.abs(fy), ctr, fy_prof)
    model1 = np.outer(Ay, np.interp(np.abs(fx), ctr, fx_hi))
    sel = (np.abs(fy)[:, None] > 0.02) & (np.abs(fx)[None, :] > 0.02)
    lr = 10 * np.log10(np.maximum(P, 1e-30) / model1)[sel]
    sep = float(max(lr.std() ** 2 - 5.57 ** 2, 0.0) ** 0.5)

    d = np.diff(noise, axis=0)
    dx = np.diff(noise, axis=1)
    cn = CircleNoise(fy_prof=fy_prof, fx_lo=fx_lo, fx_hi=fx_hi,
                     fx_lo_raw=fx_lo_raw,
                     var=float(noise.var()), sd_grey=float(255 * noise.std()),
                     lag1_along=float(1 - d.var() / (2 * noise.var())),
                     lag1_across=float(1 - dx.var() / (2 * noise.var())),
                     sep_db=sep, cover={}, n_px=int(noise.size))
    cn.cover = _coverage(cn, P, fy, fx, model1)
    if verbose:
        print(f"CIRCLE ({CAL_FRAME}, ring interior {noise.shape[0]}x{noise.shape[1]} "
              f"= {cn.n_px} px, truth known constant)")
        print(f"  droop removed by the shading model: "
              f"{255 * np.ptp(shading_model(patch).mean(1)):.1f} grey levels "
              f"top-to-bottom")
        print(f"  residual noise sd {cn.sd_grey:.2f} grey levels   "
              f"lag-1 along trace {cn.lag1_along:+.3f}, across traces "
              f"{cn.lag1_across:+.3f}")
        print(f"  along-trace shape, dB rel plateau: "
              f"fy<0.02 {10 * np.log10(fy_prof[0] / fy_prof[3:11].mean()):+.1f} "
              f"(the slow per-trace level trajectory: the streaks)   fy=0.44 "
              f"{10 * np.log10(fy_prof[21] / fy_prof[3:11].mean()):+.1f} "
              f"(the dot resampler's cliff)")
        print(f"  across-trace shape, dB rel plateau, in two along-trace regimes:")
        print(f"      |fy|<0.04 : fx<0.02 {10 * np.log10(fx_lo_raw[0]):+.1f} "
              f"(capped to {10 * np.log10(fx_lo[0]):+.1f} -- indistinguishable "
              f"from a real slide gradient)   fx=0.49 "
              f"{10 * np.log10(fx_lo[-1]):+.1f} (trace PARITY)")
        print(f"      |fy|>0.05 : fx<0.02 {10 * np.log10(fx_hi[0]):+.1f}   "
              f"fx=0.49 {10 * np.log10(fx_hi[-1]):+.1f}   -- white across traces")
        print(f"  single separable A(fy)B(fx) residual {cn.sep_db:.2f} dB rms "
              f"(above the periodogram's own 5.57 dB floor); the two-regime "
              f"model is what the filter uses")
        print(f"  CIRCLE-DOMAIN COVERAGE (model power / measured power, where "
              f"the truth is known):")
        for k in ("all", "fy<0.05", "fy 0.05-0.25", "fy>0.25", "fx=0.5 parity",
                  "arbiter HI", "arbiter LO"):
            print(f"      {k:16s} 1-regime {cn.cover[k][0]:.3f}   "
                  f"2-regime {cn.cover[k][1]:.3f}")
        print(f"  IS THE LOW CORNER NOISE, OR SHADING WE FAILED TO MODEL? Raise "
              f"the across-trace polynomial degree; it can only absorb what is "
              f"smooth across traces:")
        shading_ladder(patch)
        print(f"      -> a QUADRATIC across traces absorbs a third of the low "
              f"corner and touches nothing else; degree 5 absorbs half. The "
              f"corner is at least partly shading and the split is not "
              f"identifiable, so the template holds |fx| < {FX_CAP} flat "
              f"rather than claim that power as noise.")
    return cn


def _coverage(cn: CircleNoise, P, fy, fx, model1) -> dict:
    """How much of the KNOWN noise does the fitted model account for?

    This is the one place on the record where a noise model can be checked
    against ground truth, so it -- not the arbiter -- is what chose the
    two-regime form over the separable one.
    """
    model2 = _noise_model(cn, P, fy, fx)
    nx = len(fx)
    regions = {
        "all": np.ones_like(P, dtype=bool),
        "fy<0.05": (np.abs(fy) < 0.05)[:, None] & np.ones(nx, bool),
        "fy 0.05-0.25": ((np.abs(fy) >= 0.05) & (np.abs(fy) < 0.25))[:, None] & np.ones(nx, bool),
        "fy>0.25": (np.abs(fy) >= 0.25)[:, None] & np.ones(nx, bool),
    }
    par = np.zeros_like(P, dtype=bool)
    par[:, nx // 2] = True
    regions["fx=0.5 parity"] = par
    rad = np.hypot(np.abs(fy)[:, None], np.abs(fx)[None, :])
    regions["arbiter HI"] = (rad >= n2n_mod.HI_BAND[0]) & (rad < n2n_mod.HI_BAND[1])
    regions["arbiter LO"] = (rad >= n2n_mod.LO_BAND[0]) & (rad < n2n_mod.LO_BAND[1])
    return {k: (float(model1[m].sum() / P[m].sum()),
                float(model2[m].sum() / P[m].sum())) for k, m in regions.items()}


def _noise_model(cn: CircleNoise, P: np.ndarray, fy: np.ndarray,
                 fx: np.ndarray, raw_low_fx: bool = False) -> np.ndarray:
    """N(f) = A_frame(fy) x B_circle(fx | fy regime).

    A_frame is read from the frame's OWN noise-only band (|fx| > 0.35), which
    the circle certifies as noise-only. B_circle is the across-trace shape the
    circle measured -- including the parity spike and, crucially, the fact that
    it stays flat all the way down to low fx. That last fact is the one thing a
    photograph can never supply, because there its own picture lives.
    """
    ctr = np.linspace(0, 0.5, 25)[:-1] + 0.5 / 48
    prof = gaussian_filter1d(_bin_profile(P, np.abs(fx) > NOISE_FX, fy, 0),
                             0.8, mode="nearest")
    lo = cn.fx_lo_raw if raw_low_fx else cn.fx_lo
    Blo = np.interp(np.abs(fx), ctr, lo)
    Bhi = np.interp(np.abs(fx), ctr, cn.fx_hi)
    u = np.clip((FY_SPLIT[1] - np.abs(fy)) / (FY_SPLIT[1] - FY_SPLIT[0]), 0, 1)
    B = u[:, None] * Blo[None, :] + (1 - u)[:, None] * Bhi[None, :]
    # The frame's own read is the MEAN over |fx| > 0.35; the plateau level it
    # implies is that mean divided by B's mean over the same band.
    corr = B[:, np.abs(fx) > NOISE_FX].mean(1)
    Ay = np.interp(np.abs(fy), ctr, prof) / np.maximum(corr, 1e-9)
    return Ay[:, None] * B


# --------------------------------------------------------------------------
# 2. does it transfer?
# --------------------------------------------------------------------------


TRANSFER_FRAMES = ("L001", "L010", "L025", "L039", "L050", "L070", "L077",
                   "R000", "R010", "R025", "R039", "R050", "R070", "R077")


def porch_check(fids=("L000",) + TRANSFER_FRAMES, verbose: bool = True) -> dict:
    """The second, independent transfer check: the content-free back porch.

    Frac 0.020-0.072 of every trace carries no picture by construction --
    decode.py already clamps on it -- so it is a noise-only strip on EVERY
    frame, 512 traces x 42 samples, in the signal domain before any resampling.
    Levels are converted to grey by the frame's own sync anchors (black/white
    sit at porch +/- 0.75 x sync amplitude, decode.measure_levels), so the
    numbers are comparable between frames and between channels despite their
    different exposures.

    Two quantities, chosen to match what actually reaches the picture:
      * the within-porch sample scatter, the fine noise;
      * the per-trace level AFTER the decoder's own 7-tap same-parity median
        clamp, i.e. exactly the streak offset the decoder does NOT remove.
    """
    import math
    from scipy.signal import resample_poly
    from . import decode as decode_mod
    from . import sync as sync_mod
    from . import wav

    info = wav.probe(fuse_mod.MASTER)
    cat = catalog_mod.build()
    span = int(math.ceil((sync_mod.TRACES_PER_FRAME + 6) * sync_mod.NOMINAL_PERIOD))
    rows = []
    for fid in fids:
        fr = cat.by_id(fid)
        x = np.asarray(wav.read(info, fr.channel, fr.seed_sample - fuse_mod.PRE_384,
                                fuse_mod.PRE_384 + span), dtype=np.float64)
        x96 = resample_poly(x, 1, fuse_mod.DECIM)
        tb = sync_mod.recover(x96, period_guess=fuse_mod.NOMINAL_96,
                              n_traces=sync_mod.TRACES_PER_FRAME)
        tau = decode_mod.UNCOUPLE_TAU_384[fid[0]] * tb.period / decode_mod._NOMINAL_PERIOD_384
        x96 = decode_mod.undroop(x96, tau)
        n_tr = min(sync_mod.TRACES_PER_FRAME, len(tb.smoothed))
        starts = np.array([tb.trace_start(i) for i in range(n_tr)])
        p, ok = decode_mod._region(x96, starts, tb.period, decode_mod.PORCH_START,
                                   decode_mod.PORCH_END)
        lv = decode_mod.measure_levels(x96, starts, tb.period)
        g = 255.0 / (1.5 * abs(lv[1]))          # grey levels per raw unit
        q = p[ok][:, 6:]                        # clear of the sync edge
        level = np.median(q, axis=1)
        clamp = level.copy()
        for par in (0, 1):
            s = level[par::2]
            win = np.lib.stride_tricks.sliding_window_view(np.pad(s, 3, mode="edge"), 7)
            clamp[par::2] = np.median(win, axis=1)
        fine = (q - level[:, None]) * g
        # robust, so a dropout trace cannot pose as the noise level
        res = (level - clamp) * g
        rres = 1.4826 * np.median(np.abs(res - np.median(res)))
        rows.append({"fid": fid, "fine_sd": float(fine.std()),
                     "streak_sd": float(res.std()), "streak_rsd": float(rres)})
        if verbose:
            print(f"  {fid}  within-porch scatter {rows[-1]['fine_sd']:5.2f} grey   "
                  f"post-clamp per-trace streak {rows[-1]['streak_rsd']:5.2f} grey "
                  f"(robust; {rows[-1]['streak_sd']:5.2f} with dropouts)")
    f = np.array([r["fine_sd"] for r in rows])
    s = np.array([r["streak_rsd"] for r in rows])
    out = {"rows": rows, "fine_mean": float(f.mean()), "fine_sd": float(f.std()),
           "streak_mean": float(s.mean()), "streak_sd": float(s.std()),
           "fine_range": [float(f.min()), float(f.max())],
           "streak_range": [float(s.min()), float(s.max())]}
    if verbose:
        print(f"  the porch says the same thing on every frame: within-porch "
              f"scatter {out['fine_mean']:.2f} +/- {out['fine_sd']:.2f} grey "
              f"({out['fine_range'][0]:.2f}-{out['fine_range'][1]:.2f}), "
              f"post-clamp streak {out['streak_mean']:.2f} +/- "
              f"{out['streak_sd']:.2f} grey "
              f"({out['streak_range'][0]:.2f}-{out['streak_range'][1]:.2f}) "
              f"-- L000 is an ordinary frame, not a special one.")
    return out


def transfer_check(cn: CircleNoise, fids=TRANSFER_FRAMES, verbose: bool = True
                   ) -> dict:
    """Is the noise a property of the chain, or only of L000?

    For each frame, the shape of the noise-only band (|fx| > 0.35) against |fy|
    is compared with the circle's, each normalised to its own plateau. Shape
    agreement is quoted over fy in 0.01-0.35 -- below the resampler cliff,
    whose depth genuinely varies frame to frame with the dot-clock lock.
    """
    cat = catalog_mod.build()
    ctr = np.linspace(0, 0.5, 25)[:-1] + 0.5 / 48
    ref = cn.fy_prof / cn.fy_prof[3:11].mean()
    band = (ctr > 0.01) & (ctr < 0.35)
    rows = []
    for fid in fids:
        img = fuse_mod.decode_frame(cat.by_id(fid))
        a = (img - img.mean()) / img.std()
        P = periodogram(a)
        fy = np.fft.fftfreq(a.shape[0])
        fx = np.fft.fftfreq(a.shape[1])
        prof = _bin_profile(P, np.abs(fx) > NOISE_FX, fy, 0)
        prof = prof / prof[3:11].mean()
        d = 10 * np.log10(prof[band] / ref[band])
        rows.append({"fid": fid, "rms_db": float(np.sqrt((d ** 2).mean())),
                     "streak_db": float(10 * np.log10(prof[0])),
                     "cliff_db": float(10 * np.log10(prof[21]))})
        if verbose:
            print(f"  {fid}  shape rms vs circle {rows[-1]['rms_db']:5.2f} dB "
                  f"over fy 0.01-0.35   streak boost {rows[-1]['streak_db']:+6.2f} dB "
                  f"(circle {10 * np.log10(ref[0]):+.2f})   cliff "
                  f"{rows[-1]['cliff_db']:+7.2f} dB")
    out = {"rows": rows,
           "rms_db_mean": float(np.mean([r["rms_db"] for r in rows])),
           "rms_db_max": float(np.max([r["rms_db"] for r in rows])),
           "streak_db_min": float(np.min([r["streak_db"] for r in rows])),
           "streak_db_max": float(np.max([r["streak_db"] for r in rows])),
           "circle_streak_db": float(10 * np.log10(ref[0]))}
    if verbose:
        print(f"  SHAPE transfers: rms {out['rms_db_mean']:.2f} dB mean, "
              f"{out['rms_db_max']:.2f} dB worst, over 14 frames on both "
              f"channels and both ends of the record.")
        print(f"  AMPLITUDE does not: the streak boost spans "
              f"{out['streak_db_min']:+.1f} to {out['streak_db_max']:+.1f} dB "
              f"across frames, so the level is read per frame, not from L000.")
    return out


# --------------------------------------------------------------------------
# 3. the Wiener filter
# --------------------------------------------------------------------------


class CircleWiener:
    """Wiener filter with the noise spectrum anchored on the calibration ring.

    Duck-typed to look like n2n.Denoiser so `n2n.evaluate_triplet` can score it
    with no change to that file: it is called on a (1,1,H,W) tensor and returns
    one. There is nothing to train and no state carried between frames -- every
    frame is filtered from its own spectrum plus the circle's shape.
    """

    def __init__(self, cn: CircleNoise, flat_low_fx: bool = True,
                 strength: float = 1.0):
        # `strength` exists so the sensitivity of the result to the noise level
        # can be probed; it is 1.0 in every number this module reports and was
        # never swept against the arbiter. Sweeping it would be tuning on the
        # test, and the whole claim here is that the level is not chosen at all
        # -- it is measured.
        self.cn = cn
        self.raw_low_fx = not flat_low_fx
        self.strength = strength
        self.last: dict = {}

    def gain(self, plane: np.ndarray) -> np.ndarray:
        h, w = plane.shape
        P = periodogram(plane)
        fy = np.fft.fftfreq(h)
        fx = np.fft.fftfreq(w)
        N = _noise_model(self.cn, P, fy, fx, self.raw_low_fx) * self.strength
        Ps = gaussian_filter(P, PSD_SMOOTH, mode="wrap")
        S = np.maximum(Ps - N, 0.0)
        G = S / np.maximum(S + N, 1e-30)
        G[0, 0] = 1.0
        self.last = {"noise_frac": float(N.sum() / max(P.sum(), 1e-30)),
                     "mean_gain": float(G.mean())}
        return G

    def filter(self, plane: np.ndarray) -> np.ndarray:
        a = np.asarray(plane, dtype=np.float64)
        pad = np.pad(a, PAD, mode="reflect")
        G = self.gain(pad)
        out = np.real(np.fft.ifft2(np.fft.fft2(pad) * G))
        return out[PAD:-PAD, PAD:-PAD]

    # -- torch face, so the shared arbiter can call it -------------------
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        a = t[0, 0].detach().cpu().numpy()
        return torch.from_numpy(
            self.filter(a).astype(np.float32))[None, None].to(t.device)


# --------------------------------------------------------------------------
# 4. the shared arbiter
# --------------------------------------------------------------------------


class _Blur:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        a = t[0, 0].detach().cpu().numpy().astype(np.float64)
        return torch.from_numpy(gaussian_filter(a, self.sigma).astype(np.float32)
                                )[None, None].to(t.device)


def matched_blur_control(planes: dict, rows: list, device,
                         sigmas=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7,
                                 0.8, 0.9, 1.0)) -> dict:
    """The mandatory control: a Gaussian at MATCHED high-band strength.

    n2n.py's fixed sigma = 1 control is scored in every run above, but it
    suppresses MORE high band than this filter does, so on its own it does not
    settle the question. Here the sigma is chosen so the blur's high-band
    residual matches the filter's, and then the LOW band -- where the picture
    lives -- is what separates them.
    """
    target = float(np.mean([r["hi_res1"] for r in rows]))
    tab = []
    for s in sigmas:
        acc = [n2n_mod.evaluate_triplet(_Blur(float(s)), planes[r["n"]], device)
               for r in rows]
        for m, r in zip(acc, rows):
            m.update({f"hi_{k}": v for k, v in n2n_mod.algebra(m, "hi").items()})
        tab.append({"sigma": float(s),
                    "hi_res": float(np.mean([m["hi_res1"] for m in acc])),
                    "lo": float(np.mean([m["m_n2n1_lo"] for m in acc])),
                    "full": float(np.mean([m["m_n2n1_full"] for m in acc])),
                    "lo_wins": sum(m["m_n2n1_lo"] < r["m_id_lo"]
                                   for m, r in zip(acc, rows)),
                    "beat_wiener_lo": sum(m["m_n2n1_lo"] < r["m_n2n1_lo"]
                                          for m, r in zip(acc, rows))})
        print(f"  blur sigma {s:.1f}   hi residual {tab[-1]['hi_res']:.3e}   "
              f"low band {tab[-1]['lo']:.5f} (beats identity on "
              f"{tab[-1]['lo_wins']}/{len(rows)}, beats the filter on "
              f"{tab[-1]['beat_wiener_lo']}/{len(rows)})")
    best = min(tab, key=lambda t: abs(t["hi_res"] - target))
    print(f"  MATCHED: sigma {best['sigma']:.1f} suppresses the high band as "
          f"much as the filter ({best['hi_res']:.3e} vs {target:.3e}); there it "
          f"costs the low band {best['lo']:.5f} against the filter's "
          f"{float(np.mean([r['m_n2n1_lo'] for r in rows])):.5f} and identity's "
          f"{float(np.mean([r['m_id_lo'] for r in rows])):.5f}.")
    return {"table": tab, "matched": best, "target_hi": target,
            "wiener_lo": float(np.mean([r["m_n2n1_lo"] for r in rows])),
            "id_lo": float(np.mean([r["m_id_lo"] for r in rows]))}


def run(flat_low_fx: bool = True, matched_blur: bool = False,
        json_out: Path | None = None) -> dict:
    device = torch.device("cpu")
    cn = measure_circle()
    print("\nTRANSFER 1 -- picture domain: the same noise on other frames, or "
          "only on L000?")
    tr = transfer_check(cn)
    print("\nTRANSFER 2 -- signal domain: the content-free back porch")
    pc = porch_check()
    print()

    planes = n2n_mod.load_triplets()
    ns = sorted(planes)
    wien = CircleWiener(cn, flat_low_fx=flat_low_fx)
    print(f"ARBITER -- n2n.evaluate_triplet, unmodified, on all {len(ns)} "
          f"triplets (nothing is trained, so every scene is unseen)")
    rows = []
    for n in ns:
        r = {"n": n}
        r.update(n2n_mod.evaluate_triplet(wien, planes[n], device))
        r.update({f"hi_{k}": v for k, v in n2n_mod.algebra(r, "hi").items()})
        rows.append(r)
        print(f"  img {n:3d}  full: id {r['m_id_full']:.5f} "
              f"blur {r['m_blur_full']:.5f} wiener {r['m_n2n1_full']:.5f} "
              f"avg {r['m_avg_full']:.5f} w+mean {r['m_n2n2_full']:.5f}   "
              f"hi: N {r['hi_N']:.2e} wiener {r['hi_res1']:.2e} "
              f"({r['hi_gain1_db']:+.1f} dB)   chroma kept "
              f"{r['chroma_retention']:.2f}")

    mean = lambda k: float(np.mean([r[k] for r in rows]))
    nrow = len(rows)
    wins1 = sum(r["m_n2n1_full"] < r["m_id_full"] for r in rows)
    wins2 = sum(r["m_n2n2_full"] < r["m_avg_full"] for r in rows)
    wins_lo = sum(r["m_n2n1_lo"] < r["m_id_lo"] for r in rows)
    wins_vs_blur = sum(r["m_n2n1_full"] < r["m_blur_full"] for r in rows)
    blur_lo_worse = sum(r["m_blur_lo"] > r["m_id_lo"] for r in rows)
    g1 = np.array([r["hi_gain1_db"] for r in rows])
    gb = np.array([r["hi_gain_blur_db"] for r in rows])

    print(f"\nSUMMARY -- prediction of withheld separations, {nrow} scenes")
    print(f"  full band  id {mean('m_id_full'):.5f}   "
          f"blur s=1 {mean('m_blur_full'):.5f}   "
          f"wiener {mean('m_n2n1_full'):.5f}  beats identity on {wins1}/{nrow}, "
          f"beats blur on {wins_vs_blur}/{nrow}")
    print(f"             2-plane mean {mean('m_avg_full'):.5f}   "
          f"wiener-then-mean {mean('m_n2n2_full'):.5f}  beats plain mean on "
          f"{wins2}/{nrow}")
    print(f"  low band   wiener beats identity on {wins_lo}/{nrow}   "
          f"(blur is worse than identity on {blur_lo_worse}/{nrow})")
    print(f"  high band  N {mean('hi_N'):.3e}   wiener residual "
          f"{mean('hi_res1'):.3e}   gain {np.nanmean(g1):+.2f} dB "
          f"(median {np.nanmedian(g1):+.2f})")
    print(f"             blur s=1 control {np.nanmean(gb):+.2f} dB")
    print(f"  chroma retention {mean('chroma_retention'):.2f}")
    print(f"\n  n2n.py, same arbiter: id 0.03410  blur 0.03928  n2n1 0.03217  "
          f"avg 0.02546  hi +7.73 dB  low 15/19")

    out = {"circle": {"sd_grey": cn.sd_grey, "lag1_along": cn.lag1_along,
                      "lag1_across": cn.lag1_across, "sep_db": cn.sep_db,
                      "n_px": cn.n_px},
           "transfer": tr, "porch": pc, "rows": rows,
           "summary": {"m_id_full": mean("m_id_full"),
                       "m_blur_full": mean("m_blur_full"),
                       "m_wiener_full": mean("m_n2n1_full"),
                       "m_avg_full": mean("m_avg_full"),
                       "m_wiener_then_mean_full": mean("m_n2n2_full"),
                       "hi_N": mean("hi_N"), "hi_res1": mean("hi_res1"),
                       "hi_gain_db_mean": float(np.nanmean(g1)),
                       "hi_gain_db_median": float(np.nanmedian(g1)),
                       "hi_gain_blur_db_mean": float(np.nanmean(gb)),
                       "wins_vs_id": wins1, "wins_vs_blur": wins_vs_blur,
                       "wins_lo": wins_lo, "wins_pair_vs_avg": wins2,
                       "blur_lo_worse": blur_lo_worse,
                       "chroma_retention": mean("chroma_retention"),
                       "n_scenes": nrow, "flat_low_fx": flat_low_fx}}

    print(f"  low band means: identity {mean('m_id_lo'):.5f}   "
          f"wiener {mean('m_n2n1_lo'):.5f}   blur s=1 {mean('m_blur_lo'):.5f}")
    if matched_blur:
        print("\nMATCHED-STRENGTH BLUR CONTROL (mandatory: high-band "
              "suppression is cheap)")
        out["matched_blur"] = matched_blur_control(planes, rows, device)

    if json_out:
        json_out.write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwrote {json_out}")
    return out


# --------------------------------------------------------------------------


def circle_invariants(cn: CircleNoise | None = None) -> dict:
    """Constraint 4: filtering must not degrade the ring's own geometry."""
    from . import circle as circle_mod
    img = decode_cal()
    cn = cn or measure_circle(img, verbose=False)
    z = (img - img.mean()) / img.std()
    out = CircleWiener(cn).filter(z) * img.std() + img.mean()
    a = circle_mod.fit_ring(img)
    b = circle_mod.fit_ring(out)
    print("CIRCLE INVARIANTS (constraint 4)")
    print(f"  raw       axis ratio {a.axis_ratio:.4f}   radial rms {a.radial_rms:.3f} px")
    print(f"  filtered  axis ratio {b.axis_ratio:.4f}   radial rms {b.radial_rms:.3f} px")
    return {"raw_axis": a.axis_ratio, "raw_rms": a.radial_rms,
            "filt_axis": b.axis_ratio, "filt_rms": b.radial_rms}


def mono_coverage(cn: CircleNoise, fids=("L000", "L002", "L045", "R012",
                                         "R061", "L077")) -> dict:
    """Constraint: the route must work on the 96 single-scan frames too."""
    cat = catalog_mod.build()
    w = CircleWiener(cn)
    print("COVERAGE -- single-scan (mono) frames, which the n2n route cannot reach")
    rows = []
    for fid in fids:
        img = fuse_mod.decode_frame(cat.by_id(fid))
        z = (img - img.mean()) / img.std()
        f = w.filter(z)
        rows.append({"fid": fid, "noise_frac": w.last["noise_frac"],
                     "mean_gain": w.last["mean_gain"],
                     "removed": float((z - f).std())})
        print(f"  {fid}  noise is {100 * w.last['noise_frac']:5.2f}% of the "
              f"frame's power   mean Wiener gain {w.last['mean_gain']:.3f}   "
              f"removed {rows[-1]['removed']:.4f} z-units")
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true",
                    help="circle spectrum and the transfer check only")
    ap.add_argument("--report", action="store_true", help="everything")
    ap.add_argument("--raw-low-fx", action="store_true",
                    help="use the measured low-fx rise instead of holding it flat")
    ap.add_argument("--matched-blur", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.measure:
        cn = measure_circle()
        print("\nTRANSFER 1 -- picture domain, the noise-only across-trace band")
        transfer_check(cn)
        print("\nTRANSFER 2 -- signal domain, the content-free back porch")
        porch_check()
    elif args.report:
        run(flat_low_fx=not args.raw_low_fx, matched_blur=args.matched_blur,
            json_out=args.json)
        print()
        cn = measure_circle(verbose=False)
        circle_invariants(cn)
        print()
        mono_coverage(cn)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
