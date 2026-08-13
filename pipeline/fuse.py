"""Fuse the three separations of a colour triplet: registration, noise, resolution.

Twenty of the 116 images are three consecutive frames that scanned the SAME
scene through three colour filters. Three scans of one scene is three
measurements, so luminance noise can be averaged down; and if the three scans
happened to sample on different sub-dot phases, some resolution could in
principle be unfolded too. This module measures both, on the master, and ships
the part that survived measurement.

Everything below was measured with this file on data/master/384kHzStereo.wav,
decoded through the ordinary pipeline (sync.recover -> decode.decode at 96 kHz,
377 x 512 square-pixel render, per-channel uncouple). 19 triplets; image 8
(the solar spectrum) is excluded from every statistic because its content
genuinely moves with wavelength (~90 px), so it is not a repeat measurement of
one scene.

1. THE OLD REGISTRATION NUMBER WAS MEASURING THE DECODER, NOT THE PICTURE
-------------------------------------------------------------------------
Plain (full-band) phase correlation -- what pipeline/colour.py uses -- is
pulled toward ZERO along-trace shift on this material. The three decodes share
a fixed pattern locked to the output row grid (the 230 dot rows resampled to
377 display rows, plus parity residue), and phase correlation whitens every
frequency bin, so that shared, zero-lag structure outvotes the picture.
Measured on the master:

  * full band, the 1->2 along-trace shift comes out at -0.02, -0.02, -0.01,
    -0.01 px on images 46/89/112/55 -- "the separations are already perfectly
    registered". Band-limited to 0.25 cyc/px the SAME pairs measure +0.32,
    +0.31, +0.24, +0.13 px, and the estimate is then stable from fmax 0.30 all
    the way down to 0.10 (46: 0.31/0.32/0.32/0.32/0.30).
  * the three pairwise estimates of a triplet must close: est(1->2) must equal
    est(3->2) + est(1->3). Full band the closure error reaches 0.75 px
    (mean |err| 0.28 along-trace over 19 triplets); band-limited it is
    <= 0.05 px along-trace on 18 of 19 (0.11 on image 114, the noisiest
    triplet) and <= 0.14 across-trace on 18 of 19 (0.23 on 114).
  * the estimator is not shrinking, it is offset: injecting a known extra shift
    into one separation moves the estimate by exactly that amount at every
    fmax (slope 1.000 +- 0.004 over -1..+1 px, 6 pairs).

So: register in the band where the picture actually lives. 0.25 cyc/px is not
a tuning knob -- it is where the 1977 camera's response ends. The measured
inter-separation coherence along the trace falls 0.87 (f < 0.2) -> 0.60
(0.20-0.25) -> 0.37 (0.25-0.305, the dot Nyquist) -> 0.09 above it.

REFINEMENT: parabolic interpolation of the correlation peak (colour.py) has a
+-0.12 px bias; the upsampled-DFT refinement used here recovers known Fourier
shifts of real decoded frames to < 0.001 px, with 0.02 px scatter at the
frames' own noise level and 0.08 px at 2x that noise.

2. WHAT THE SEPARATIONS ACTUALLY DO (19 triplets, measured, display px)
-----------------------------------------------------------------------
Across-trace (dx, 1 px = 1 trace) is RANDOM: sep1->sep2 mean -0.10 sd 0.87,
sep3->sep2 mean -0.02 sd 0.55, both signs, max 2.26 px (image 60, the Jane
Goodall triplet Barry noticed was misaligned).

Along-trace (dy, 1 px = 0.611 converter dots) is SYSTEMATIC: sep1->sep2
+0.269 +- 0.154 px, sep2->sep3 +0.389 +- 0.182 px, and sep1->sep3 is positive
on 19 of 19 triplets (+0.658 mean; sign test p = 4e-6). The picture creeps one
way along the trace, +0.33 px = +0.20 dots = +2.4 samples at 384 kHz per frame.

That creep is IN THE PICTURE, not in the timebase. Two scene-free instruments
say the recording chain held still between the frames of a triplet:

  * the converter's front-porch dip (geometry.py's bins 3044-3063; nothing but
    converter blanking in it) measured on the 512-trace median fold moves
    -0.04 +- 0.04 samples per frame at 96 kHz -- the picture moved +0.61;
  * the converter's dot-clock phase relative to the trace start
    (dotclock.track_phase, the same instrument decode.py samples on) moves
    +0.03 +- 0.03 dots per frame -- the picture moved +0.20.

Both exclude a timing origin at ~6 sigma, so the slide/camera image itself
drifted between the three exposures: a mechanical cause in the 1977 chain.

The mapping is a translation, not a warp: 3x3 tile registration reproduces the
global fit (tile mean within 0.1 px of it) with a tile-to-tile sd of 0.04-0.3
px on 15 of 19 triplets; the four exceptions (23, 34, 35, 114) have individual
tiles with no usable texture, not a warp -- their other tiles agree.

3. FUSION FOR NOISE: THE ALGORITHM IS IDEAL, THE MATERIAL IS NOT A REPEAT
-------------------------------------------------------------------------
Fusion = register, then average the three separations. On INDEPENDENT noise it
is exactly ideal: inject white noise at 30% of picture rms into each separation
and the mean suppresses it by 4.77 dB (mean over 20 triplets; sqrt(3) = 4.77),
i.e. registration and the Fourier shift cost nothing.

On the pictures' OWN noise the honest number is smaller, and the reason is
physical, not algorithmic: the three scans are not repeat measurements of one
signal. Each filter renders the scene differently, so part of what the
separations do not share is real colour information, and averaging removes it
on purpose. Measured decomposition (per 2-D frequency cell, over 19 triplets):

  * of each separation's picture energy, 7.3% on average is NOT shared with
    the other two below 0.30 cyc/px (real chroma + a little noise), and 0.28%
    above it (essentially all noise);
  * inter-separation correlation of the residual falls to 0.075 / 0.055 /
    0.047 / 0.026 in radial bands 0.48-0.52 / 0.52-0.56 / 0.60-0.65 /
    0.65-0.71 cyc/px, i.e. the noise IS independent between scans; the
    implied fusion gain in those bands is 4.2-4.6 dB;
  * extrapolating that noise density over the whole plane, noise is 5.9% of
    picture rms on average (3.96% image 45 .. 12.9% image 46), and fusion
    takes it to 3.4% (2.3% .. 7.5%).
  * a direct flat-patch measurement (60 flattest 12x12 patches per triplet,
    local plane removed) gives 3.2 dB mean, 0.5-4.7 dB spread; it reads low on
    busy pictures because "flat" patches of a forest floor still carry texture
    that differs between separations.

So the deliverable claim is: 4.8 dB on the independent noise, which is 5.9% ->
3.4% of picture rms; NOT a 4.8 dB improvement of the picture.

The visible part is the vertical streaking: it is per-trace, hence independent
between the three scans, and it is what you see go away in the fused luminance.
Measured, the frame-specific column-constant (streak) energy is 0.363% of
picture power, i.e. 6.0% of picture rms, and the 3-scan mean takes it to
0.121% / 3.5% -- the largest single visible gain fusion offers here.

4. SUPER-RESOLUTION: NO. MEASURED, NOT ASSUMED
-----------------------------------------------
The phase diversity IS there -- the premise that the scans sample at nearly the
same sub-dot phase is wrong. Sub-dot phase distance between separations
(|dy| in dots, mod 1, folded to 0..0.5) over 57 pairs: mean 0.256, median
0.250, max 0.493. The report's per-triplet column is the mean of a triplet's
three pairwise distances, which for three points on a circle cannot exceed
1/3 and reaches it only for perfect 0, 1/3, 2/3 staggering: measured mean
0.254, and 8 of 19 triplets sit at the 1/3 optimum. Only 13 and 78 are
degenerate (0.065, 0.086).

What is missing is anything to unfold. Fit, at each along-trace frequency in
the top of the sampled band (0.20-0.305 cyc/px), the two-component model
    S_i = T + A exp(-i 2 pi delta_i)
(three separations, two unknowns, one residual degree of freedom for noise),
debias |A|^2 by the noise the residual measures, and the aliased energy comes
out at 6.3% +- 5.6% (sem) of the band's power -- not distinguishable from
zero, and no better than the same fit run with ANOTHER triplet's phases
(20.9% vs 27.9% of noise power, a wrong model), while the same estimator
reports 85% of noise power in a mid band where aliasing is impossible (that is
its leakage floor). Injecting a known alias shows the estimator would report
0.85 x the true fraction, so the measurement bounds the real aliased energy at
< 0.2 of the near-Nyquist band power (95%), and that band is itself 0.37% of
picture power, where measured coherence is already only 0.37 -- noise there
already exceeds signal.

And unfolding is not free. Solving for T and A instead of averaging multiplies
the luminance noise variance by cT = (M^H M)^-1_00 x 3 relative to the plain
mean: +2.6 dB on average, +0.04 dB at best, +11.9 dB on the two triplets whose
phases nearly coincide (13, 78). Super-resolution would spend the entire
sqrt(3) noise gain, and more, to recover a component that measures zero.

NOT SHIPPED. The 3-frame mean is the right estimator for this material.

5. COLOUR: FUSION HELPS, AND THE MEASUREMENT IS INDEPENDENT OF THE FITTER
--------------------------------------------------------------------------
Residual misregistration measured in IMAGE space, by regressing the colour
difference on the luminance gradient ((R-G) ~ a dL/dy + b dL/dx, so (a,b) is
the leftover shift in px) -- an estimator that has nothing to do with the phase
correlation that did the registering:

    unregistered                       R 0.82  B 0.63 px
    colour.py (full band + spline)     R 0.40  B 0.23 px
    this module (band + Fourier shift) R 0.09  B 0.08 px

Edge chroma rms (10% highest-gradient pixels) 0.672 -> 0.542 -> 0.518, and the
same change RAISES luminance high-frequency energy (0.15-0.30 cyc/px) by
14.5%, so it is not the usual "less fringing because blurrier": ndimage.shift
order=1, which colour.py uses, is a triangle filter that costs that 14.5%.

To spend the luminance noise gain on the colour image the planes must be
rebuilt as luminance + chroma, out_i = L + g(f) (s_i - L), and g must be
measured, not chosen: the incoherent (chroma) density stays 39x, 21x, 12x,
6.8x, 3.9x the noise density in bands 0.14-0.18 ... 0.30-0.35 cyc/px, i.e.
colour detail on this material extends nearly as far as luminance detail, and
a conventional narrow chroma low-pass would throw real colour away. The Wiener
gain g = 1 - N/I built from those measurements averages 0.59-0.62 over the
plane and buys ~2.0 dB of plane noise while leaving chroma untouched wherever
chroma dominates.

Usage:
    python -m pipeline.fuse --report            # the whole measurement suite
    python -m pipeline.fuse --report --quick    # registration + noise only
    python -m pipeline.fuse --images 60,89 --out /tmp/f   # write fused PNGs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import catalog as catalog_mod
from . import decode as decode_mod
from . import sync as sync_mod
from . import wav

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "data" / "master" / "384kHzStereo.wav"
DECIM = 4
NOMINAL_96 = sync_mod.NOMINAL_PERIOD / DECIM
PRE_384 = int(round(1.5 * sync_mod.NOMINAL_PERIOD))

#: Registration band. The camera's response ends here: measured
#: inter-separation coherence along the trace is 0.87 below 0.2 cyc/px and
#: 0.09 above the dot Nyquist (0.305). Above ~0.3 the correlation is dominated
#: by the decoder's shared row-grid pattern, which pulls the estimate to zero.
REG_FMAX = 0.25

#: Dots inside the picture gate (0.8775 of a line x 262.5 dots); one display
#: row of the 377-row render is DOTS_PER_GATE / 377 = 0.611 dots.
DOTS_PER_GATE = 0.8775 * 262.5


# --------------------------------------------------------------------------
# decoding one triplet from the master
# --------------------------------------------------------------------------


def decode_frame(frame: catalog_mod.Frame, height: int = decode_mod.SQUARE_ROWS,
                 master: Path = MASTER) -> np.ndarray:
    """Decode one frame from the master exactly as pipeline/build.py does."""
    info = wav.probe(master)
    span = int(math.ceil((sync_mod.TRACES_PER_FRAME + 6) * sync_mod.NOMINAL_PERIOD))
    start = frame.seed_sample - PRE_384
    if start < 0:
        raise ValueError(f"{frame.id}: seed too close to file start")
    x = np.asarray(wav.read(info, frame.channel, start, PRE_384 + span), dtype=np.float64)
    x96 = resample_poly(x, 1, DECIM)
    tb = sync_mod.recover(x96, period_guess=NOMINAL_96, n_traces=sync_mod.TRACES_PER_FRAME)
    cfg = decode_mod.Settings(channel=frame.id[0], height=height)
    return decode_mod.decode(np.asarray(x96, dtype=np.float32), cfg, tb).image.astype(np.float64)


def decode_triplet(cat: catalog_mod.Catalog, image: catalog_mod.Image,
                   height: int = decode_mod.SQUARE_ROWS) -> list[np.ndarray]:
    """The three separations of one colour image, in record order (r, g, b)."""
    if not image.color:
        raise ValueError(f"image {image.n} is not a colour image")
    return [decode_frame(cat.by_id(fid), height) for fid in image.frames]


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def _window(h: int, w: int) -> np.ndarray:
    return np.hanning(h)[:, None] * np.hanning(w)[None, :]


def _z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-12)


def fourier_shift(a: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Band-limited translation. Not ndimage.shift(order=1): that triangle
    kernel costs 14.5% of the 0.15-0.30 cyc/px luminance energy (measured)."""
    h, w = a.shape
    F = np.fft.rfft2(a)
    F *= np.exp(-2j * np.pi * (np.fft.fftfreq(h)[:, None] * dy
                               + np.fft.rfftfreq(w)[None, :] * dx))
    return np.fft.irfft2(F, a.shape)


def phase_offset(a: np.ndarray, b: np.ndarray, *, fmax: float = REG_FMAX,
                 upsample: int = 100) -> tuple[float, float, float]:
    """Sub-pixel (dy, dx) that lands `b` on `a`, plus the peak height.

    Phase correlation restricted to |f| <= fmax (see the module docstring: the
    unrestricted version measures the decoder's shared row grid, not the
    picture) and refined by a locally upsampled inverse DFT rather than a
    parabola (which is biased by up to 0.12 px).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    h, w = a.shape
    win = _window(h, w)
    A = np.fft.fft2(_z(a) * win)
    B = np.fft.fft2(_z(b) * win)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    if fmax > 0:
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        R *= np.hypot(fy, fx) <= fmax
    c = np.real(np.fft.ifft2(R))
    ky, kx = np.unravel_index(int(np.argmax(c)), c.shape)
    peak = float(c[ky, kx])

    span = 1.5
    n = int(2 * span * upsample) + 1
    oy = ky + np.linspace(-span, span, n)
    ox = kx + np.linspace(-span, span, n)
    Ey = np.exp(2j * np.pi * np.outer(oy, np.fft.fftfreq(h)))
    Ex = np.exp(2j * np.pi * np.outer(ox, np.fft.fftfreq(w)))
    cu = np.real(Ey @ R @ Ex.T)
    iy, ix = np.unravel_index(int(np.argmax(cu)), cu.shape)
    dy, dx = float(oy[iy]), float(ox[ix])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dy, dx, peak


def register(seps: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict]]:
    """Align the outer separations onto the middle one (green).

    Returns the registered planes in record order and the measured offsets.
    Offsets are per triplet: the across-trace part is random, the along-trace
    part is a systematic creep (module docstring section 2).
    """
    if len(seps) != 3:
        raise ValueError(f"need 3 separations, got {len(seps)}")
    out = [None, seps[1], None]
    offsets = []
    for i in (0, 2):
        dy, dx, peak = phase_offset(seps[1], seps[i])
        out[i] = fourier_shift(seps[i], dy, dx)
        offsets.append({"index": i, "dy": dy, "dx": dx, "peak": peak,
                        "dy_dots": dy * DOTS_PER_GATE / seps[1].shape[0]})
    return out, offsets


def closure(seps: list[np.ndarray]) -> tuple[float, float]:
    """Triangle-closure error of the three pairwise estimates, in px.

    est(1->2) - est(3->2) - est(1->3) must be zero for any consistent
    estimator; it is 0.01-0.05 px band-limited and up to 0.62 px full band.
    Cheap self-check on a triplet whose registration is in doubt.
    """
    d12 = phase_offset(seps[1], seps[0])
    d32 = phase_offset(seps[1], seps[2])
    d13 = phase_offset(seps[2], seps[0])
    return (d12[0] - d32[0] - d13[0], d12[1] - d32[1] - d13[1])


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def fuse_luminance(registered: list[np.ndarray]) -> np.ndarray:
    """The 3-scan mean, in each separation's own z-scale.

    Each separation is z-scored first because the 1977 chain gave the three
    exposures different gain and offset (pipeline/colour.py's docstring: whole
    frame means are not even monotone in scene hue). Averaging then estimates
    the common (luminance) component with the noise down by sqrt(3): measured
    4.77 dB on injected independent noise, the ideal.
    """
    return np.mean([_z(r) for r in registered], axis=0)


def chroma_gain(registered: list[np.ndarray], noise_from: float = 0.45
                ) -> tuple[np.ndarray, float]:
    """Wiener gain for the chroma channels, measured from the triplet itself.

    The three separations differ by real colour AND by noise. At each spatial
    frequency the incoherent density I(f) is what they do not share; the noise
    density N is that same quantity where no scene can survive (radial
    f > 0.45 cyc/px, camera MTF < 0.06). The Wiener gain 1 - N/I keeps colour
    where colour dominates and kills the rest. Measured on this master, colour
    detail runs out to 0.35-0.40 cyc/px, so a fixed narrow chroma low-pass
    would discard real colour: this is measured per image instead.
    """
    F = [np.fft.fft2(_z(r)) for r in registered]
    L = np.mean(F, axis=0)
    inc = 1.5 * np.mean([np.abs(f - L) ** 2 for f in F], axis=0)  # 3/2 debias
    h, w = registered[0].shape
    rad = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    edges = np.linspace(0.0, 0.72, 40)
    idx = np.clip(np.digitize(rad, edges) - 1, 0, len(edges) - 2)
    cnt = np.maximum(np.bincount(idx.ravel(), minlength=len(edges) - 1), 1)
    dens = np.bincount(idx.ravel(), inc.ravel(), len(edges) - 1) / cnt
    band = (edges[:-1] >= noise_from) & (edges[:-1] < 0.60)
    n_lev = float(dens[band].mean())
    g = np.clip(1.0 - n_lev / np.maximum(dens, 1e-30), 0.0, 1.0)[idx]
    return g, n_lev


def fuse_colour(seps: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict], np.ndarray]:
    """Register, fuse luminance, put the measured chroma back.

    out_i = L + g(f) (s_i - L): the fused luminance carries the detail of all
    three scans, the chroma carries whatever of the separation's own signal is
    above its noise. Returns (planes r,g,b), the offsets, and the gain map.
    """
    reg, offsets = register(seps)
    zs = [_z(r) for r in reg]
    L = np.mean(zs, axis=0)
    g, _ = chroma_gain(reg)
    FL = np.fft.fft2(L)
    out = [np.real(np.fft.ifft2(FL + g * (np.fft.fft2(s) - FL))) for s in zs]
    return out, offsets, g


# --------------------------------------------------------------------------
# measurement suite (the numbers in the docstring come from here)
# --------------------------------------------------------------------------


def _cells(shape, nb=16):
    h, w = shape
    fy = np.abs(np.fft.fftfreq(h))[:, None] * np.ones((1, w))
    fx = np.ones((h, 1)) * np.abs(np.fft.fftfreq(w))[None, :]
    iy = np.clip((fy / 0.5 * nb).astype(int), 0, nb - 1)
    ix = np.clip((fx / 0.5 * nb).astype(int), 0, nb - 1)
    return iy * nb + ix, np.hypot(fy, fx)


def noise_report(registered: list[np.ndarray], seed: int = 11) -> dict:
    """Noise accounting for one triplet (see docstring section 3)."""
    rng = np.random.default_rng(seed)
    zs = [_z(r) for r in registered]
    h, w = zs[0].shape
    idx, rad = _cells((h, w))
    F = [np.fft.fft2(z * _window(h, w)) for z in zs]
    P = np.mean([np.abs(f) ** 2 for f in F], axis=0)
    X = np.mean([np.real(F[i] * np.conj(F[j])) for i, j in ((0, 1), (0, 2), (1, 2))], axis=0)
    nb = idx.max() + 1
    rho = np.clip(np.bincount(idx.ravel(), X.ravel(), nb)
                  / np.maximum(np.bincount(idx.ravel(), P.ravel(), nb), 1e-30), 0.0, 1.0)
    inc = P - rho[idx] * P
    hi = rad >= 0.30
    dens = inc[hi].sum() / max(hi.sum(), 1)
    noise = dens * P.size
    total = P.sum()
    nsr = float(np.sqrt(noise / max(total - noise, 1e-30)))

    lvl = 0.3
    fused = np.mean(zs, axis=0)
    fused_n = np.mean([z + rng.normal(0, lvl, z.shape) for z in zs], axis=0)
    injected_db = float(20 * np.log10(lvl / (fused_n - fused).std()))
    return {
        "nsr_single_pct": 100 * nsr,
        "nsr_fused_pct": 100 * nsr / math.sqrt(3.0),
        "incoherent_lo_pct": float(100 * inc[~hi].sum() / total),
        "incoherent_hi_pct": float(100 * inc[hi].sum() / total),
        "injected_noise_gain_db": injected_db,
    }


def sr_report(registered: list[np.ndarray], offsets: list[dict]) -> dict:
    """Aliased-energy fit and the cost of unfolding (docstring section 4)."""
    h, w = registered[0].shape
    dots = np.array([offsets[0]["dy_dots"], 0.0, offsets[1]["dy_dots"]])
    F = np.stack([np.fft.fft2(_z(r) * _window(h, w)) for r in registered])
    fy = np.abs(np.fft.fftfreq(h))[:, None] * np.ones((1, w))
    fx = np.ones((h, 1)) * np.abs(np.fft.fftfreq(w))[None, :]
    dot_ny = 0.5 * DOTS_PER_GATE / h
    m = (fy >= 0.20) & (fy < dot_ny) & (fx < 0.25)
    S = np.stack([f[m] for f in F])
    M = np.stack([np.ones(3), np.exp(-2j * np.pi * dots)], 1)
    coef = np.linalg.pinv(M) @ S
    resid = S - M @ coef
    N = float(np.mean(np.sum(np.abs(resid) ** 2, axis=0)))  # 1 dof
    inv = np.linalg.inv(M.conj().T @ M)
    alias = float(np.mean(np.abs(coef[1]) ** 2) - N * float(np.real(inv[1, 1])))
    power = float(np.mean(np.mean(np.abs(S) ** 2, axis=0)))
    phase = np.abs(dots) % 1.0
    phase = np.minimum(phase, 1 - phase)
    return {
        "phase_dots": [float(dots[0]), float(dots[2])],
        "phase_distance": float(np.mean([phase[0], phase[2],
                                         min(abs(dots[0] - dots[2]) % 1.0,
                                             1 - abs(dots[0] - dots[2]) % 1.0)])),
        "alias_over_power": alias / power,
        "unfold_penalty_db": float(10 * np.log10(float(np.real(inv[0, 0])) * 3.0)),
    }


def chroma_noise_gain(registered: list[np.ndarray]) -> tuple[float, float]:
    """Noise reduction the luma/chroma rebuild gives each output RGB plane.

    out_i = L + g (s_i - L) with independent per-scan noise N leaves
    N/3 + (2/3) g(f)^2 N, so the gain is -10 log10 <1/3 + 2/3 g^2> over the
    plane. Measured 1.99 dB mean (1.70 .. 2.52) over the 19 triplets, and an
    end-to-end injection of independent noise through the same rebuild agrees
    to 0.02 dB.
    """
    g, _ = chroma_gain(registered)
    return float(-10 * np.log10(np.mean(1 / 3 + 2 / 3 * g ** 2))), float(g.mean())


def fringe_report(planes: list[np.ndarray]) -> dict:
    """Residual misregistration measured in image space (docstring section 5)."""
    L = np.mean([_z(p) for p in planes], axis=0)
    gy, gx = np.gradient(L)
    sl = np.s_[20:-20, 20:-20]
    A = np.stack([gy[sl].ravel(), gx[sl].ravel(), np.ones(gy[sl].size)], 1)
    out = {}
    for i, name in ((0, "r"), (2, "b")):
        d = (_z(planes[i]) - _z(planes[1]))[sl].ravel()
        c, *_ = np.linalg.lstsq(A, d, rcond=None)
        out[f"residual_shift_{name}"] = float(math.hypot(c[0], c[1]))
    g = np.hypot(gy, gx)[sl]
    ch = (_z(planes[0]) - _z(planes[2]))[sl]
    ch = ch - ch.mean()
    out["edge_chroma_rms"] = float(np.sqrt(np.mean(ch[g >= np.percentile(g, 90)] ** 2)))
    db, mean_g = chroma_noise_gain(planes)
    out["plane_noise_gain_db"] = db
    out["mean_chroma_gain"] = mean_g
    return out


def report(images: list[int] | None = None, quick: bool = False) -> dict:
    """Run the whole measurement suite on the master and print the tables."""
    cat = catalog_mod.build()
    targets = [im for im in cat.images if im.color
               and (images is None or im.n in images)]
    rows = []
    print(f"{'img':>4} {'frames':11} {'1->2 dy,dx':>16} {'3->2 dy,dx':>16} "
          f"{'dy dots':>14} {'closure':>14}")
    cache = {}
    for im in targets:
        seps = decode_triplet(cat, im)
        cache[im.n] = seps
        reg, off = register(seps)
        cl = closure(seps)
        rows.append({"n": im.n, "title": im.title, "frames": list(im.frames),
                     "offsets": off, "closure": cl})
        print(f"{im.n:4d} {im.frames[0]}-{im.frames[2]:5} "
              f"{off[0]['dy']:+7.3f},{off[0]['dx']:+7.3f} "
              f"{off[1]['dy']:+7.3f},{off[1]['dx']:+7.3f} "
              f"{off[0]['dy_dots']:+6.3f},{off[1]['dy_dots']:+6.3f} "
              f"{cl[0]:+6.3f},{cl[1]:+6.3f}")

    sel = [r for r in rows if r["n"] != 8]
    if sel:
        a = np.array([[r["offsets"][0]["dy"], r["offsets"][0]["dx"],
                       r["offsets"][1]["dy"], r["offsets"][1]["dx"]] for r in sel])
        print(f"\nn={len(sel)} (image 8, the spectrum, excluded: its content moves)")
        print(f"  1->2  dy {a[:,0].mean():+.3f} +- {a[:,0].std(ddof=1):.3f}   "
              f"dx {a[:,1].mean():+.3f} +- {a[:,1].std(ddof=1):.3f}")
        print(f"  3->2  dy {a[:,2].mean():+.3f} +- {a[:,2].std(ddof=1):.3f}   "
              f"dx {a[:,3].mean():+.3f} +- {a[:,3].std(ddof=1):.3f}")
        print(f"  along-trace creep per frame: "
              f"{0.5*(a[:,0]-a[:,2]).mean():+.3f} px = "
              f"{0.5*(a[:,0]-a[:,2]).mean()*DOTS_PER_GATE/377:+.3f} dots "
              f"(positive on {int(((a[:,0]-a[:,2])>0).sum())}/{len(sel)} triplets)")

    if quick:
        return {"registration": rows}

    print(f"\n{'img':>4} {'noise/sig %':>11} {'fused %':>8} {'lo-f incoh %':>12} "
          f"{'hi-f incoh %':>12} {'injected dB':>11} | {'phase dots':>10} "
          f"{'alias/pwr':>9} {'unfold dB':>9} | {'res R':>6} {'res B':>6} {'edge chr':>8} {'plane dB':>8}")
    for r in rows:
        seps = cache[r["n"]]
        reg, off = register(seps)
        nr = noise_report(reg)
        sr = sr_report(reg, off)
        fr = fringe_report(reg)
        r.update(noise=nr, sr=sr, fringe=fr)
        print(f"{r['n']:4d} {nr['nsr_single_pct']:11.2f} {nr['nsr_fused_pct']:8.2f} "
              f"{nr['incoherent_lo_pct']:12.2f} {nr['incoherent_hi_pct']:12.3f} "
              f"{nr['injected_noise_gain_db']:11.2f} | {sr['phase_distance']:10.3f} "
              f"{sr['alias_over_power']:9.3f} {sr['unfold_penalty_db']:9.2f} | "
              f"{fr['residual_shift_r']:6.3f} {fr['residual_shift_b']:6.3f} "
              f"{fr['edge_chroma_rms']:8.4f} {fr['plane_noise_gain_db']:8.2f}")
    if sel:
        def mean(path):
            return float(np.mean([_dig(r, path) for r in rows if r["n"] != 8]))
        print(f"MEAN {mean('noise.nsr_single_pct'):11.2f} {mean('noise.nsr_fused_pct'):8.2f} "
              f"{mean('noise.incoherent_lo_pct'):12.2f} "
              f"{mean('noise.incoherent_hi_pct'):12.3f} "
              f"{mean('noise.injected_noise_gain_db'):11.2f} | "
              f"{mean('sr.phase_distance'):10.3f} {mean('sr.alias_over_power'):9.3f} "
              f"{mean('sr.unfold_penalty_db'):9.2f} | "
              f"{mean('fringe.residual_shift_r'):6.3f} "
              f"{mean('fringe.residual_shift_b'):6.3f} "
              f"{mean('fringe.edge_chroma_rms'):8.4f} "
              f"{mean('fringe.plane_noise_gain_db'):8.2f}")
        print("\nideal independent-noise gain sqrt(3) = 4.77 dB")
    return {"registration": rows}


def _dig(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="", help="comma-separated canonical numbers")
    ap.add_argument("--report", action="store_true", help="run the measurement suite")
    ap.add_argument("--quick", action="store_true", help="registration only")
    ap.add_argument("--out", type=Path, default=None, help="write fused PNGs here")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    want = [int(s) for s in args.images.split(",") if s.strip()] or None

    if args.report or args.json:
        res = report(want, quick=args.quick)
        if args.json:
            args.json.write_text(json.dumps(res, indent=1, default=float))
    if args.out:
        from PIL import Image
        cat = catalog_mod.build()
        args.out.mkdir(parents=True, exist_ok=True)
        for im in cat.images:
            if not im.color or (want is not None and im.n not in want):
                continue
            planes, offsets, _ = fuse_colour(decode_triplet(cat, im))
            from . import build as build_mod  # deferred: heavy imports
            turns = cat.by_id(im.frames[0]).orientation
            planes = [build_mod.orient(p, turns) for p in planes]
            arr = np.stack([np.clip((p - np.percentile(p, 0.5))
                                    / (np.percentile(p, 99.5) - np.percentile(p, 0.5)),
                                    0, 1) for p in planes], -1)
            arr = (arr ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
            Image.fromarray(arr, "RGB").save(args.out / f"image{im.n:03d}.png")
            print(f"image {im.n:3d} -> {args.out}/image{im.n:03d}.png")
    if not (args.report or args.json or args.out):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
