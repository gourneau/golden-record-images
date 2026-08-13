"""Noise2Noise across the twenty colour triplets: the record's own training pairs.

Twenty of the 116 images were scanned three times through red, green and blue
filters, so the record carries twenty sets of THREE INDEPENDENT NOISY
OBSERVATIONS OF THE SAME SCENE GEOMETRY. Noise2Noise (Lehtinen et al. 2018)
proves that a network trained to map one noisy observation to another
independent noisy observation of the same scene converges to the same estimator
as training on clean targets: the network cannot predict the target's noise
from the input, so the noise contributes only an irreducible constant to the
loss and what the network learns is the scene. The literature's stated obstacle
is that independent noisy pairs are hard to obtain; here the artifact itself
provides sixty frames of them. The independence premise is MEASURED, not
assumed: fuse.py's residual inter-separation correlation is 0.026-0.075 in the
noise-dominated bands (0.48-0.71 cyc/px), i.e. the per-scan noise (per-trace
streaks, tape noise) really is independent between the three scans.

PROVENANCE: TIER 1 (universal priors). The training corpus is the record
itself -- the sixty colour-separation frames -- and nothing else. No reference
image, no Earth photograph, no external corpus enters at any point; an alien
holding this record could run this file. The only catalog fact used is the
triplet GROUPING (which three frames repeat one scene), which is tier 0
(colour.py: detectable by correlation alone). Which separation is "red" is
never used -- the method is symmetric in the three planes.

REGISTRATION FIRST, or the network is trained to blur a misalignment. The
separations are misregistered by up to 2.26 px across-trace (image 60) and
creep systematically +0.33 px/frame along-trace (fuse.py section 2, measured on
19 triplets). This module reuses fuse.register: band-limited phase correlation
(<= 0.25 cyc/px, where the 1977 camera's response lives) with upsampled-DFT
refinement, validated to < 0.001 px on known shifts and 0.08-0.09 px residual
by an independent image-space estimator. Registration happens ONCE, before
anything is trained, and the same registered planes feed training, baselines
and evaluation, so no arm gets a private alignment.

THE TONE PROBLEM, and the deliberate choice made here. The three separations
are different colour channels of a colour scene: they differ in TONE (7.3% of
each plane's energy below 0.30 cyc/px is not shared with the other two -- real
chroma) as well as in noise. A naive Noise2Noise would spend its capacity
learning the i->j tone mapping and call it denoising. Handled two ways at once:

  1. LOCAL AFFINE TONE NORMALISATION of the target before the loss: the target
     plane is mapped onto the input's tone frame by a guided-filter-style
     per-pixel affine a(x)*t + b(x), with a and b fitted between the two planes
     Gaussian-smoothed at sigma = 16 px and themselves smoothed. This removes
     the coarse tone/chroma differences (below ~0.01 cyc/px) exactly, so there
     is no systematic global mapping left for the network to learn. Chosen over
     predicting a high-pass band only, because the dominant visible noise --
     the per-trace streaks -- is column-constant and therefore lives at ALL
     across-trace frequencies including low ones; a high-pass target would put
     most of the streak energy out of reach of the loss.
  2. CORPUS SYMMETRY for what the affine cannot remove: chroma between ~0.01
     and 0.35 cyc/px stays in the loss, but training uses all SIX ordered pairs
     of every triplet and never tells the network which channel it is looking
     at, so the residual chroma enters with both signs and averages toward zero
     like noise. Any per-scene chroma the network memorises anyway is caught by
     the arbiter, because evaluation scenes are excluded from training.

  LANDMINES, named:
  * The affine fields are fitted from noisy planes. The bias is a slight
    shrinkage of the target (regression attenuation), which is conservative --
    it biases the network toward doing LESS, never toward inventing.
  * The b(x) field carries the input's own low-frequency content into the
    target, so below ~0.01 cyc/px the network is trained to the identity: NO
    DENOISING IS CLAIMED in that band. (Tone and scene DC are not separable
    from chroma there without knowing what colour things are, which would be
    Earth knowledge.)

THE ARBITER (the only one): predict measurements that were never seen. Image
metrics are known unreliable here -- quality.py's composite rewards blur and
improves monotonically past correct settings -- so no metric chooses anything
in this file. Instead: triplets are split into folds; the network never sees
the evaluation scenes. On an evaluation triplet, one separation x_k is the
HELD-OUT MEASUREMENT and predictors built from the other two are scored by
mean-squared prediction error against it. Since x_k's own noise adds the same
constant to every predictor's MSE, the ranking against x_k IS the ranking
against the unknown clean scene. The honest baselines on the same withheld
plane are the classical estimators: the raw registered plane (identity) and
the registered mean of the other two separations -- the estimator fuse.py
ships, and a strong one (ideal sqrt(2) = 3.01 dB; sqrt(3) = 4.77 dB for the
three-plane mean it stands in for). A blur-control arm (Gaussian sigma = 1 px)
is scored alongside, so a win by indiscriminate smoothing cannot be
misread as denoising.

  Because the held-out plane is a different COLOUR CHANNEL, not a true repeat,
  full-band MSE against it contains a chroma term, and a predictor could cheat
  it by stripping the input's chroma (desaturating toward the shared
  luminance). Three guards:
  * the noise algebra is also computed in the HIGH band (0.40-0.71 cyc/px
    radial) where chroma is measured absent (incoherent energy above 0.30
    cyc/px is 0.28% of picture power and essentially all noise -- fuse.py), so
    the dB gains quoted against sqrt(3) are chroma-clean;
  * a CHROMA RETENTION ratio (input-minus-luminance energy surviving the
    network, 0.10-0.35 cyc/px) is reported, so desaturation is visible;
  * the LOW band (0.02-0.35 cyc/px) error is reported separately, so real
    structure destroyed by the network shows up where the structure lives.

  Noise algebra, exact under independence (which is measured, above): with
  per-plane noise power N, m_id = MSE(x_i -> x_k) = N + N_k + C and
  m_avg = MSE(mean(x_i,x_j) -> x_k) = N/2 + N_k + C, so N = 2*(m_id - m_avg)
  with the unknowable N_k + C cancelling; the residual noise of a denoised
  plane is N_res = m_n2n - m_id + N by the same cancellation. In the HIGH band
  C ~ 0 and this is clean; full-band, C differs slightly between the id and avg
  arms (averaging two channels' chroma is not one channel's chroma), so the
  full-band N is reported as a cross-check, not used for the dB claims.

WHAT THIS CANNOT DO, stated up front: it cannot recover detail the 1977 camera
did not resolve (we already sample 1.4-1.9x above the camera's Nyquist in both
axes -- breakthrough_spec.md section 6), and it does not try: the network is a
denoiser fitted to this record's own repeats, its output is scored purely on
predicting unseen measurements, and hallucinated high-frequency content RAISES
that score's error, it does not lower it.

MEASURED RESULT (2026-08, data/master/384kHzStereo.wav; 19 triplets -- image
8, the solar spectrum, excluded because its content moves with wavelength;
4 folds x 3000 steps, every triplet evaluated exactly once as a scene the
network never saw). EXACTLY ONE full run exists. A draft of this docstring
briefly contained illustrative placeholder figures written before anything had
been run; they were never measurements, and any text describing them as a
"first run" to be compared against is describing numbers that were invented.
The table below is the run.

  held-out full-band MSE predicting the withheld separation, mean over the 19
  unseen scenes (every predictor built from the other two planes only):
      identity            0.03410     the raw registered plane
      blur sigma=1        0.03928     control; beats identity on only 4/19
      n2n (single plane)  0.03217     beats identity 19/19, beats blur 19/19
      2-plane mean        0.02546     the fuse.py estimator, and it is strong
      n2n then mean       0.02524     beats the plain mean on 14/19
  high band (0.40-0.71 cyc/px, chroma-free) noise algebra, mean:
      per-plane noise N           1.18e-3
      after n2n                   2.40e-4   -> +7.73 dB  (median +7.45)
      2-plane mean, ideal         5.92e-4   -> +3.01 dB  (sqrt 2)
      3-plane mean, ideal         3.95e-4   -> +4.77 dB  (sqrt 3, averaging's
                                               ceiling)
      n2n then 2-plane mean       1.93e-4   -> +8.82 dB
      blur control                3.26e-4   -> +5.60 dB  <- READ THIS ONE
  low band (0.02-0.35 cyc/px): n2n beats identity on 15/19; blur is WORSE
  than identity there on 17/19. Chroma retention 0.93 mean (0.78-1.16).

What is and is not established, in order of strength:

  * NOISE2NOISE WORKS ON THIS ARTIFACT: on 19 of 19 scenes the network never
    saw, the denoised plane predicts a withheld separation better than the
    plane it was given -- real, held-out, and not blur (the blur arm fails
    the same test on 15/19 and is beaten by n2n on every scene).
  * The hi-band +7.7 dB exceeds the sqrt(3) averaging ceiling on 17/19
    scenes, BUT the blur control's +5.6 dB shows hi-band suppression alone is
    cheap -- there is almost no camera signal above 0.40 cyc/px to protect.
    The claim n2n earns and blur does not is the CONJUNCTION: blur-level
    noise suppression at high frequency while the low band, where the picture
    lives, gets better on 15/19 rather than worse on 17/19.
  * AGAINST SIMPLE AVERAGING (the deliverable question): a single denoised
    plane does NOT beat even the 2-plane mean full-band (0.03217 vs 0.02546)
    -- averaging removes noise at every frequency and this network earns its
    keep mostly above ~0.1 cyc/px. Denoise-THEN-average beats plain
    averaging, but modestly: 14/19 scenes, -0.9% mean MSE. The margin is
    small because most of the full-band error against a different colour
    channel is the target's own noise plus real chroma, which no predictor
    of any kind can remove. So: n2n is a genuine, held-out improvement ON
    TOP of fusion, not a replacement for it, and plain 3-plane averaging
    remains a strong, nearly-free baseline.

NEGATIVES, per project policy, none quietly dropped:

  * Low-band error RISES on 4 of 19 scenes: images 12 (+6.2%), 46 (+4.0%),
    13 and 112 (<1%). 46 is the noisiest triplet on the record (12.9% noise,
    fuse.py). On roughly a fifth of scenes the network costs something where
    real structure lives; "no structure destroyed" is NOT a claim this
    evidence supports, only "structure preserved or improved on 15/19".
  * Chroma retention dips to 0.78-0.84 on the noisiest triplets (78, 114,
    55, 23): mild desaturation, so part of those scenes' full-band win may
    be chroma geometry rather than denoising. Their hi-band gains, which
    cannot be bought that way, are still +4.9 to +13.0 dB.
  * The training loss sits on the irreducible chroma+noise floor and is
    visibly noisy at batch 16; the budget was fixed, not extended to chase
    convergence, so these numbers are a floor on the method, not a ceiling.

Disclosure: two defects were found and fixed on smoke runs of fold 0 before
the full run -- the regression-slope tone map (see tone_fields: it taught the
network a ~30% damping of shared detail) and a zero last-layer init that
stalled training. Both fixes are structural, neither is per-scene; fold 0's
five scenes were observed during that debugging, folds 1-3 (14 of 19 scenes)
were not, and the fold-0 numbers are indistinguishable from the rest.

Usage:
    python -m pipeline.n2n --report            # build cache, train folds, evaluate
    python -m pipeline.n2n --report --quick    # one fold only (fast sanity run)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from . import catalog as catalog_mod
from . import fuse as fuse_mod

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "cache" / "n2n_triplets.npz"

#: Image 8 is the solar spectrum: its content genuinely moves with wavelength
#: (~90 px between separations, fuse.py), so its three frames are NOT repeat
#: observations of one scene and it is excluded from training AND evaluation.
EXCLUDE_IMAGES = (8,)

#: Tone-normalisation scale, px. The affine fields are fitted between planes
#: smoothed at this sigma, so tone/chroma below ~1/(2*pi*sigma) ~ 0.01 cyc/px
#: is removed from the loss exactly and NOT denoised (see module docstring).
TONE_SIGMA = 16.0
#: Regulariser on the local variance in the affine fit. In z-units (plane
#: variance 1) flat regions have local variance ~ the noise power ~4e-3, well
#: under this floor, so their fitted slope collapses to ~0 and the target
#: falls back to the smoothed level -- no noise is transferred through `a`.
TONE_EPS = 0.02

#: Evaluation ignores this border: Fourier-shift wraparound from registration
#: and the network's receptive field both live there.
EVAL_CROP = 24
#: Training loss ignores this patch border (receptive field radius is 8).
LOSS_MARGIN = 8

#: Radial frequency bands, cyc/px in the 377x512 square-pixel render.
#: HI: incoherent (non-shared) energy above 0.30 cyc/px is 0.28% of picture
#: power and essentially all noise (fuse.py section 3), so above 0.40 the
#: chroma term in the hold-out algebra is negligible: this band gives the
#: chroma-clean dB numbers. LO: where the camera's picture actually lives.
HI_BAND = (0.40, 0.71)
LO_BAND = (0.02, 0.35)
#: Band where chroma dominates noise by 39x..3.9x (fuse.py section 5): the
#: right place to measure whether the network strips channel identity.
CHROMA_BAND = (0.10, 0.35)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-12)


# --------------------------------------------------------------------------
# data: decode + register once, cache
# --------------------------------------------------------------------------


def build_cache(path: Path = CACHE, verbose: bool = True) -> None:
    """Decode all colour triplets from the master and register each in-triplet.

    Decoding is fuse.decode_triplet -- the ordinary shipping chain (sync.recover
    -> decode.decode, per-channel uncouple, 377x512 square-pixel render) -- and
    registration is fuse.register (band-limited phase correlation + Fourier
    shift, the validated estimator; see module docstring). Planes are stored
    z-scored AFTER registration because the 1977 chain gave the three exposures
    different gain and offset (colour.py).
    """
    cat = catalog_mod.build()
    out: dict[str, np.ndarray] = {}
    for im in cat.images:
        if not im.color or im.n in EXCLUDE_IMAGES:
            continue
        seps = fuse_mod.decode_triplet(cat, im)
        reg, off = fuse_mod.register(seps)
        arr = np.stack([_z(r) for r in reg]).astype(np.float32)
        out[f"im{im.n:03d}"] = arr
        if verbose:
            print(f"  cached image {im.n:3d}  {im.frames[0]}-{im.frames[2]}  "
                  f"dy,dx 1->2 {off[0]['dy']:+.2f},{off[0]['dx']:+.2f}  "
                  f"3->2 {off[1]['dy']:+.2f},{off[1]['dx']:+.2f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    if verbose:
        print(f"  wrote {path} ({len(out)} triplets)")


def load_triplets(path: Path = CACHE) -> dict[int, np.ndarray]:
    """{image number -> (3, H, W) registered, z-scored separations}."""
    if not path.exists():
        print(f"cache {path} missing; decoding 57 frames from the master "
              f"(a few minutes, once)")
        build_cache(path)
    with np.load(path) as z:
        return {int(k[2:]): z[k].astype(np.float32) for k in z.files}


# --------------------------------------------------------------------------
# local affine tone normalisation
# --------------------------------------------------------------------------


def tone_fields(src: np.ndarray, dst: np.ndarray, sigma: float = TONE_SIGMA,
                eps: float = TONE_EPS) -> tuple[np.ndarray, np.ndarray]:
    """Smooth per-pixel affine (a, b) such that a*src + b matches dst's tone.

    THE SLOPE IS THE SYMMETRIC (variance-matching) ONE, a = sqrt((var_dst+eps)
    / (var_src+eps)), NOT the regression slope cov/(var_src+eps) -- and that is
    a measured lesson, not a preference. The first version of this file used
    the regression slope; it is attenuated by the source plane's own noise and
    chroma variance and by the eps floor (local detail variance in weak-texture
    regions is ~0.01 in z-units, BELOW eps = 0.02), so every training target
    carried systematically shrunken detail, and the network faithfully learned
    to DAMP shared structure by ~30%. Measured on a fold-0 smoke run: held-out
    full-band MSE 0.0363 vs 0.0286 for the raw plane (worse on 5/5 scenes),
    hi-band untouched, chroma-retention 1.5 because (0.3)^2 of the SHARED
    detail power leaked into the deviation-from-luminance metric. That is the
    tone-mapping trap this module exists to avoid, in its subtlest form: a
    biased normalisation becomes a learned distortion. The symmetric slope has
    no attenuation (exact when the two planes' noise levels match), so the
    target keeps full-scale detail AND full-scale independent noise -- which is
    what Noise2Noise needs. Cost: where the local inter-channel correlation is
    genuinely negative (opposite-signed chroma edges; rare) the unsigned slope
    mismatches the detail sign, which enters the loss as more zero-mean chroma
    noise; accepted.

    Fields are built from sigma-smoothed statistics, so the map can only
    follow tone at scales >> the noise grain; the target's high-frequency
    noise cannot leak through it. Returned as FIELDS rather than an applied
    result so the SAME map can be applied to a raw plane and to its denoised
    version -- the evaluation depends on that (an arm must never get a private
    tone map).
    """
    g = lambda v: gaussian_filter(v.astype(np.float64), sigma, mode="reflect")
    mu_s, mu_d = g(src), g(dst)
    var_s = np.maximum(g(src * src) - mu_s * mu_s, 0.0)
    var_d = np.maximum(g(dst * dst) - mu_d * mu_d, 0.0)
    a = np.sqrt((var_d + eps) / (var_s + eps))
    b = mu_d - a * mu_s
    return g(a), g(b)


def tone_map(src: np.ndarray, dst: np.ndarray, **kw) -> np.ndarray:
    a, b = tone_fields(src, dst, **kw)
    return a * src + b


# --------------------------------------------------------------------------
# the denoiser
# --------------------------------------------------------------------------


class Denoiser(torch.nn.Module):
    """Small residual CNN, receptive field 17 px. Universal prior only.

    Deliberately small (~130k parameters against ~11M training pixels): the
    point is a denoiser, not a scene memoriser, and the fold structure would
    expose memorisation anyway. No normalisation layers -- inputs are z-scored
    planes and batch statistics over 16 patches of 15 scenes are exactly the
    kind of shortcut that lets a network smuggle scene identity. Reflect
    padding for the same reason as forward.py's convolutions: zero padding
    invents a black border for the optimiser to fit.
    """

    def __init__(self, width: int = 48, depth: int = 8):
        super().__init__()
        conv = lambda ci, co: torch.nn.Conv2d(ci, co, 3, padding=1,
                                              padding_mode="reflect")
        layers: list[torch.nn.Module] = [conv(1, width), torch.nn.ReLU()]
        for _ in range(depth - 2):
            layers += [conv(width, width), torch.nn.ReLU()]
        layers += [conv(width, 1)]
        self.net = torch.nn.Sequential(*layers)
        # Residual form: the identity is the starting point and the network
        # learns the (small) correction. The last layer starts SMALL but not
        # zero: an exactly-zero init was tried first and it stalls -- with the
        # last layer at zero, every earlier layer's gradient is zero too, so
        # nothing below it trains until the last layer has drifted away from
        # the very identity it was pinned to.
        with torch.no_grad():
            self.net[-1].weight *= 0.05
            torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def denoise_plane(model: torch.nn.Module, plane: np.ndarray,
                  device: torch.device) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(plane, dtype=np.float32))
        out = model(t[None, None].to(device))[0, 0]
    return out.cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


@dataclass
class TrainCfg:
    steps: int = 3000
    batch: int = 16
    patch: int = 64
    lr: float = 1e-3       # decays x0.3 at 60% and 85% of steps
    seed: int = 0


def build_pairs(planes: dict[int, np.ndarray],
                train_ns: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """All six ordered (input, tone-normalised target) pairs of each triplet.

    Six ordered pairs, not three unordered: Noise2Noise is symmetric in which
    observation is input and which is target, and using both directions is
    what makes the residual chroma enter the loss with both signs (module
    docstring). The target is tone-mapped ONTO THE INPUT's frame, so the
    network's output lives in its own input's tone and never learns a
    channel-to-channel mapping.
    """
    pairs = []
    for n in train_ns:
        reg = planes[n]
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                y = tone_map(reg[j], reg[i]).astype(np.float32)
                pairs.append((torch.from_numpy(reg[i].copy()),
                              torch.from_numpy(y)))
    return pairs


def train(pairs: list[tuple[torch.Tensor, torch.Tensor]], cfg: TrainCfg,
          device: torch.device, verbose: bool = False
          ) -> tuple[Denoiser, list[float]]:
    """Fixed-budget Noise2Noise fit. NOTHING here is tuned on any outcome:
    steps, lr and schedule were fixed before the first full run (constraint:
    the image metrics that could tune them are known unreliable, and the
    hold-out must stay a test, not a training signal)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = Denoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(0.6 * cfg.steps), int(0.85 * cfg.steps)], gamma=0.3)
    h, w = pairs[0][0].shape
    m = LOSS_MARGIN
    hist: list[float] = []
    for step in range(cfg.steps):
        xs, ys = [], []
        for b in rng.integers(0, len(pairs), cfg.batch):
            x, y = pairs[b]
            r0 = int(rng.integers(0, h - cfg.patch))
            c0 = int(rng.integers(0, w - cfg.patch))
            xp = x[r0:r0 + cfg.patch, c0:c0 + cfg.patch]
            yp = y[r0:r0 + cfg.patch, c0:c0 + cfg.patch]
            # Axis-preserving flips only: the streak noise is column-constant,
            # so a transpose would present the network with a noise model that
            # does not exist on the record.
            if rng.random() < 0.5:
                xp, yp = torch.flip(xp, (0,)), torch.flip(yp, (0,))
            if rng.random() < 0.5:
                xp, yp = torch.flip(xp, (1,)), torch.flip(yp, (1,))
            xs.append(xp)
            ys.append(yp)
        X = torch.stack(xs)[:, None].to(device)
        Y = torch.stack(ys)[:, None].to(device)
        opt.zero_grad(set_to_none=True)
        out = model(X)
        loss = torch.mean((out - Y)[..., m:-m, m:-m] ** 2)
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss.item()))
        if verbose and (step % 500 == 0 or step == cfg.steps - 1):
            print(f"    step {step:5d}  n2n loss {hist[-1]:.5f}")
    model.eval()
    return model, hist


# --------------------------------------------------------------------------
# evaluation: predict the withheld separation
# --------------------------------------------------------------------------


def band_mse(err: np.ndarray, band: tuple[float, float] | None) -> float:
    """Mean-squared error restricted to a radial frequency band (cyc/px).

    Hann-windowed so the crop boundary does not leak broadband energy into the
    high band; normalised so that band=None reproduces the windowed MSE. Every
    arm goes through the identical transform, so the windowing cancels in
    every comparison.
    """
    if band is None:
        return float(np.mean(err ** 2))
    hh, ww = err.shape
    win = np.hanning(hh)[:, None] * np.hanning(ww)[None, :]
    F = np.fft.fft2(err * win)
    fy = np.abs(np.fft.fftfreq(hh))[:, None]
    fx = np.abs(np.fft.fftfreq(ww))[None, :]
    rad = np.hypot(fy, fx)
    mask = (rad >= band[0]) & (rad < band[1])
    return float((np.abs(F) ** 2)[mask].sum() / (err.size ** 2)
                 / (win ** 2).mean())


def evaluate_triplet(model: Denoiser, reg: np.ndarray,
                     device: torch.device) -> dict:
    """Score every predictor of each withheld separation of one UNSEEN scene.

    For each k, predictors are built ONLY from the other two planes; each
    plane's tone map onto x_k is fitted once from the RAW plane and applied to
    both its raw and its denoised version, so the n2n arm cannot acquire a
    different tone frame than its own baseline. MSEs are per-k and averaged
    over the three rotations of k.
    """
    sl = np.s_[EVAL_CROP:-EVAL_CROP, EVAL_CROP:-EVAL_CROP]
    den = [denoise_plane(model, reg[i], device) for i in range(3)]
    blur = [gaussian_filter(reg[i].astype(np.float64), 1.0) for i in range(3)]

    bands = {"full": None, "hi": HI_BAND, "lo": LO_BAND}
    acc = {arm: {b: [] for b in bands} for arm in
           ("id", "avg", "n2n1", "n2n2", "blur")}
    for k in range(3):
        i, j = [p for p in range(3) if p != k]
        maps = {p: tone_fields(reg[p].astype(np.float64), reg[k].astype(np.float64))
                for p in (i, j)}
        tgt = reg[k].astype(np.float64)

        def mapped(p: int, arr: np.ndarray) -> np.ndarray:
            a, b = maps[p]
            return a * arr + b

        arms = {
            "id":   [mapped(p, reg[p].astype(np.float64)) for p in (i, j)],
            "blur": [mapped(p, blur[p]) for p in (i, j)],
            "n2n1": [mapped(p, den[p]) for p in (i, j)],
        }
        arms["avg"] = [0.5 * (arms["id"][0] + arms["id"][1])]
        arms["n2n2"] = [0.5 * (arms["n2n1"][0] + arms["n2n1"][1])]
        for arm, preds in arms.items():
            for bname, band in bands.items():
                acc[arm][bname].append(
                    float(np.mean([band_mse((p - tgt)[sl], band) for p in preds])))

    out = {f"m_{arm}_{b}": float(np.mean(v[b]))
           for arm, v in acc.items() for b in bands}

    # Chroma retention: how much of the input's channel-specific (not-shared)
    # mid-band energy survives the network. Stripping chroma would fake a
    # full-band win; this makes that visible.
    lum = reg.astype(np.float64).mean(axis=0)
    num = np.mean([band_mse((den[i] - lum)[sl], CHROMA_BAND) for i in range(3)])
    dnm = np.mean([band_mse((reg[i].astype(np.float64) - lum)[sl], CHROMA_BAND)
                   for i in range(3)])
    out["chroma_retention"] = float(num / max(dnm, 1e-30))
    return out


def algebra(m: dict, band: str) -> dict:
    """Noise powers from the hold-out MSEs (module docstring): the target's
    own noise and the shared chroma floor cancel in every difference."""
    n = 2.0 * (m[f"m_id_{band}"] - m[f"m_avg_{band}"])
    res1 = m[f"m_n2n1_{band}"] - m[f"m_id_{band}"] + n
    res2 = m[f"m_n2n2_{band}"] - m[f"m_avg_{band}"] + n / 2.0
    res_blur = m[f"m_blur_{band}"] - m[f"m_id_{band}"] + n
    db = lambda r: 10.0 * math.log10(n / r) if (n > 0 and r > 0) else float("nan")
    return {"N": n, "res1": res1, "res2": res2, "res_blur": res_blur,
            "gain1_db": db(res1), "gain2_db": db(res2),
            "gain_blur_db": db(res_blur)}


# --------------------------------------------------------------------------


def run(folds: int = 4, steps: int = 3000, seed: int = 0,
        quick: bool = False, json_out: Path | None = None) -> dict:
    device = _device()
    planes = load_triplets()
    ns = sorted(planes)
    print(f"device: {device};  {len(ns)} triplets (image 8 excluded)")
    print(f"folds: {folds}  steps: {steps}  (fixed a priori; the hold-out is "
          f"a test, never a tuning signal)\n")

    rows: list[dict] = []
    n_folds = 1 if quick else folds
    for f in range(n_folds):
        eval_ns = ns[f::folds]
        train_ns = [n for n in ns if n not in eval_ns]
        print(f"fold {f}: train on {len(train_ns)} scenes, "
              f"evaluate on unseen {eval_ns}")
        pairs = build_pairs(planes, train_ns)
        model, hist = train(pairs, TrainCfg(steps=steps, seed=seed + f), device,
                            verbose=True)
        for n in eval_ns:
            r = {"n": n, "fold": f}
            r.update(evaluate_triplet(model, planes[n], device))
            r.update({f"hi_{k}": v for k, v in algebra(r, "hi").items()})
            rows.append(r)
            print(f"    img {n:3d}  full: id {r['m_id_full']:.5f} "
                  f"blur {r['m_blur_full']:.5f} n2n1 {r['m_n2n1_full']:.5f} "
                  f"avg {r['m_avg_full']:.5f} n2n2 {r['m_n2n2_full']:.5f}   "
                  f"hi: N {r['hi_N']:.2e} n2n {r['hi_res1']:.2e} "
                  f"({r['hi_gain1_db']:+.1f} dB)   chroma kept "
                  f"{r['chroma_retention']:.2f}")
        print()

    mean = lambda k: float(np.mean([r[k] for r in rows]))
    wins1 = sum(r["m_n2n1_full"] < r["m_id_full"] for r in rows)
    wins2 = sum(r["m_n2n2_full"] < r["m_avg_full"] for r in rows)
    wins_lo = sum(r["m_n2n1_lo"] < r["m_id_lo"] for r in rows)
    blur_wins = sum(r["m_blur_full"] < r["m_id_full"] for r in rows)
    beats_avg3_hi = sum(r["hi_res1"] < r["hi_N"] / 3.0 for r in rows)
    # nan-aware: a scene where the hold-out algebra degenerates (N <= 0 or a
    # negative residual) reports NaN rather than a fabricated gain; it still
    # counts in every direct win/loss tally above.
    g1 = np.array([r["hi_gain1_db"] for r in rows])
    g2 = np.array([r["hi_gain2_db"] for r in rows])
    gm1, gmed1 = float(np.nanmean(g1)), float(np.nanmedian(g1))
    gm2, gmed2 = float(np.nanmean(g2)), float(np.nanmedian(g2))

    print("SUMMARY -- prediction of withheld separations, "
          f"{len(rows)} unseen scenes")
    print(f"  full band  id {mean('m_id_full'):.5f}   blur "
          f"{mean('m_blur_full'):.5f} (wins {blur_wins}/{len(rows)} -- "
          f"the arbiter {'REWARDS' if blur_wins > len(rows)/2 else 'punishes'} blur)")
    print(f"             n2n(single) {mean('m_n2n1_full'):.5f}  beats identity "
          f"on {wins1}/{len(rows)}")
    print(f"             2-plane mean {mean('m_avg_full'):.5f}  vs n2n-then-mean "
          f"{mean('m_n2n2_full'):.5f}  -- n2n beats plain averaging on "
          f"{wins2}/{len(rows)}")
    print(f"  low band   n2n(single) beats identity on {wins_lo}/{len(rows)} "
          f"(structure preserved where the picture lives)")
    print(f"  high band (chroma-free) noise algebra, mean over scenes:")
    print(f"             per-plane noise N       {mean('hi_N'):.3e}")
    print(f"             n2n residual            {mean('hi_res1'):.3e}   "
          f"gain {gm1:+.2f} dB (median {gmed1:+.2f})")
    print(f"             2-plane mean ideal      {mean('hi_N')/2:.3e}   +3.01 dB")
    print(f"             3-plane mean ideal      {mean('hi_N')/3:.3e}   +4.77 dB "
          f"(sqrt(3), the averaging ceiling)")
    print(f"             n2n-then-mean residual  {mean('hi_res2'):.3e}   "
          f"gain {gm2:+.2f} dB (median {gmed2:+.2f})")
    print(f"             single n2n plane beats the 3-plane average on "
          f"{beats_avg3_hi}/{len(rows)} scenes")
    gb = np.array([r["hi_gain_blur_db"] for r in rows])
    blur_lo_worse = sum(r["m_blur_lo"] > r["m_id_lo"] for r in rows)
    print(f"             CONTEXT: the blur control also reaches "
          f"{float(np.nanmean(gb)):+.2f} dB here -- hi-band dB alone cannot "
          f"tell denoising from smoothing. The difference is the low band, "
          f"where blur is worse than identity on {blur_lo_worse}/{len(rows)} "
          f"scenes and n2n is better on {wins_lo}/{len(rows)}.")
    print(f"  chroma retention {mean('chroma_retention'):.2f} "
          f"(1.0 = untouched; ~0.5 would mean the wins were bought by "
          f"desaturation)")

    out = {"rows": rows, "summary": {
        "m_id_full": mean("m_id_full"), "m_avg_full": mean("m_avg_full"),
        "m_n2n1_full": mean("m_n2n1_full"), "m_n2n2_full": mean("m_n2n2_full"),
        "m_blur_full": mean("m_blur_full"),
        "hi_N": mean("hi_N"), "hi_res1": mean("hi_res1"),
        "hi_res2": mean("hi_res2"),
        "hi_gain1_db_mean": gm1, "hi_gain1_db_median": gmed1,
        "hi_gain2_db_mean": gm2, "hi_gain2_db_median": gmed2,
        "wins_single_vs_id": wins1, "wins_pair_vs_avg": wins2,
        "wins_lo": wins_lo, "beats_avg3_hi": beats_avg3_hi,
        "hi_gain_blur_db_mean": float(np.nanmean(gb)),
        "blur_lo_worse": blur_lo_worse,
        "chroma_retention": mean("chroma_retention"), "n_scenes": len(rows)}}
    if json_out:
        json_out.write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwrote {json_out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="train the folds and run the hold-out evaluation")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="first fold only")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args(argv)
    if args.rebuild_cache:
        build_cache()
    if args.report:
        run(folds=args.folds, steps=args.steps, seed=args.seed,
            quick=args.quick, json_out=args.json)
    elif not args.rebuild_cache:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
