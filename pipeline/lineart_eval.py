"""Does the shipped denoiser help, do nothing, or HARM the line art?

THE GAP THIS FILE EXISTS TO CLOSE. `n2n.py` trains a Noise2Noise denoiser on
the record's own colour repeats and validates it by withholding one separation
of an unseen scene: 19/19 improved, the blur control fails the same test.
`reconstruct.py` then applies that network to ALL 156 frames. Every scene in
that validation is a PHOTOGRAPH. Roughly a fifth of the mono frames are not
photographs at all -- they are DIAGRAMS AND LINE ART (the calibration circle,
the mathematical definitions, the DNA and anatomy plates): bimodal histograms,
hard thin strokes, large fields that are uniform by design. Different image
statistics, and the denoiser has never been scored on them.

There is no held-out scan for a mono frame, so the hold-out arbiter that
settled the colour case is unavailable. Everything below is therefore built on
criteria that need no reference and that a smoother cannot buy:

  1. THE CALIBRATION CIRCLE. Axis ratio and radial rms of the ring on L000,
     `quality.circle_metrics`, run on the FLOAT decode (that distinction
     already mattered once -- reconstruct.py's own retraction). Geometry is set
     by the timebase; a denoiser has no business moving it in either direction.
  2. FLAT FIELDS, chosen blindly and measured PER CONNECTED REGION. Regions
     whose local standard deviation in the UNDENOISED decode is below a fixed
     absolute threshold. Within each region a plane is fitted and removed;
     the residual deviation should FALL (that is what denoising is) while the
     region's mean and the plane's tilt must NOT move (that would be a re-grade
     or an invented gradient). Measuring per region matters: pooling every flat
     pixel in a frame and fitting one plane measures the DIFFERENCE BETWEEN
     REGIONS, which is scene content, not noise.
  3. EDGE WIDTH. A super-resolved edge-spread function built from isolated
     steps found in the UNDENOISED image, fitted with a free-amplitude erf,
     reported as the 10-90 width in pixels -- separately across-trace and
     along-trace, because the network's training augmentation was deliberately
     anisotropic. A denoiser that widens hard edges is destroying the content.
  4. THIN-STROKE AMPLITUDE. Line art is mostly strokes 1-3 px wide, where "edge
     width" is not defined at all. Each stroke's amplitude against its own
     local baseline is measured at pixels fixed from the undenoised image;
     retention below 1 means strokes are being erased.
  5. HISTOGRAM VALLEY. Line art has two designed levels. Smoothing manufactures
     intermediate greys along every edge and fills the valley between the
     modes; denoising should empty it. Reported with the mode gap, so a
     contrast change cannot be mistaken for either.

  A BLUR CONTROL RUNS IN EVERY ONE OF THEM, at MATCHED STRENGTH: for each frame
  the Gaussian sigma is bisected until its rms change from the decode equals
  the network's rms change on that same frame, to 1%. Any claim of the form
  "the network improved X" that the matched blur also makes is not a claim
  about denoising.

  AN ADDED-NOISE ARM VALIDATES THE INSTRUMENT, also at matched strength. White
  noise cannot move an edge, so a correct edge-width estimator must report the
  noise arm as unchanged. It is there because of a specific bias risk: the
  sub-pixel registration of each edge is estimated from the undenoised profile,
  and if that estimate were correlated with the undenoised profile's own noise
  the identity arm would look artificially sharp. (Mitigated as well by
  estimating the crossing from a profile smoothed ALONG the edge, which does
  not touch its width across the edge.) The noise arm measures whatever bias
  survives, in the direction it acts.

WHICH FRAMES ARE LINE ART IS DECIDED BLINDLY, BY ONE LOCAL STATISTIC. Titles
are Earth knowledge (tier 2) and never enter the classifier. The statistic is

    tile_occ   Take 32x32 tiles on a 16 px stride. Keep the tiles whose 5-95
               percentile range exceeds 0.10 of black-to-white, i.e. the tiles
               that contain something. Inside each, the share of pixels lying
               within 20% of the tile's own 5th or 95th percentile. Report the
               median over kept tiles. A tile of line art is a field plus
               strokes -- two levels, so the share is high. A tile of
               continuous tone fills the space between its extremes, so it is
               low.

and the split is 1-D 2-means on the 156 frame values (equivalently, Otsu on
the statistic itself). ONE statistic, one unsupervised split, no weights.

  WHY LOCAL, and this is a measured lesson rather than a preference. The first
  version of this classifier used GLOBAL histogram statistics -- Otsu
  separability, valley depth, two-level occupancy over the whole frame. It put
  the Golden Gate Bridge and Snake River in the line-art cluster and left
  MATHEMATICAL DEFINITIONS and PAGE OF BOOK out of it. The mechanism is visible
  in the decode: L002's field ramps from mid-grey at one edge to white at the
  other across the whole frame, so its global histogram is broad and unimodal
  even though every 32-px neighbourhood of it is strictly two-level; while a
  photograph with a large sky is globally bimodal for reasons that have nothing
  to do with line art. The class is a LOCAL property and only a local statistic
  finds it.

  DISCLOSURE, because it bears on how much the agreement below is worth: that
  revision was made after seeing WHICH frames the global version misplaced,
  i.e. after seeing titles. The computation is title-free and an alien could
  run it, but the CHOICE of statistic was informed by Earth knowledge, so the
  agreement figure this file reports is optimistic and is not a cross-validated
  estimate. It is quoted as a sanity check on the grouping, never as a claim
  about a classifier's accuracy.

The catalogue titles are printed afterwards as an evaluation, and the errors
are reported as errors. Every quantitative conclusion is ALSO reported for a
title-defined line-art set (tier 2, evaluation only) so that the verdict does
not depend on the classifier being right, and against the continuous blind
score so it does not depend on where the boundary fell.

MEASURED RESULT (2026-08, data/master/384kHzStereo.wav, all 156 frames decoded
through fuse.decode_frame, the SHIPPED model data/cache/n2n_model.pt, one full
run). The blind cluster holds 31 frames, of which 23 are in the title set; 8
title-line-art frames are missed (Page of book, Violin with music score, Cells
and cell division, and four anatomy plates), 8 photographs are false positives.
Every number below is quoted for the blind cluster and, in brackets, for the
title set: THEY AGREE, so nothing here depends on the classifier.

  1. THE CALIBRATION CIRCLE CANNOT SETTLE THIS, AND THAT IS THE FIRST FINDING.
     On the float decode of L000, every arm at rms change 1.639 of 255:
                        id        n2n       blur      noise (8 seeds)
        axis ratio    1.00501   1.00535   1.00476   1.00545 +- 0.00035
        radial rms    0.8633    0.8552    0.8485    0.8629  +- 0.0095 px
     The network moves axis ratio by 0.00035 and radial rms by 0.008 px. An
     information-free perturbation of the same size -- white noise, which
     cannot possibly improve a ring's geometry -- moves them by 0.00045 and
     0.0004 +- 0.0095. BOTH OF THE NETWORK'S MOVEMENTS ARE INSIDE THE FLOOR,
     and the matched blur "improves" both metrics MORE than the network does.
     `reconstruct.py` reports "radial rms 0.837 -> 0.834, slightly BETTER,
     PASS" and calls this "the check a denoiser cannot game". At this
     correction size it is not a check at all: it has no power to certify the
     denoiser and no power to condemn it, in either direction. That is a
     statement about the instrument's resolution, not about the network.

  2-5. WHAT THE NETWORK ACTUALLY DOES TO LINE ART, at matched strength
     (median ratio to the undenoised decode; frames moving down / frames
     measured; blur and noise arms at the identical rms change):

                              LINE ART        [title set]     PHOTOGRAPHS
        flat-field sd       0.946x 31/31      [0.946x]       0.942x 121/121
          matched blur      0.961x 31/31      [0.962x]       0.962x 121/121
          matched noise     1.073x  0/31      [1.074x]       1.055x   0/121
        ESF 10-90 across    1.059x  0/23      [1.059x]       1.048x   1/91
          matched blur      1.267x            [1.242x]       1.207x
        ESF 10-90 along     1.051x  0/27      [1.051x]       1.043x   3/99
          matched blur      1.155x            [1.140x]       1.133x
        stroke amp across   0.919x 27/27      [0.919x]       0.925x 118/118
          matched blur      0.868x            [0.867x]       0.873x
        stroke amp along    0.924x 28/28      [0.926x]       0.922x  96/96
        histogram valley    1.020x  3/31      [1.018x]       1.019x   6/125

     Read as physics rather than ratios: the network's edge damage is
     equivalent to an EXTRA GAUSSIAN OF SIGMA 0.26 px across-trace and 0.33 px
     along-trace, against 0.55-0.60 px for the blur arm that displaces the
     picture by the same amount. The conjunction that justified the network on
     photographs therefore survives on line art -- it removes 5.4% of
     flat-field deviation for 5.9% of edge widening, where the matched blur
     removes 3.9% for 26.7%, and the network wins that trade on 23/23 and
     27/27 line-art frames.

     BUT THE TRADE IS ABOUT 1.7x WORSE ON LINE ART THAN ON PHOTOGRAPHS (0.90
     and 0.80 against 1.41 and 1.45), and the mechanism is not that the
     network behaves differently: the ABSOLUTE damage is the same to two
     decimal places (equivalent sigma 0.26 / 0.33 px on line art against 0.26 /
     0.33 px on photographs). It is that line art's edges are SHARPER to begin
     with -- 1.97 px 10-90 across-trace against 2.13 -- so the same fixed
     low-pass costs a larger fraction of them. The class does not change the
     operator. It changes what the operator costs.

  NEGATIVES, none dropped:
  * The network is NOT DC-NEUTRAL. Flat-field means move UP by 0.00087 of
    black-to-white on 30 of 31 line-art frames (0.00094 on 96 of 121
    photographs) -- about a quarter of a grey level in 255. The blur and noise
    arms move them by 0.00000. Small, systematic, and nobody had measured it.
  * FLAT-FIELD TILT falls 1.0% on 31/31 and 120/121 frames. The blur arm does
    not do this. The network is very slightly flattening real gradients, which
    is exactly the class of thing a prior can invent or destroy.
  * THE HISTOGRAM VALLEY RISES, 1.020x on line art, i.e. the network creates
    intermediate greys between the two designed levels rather than removing
    them. It is confounded with the DC shift above (the mode band is fixed
    from the undenoised frame), so this is reported as a direction, not a size.
  * The stroke-amplitude loss was 0.903x before the leave-one-out selection
    de-bias and 0.919x after: 1.6 percentage points of the apparent damage was
    regression to the mean from selecting strokes on their own noise. Anyone
    repeating this measurement without `_loo` will overstate the damage.
  * The 8 line-art frames the blind rule misses are missed for a real reason,
    not a tuning failure: on R068 (a printed page) and R077 (a music score) the
    1977 camera did not resolve the strokes, so the decoded frame is not a
    two-level image and no local statistic can say it is. The frames whose
    content is most obviously "line art" to a human are the ones where that
    fact did not survive the camera.
  * Only ONE full run exists. Nothing here is a repeated measurement except the
    circle's noise arm.

  6. A DEPLOYMENT DEFECT FOUND ON THE WAY, unrelated to image class.
     `reconstruct.decode_float` applies Barry's quarter turn and THEN denoises,
     so the ~60 shipped frames with a non-zero turn are fed to the network with
     their trace axis and their along-trace axis swapped -- and the network's
     training augmentation was axis-preserving flips ONLY, on purpose, because
     the record's structured noise is column-constant. Measured on 16 frames:
     rotating-then-denoising and denoising-then-rotating disagree by 0.66x the
     size of the correction itself. The rotated order is WORSE on both counts:
     it removes less flat-field deviation (0.965x against 0.947x) and widens
     across-trace edges more (1.092x against 1.046x). Denoise before the turn.

PROVENANCE. Tier 1 evaluation. Reads the master WAV, `catalog` (seed samples
and channel only -- `orientation` never reaches a pixel, and `title` is used
only in `TITLE_LINEART` and in the printed evaluation), and the trained model
`n2n.py` produced. No file under docs/reference is opened; `decode.py` is
untouched and still imports numpy + dotclock + sync.

Usage:
    python -m pipeline.lineart_eval --report          # all 156 frames
    python -m pipeline.lineart_eval --report --quick  # every 7th, smoke test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label, uniform_filter

from . import catalog as catalog_mod
from . import fuse as fuse_mod
from . import n2n as n2n_mod
from . import quality

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "cache" / "lineart_frames.npz"
MODEL = REPO / "data" / "cache" / "n2n_model.pt"
REPORT = REPO / "docs" / "lineart_eval.json"

#: Border dropped before any measurement: the outermost columns carry
#: sync-adjacent junk from the neighbouring frame (quality.BORDER_TRIM exists
#: for the same reason) and the network's receptive field is 17 px.
CROP = 16

#: A pixel is FLAT when its local sd over a 15x15 window is below this, in
#: units of black-to-white: ~5 grey levels of 255, fixed a priori. The decode
#: is level-anchored to the record's own porch and sync amplitude, so the
#: absolute threshold means the same thing on every frame.
FLAT_SD = 0.02
FLAT_WIN = 15
#: Smallest connected flat region worth a statistic.
FLAT_MIN_PX = 1500

#: An edge enters the ESF when its step exceeds this fraction of black-to-white
#: and no competing gradient above half its own lies within EDGE_HALF px.
EDGE_STEP = 0.25
EDGE_HALF = 6
#: A stroke enters the amplitude test when the 2-px line filter exceeds this.
STROKE_AMP = 0.20

#: Blind line-art statistic: tile size, stride, and the 5-95 range a tile must
#: carry before it counts as containing something.
TILE = 32
TILE_STRIDE = 16
TILE_MIN_RANGE = 0.10

ARMS = ("id", "n2n", "blur", "noise")

#: ------------------------------------------------------------------------
#: TIER 2, EVALUATION ONLY. The frames whose CATALOGUE TITLE says diagram,
#: definition, anatomical plate, sketch or printed page. This set never reaches
#: a decision, a threshold or a pixel: it exists so the classifier can be
#: scored, and so every aggregate below can be recomputed on a grouping that
#: does not depend on the classifier at all. If the two groupings disagree
#: about the verdict, the verdict is not established.
TITLE_LINEART = frozenset("""
L000 L001 L002 L003 L004 L005 L019 L020 L021 L022 L023 L024 L025 L026 L027
L028 L029 L030 L031 L032 L033 L034 L037 L039 L050 L052 L053 L075 R010 R068
R077""".split())


def _device() -> torch.device:
    """CPU on purpose: 130k parameters on a 377x512 image, and the GPU on this
    machine is under other load."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# decode cache
# ---------------------------------------------------------------------------


def build_cache(path: Path = CACHE, verbose: bool = True) -> None:
    """Decode all 156 frames through the SAME chain n2n was trained on.

    `fuse.decode_frame` is the path that produced the training corpus: 384 kHz
    master decimated to 96 kHz, `decode.Settings(channel=...)`, 377x512
    square-pixel render, NO rotation. Using it here means anything measured
    below is the image class and not a change of decoder. (The shipped
    `reconstruct.decode_float` differs on both counts -- it decodes at 384 kHz
    and applies Barry's quarter turn BEFORE denoising. That second mismatch is
    measured on its own, in `rotation_sensitivity`.)
    """
    cat = catalog_mod.build()
    out: dict[str, np.ndarray] = {}
    for k, fr in enumerate(cat.frames):
        out[fr.id] = fuse_mod.decode_frame(fr).astype(np.float32)
        if verbose and k % 20 == 0:
            print(f"  decoded {k}/{len(cat.frames)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    if verbose:
        print(f"  wrote {path} ({len(out)} frames)")


def load_frames(path: Path = CACHE) -> dict[str, np.ndarray]:
    if not path.exists():
        print(f"cache {path} missing; decoding 156 frames from the master")
        build_cache(path)
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


# ---------------------------------------------------------------------------
# blind classification
# ---------------------------------------------------------------------------


def _otsu(a: np.ndarray, bins: int = 256):
    """Otsu threshold, class means, and the histogram, from pixels alone."""
    h, edges = np.histogram(a, bins=bins, range=(0.0, 1.0))
    p = h.astype(np.float64) / max(h.sum(), 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    m0c = np.cumsum(p * c)
    mt = m0c[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = (mt * w0 - m0c) ** 2 / (w0 * (1.0 - w0))
    k = int(np.argmax(np.nan_to_num(sb, nan=0.0, posinf=0.0)))
    thr = float(c[k])
    lo, hi = p[:k + 1], p[k + 1:]
    m0 = float((c[:k + 1] * lo).sum() / max(lo.sum(), 1e-12))
    m1 = float((c[k + 1:] * hi).sum() / max(hi.sum(), 1e-12))
    return thr, m0, m1, p, c


def features(img: np.ndarray) -> dict:
    """The blind line-art statistic, plus diagnostics. Pixels only.

    `tile_occ` is the classifier's whole input (module docstring). `tile_cov`,
    `flat_frac` and `mode_gap` are printed for context and take part in no
    decision -- a second feature only added false positives when it was tried
    (2-means on (tile_occ, flat_frac): cluster of 53, of which 26 were
    photographs, against 31 and 8 for tile_occ alone).
    """
    a = np.asarray(img, dtype=np.float64)[CROP:-CROP, CROP:-CROP]
    h, w = a.shape
    occ, kept, total = [], 0, 0
    for r0 in range(0, h - TILE + 1, TILE_STRIDE):
        for c0 in range(0, w - TILE + 1, TILE_STRIDE):
            t = a[r0:r0 + TILE, c0:c0 + TILE].ravel()
            total += 1
            lo, hi = np.percentile(t, [5.0, 95.0])
            if hi - lo < TILE_MIN_RANGE:
                continue
            kept += 1
            u = (t - lo) / (hi - lo)
            occ.append(float(np.mean((u < 0.2) | (u > 0.8))))

    mu = uniform_filter(a, FLAT_WIN, mode="reflect")
    sd = np.sqrt(np.maximum(uniform_filter(a * a, FLAT_WIN, mode="reflect") - mu * mu, 0.0))
    thr, m0, m1, _, _ = _otsu(a)
    return {"tile_occ": float(np.median(occ)) if occ else float("nan"),
            "tile_cov": kept / max(total, 1),
            "flat_frac": float(np.mean(sd < FLAT_SD)),
            "mode_gap": float(m1 - m0)}


def _kmeans2(X: np.ndarray, iters: int = 100):
    """Deterministic 2-means, centroids seeded at the extremes of the first
    principal direction so there is no random restart anywhere to tune."""
    Xc = X - X.mean(axis=0)
    u = np.linalg.svd(Xc, full_matrices=False)[2][0]
    t = Xc @ u
    c = np.stack([X[np.argmin(t)], X[np.argmax(t)]]).astype(np.float64)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in (0, 1):
            if (lab == j).any():
                c[j] = X[lab == j].mean(axis=0)
    return lab, c


def classify(feats: dict[str, dict]) -> tuple[dict[str, bool], dict[str, float], float]:
    """{id -> is_lineart}, {id -> blind score}, and the split threshold.

    1-D 2-means on `tile_occ`. The score is (tile_occ - boundary) scaled so the
    two centroids sit at -1 and +1, so every conclusion can be checked against
    a continuum instead of against a hard split that some frames will straddle.
    """
    ids = sorted(feats)
    x = np.array([[feats[i]["tile_occ"]] for i in ids], dtype=np.float64)
    lab, cen = _kmeans2(x)
    art = 0 if cen[0, 0] > cen[1, 0] else 1   # the two-level cluster
    mid = 0.5 * (cen[0, 0] + cen[1, 0])
    half = 0.5 * abs(cen[0, 0] - cen[1, 0])
    s = (x[:, 0] - mid) / max(half, 1e-12)    # positive = line art
    return ({i: bool(lab[k] == art) for k, i in enumerate(ids)},
            {i: float(s[k]) for k, i in enumerate(ids)}, float(mid))


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------


def load_model() -> torch.nn.Module:
    """The SHIPPED model -- data/cache/n2n_model.pt, the one reconstruct.py
    applies to all 156 frames. Nothing is retrained here."""
    dev = _device()
    m = n2n_mod.Denoiser().to(dev)
    m.load_state_dict(torch.load(MODEL, map_location=dev))
    m.eval()
    return m


def denoise(model: torch.nn.Module, img: np.ndarray) -> np.ndarray:
    """Exactly reconstruct.py's call: z-score, run, restore mean and sd."""
    a = np.asarray(img, dtype=np.float64)
    mu, sd = a.mean(), a.std()
    if sd < 1e-9:
        return a.copy()
    return n2n_mod.denoise_plane(model, (a - mu) / sd, _device()) * sd + mu


def _rms(d: np.ndarray) -> float:
    return float(np.sqrt(np.mean(d[CROP:-CROP, CROP:-CROP] ** 2)))


def matched_blur(img: np.ndarray, target: float, tol: float = 0.01) -> tuple[np.ndarray, float]:
    """Gaussian whose rms change from `img` equals the network's, per frame.

    Matching STRENGTH rather than fixing sigma is what makes this a control: at
    equal displacement from the decode, "which one moved the edges less and the
    flat fields more" has an answer, where a fixed-sigma blur could always be
    dismissed as too strong or too weak.
    """
    a = np.asarray(img, dtype=np.float64)
    if target <= 0:
        return a.copy(), 0.0
    lo, hi = 0.02, 8.0
    if _rms(gaussian_filter(a, hi) - a) < target:
        return gaussian_filter(a, hi), hi
    mid = hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = _rms(gaussian_filter(a, mid) - a)
        if abs(r - target) < tol * target:
            break
        if r < target:
            lo = mid
        else:
            hi = mid
    return gaussian_filter(a, mid), mid


def matched_noise(img: np.ndarray, target: float, seed: int) -> np.ndarray:
    """White noise at the same rms. The instrument's null: noise cannot move an
    edge, so a correct ESF estimator must report this arm as unchanged."""
    rng = np.random.default_rng(seed)
    return np.asarray(img, dtype=np.float64) + rng.normal(0.0, target, img.shape)


# ---------------------------------------------------------------------------
# criterion 2: flat fields, per connected region
# ---------------------------------------------------------------------------


def flat_regions(img: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Connected flat regions, fixed once on the UNDENOISED decode.

    Fixed on the original so every arm is measured on identical pixels;
    letting each arm nominate its own flat regions would let a smoother pick
    its successes.
    """
    a = np.asarray(img, dtype=np.float64)
    mu = uniform_filter(a, FLAT_WIN, mode="reflect")
    sd = np.sqrt(np.maximum(uniform_filter(a * a, FLAT_WIN, mode="reflect") - mu * mu, 0.0))
    m = sd < FLAT_SD
    b = CROP + FLAT_WIN
    m[:b] = m[-b:] = False
    m[:, :b] = False
    m[:, -b:] = False
    lab, n = label(m)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) >= FLAT_MIN_PX:
            out.append((ys, xs))
    return out


def flat_stats(img: np.ndarray, regions) -> dict:
    """Per-region plane fit; area-weighted means of sd, |mean|, tilt.

    The plane is removed before the deviation is measured so a genuine slow
    gradient inside a region is not counted as noise, and the plane's own slope
    is reported, because a denoiser that changes it has invented or destroyed a
    gradient rather than suppressed noise.
    """
    a = np.asarray(img, dtype=np.float64)
    if not regions:
        return {"n_regions": 0, "px": 0, "sd": float("nan"),
                "mean": float("nan"), "tilt": float("nan")}
    sds, mus, tilts, ws = [], [], [], []
    for ys, xs in regions:
        v = a[ys, xs]
        A = np.stack([np.ones_like(ys, dtype=np.float64),
                      (ys - ys.mean()) / 100.0, (xs - xs.mean()) / 100.0], axis=1)
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        sds.append((v - A @ coef).std())
        mus.append(coef[0])
        tilts.append(math.hypot(coef[1], coef[2]))
        ws.append(len(ys))
    w = np.array(ws, dtype=np.float64)
    w /= w.sum()
    return {"n_regions": len(regions), "px": int(sum(ws)),
            "sd": float(np.dot(w, sds)), "mean": float(np.dot(w, mus)),
            "tilt": float(np.dot(w, tilts))}


# ---------------------------------------------------------------------------
# criterion 3: super-resolved edge width
# ---------------------------------------------------------------------------

_ESF_EDGES = np.arange(-4.0, 4.0001, 0.25)
_ESF_T = 0.5 * (_ESF_EDGES[:-1] + _ESF_EDGES[1:])
_SIGMAS = np.arange(0.10, 3.001, 0.01)
_T0S = np.arange(-1.0, 1.0001, 0.05)
_PHI = 0.5 * (1.0 + np.vectorize(math.erf)(
    (_ESF_T[None, None, :] - _T0S[None, :, None]) / (_SIGMAS[:, None, None] * math.sqrt(2.0))))


def _grad(a: np.ndarray) -> np.ndarray:
    g = np.zeros_like(a)
    g[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2])
    return g


def _loo(a: np.ndarray) -> np.ndarray:
    """Mean of the four neighbours ALONG the feature, excluding the row itself.

    The de-bias that both selections below need. An edge or a stroke is picked
    out by a statistic computed from the picture, and then a value at that same
    pixel is compared between arms. If the selecting statistic used the pixel's
    own noise, the identity arm would be measured exactly where its noise made
    the feature look strongest, and every smoother would show a drop that is
    regression to the mean rather than damage. Edges and strokes run for many
    pixels along their length, so rows +-1 and +-2 carry the same feature and
    independent noise: selecting on them makes the identity arm's measurement
    noise exactly independent of its selection. (The network's output at the
    row does depend on the neighbouring rows, so ITS noise stays slightly
    correlated with the selection -- a bias in the conservative direction, i.e.
    against finding damage.)
    """
    p = np.pad(a, ((2, 2), (0, 0)), mode="reflect")
    return 0.25 * (p[:-4] + p[1:-3] + p[3:-1] + p[4:])


def find_edges(img: np.ndarray, axis: int):
    """Isolated strong steps along `axis`, located on the UNDENOISED image.

    `axis=1` walks across traces, `axis=0` along them. Requirements: the step
    exceeds EDGE_STEP of black-to-white; |gradient| is a strict local maximum
    along the profile and at least 3x the perpendicular gradient, so a diagonal
    contour never enters a 1-D profile; and no other gradient above half the
    peak lies within EDGE_HALF px -- which excludes thin strokes, whose two
    edges overlap and for which a 10-90 width is not defined. Those are
    measured separately, by amplitude.

    Returns (rows, cols, subpixel 50% crossing, gradient sign). THE CROSSING IS
    ESTIMATED FROM THE FOUR NEIGHBOURING ROWS ALONG THE EDGE AND NOT FROM THE
    EDGE PIXEL'S OWN ROW (`_loo`), so the identity arm cannot align with its
    own noise and look artificially sharp; the `noise` arm measures whatever
    bias survives.
    """
    a = np.asarray(img, dtype=np.float64)
    if axis == 0:
        a = a.T
    g = _grad(a)
    ag = np.abs(g)
    gp = np.zeros_like(a)
    gp[1:-1, :] = 0.5 * (a[2:, :] - a[:-2, :])

    peak = np.zeros_like(a, dtype=bool)
    peak[:, 1:-1] = (ag[:, 1:-1] > ag[:, :-2]) & (ag[:, 1:-1] >= ag[:, 2:])
    ok = peak & (ag > 3.0 * np.abs(gp))
    b = EDGE_HALF + CROP
    ok[:, :b] = False
    ok[:, -b:] = False
    ok[:CROP, :] = False
    ok[-CROP:, :] = False

    rr, cc = np.nonzero(ok)
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0))
    if len(rr) == 0:
        return empty

    off = np.arange(-EDGE_HALF, EDGE_HALF + 1)
    sgn = np.where(g[rr, cc] >= 0, 1.0, -1.0)
    prof = a[rr[:, None], cc[:, None] + off[None, :]]
    prof = np.where(sgn[:, None] > 0, prof, prof[:, ::-1])
    step = prof[:, -2:].mean(axis=1) - prof[:, :2].mean(axis=1)

    gnb = np.abs(g[rr[:, None], cc[:, None] + off[None, :]])
    gnb[:, EDGE_HALF - 1:EDGE_HALF + 2] = 0.0
    keep = (step > EDGE_STEP) & (gnb.max(axis=1) < 0.5 * np.abs(g[rr, cc]))
    rr, cc, sgn = rr[keep], cc[keep], sgn[keep]
    if len(rr) == 0:
        return empty

    # crossing from the neighbouring rows only (leave-one-out along the edge)
    s = _loo(a)
    ps = s[rr[:, None], cc[:, None] + off[None, :]]
    ps = np.where(sgn[:, None] > 0, ps, ps[:, ::-1])
    base = ps[:, :2].mean(axis=1)
    st = ps[:, -2:].mean(axis=1) - base
    good = st > 1e-6
    rr, cc, sgn, ps, base, st = rr[good], cc[good], sgn[good], ps[good], base[good], st[good]
    n = (ps - base[:, None]) / st[:, None]
    x0 = np.full(len(rr), np.nan)
    for j in range(EDGE_HALF - 2, EDGE_HALF + 2):
        d = n[:, j + 1] - n[:, j]
        hit = np.isnan(x0) & (n[:, j] <= 0.5) & (n[:, j + 1] > 0.5) & (np.abs(d) > 1e-9)
        x0[hit] = (j - EDGE_HALF) + (0.5 - n[hit, j]) / d[hit]
    ok2 = np.isfinite(x0)
    return rr[ok2], cc[ok2], x0[ok2], sgn[ok2]


def esf_width(img: np.ndarray, axis: int, rr, cc, x0, sgn) -> float:
    """10-90 width, px, of the pooled super-resolved edge-spread function.

    Edge set and registration are passed in from the undenoised image, so every
    arm's profiles land on the same grid. Each edge is normalised by ITS OWN
    asymptotes in the arm being measured -- a 10-90 width is a shape, and a
    contrast change belongs in a different number -- the samples are pooled
    into 0.25 px bins and reduced by median, and a free-amplitude erf is fitted
    to the binned curve over a (t0, sigma) grid with amplitude and offset
    solved in closed form at every node. 10-90 = 2.5631 sigma.
    """
    a = np.asarray(img, dtype=np.float64)
    if axis == 0:
        a = a.T
    if len(rr) == 0:
        return float("nan")
    off = np.arange(-EDGE_HALF, EDGE_HALF + 1)
    prof = a[rr[:, None], cc[:, None] + off[None, :]]
    prof = np.where(sgn[:, None] > 0, prof, prof[:, ::-1])
    base = prof[:, :2].mean(axis=1)
    step = prof[:, -2:].mean(axis=1) - base
    keep = np.abs(step) > 1e-6
    t = (off[None, :] - x0[keep][:, None]).ravel()
    v = ((prof[keep] - base[keep, None]) / step[keep, None]).ravel()

    idx = np.digitize(t, _ESF_EDGES) - 1
    good = (idx >= 0) & (idx < len(_ESF_T))
    idx, v = idx[good], v[good]
    med = np.full(len(_ESF_T), np.nan)
    cnt = np.bincount(idx, minlength=len(_ESF_T))
    order = np.argsort(idx, kind="stable")
    idx_s, v_s = idx[order], v[order]
    bounds = np.searchsorted(idx_s, np.arange(len(_ESF_T) + 1))
    for b in range(len(_ESF_T)):
        if cnt[b] >= 8:
            med[b] = np.median(v_s[bounds[b]:bounds[b + 1]])
    m = np.isfinite(med)
    if m.sum() < 12:
        return float("nan")

    y = med[m]
    P = _PHI[:, :, m]                              # (n_sig, n_t0, n_bins)
    n = float(m.sum())
    Sp = P.sum(axis=2)
    Spp = (P * P).sum(axis=2)
    Sv = float(y.sum())
    Svv = float((y * y).sum())
    Spv = (P * y[None, None, :]).sum(axis=2)
    det = n * Spp - Sp * Sp
    with np.errstate(invalid="ignore", divide="ignore"):
        A = (n * Spv - Sp * Sv) / det
        B = (Spp * Sv - Sp * Spv) / det
        sse = Svv - A * Spv - B * Sv
    sse = np.where((det > 1e-12) & (A > 0.2) & np.isfinite(sse), sse, np.inf)
    k = int(np.argmin(sse))
    return float(2.5631 * _SIGMAS[k // len(_T0S)])


# ---------------------------------------------------------------------------
# criterion 4: thin strokes
# ---------------------------------------------------------------------------


def find_strokes(img: np.ndarray, axis: int):
    """Thin lines: extrema of 2*v[c] - v[c-2] - v[c+2].

    Computed on `_loo` -- the mean of the four rows along the stroke, excluding
    the row whose amplitude will be measured -- so that the identity arm is not
    sampled exactly where its own noise made a stroke look strongest. Without
    this the whole stroke-amplitude result would be partly regression to the
    mean; with it, the identity's measurement noise is independent of the
    selection by construction.
    """
    a = np.asarray(img, dtype=np.float64)
    if axis == 0:
        a = a.T
    s = _loo(a)
    lap = np.zeros_like(a)
    lap[:, 2:-2] = 2.0 * s[:, 2:-2] - s[:, :-4] - s[:, 4:]
    al = np.abs(lap)
    peak = np.zeros_like(a, dtype=bool)
    peak[:, 1:-1] = (al[:, 1:-1] > al[:, :-2]) & (al[:, 1:-1] >= al[:, 2:])
    ok = peak & (al > STROKE_AMP)
    b = CROP + 6
    ok[:, :b] = False
    ok[:, -b:] = False
    ok[:CROP, :] = False
    ok[-CROP:, :] = False
    rr, cc = np.nonzero(ok)
    return rr, cc, (np.sign(lap[rr, cc]) if len(rr) else np.zeros(0))


def stroke_amp(img: np.ndarray, axis: int, rr, cc, sgn) -> float:
    """Median signed amplitude of the FIXED stroke set against its own flanks."""
    a = np.asarray(img, dtype=np.float64)
    if axis == 0:
        a = a.T
    if len(rr) == 0:
        return float("nan")
    base = 0.25 * (a[rr, cc - 4] + a[rr, cc - 5] + a[rr, cc + 4] + a[rr, cc + 5])
    return float(np.median(sgn * (a[rr, cc] - base)))


# ---------------------------------------------------------------------------
# criterion 5: histogram valley
# ---------------------------------------------------------------------------


def valley(img: np.ndarray, m0: float, m1: float) -> float:
    """Share of pixels in the central 40% of the mode gap, at FIXED modes.

    The modes come from the undenoised frame so the arms are compared on the
    same intervals. Smoothing creates intermediate greys along every edge and
    raises this; denoising should lower it.
    """
    a = np.asarray(img, dtype=np.float64)[CROP:-CROP, CROP:-CROP]
    lo, hi = m0 + 0.3 * (m1 - m0), m0 + 0.7 * (m1 - m0)
    return float(np.mean((a > lo) & (a < hi)))


# ---------------------------------------------------------------------------
# per-frame evaluation
# ---------------------------------------------------------------------------


def build_arms(model, img: np.ndarray, seed: int = 0) -> tuple[dict, dict]:
    a = np.asarray(img, dtype=np.float64)
    n = denoise(model, a)
    r = _rms(n - a)
    b, sigma = matched_blur(a, r)
    z = matched_noise(a, r, seed)
    arms = {"id": a, "n2n": n, "blur": b, "noise": z}
    info = {"rms_n2n": r, "rms_blur": _rms(b - a), "rms_noise": _rms(z - a),
            "blur_sigma": sigma}
    return arms, info


def evaluate_frame(model, fid: str, img: np.ndarray, feat: dict) -> dict:
    arms, info = build_arms(model, img, seed=abs(hash(fid)) % (2 ** 31))
    row = {"frame": fid, **info}

    regions = flat_regions(img)
    base = flat_stats(arms["id"], regions)
    for name, arr in arms.items():
        s = flat_stats(arr, regions)
        for k, v in s.items():
            row[f"flat_{k}_{name}"] = v
        # absolute, signed: a ratio near 1.000 hides a systematic DC shift
        row[f"flat_dmean_{name}"] = s["mean"] - base["mean"]

    thr, m0, m1, _, _ = _otsu(np.asarray(img)[CROP:-CROP, CROP:-CROP])
    for name, arr in arms.items():
        row[f"valley_{name}"] = valley(arr, m0, m1)

    for axis, tag in ((1, "x"), (0, "y")):
        rr, cc, x0, sg = find_edges(img, axis)
        row[f"n_edges_{tag}"] = int(len(rr))
        for name, arr in arms.items():
            row[f"esf_{tag}_{name}"] = (esf_width(arr, axis, rr, cc, x0, sg)
                                        if len(rr) >= 40 else float("nan"))
        sr, sc, sg2 = find_strokes(img, axis)
        row[f"n_strokes_{tag}"] = int(len(sr))
        for name, arr in arms.items():
            row[f"stroke_{tag}_{name}"] = (stroke_amp(arr, axis, sr, sc, sg2)
                                           if len(sr) >= 40 else float("nan"))
    return row


# ---------------------------------------------------------------------------
# the circle, L000's designed-flat field, and the rotation mismatch
# ---------------------------------------------------------------------------


def circle_check(model, img: np.ndarray, n_seeds: int = 8) -> dict:
    """quality.circle_metrics on the FLOAT decode of L000, all four arms.

    Scaled to 0-255 before the call, as reconstruct.py does, because the ridge
    detector's absolute floor ("p99 < 0.005 means a blank frame") is written in
    those units. This is the float decode, not a re-read of the 8-bit thumbnail.

    THE NOISE ARM IS RUN WITH `n_seeds` DIFFERENT SEEDS and its spread is
    reported, because that spread is the resolution of the instrument. The
    circle is a physical invariant, but `circle_metrics` is an ESTIMATOR of it:
    it picks the strongest band-pass ridge pixel per angular bin, and how far
    that pixel moves when the picture is perturbed at all sets a floor on what
    any before/after difference can mean. A perturbation that carries no
    geometric information -- white noise at the same rms -- must not move the
    numbers by more than that floor, and any arm whose movement is inside the
    floor has not been measured, whichever way it went.
    """
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 255.0
    mu, sd = a.mean(), a.std()
    n = np.clip(n2n_mod.denoise_plane(model, (a - mu) / sd, _device()) * sd + mu, 0, 255)
    r = _rms(n - a)
    b, sigma = matched_blur(a, r)
    out = {"blur_sigma": sigma, "rms_change_255": r}
    for name, arr in (("id", a), ("n2n", n), ("blur", b)):
        m = quality.circle_metrics(arr)
        out[f"axis_ratio_{name}"] = float(m["axis_ratio"])
        out[f"radial_rms_{name}"] = float(m["radial_rms"])
        out[f"inliers_{name}"] = int(m["inliers"])
    ax, rr, inl = [], [], []
    for s in range(n_seeds):
        m = quality.circle_metrics(matched_noise(a, r, 1000 + s))
        ax.append(float(m["axis_ratio"]))
        rr.append(float(m["radial_rms"]))
        inl.append(int(m["inliers"]))
    out.update(axis_ratio_noise=float(np.mean(ax)), axis_ratio_noise_sd=float(np.std(ax)),
               radial_rms_noise=float(np.mean(rr)), radial_rms_noise_sd=float(np.std(rr)),
               inliers_noise=int(np.mean(inl)), n_seeds=n_seeds)
    return out


def rotation_sensitivity(model, frames: dict[str, np.ndarray], fids: list[str]) -> dict:
    """Is the network the same operator on a rotated frame?

    It should not be, and the question matters. The training augmentation was
    axis-preserving flips ONLY, deliberately, because the record's dominant
    structured noise is column-constant (n2n.py). `reconstruct.py` denoises
    AFTER applying Barry's quarter turn, so the ~60 shipped frames with a
    non-zero turn are fed to the network with their trace axis where their
    along-trace axis was. This measures that mismatch with no reference at all:
    the discrepancy between rotating-then-denoising and denoising-then-rotating,
    as a fraction of the correction itself.
    """
    rows = []
    for fid in fids:
        a = frames[fid]
        up = denoise(model, a)
        ro = np.rot90(denoise(model, np.rot90(a, 1)), -1)
        d1, d2 = up - a, ro - a
        row = {"frame": fid, "rms_upright": _rms(d1), "rms_rotated": _rms(d2),
               "rms_disagreement": _rms(d1 - d2),
               "relative": _rms(d1 - d2) / max(_rms(d1), 1e-12)}
        # and does the mismatch COST anything, on the same fixed measurements?
        reg = flat_regions(a)
        s0, s1, s2 = (flat_stats(x, reg)["sd"] for x in (a, up, ro))
        row["flat_sd_ratio_upright"] = s1 / s0 if s0 else float("nan")
        row["flat_sd_ratio_rotated"] = s2 / s0 if s0 else float("nan")
        for axis, tag in ((1, "x"), (0, "y")):
            rr, cc, x0, sg = find_edges(a, axis)
            if len(rr) >= 40:
                w0 = esf_width(a, axis, rr, cc, x0, sg)
                row[f"esf_{tag}_ratio_upright"] = esf_width(up, axis, rr, cc, x0, sg) / w0
                row[f"esf_{tag}_ratio_rotated"] = esf_width(ro, axis, rr, cc, x0, sg) / w0

        rows.append(row)
    agg = {f"mean_{k}": float(np.nanmean([r.get(k, np.nan) for r in rows]))
           for k in ("relative", "flat_sd_ratio_upright", "flat_sd_ratio_rotated",
                     "esf_x_ratio_upright", "esf_x_ratio_rotated",
                     "esf_y_ratio_upright", "esf_y_ratio_rotated")}
    return {"frames": rows, "mean_relative": agg["mean_relative"], **agg}


# ---------------------------------------------------------------------------
# aggregation and report
# ---------------------------------------------------------------------------


def _ratio(rows, key: str, arm: str):
    """Median per-frame ratio arm/id, and how many frames fell."""
    a = np.array([r.get(f"{key}_{arm}", np.nan) for r in rows], dtype=np.float64)
    b = np.array([r.get(f"{key}_id", np.nan) for r in rows], dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-9)
    if not m.any():
        return float("nan"), 0, 0
    r = a[m] / b[m]
    return float(np.median(r)), int((r < 1.0).sum()), int(m.sum())


def _med(rows, key: str) -> float:
    v = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")


def trade(rows, arm: str, tag: str = "x"):
    """Flat-field noise removed per unit of edge widening, per frame.

    The single scalar that expresses the CONJUNCTION n2n.py's colour result
    rested on: suppression where there is nothing to protect, divided by damage
    where the picture lives. Larger is better; a value below the matched blur's
    means the network is doing worse than smoothing on this frame.
    """
    out = []
    for r in rows:
        sd0, sd1 = r.get("flat_sd_id"), r.get(f"flat_sd_{arm}")
        w0, w1 = r.get(f"esf_{tag}_id"), r.get(f"esf_{tag}_{arm}")
        if not all(np.isfinite([sd0 or np.nan, sd1 or np.nan, w0 or np.nan, w1 or np.nan])):
            continue
        gain = 1.0 - sd1 / sd0
        cost = w1 / w0 - 1.0
        out.append((r["frame"], gain / cost if cost > 1e-6 else np.inf))
    return out


def run(quick: bool = False, json_out: Path | None = REPORT) -> dict:
    frames = load_frames()
    ids = sorted(frames)
    if quick:
        ids = ids[::7]
    frames = {i: frames[i] for i in ids}

    print(f"{len(ids)} frames; classifying blindly (pixels only, no titles)")
    feats = {i: features(frames[i]) for i in ids}
    art, score, thr = classify(feats)
    art_ids = [i for i in ids if art[i]]
    pho_ids = [i for i in ids if not art[i]]
    print(f"  line-art cluster {len(art_ids)}   photograph cluster {len(pho_ids)}"
          f"   (tile_occ split at {thr:.3f})")

    model = load_model()
    print("evaluating: id / n2n / matched blur / matched noise")
    rows = []
    for k, fid in enumerate(ids):
        r = evaluate_frame(model, fid, frames[fid], feats[fid])
        r["lineart"] = bool(art[fid])
        r["title_lineart"] = fid in TITLE_LINEART   # tier 2, evaluation only
        r["score"] = score[fid]
        r.update({f"f_{a}": b for a, b in feats[fid].items()})
        rows.append(r)
        if k % 20 == 0:
            print(f"  {k}/{len(ids)}")

    circ = circle_check(model, frames["L000"]) if "L000" in frames else {}
    rot = rotation_sensitivity(model, frames, art_ids[:8] + pho_ids[:8])

    out = {"n_frames": len(ids), "statistic": "tile_occ", "threshold": thr,
           "lineart_frames": art_ids, "photo_frames": pho_ids,
           "circle": circ, "rotation": rot, "rows": rows}
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {json_out}")
    return out


METRICS = (("flat-field sd (want DOWN)", "flat_sd"),
           ("flat-field mean (want SAME)", "flat_mean"),
           ("flat-field tilt (want SAME)", "flat_tilt"),
           ("histogram valley (want DOWN)", "valley"),
           ("ESF 10-90 across-trace (SAME)", "esf_x"),
           ("ESF 10-90 along-trace (SAME)", "esf_y"),
           ("stroke amp across-trace (SAME)", "stroke_x"),
           ("stroke amp along-trace (SAME)", "stroke_y"))


def report(out: dict) -> None:
    rows = out["rows"]
    art = [r for r in rows if r["lineart"]]
    pho = [r for r in rows if not r["lineart"]]
    tart = [r for r in rows if r["title_lineart"]]
    title = {f.id: (f.title or "") for f in catalog_mod.build().frames}

    print("\n" + "=" * 92)
    print("BLIND CLASSIFICATION -- one statistic on decoded pixels. Titles are EVALUATION.")
    print("=" * 92)
    for r in sorted(art, key=lambda r: -r["score"]):
        mark = "ok " if r["title_lineart"] else "FP "
        print(f"  {mark}{r['frame']}  score {r['score']:+5.2f}  tile_occ "
              f"{r['f_tile_occ']:.3f}  cov {r['f_tile_cov']:.2f}  flat "
              f"{r['f_flat_frac']:.2f}   {title.get(r['frame'], '')[:40]}")
    print("  --- title-line-art frames the blind rule MISSED ---")
    for r in sorted([x for x in pho if x["title_lineart"]], key=lambda r: -r["score"]):
        print(f"  FN {r['frame']}  score {r['score']:+5.2f}  tile_occ "
              f"{r['f_tile_occ']:.3f}   {title.get(r['frame'], '')[:40]}")
    tp = sum(1 for r in art if r["title_lineart"])
    print(f"  agreement with the title set: {tp} of {len(art)} in the cluster, "
          f"{tp} of {len(tart)} of the title set found "
          f"({len(art) - tp} false positives, {len(tart) - tp} missed). "
          f"OPTIMISTIC: the statistic was chosen after seeing the previous "
          f"version's errors (module docstring).")

    c = out.get("circle") or {}
    if c:
        print("\n" + "=" * 92)
        print("1. CALIBRATION CIRCLE, FLOAT DECODE OF L000")
        print("=" * 92)
        print(f"  all arms at rms change {c['rms_change_255']:.3f} of 255; "
              f"matched blur sigma {c['blur_sigma']:.3f} px")
        for k, f in (("axis_ratio", "{:.5f}"), ("radial_rms", "{:.4f}"), ("inliers", "{:.0f}")):
            vals = "  ".join(f"{a} " + f.format(c[f'{k}_{a}']) for a in ARMS)
            sdk = c.get(f"{k}_noise_sd")
            tail = f"  (noise arm over {c['n_seeds']} seeds, sd {sdk:.5f})" if sdk else ""
            print(f"    {k:<11s}  {vals}{tail}")
        for k in ("axis_ratio", "radial_rms"):
            sdk = c.get(f"{k}_noise_sd", 0.0)
            dn = abs(c[f"{k}_n2n"] - c[f"{k}_id"])
            dz = abs(c[f"{k}_noise"] - c[f"{k}_id"])
            print(f"    {k}: n2n moves it {dn:.5f}; a pure-noise perturbation of "
                  f"the same size moves it {dz:.5f} +- {sdk:.5f}. "
                  f"{'RESOLVED' if dn > dz + 2 * sdk else 'INSIDE THE INSTRUMENT FLOOR'}")

    print("\n" + "=" * 92)
    print("2-5. PER-CLASS MEDIANS. n2n, blur and noise are all at MATCHED rms change.")
    print("     ratio to the undenoised decode, and how many frames moved DOWN.")
    print("=" * 92)
    for name, g in (("LINE ART (blind cluster)", art), ("PHOTOGRAPHS (blind)", pho),
                    ("LINE ART (title set, tier 2 cross-check)", tart)):
        if not g:
            continue
        print(f"\n  {name}  n={len(g)}   rms change {_med(g,'rms_n2n'):.4f} of "
              f"black-white   matched blur sigma {_med(g,'blur_sigma'):.3f} px")
        for lbl, key in METRICS:
            line = f"    {lbl:<31s} id {_med(g, key+'_id'):8.4f}"
            for arm in ("n2n", "blur", "noise"):
                m, w, n = _ratio(g, key, arm)
                line += f"   {arm} {m:6.3f}x {w:3d}/{n}"
            print(line)
        for arm in ("n2n", "blur", "noise"):
            d = np.array([r.get(f"flat_dmean_{arm}", np.nan) for r in g], dtype=np.float64)
            d = d[np.isfinite(d)]
            if d.size:
                print(f"    flat-field mean SHIFT {arm:<6s} {np.median(d):+.5f} of "
                      f"black-white, up on {int((d > 0).sum())}/{d.size}")
        for lbl, key in (("across-trace", "esf_x"), ("along-trace", "esf_y")):
            w0 = _med(g, f"{key}_id")
            for arm in ("n2n", "blur"):
                m = _ratio(g, key, arm)[0]
                if np.isfinite(m) and np.isfinite(w0) and m > 1.0:
                    eq = (w0 / 2.5631) * math.sqrt(m * m - 1.0)
                    print(f"    edge widening {arm:<5s} {lbl:<13s} = a Gaussian of "
                          f"sigma {eq:.3f} px")
        for tag in ("x", "y"):
            tn = dict(trade(g, "n2n", tag))
            tb = dict(trade(g, "blur", tag))
            both = [k for k in tn if k in tb and np.isfinite(tn[k]) and np.isfinite(tb[k])]
            if both:
                print(f"    TRADE {tag} (flat-sd drop per unit of edge widening): "
                      f"n2n {np.median([tn[k] for k in both]):.3f}   "
                      f"blur {np.median([tb[k] for k in both]):.3f}   "
                      f"n2n better on {sum(tn[k] > tb[k] for k in both)}/{len(both)}")

    rot = out.get("rotation") or {}
    if rot:
        print("\n" + "=" * 92)
        print("6. ROTATION MISMATCH -- reconstruct.py denoises AFTER Barry's quarter turn")
        print("=" * 92)
        print(f"  mean |rot(f(x)) - f(rot(x))| / |f(x) - x| = {rot['mean_relative']:.3f}"
              f" over {len(rot['frames'])} frames -- the two orders are not the "
              f"same operator")
        print(f"    flat-field sd    upright {rot['mean_flat_sd_ratio_upright']:.4f}x"
              f"   applied-rotated {rot['mean_flat_sd_ratio_rotated']:.4f}x")
        for tag in ("x", "y"):
            print(f"    ESF 10-90 {tag}      upright "
                  f"{rot[f'mean_esf_{tag}_ratio_upright']:.4f}x   applied-rotated "
                  f"{rot[f'mean_esf_{tag}_ratio_rotated']:.4f}x")
        for r in rot["frames"][:5]:
            print(f"    {r['frame']}  upright {r['rms_upright']:.4f}  rotated "
                  f"{r['rms_rotated']:.4f}  disagreement {r['rms_disagreement']:.4f} "
                  f"({r['relative']:.2f}x)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--quick", action="store_true", help="every 7th frame")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--json", type=Path, default=REPORT)
    a = ap.parse_args(argv)
    if a.rebuild_cache:
        build_cache()
    if a.report:
        report(run(quick=a.quick, json_out=a.json))
    elif not a.rebuild_cache:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
