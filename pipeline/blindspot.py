"""Blind-spot denoising (Noise2Void / Noise2Self), with the blind spot's SHAPE
measured from the calibration circle instead of guessed.

    python -m pipeline.blindspot --calibrate       # the circle measurement alone
    python -m pipeline.blindspot --report          # train the arms, run the arbiter
    python -m pipeline.blindspot --linear          # closed form, per plane, no GPU
    python -m pipeline.blindspot --linear --hp 1.0 # ... on the fine grain only
    python -m pipeline.blindspot --circle-check    # the ring must not degrade
    python -m pipeline.blindspot --single          # one net per frame, no repeats

THE QUESTION. pipeline/n2n.py buys its denoiser from the twenty COLOUR images:
three filtered scans of one scene are three independent noisy observations, so
Noise2Noise applies. That is tier 1 and it works (+7.73 dB in the chroma-free
high band on scenes it never saw, and -- the claim blur cannot buy -- the low
band improves on 15/19 where blur worsens it on 17/19). But it only has
training data for 20 of the 116 images; on the other 96, scanned once, it
extrapolates. A route that needs no repeats would cover all 156 frames.

Blind-spot methods are that route in principle. Train a network to predict each
pixel from its NEIGHBOURS ONLY, with the pixel itself excluded from the
receptive field; the network then cannot learn that pixel's noise, so it
converges on the signal, from ONE noisy image and no clean target (Krull et al.
2019, Noise2Void; Batson & Royer 2019, Noise2Self, which states the premise as
J-invariance and proves it needs the noise to be independent across the
partition).

THE PREMISE THIS RECORD BREAKS, and the circle is how we find out by how much.
The independence premise is false here and was already known to be false in the
audio domain: the chain's noise on the content-free porch has lag-1
autocorrelation +0.931 (L002) and +0.986 (R040). Structured Noise2Void
(Broaddus, Krull, Weigert, Myers, ISBI 2020) is the published fix: replace the
one-pixel blind spot with a LINE-SHAPED one, oriented along the noise
correlation and long enough to cover it, so the correlated component cannot be
read off the surviving neighbours either. What StructN2V does not tell you is
how long the line must be on YOUR instrument -- their masks are chosen from the
visible streak direction and a hand-picked length. That is exactly the number
the calibration slide can supply.

WHAT THE CIRCLE SUPPLIES (tier 0; the record's cover states image 1 is a
circle, so the field's uniformity is the artifact's own claim, not ours).
Inside the ring, L000's field is uniform BY DESIGN, so every deviation is the
chain's error, observed over ~152,000 pixels in the image domain -- the domain
the network actually works in, not the audio domain. Two things come out:

  1. THE IMAGE-DOMAIN NOISE AUTOCORRELATION, measured (this file, --calibrate):
     along the trace (down a column of the 377x512 render) it does not decay at
     all -- 0.986 at lag 1 and still 0.78 at lag 24 -- because 61% of the noise
     power inside the ring is a PER-TRACE CONSTANT (the streak). Across traces
     it falls to 0.20 by lag 3, with the even/odd alternation the half-dot
     trace phase predicts (lag 2 correlates 0.62, higher than lag 1's 0.48).
     So the correlation length along the trace is not "a few pixels" that a
     StructN2V line could cover: it is the whole trace.

  2. THE DESIGN CRITERION, which is better than a correlation length: for a
     CANDIDATE blind-spot shape, how much of a pixel's own noise can still be
     predicted from the neighbours the shape leaves visible? Fit the best
     linear predictor of the centre from the surviving 17x17 receptive field
     on half the ring's pixels, score it on the other half. That R^2 IS the
     leak. R^2 -> 0 means the mask works; R^2 -> 1 means the network can
     reconstruct the pixel's noise from its context and will converge to the
     IDENTITY while its training loss falls perfectly normally.

     Measured, out-of-sample, on 120,840 sites (see --calibrate):

       blind spot           fine noise   2px    4px    16px  <- scale called noise
       1 px (plain N2V)         0.996   0.998  0.999  1.000
       vertical line +-1        0.780   0.717  0.864  0.951
       vertical line +-3        0.705   0.595  0.776  0.918
       vertical line +-8        0.506   0.257  0.607  0.858
       whole trace (w=0)        0.506   0.257  0.607  0.858
       whole trace +-1 trace    0.093   0.196  0.484  0.811
       whole trace +-2 traces   0.086   0.181  0.271  0.693
       whole trace +-4 traces   0.065   0.103  0.184  0.492

     Plain Noise2Void is dead on arrival on this artifact -- 0.996, i.e. the
     network can recover 99.6% of the pixel it is forbidden to see. A
     StructN2V line of any practical length is not enough either (and note the
     line of +-8 and the whole trace give the SAME number, because within a
     17 px receptive field they are the same mask -- the correlation does not
     end anywhere inside the network's reach). Only masking the WHOLE TRACE
     and its immediate neighbours brings the leak near zero, and only for the
     fine-grained part of the noise; the coarse part (the droop residual,
     which belongs to dedroop.py, not to a denoiser) stays predictable at
     every mask size and is therefore untouchable by any blind-spot method,
     which is worth knowing on its own.

  3. THE NOISE VARIANCE, which is the thing a single noisy image provably does
     not contain, and the reason this route needs a calibration target rather
     than merely benefiting from one. A blind-spot network outputs f(x), the
     best estimate of a pixel from context that IGNORES the pixel. That throws
     away a real measurement. The optimal estimator blends them back,
     y = a*x + (1-a)*f(x) with a = sc^2/(sn^2 + sc^2), where sc^2 is the
     blind prediction's error and sn^2 the pixel's noise. The network can
     measure their SUM from its own residual, E[(f-x)^2] = sc^2 + sn^2, and
     nothing in one noisy frame can split it. The circle splits it: inside the
     ring the true value is known, so sn is observed -- and per mask shape, the
     part of it the mask actually hides, (1 - leakage) x noise variance, since
     a leaked component is not independent of f and must not be counted twice.
     See `shrink()`. The measured values, in grey levels of 255:
     1 px 0.20, line +-3 3.13, whole trace 4.13, +-2 traces 6.06, +-4 7.80
     -- i.e. plain N2V's mask hides 0.2 grey levels of a 10.95-grey-level
     noise, so the circle says in advance that its output is worth nothing and
     the shrinkage will (correctly) discard it.
     Carried onto a z-scored plane through that frame's own contrast
     (data/cache/blindspot_contrast.json), which is the stationarity
     assumption a chain calibration is allowed to make and the only one the
     single circle cannot check.

  The mask this file calls CALIBRATED is therefore: the entire trace, plus
  MASK_W traces either side, with MASK_W = 2 read off the table above (the
  smallest width whose fine-noise leak is under 0.10 with margin). It is
  measured, not tuned: no arbiter number was consulted to pick it, and the
  runs report w = 0 alongside it so the choice can be seen.

DOES THE CALIBRATION TRANSFER OFF L000? There is only one circle, so this has
to be checked some other way, and caltarget.gap_lines gives one: the blank
traces in the gap BEFORE a frame decode through the same chain and must, like
the ring's interior, come out constant -- on a different frame, with no slide
involved (scratch/transfer, reproduced by running caltarget.gap_lines and this
file's noise_acf / leakage on the result). Measured on four frames, both
channels:

    frame   noise sd   per-trace constant   ACF along trace, lag 1 / 8 / 24
    L000 ring  10.95        60.7%             0.986  0.847  0.928
    L002        3.33        37.0%             0.958  0.946  0.923
    R040        6.05        72.3%             0.992  0.989  0.985
    L020       22.63        66.7%             0.988  0.859  0.800
    L055        7.26        11.6%             0.947  0.629  0.369

  What transfers, on every frame tested: the along-trace correlation is 0.95+
  at lag 1 and has not decayed by lag 24, and a large fraction of the noise is
  a per-trace constant. So the finding that kills plain Noise2Void here is a
  property of the CHAIN, not of L000 -- the plain-N2V leak measured on these
  gap fields is 0.81 to 0.99. What does NOT transfer is the LEVEL: the noise
  ranges over 3.3 to 22.6 grey levels and the per-trace share over 12 to 72%,
  so the single sigma the circle supplies is a chain average, not a per-frame
  truth, and the shrinkage below is correspondingly approximate. (These gap
  fields cannot check the ACROSS-trace numbers or the fine-grain leak: the
  usable gap traces are not contiguous and the blank lines are resampled by
  linear interpolation rather than dot-locked, which changes the fine grain.)

ARCHITECTURAL BLIND SPOT, NOT MASKING. N2V and StructN2V blank the blind spot
in the INPUT and put the loss on it. That has a train/test mismatch -- at
inference nothing is blanked -- and here the mismatch would be severe, because
a whole-trace mask blanks 5 of every 16 columns. Instead the blind spot is
built into the ARCHITECTURE (Laine et al. 2019): the output at column c is
computed from two strictly one-sided branches, one that can see only columns
<= c-w-1 and one only columns >= c+w+1, combined by a 1x1 head. Four branches
(left, right, up, down) give a rectangular blind spot instead, which is how the
plain-N2V and StructN2V-line controls in this file are built. Consequences,
all good: the blind spot is EXACT rather than statistical; the loss is on every
pixel, not on 3% of them; and the network runs at inference exactly as it
trained. Padding inside the branches is ZERO, never reflect -- reflect padding
at the frame edge would mirror a column from the far side of the blind spot
straight into it, and the blind spot would silently stop being blind.
NOT residual: n2n.py's `x + net(x)` would hand the network its own input and
destroy the blind spot. The output is the network's, whole.

THE ARBITER IS n2n.py's, UNMODIFIED, so the numbers line up. n2n.evaluate_triplet
and n2n.algebra are imported and called; the fold split is n2n's `ns[f::folds]`;
the corpus is n2n's cached, registered, z-scored triplets. What changes is only
what training is allowed to use: n2n pairs separations, this file never does --
every training example is ONE plane predicting ITSELF through a blind spot, so
the method transfers verbatim to the 96 single-scan images. Evaluation still
withholds a separation and scores prediction of it, so both routes are judged
on the same unseen measurements.

CONTROLS in every arm, because high-band suppression is cheap:
  * isotropic Gaussian blur sigma=1 (n2n's own control, already in the table);
  * a MATCHED-STRENGTH blur, whose sigma is solved per arm so that the blur
    moves the plane by the same rms as the network does -- the honest question
    is not "does it beat some blur" but "does it beat the blur that changes
    the picture as much as it does";
  * a matched-strength HORIZONTAL-ONLY blur, which is the shape-matched
    control for a whole-trace blind spot: a network that can only see other
    traces has an obvious way to cheat, and this is it.

THE FAILURE MODE, CHECKED EXPLICITLY. A blind-spot network whose mask leaks
converges to the identity: output minus input goes to zero and the held-out MSE
sits exactly on the raw plane's, which reads as "a small, safe improvement"
rather than as nothing. Every arm reports `delta_rms` (rms(output - input) /
rms(input), over the same crop the arbiter uses) and `id_gap` (held-out MSE
minus the identity's). An arm with delta_rms below IDENTITY_RMS is declared
IDENTITY-COLLAPSED in the output and its MSE is not offered as a win.

RESULT (2026-08, data/master/384kHzStereo.wav; n2n.py's arbiter, its 19
triplets, image 8 excluded as it excludes it).

  THE ANSWER, first, because it is a negative: NO. The calibration circle
  cannot buy what the colour repeats buy. On the shared arbiter the best
  blind-spot arm reaches 0.03348 against the identity's 0.03410 and n2n.py's
  0.03217, gives +1.34 dB in the chroma-free high band where n2n gives +7.73
  and a plain Gaussian blur gives +5.60, and beats its own matched-strength
  blur on 9 of 19 scenes -- a coin toss, i.e. the one thing n2n earns and this
  does not. Given the held-out answer itself (tier 3, section B) the entire
  family still loses to a blurred baseline. The route is a null.

  WHAT THE CIRCLE DID DELIVER is the reason the null is worth having: it
  predicts the failure IN ADVANCE and quantitatively, from 152,000 pixels and
  36 seconds of arithmetic, without training anything. It says plain
  Noise2Void will converge to the identity here (leak 0.996) -- and the
  closed-form optimum turns out to BE the identity to within 0.57% of the
  plane's rms. It says the mask that would work must cover the whole trace --
  and a network so masked cannot predict its own input to better than 0.109 of
  the plane's variance, against a removable noise worth about 0.002. That
  measurement is reusable, costs nothing, and is the deliverable here; the
  network is not.

THE FAILURE, and the circle is what makes it legible rather than merely
observed.

A. CLOSED FORM, PER PLANE, ALL 19 SCENES (`--linear`). The best LINEAR
   blind-spot predictor of each plane from itself -- no training budget, no
   fold structure, every scene unseen by construction, and the purest form of
   the assigned route: one image, no repeats, no clean target.

     arm      raw MSE   vs id   shrunk   vs id   vs matched blur   lo    hi dB
     n2v      0.03409   19/19   0.03408  19/19        0/19        15/19  +0.03
     line3    0.03729    3/19   0.03588   3/19        2/19         6/19  -3.53
     trace0   0.03750    3/19   0.03663   3/19        2/19         5/19  -3.70
     trace2   0.11526    0/19   0.05210   0/19        0/19         0/19  +1.63
     identity 0.03410                              (n2n.py: 0.03217, 19/19)

   Read the first row. The optimal linear blind-spot predictor of a plane from
   its punctured neighbourhood IS THE PLANE, to within 0.57% of its rms: it
   "wins" 19/19 by 0.03%, which is the identity wearing a rosette. That is the
   collapse the circle predicted from a leak of 0.996, arrived at here in
   closed form, so it cannot be blamed on optimisation. And the CALIBRATED mask
   -- the one the circle's own table selects -- is the worst arm on the record:
   3.4x the identity's error raw, still 1.5x after the shrinkage, 0/19
   everywhere, chroma retention 21.8 because it has stopped resembling the
   plane at all.

B. THE ORACLE CEILING (tier 3; alpha picked using the held-out answer, so no
   claim rests on it -- it exists to rule out "the calibration constant was
   just wrong"). Blend each blind prediction with the plane at the BEST
   possible weight:

     n2v 0.03408   line3 0.03300   trace0 0.03289   trace2 0.03327
     the same oracle applied to a plain sigma=1 Gaussian blur:  0.03264
     n2n.py, with no oracle of any kind:                        0.03217

   Every blind-spot arm, given the answer, still loses to a Gaussian blur given
   the answer (trace0 beats it on 7/19 scenes, the rest on 0-4/19). And n2n
   beats all of them without being told anything. So the negative is not a
   mis-set sigma and not an undertrained net: there is no blend of any of these
   blind predictions with the measurement that reaches what the colour repeats
   give for free.

C. THE BEST VERSION OF THE ROUTE, which the circle's own table argues for
   (`--linear --hp 1.0`). The leak is only small for the FINE grain, so give
   the blind spot only the fine grain: predict the detail above sigma=1 px and
   pass the smooth part through untouched, so the low band cannot be damaged
   by construction. The blind spot is dilated by 3 traces to stay J-invariant
   through the split.

     arm         raw MSE   shrunk   vs matched blur   lo    hi dB   ORACLE
     trace0-hp   0.03924   0.03674       1/19        1/19  +4.62   0.03272
     trace2-hp   0.03929   0.03784       1/19        2/19  +5.30   0.03270
     sigma=1 blur control                                  +5.60   0.03264
     n2n.py                                                +7.73   0.03217

   This is the sharpest result in the file, because it is the failure in n2n's
   own vocabulary. The band-limited arms DO reach blur-level high-band
   suppression (+4.6 and +5.3 dB against blur's +5.60). And in the low band,
   where the picture lives, they are worse than the raw plane on 17-18 of 19
   scenes -- which is the blur control's signature exactly (blur is worse
   there on 17/19). They beat their own matched blur on 1 of 19. The
   conjunction n2n earns -- blur-level suppression high up WHILE the low band
   improves, 15/19 -- is not merely missed here, it is inverted.

D. TRAINED NETWORKS, n2n's PROTOCOL (`--report`, 2 folds x 1500 steps, all 19
   scenes, each evaluated as a scene its network never saw; every arm passed
   the finite-difference blind-spot check before training).

   raw output, which is what N2V and StructN2V actually produce:
     arm      d_rms   MSE       vs id   lo wins   hi dB   chroma kept
     n2v      0.135   0.03400   13/19    4/19     +2.71     1.13
     line3    0.185   0.03761    4/19    6/19     -4.65     1.41
     trace0   0.194   0.03731    4/19    6/19     -4.65     1.35
     trace2   0.334   0.11669    0/19    0/19     -1.88    21.72
     identity 0.000   0.03410

   after the circle's shrinkage:
     arm      alpha  d_rms   MSE       vs id   lo      hi dB   beats its
                                                              matched blur
     n2v      0.996  0.0004  0.03409   19/19   19/19   +0.03      0/19
     line3    0.599  0.0608  0.03348   17/19   12/19   +1.34      9/19
     trace0   0.422  0.0930  0.03439   11/19   10/19   -0.69     12/19
     trace2   0.579  0.1260  0.04208    0/19    0/19   +2.56      5/19
     n2n.py                  0.03217   19/19   15/19   +7.73     19/19

   The best arm on the record is line3 -- the StructN2V line, NOT the
   circle-calibrated whole-trace mask -- at 0.03348 against the identity's
   0.03410: a 1.8% held-out improvement on 17 of 19 unseen scenes, real but
   a third of n2n's 5.7%, and it beats its own matched-strength blur on 9 of
   19 scenes, which is a coin toss. Its high-band gain is +1.34 dB where a
   plain sigma=1 Gaussian gets +5.60 and n2n gets +7.73. n2v collapses to the
   identity under the shrinkage exactly as the leak of 0.996 said it would:
   0.03409 against 0.03410, flagged in the output rather than reported as a
   19/19 win. And the calibrated mask, trace2, is last by a wide margin.

   One negative that is not about performance: chroma retention comes out
   ABOVE 1 for every raw arm (1.13-1.41, and 21.7 for trace2). These
   predictors do not desaturate to buy a full-band win -- they do the
   opposite, injecting channel-specific structure that was not in the
   luminance. For trace2 that number alone is a disqualification.

E. CONSTRAINT: THE CIRCLE'S OWN GEOMETRY MUST NOT DEGRADE (`--circle-check`,
   ring refitted from the denoised L000 by circle.fit_ring, against the same
   fuse-path decode as the baseline):

     arm        axis ratio     radial rms      ring width
     decode       1.0050        0.863 px         5.50 px
     n2v          1.0050        0.845 px         5.50 px    ok
     line3        1.0045        0.863 px         5.60 px    ok
     trace0       1.0047        0.874 px         5.60 px    ok
     trace2       1.0039        0.955 px         6.10 px    FAILS  (+11% rms,
                                                            the ring smeared
                                                            across traces)

   The calibrated mask is the one that breaks the calibration target's own
   geometry, which is as clean a self-refutation as this file can produce.

WHY, in one line each, all three measured here:
  * the mask that blocks the noise is wider than the picture's own correlation
    length. The circle says hide the whole trace and +-2 traces; a network so
    masked cannot predict its own input better than 0.109 of the plane's
    variance (the trace2 training loss), while the noise it is trying to
    remove is worth about 0.002 (n2n's held-out gain over the identity). The
    cure is fifty times the size of the disease.
  * the noise and the picture occupy the SAME band. The decode resamples ~231
    dots to 377 rows and decimates 384 kHz to 96 kHz, so by the time a frame
    is an image its noise is band-limited to roughly where the 1977 camera's
    signal is. Blind-spot methods need a band where noise lives and signal
    does not; this chain does not provide one.
  * what IS separable -- the per-trace streak, 61% of the ring's noise power --
    is exactly what a blind spot cannot touch, because it is constant down the
    trace and therefore perfectly predictable from any pixel of the same
    trace. It needs an estimator that knows it is a per-trace constant
    (destripe.py's problem), not one that is forbidden to look.

NEGATIVES AND LIMITS OF THIS NEGATIVE, none omitted:

  * THE CIRCLE'S NOISE LEVEL DOES NOT TRANSFER WELL, and this file's own
    shrinkage is the casualty. L000's field is uniform WHITE, which drives the
    AC-coupling droop harder than any photograph does, so the ring's error is
    dominated by smooth droop residual -- the most predictable component there
    is. The leak it reports is therefore an OVER-estimate for picture frames
    (gap-line leak for a 1 px blind spot: 0.813 on L055 against 0.996 on the
    ring), so sigma_indep comes out too small for small masks and the
    shrinkage stands aside when it should not; and the absolute level is a
    chain average over frames whose own noise ranges 3.3 to 22.6 grey levels,
    so for the wide masks it comes out too large and the shrinkage over-trusts
    the network. Both errors were seen: n2v's alpha lands at 0.99 (throwing
    away a real if tiny win) and trace2's at 0.43 (keeping far too much of a
    prediction that is 3x worse than the plane). The ORACLE row in B is what
    bounds this: correcting the constant perfectly still does not reach a
    blurred baseline, so the mis-transfer is not what killed the route.
  * COMPUTE. The trained-network table in C is 2 folds x 1500 steps, not
    n2n.py's 4 x 3000: the GPU was shared and throughput fell about 8x below
    the benchmark mid-run. Stated rather than hidden. Two things bound the
    damage: the losses are flat over the last quarter of every run (printed),
    and the closed-form table in A needs no training at all and agrees.
  * ONLY ONE CIRCLE EXISTS. Everything here rests on one frame's uniform
    field. The gap-line check (four frames, both channels) confirms the
    along-trace non-decay and the large per-trace-constant share that kill
    plain Noise2Void, but it cannot check the across-trace numbers or the
    fine-grain leak, because the usable gap traces are not contiguous.
  * WHAT WOULD CHANGE THE ANSWER. A blind spot works where noise and signal
    occupy different bands. If a future dedroop/destripe pass removes the
    per-trace level trajectories FIRST, the residual might be white enough for
    a 1 px blind spot to have something to remove -- the leakage measurement
    in this file is exactly the test to re-run on that residual, and it costs
    36 seconds. This module's real deliverable is that test, not its network.

EVERY NUMBER ABOVE IS IN data/cache/, per row and per scene, not just as a
mean: blindspot_calibration.json (the circle: ACF, leakage, sigma_indep),
blindspot_report.json (the trained arms), blindspot_linear.json and
blindspot_linear_hp.json (the closed-form and band-limited arms, with the
oracle), blindspot_contrast.json (the per-frame decoder-unit std that carries
the circle's absolute sigma onto z-scored planes).

PROVENANCE: TIER 0 + TIER 1, except where labelled. The circle is tier 0 (the
cover says image 1 is a circle). The networks and the closed-form predictors
are tier 1: fitted to this record's own frames, no Earth photograph, no
pretrained weights, no external corpus, no reference image; an alien holding
the record could run all of it. Nothing under docs/reference is read, and
nothing in this file is imported by any shipping module. The single exception
is the ORACLE row of section B and its printout, which chooses a blend weight
using the held-out answer: it is tier 3 by construction, exists only to bound
the negative, and no claim anywhere rests on it.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from . import n2n as n2n_mod

REPO = Path(__file__).resolve().parent.parent
CAL_JSON = REPO / "data" / "cache" / "blindspot_calibration.json"

#: The calibration frame. The record's cover says image 1 is a circle; this is
#: the only frame on the record whose content is stated by the artifact.
CAL_FRAME = "L000"

#: Receptive-field radius used for the leakage measurement, matching the
#: 17x17 field of n2n.py's denoiser so the two are the same size of network.
LEAK_RF = 8

#: Below this, output-minus-input is indistinguishable from the identity and
#: the arm's MSE is not reported as a win. 1% of the plane's rms; the smallest
#: change any arm here makes that is NOT collapse is ~8x this.
IDENTITY_RMS = 0.01


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# 1. the circle: what shape must the blind spot be?
# --------------------------------------------------------------------------


def _masked_smooth(E: np.ndarray, M: np.ndarray, sigma: float) -> np.ndarray:
    num = gaussian_filter(np.where(M, E, 0.0), sigma, mode="reflect")
    den = gaussian_filter(M.astype(float), sigma, mode="reflect")
    return num / np.maximum(den, 1e-6)


def calibration_field() -> tuple[np.ndarray, np.ndarray]:
    """L000's error field and the mask of pixels the record's spec pins down.

    Decoded through fuse.decode_frame -- the SAME path that built n2n.py's
    cached planes -- so the correlation measured here is the correlation in
    the domain the network will actually see. (The 96 kHz decimation and the
    231-dot -> 377-row resample are part of what makes the noise correlated;
    measuring at 384 kHz would understate it.) The ideal slide, the ring fit
    and the field mask are caltarget.build_ideal's, unmodified.
    """
    from . import caltarget as cal_mod
    from . import catalog as catalog_mod
    from . import circle as circle_mod
    from . import fuse as fuse_mod

    cat = catalog_mod.build()
    img = fuse_mod.decode_frame(cat.by_id(CAL_FRAME))
    ring = circle_mod.fit_ring(img)
    if ring is None:
        raise RuntimeError("the calibration ring did not fit; cannot proceed")
    ideal = cal_mod.build_ideal(img, ring=ring)
    return img - ideal.img, ideal.field


def noise_acf(E: np.ndarray, M: np.ndarray, axis: int, maxlag: int = 24,
              sigma: float = 16.0) -> np.ndarray:
    """Mask-aware autocorrelation of the ring interior's deviation.

    Structure coarser than `sigma` is removed first: on the calibration slide
    that is the droop residual, a different defect with a different owner
    (dedroop.py), and leaving it in would report the droop's correlation
    instead of the noise's.
    """
    R = np.where(M, E - _masked_smooth(E, M, sigma), 0.0)
    W = M.astype(float)
    out = []
    for L in range(maxlag + 1):
        sl_a = (slice(None, -L or None), slice(None)) if axis == 0 else \
               (slice(None), slice(None, -L or None))
        sl_b = (slice(L, None), slice(None)) if axis == 0 else \
               (slice(None), slice(L, None))
        a, b, w = R[sl_a], R[sl_b], W[sl_a] * W[sl_b]
        n = w.sum()
        num = (a * b * w).sum() / n
        d1, d2 = (a * a * w).sum() / n, (b * b * w).sum() / n
        out.append(float(num / math.sqrt(d1 * d2)))
    return np.array(out)


def blind_offsets(kind: str, w: int = 0, h: int = 0,
                  rf: int = LEAK_RF) -> set[tuple[int, int]]:
    """The (dy, dx) offsets a given blind-spot shape hides, within +-rf."""
    if kind == "point":
        return {(0, 0)}
    if kind == "line":            # StructN2V: vertical segment, 2h+1 tall
        return {(dy, 0) for dy in range(-h, h + 1)}
    if kind == "trace":           # whole trace, 2w+1 traces wide
        return {(dy, dx) for dy in range(-rf, rf + 1)
                for dx in range(-w, w + 1)}
    raise ValueError(kind)


def leakage(R: np.ndarray, M: np.ndarray, hidden: set[tuple[int, int]],
            rf: int = LEAK_RF, seed: int = 0) -> float:
    """Out-of-sample R^2 of the best linear predictor of a pixel's noise from
    the neighbours a blind spot leaves visible. This is the number that decides
    whether a blind-spot network can work at all: it is an upper bound on how
    much noise the mask removes (1 - R^2) and a lower bound on how much of the
    pixel the network can rebuild from context without learning any signal.

    Fitted on a random half of the ring's interior and scored on the other
    half, so the ~280 free coefficients cannot inflate it.
    """
    H, W = R.shape
    offs = [(dy, dx) for dy in range(-rf, rf + 1) for dx in range(-rf, rf + 1)
            if (dy, dx) not in hidden]
    ys, xs = np.mgrid[rf:H - rf, rf:W - rf]
    ys, xs = ys.ravel(), xs.ravel()
    ok = np.ones(ys.shape, bool)
    for dy in range(-rf, rf + 1):
        for dx in range(-rf, rf + 1):
            ok &= M[ys + dy, xs + dx]
    ys, xs = ys[ok], xs[ok]
    perm = np.random.default_rng(seed).permutation(len(ys))
    ys, xs = ys[perm], xs[perm]
    half = len(ys) // 2
    A = np.stack([R[ys + dy, xs + dx] for dy, dx in offs]
                 + [np.ones(len(ys))], axis=1)
    y = R[ys, xs]
    coef, *_ = np.linalg.lstsq(A[:half], y[:half], rcond=None)
    resid = y[half:] - A[half:] @ coef
    return float(1.0 - resid.var() / y[half:].var())


def calibrate(verbose: bool = True, json_out: Path | None = CAL_JSON) -> dict:
    """Everything the circle says about blind-spot geometry."""
    E, M = calibration_field()
    g = lambda v: v * 255.0  # decoder units -> grey levels of 255

    rep: dict = {"frame": CAL_FRAME, "field_px": int(M.sum())}
    if verbose:
        print(f"CALIBRATION CIRCLE {CAL_FRAME}: {int(M.sum())} pixels whose true "
              f"value the record's cover fixes (uniform field, ring excluded)\n")

    # --- what the deviation is made of
    R16 = np.where(M, E - _masked_smooth(E, M, 16.0), 0.0)
    colm = np.array([R16[:, j][M[:, j]].mean() if M[:, j].sum() > 40 else np.nan
                     for j in range(E.shape[1])])
    ok = np.isfinite(colm)
    rep["noise_sd_grey"] = float(g(R16[M].std()))
    rep["trace_constant_share"] = float(np.nanvar(colm[ok]) / R16[M].var())
    if verbose:
        print(f"  noise sd inside the ring            "
              f"{rep['noise_sd_grey']:6.2f} grey levels")
        print(f"  of which a PER-TRACE CONSTANT       "
              f"{rep['trace_constant_share']*100:5.1f}% of the power\n")

    # --- autocorrelation, the two axes of the render
    av = noise_acf(E, M, axis=0)
    ah = noise_acf(E, M, axis=1)
    rep["acf_along_trace"] = av.tolist()
    rep["acf_across_trace"] = ah.tolist()
    if verbose:
        print("  image-domain noise autocorrelation (the domain the net sees):")
        print("    lag        " + " ".join(f"{i:6d}" for i in [1, 2, 3, 4, 6, 8, 12, 16, 24]))
        for nm, a in (("along trace ", av), ("across trace", ah)):
            print(f"    {nm} " + " ".join(f"{a[i]:6.3f}" for i in
                                          [1, 2, 3, 4, 6, 8, 12, 16, 24]))
        print("    along-trace correlation NEVER decays -- the streak is a "
              "per-trace constant.")
        print("    across-trace lag 2 > lag 1: the half-dot trace phase, "
              "so traces of\n    the same parity resemble each other more "
              "than adjacent ones.\n")

    # --- the leak, per candidate shape and per scale-called-noise
    shapes = [("n2v", "1 px (plain N2V)", blind_offsets("point")),
              ("line1", "line +-1 (StructN2V)", blind_offsets("line", h=1)),
              ("line3", "line +-3 (StructN2V)", blind_offsets("line", h=3)),
              ("line8", "line +-8 (StructN2V)", blind_offsets("line", h=8)),
              ("trace0", "whole trace w=0", blind_offsets("trace", w=0)),
              ("trace1", "whole trace w=1", blind_offsets("trace", w=1)),
              ("trace2", "whole trace w=2", blind_offsets("trace", w=2)),
              ("trace4", "whole trace w=4", blind_offsets("trace", w=4))]
    scales = (1.0, 2.0, 4.0, 16.0)
    table, sig, sig_all = {}, {}, {}
    for s in scales:
        Rs = np.where(M, E - _masked_smooth(E, M, s), 0.0)
        for key, name, hid in shapes:
            r2 = leakage(Rs, M, hid)
            table.setdefault(name, {})[f"hp{s:g}"] = r2
            sig_all.setdefault(f"hp{s:g}", {})[key] = float(
                math.sqrt(max(1.0 - r2, 0.0) * Rs[M].var()))
            if s == 16.0:
                # The noise this shape actually HIDES, in the decoder's own
                # absolute units: the leaked part is not independent of the
                # network's prediction and must not be counted as removable.
                # This is the sigma_n the shrinkage needs and that a single
                # noisy frame cannot supply.
                sig[key] = float(math.sqrt(max(1.0 - r2, 0.0) * Rs[M].var()))
    rep["leakage"] = table
    rep["sigma_indep_abs"] = sig
    rep["sigma_indep_abs_by_scale"] = sig_all
    if verbose:
        print("  NOISE LEAKAGE R^2 -- how much of the hidden pixel's own noise the")
        print("  network can still rebuild from what it may see. 1.0 = it will")
        print("  learn the IDENTITY; 0.0 = the mask works. Out-of-sample.")
        print("    blind spot              " + "".join(
            f"{'hp'+format(s,'g'):>10s}" for s in scales) + "   sigma_indep")
        for key, name, _ in shapes:
            print(f"    {name:22s}  " + "".join(
                f"{table[name][f'hp{s:g}']:10.3f}" for s in scales)
                  + f"{sig[key]*255:11.2f} grey")
        print("    (columns: the scale above which deviation is called noise --")
        print("     hp1 is the fine grain, hp16 includes the droop residual,")
        print("     which is dedroop.py's defect and no blind spot can reach it.)\n")
        print(f"  CALIBRATED CHOICE: whole trace +- {MASK_W} traces "
              f"(leak {table[f'whole trace w={MASK_W}']['hp1']:.3f} on the fine "
              f"grain).\n  Chosen from this table alone; no arbiter number was "
              f"consulted.\n")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rep, indent=1))
    return rep


#: Read off the leakage table above: the smallest whole-trace width whose
#: fine-grain leak is comfortably under 0.10. Fixed here, before any arbiter
#: run; w = 0, 1 and 4 are reported alongside so the choice is visible.
MASK_W = 2


# --------------------------------------------------------------------------
# 2. the network: an EXACT blind spot, built into the architecture
# --------------------------------------------------------------------------


def _oriented(x: torch.Tensor, d: int) -> torch.Tensor:
    """Map direction d onto 'left', so one implementation serves all four.

    0 left, 1 right, 2 up, 3 down. `_unoriented` is its inverse.
    """
    if d == 0:
        return x
    if d == 1:
        return torch.flip(x, (-1,))
    y = x.transpose(-1, -2)
    return y if d == 2 else torch.flip(y, (-1,))


def _unoriented(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0:
        return x
    if d == 1:
        return torch.flip(x, (-1,))
    if d == 2:
        return x.transpose(-1, -2)
    return torch.flip(x, (-1,)).transpose(-1, -2)


class OneSided(torch.nn.Module):
    """A stack of 3x3 convolutions whose output at column c depends ONLY on
    input columns <= c - 1 - gap, and on every row.

    Each layer pads 2 columns on the left, none on the right, convolves valid
    and drops the overhang, so output[c] = f(in[c-2], in[c-1], in[c]) -- reach
    2 columns to the left, and CRUCIALLY no minimum gap. The single gap is
    opened once, at the end, by shifting the branch right by gap+1 (Laine et
    al.'s construction). Getting this wrong is easy and silent: a first draft
    padded 3 and dropped the centre in every layer, so the gaps accumulated and
    the blind spot was 15 traces wide instead of 1 -- the network still trained,
    the loss still fell, and the mask under test was not the mask measured. The
    finite-difference check in verify_blind() is what caught it, which is why it
    runs before every report and raises rather than warns.

    Rows are padded by replication (this branch has no vertical blind spot);
    columns are padded with ZEROS, never reflect -- reflect would fold column
    +1 into position -1 at the frame edge and quietly unblind the spot on the
    first columns of every frame.
    """

    def __init__(self, width: int, depth: int, gap: int):
        super().__init__()
        self.gap = int(gap)
        chans = [1] + [width] * (depth - 1) + [width]
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(chans[i], chans[i + 1], 3) for i in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, cv in enumerate(self.convs):
            h = F.pad(h, (0, 0, 1, 1), mode="replicate")   # rows: free
            h = F.pad(h, (2, 0, 0, 0), mode="constant", value=0.0)
            h = cv(h)[..., :x.shape[-1]]
            if i < len(self.convs) - 1:
                h = F.relu(h)
        h = F.pad(h, (self.gap + 1, 0, 0, 0), mode="constant",
                  value=0.0)[..., :x.shape[-1]]
        return h


class BlindSpotNet(torch.nn.Module):
    """Predict every pixel from context that provably excludes a rectangle.

    `dirs` picks the branches and therefore the blind spot's shape:
      (0, 1)          left+right only            -> the WHOLE TRACE, 2*gx+1 wide
      (0, 1, 2, 3)    all four half-planes       -> a (2*gy+1) x (2*gx+1) box
    The visible set is the union of the branches' half-planes, so the blind
    spot is the complement of that union -- exact, not statistical.

    Capacity is matched to n2n.py's denoiser (~130k parameters, 17 px reach per
    layer stack): the comparison must be about the training signal, not about
    who had the bigger network. No normalisation layers, same reason as n2n.py.
    """

    def __init__(self, dirs=(0, 1), gx: int = MASK_W, gy: int = 0,
                 width: int = 32, depth: int = 8):
        super().__init__()
        self.dirs = tuple(dirs)
        gaps = {0: gx, 1: gx, 2: gy, 3: gy}
        self.branches = torch.nn.ModuleList(
            [OneSided(width, depth, gaps[d]) for d in self.dirs])
        n = width * len(self.dirs)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(n, 64, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [_unoriented(b(_oriented(x, d)), d)
                 for d, b in zip(self.dirs, self.branches)]
        return self.head(torch.cat(feats, dim=1))


#: The arms. Branch width is set so every arm has ~135k parameters, within 5%
#: of n2n.py's denoiser: the comparison is about what the training signal can
#: support, not about who was given the larger network.
ARMS: dict[str, dict] = {
    # name            branches                 blind spot        what it is
    "n2v":     dict(dirs=(0, 1, 2, 3), gx=0, gy=0, width=22),  # 1x1   plain N2V
    "line3":   dict(dirs=(0, 1, 2, 3), gx=0, gy=3, width=22),  # 7x1   StructN2V
    "trace0":  dict(dirs=(0, 1), gx=0, width=32),              # 377x1
    "trace1":  dict(dirs=(0, 1), gx=1, width=32),              # 377x3
    "trace2":  dict(dirs=(0, 1), gx=MASK_W, width=32),         # 377x5 CALIBRATED
    "trace4":  dict(dirs=(0, 1), gx=4, width=32),              # 377x9
}


def verify_blind(cfg: dict, size: int = 96, seed: int = 1) -> dict:
    """Prove the blind spot is exactly where it is claimed, by finite difference.

    Not a formality: an off-by-one in a pad, or a reflect where a zero belongs,
    turns a blind-spot network into an autoencoder, and its training loss looks
    BETTER when that happens. Here: perturb ONE input pixel by a large amount
    and read the output AT THE PREDICTED PIXEL. Every offset the arm claims to
    hide must move it by exactly zero; the nearest offset outside the claimed
    shape must move it by something. Run in float64 on the CPU, so "exactly
    zero" means exactly zero and not "small on this accelerator".
    """
    torch.manual_seed(seed)
    net = BlindSpotNet(**cfg).double().eval()
    x = torch.randn(1, 1, size, size, dtype=torch.double)
    c = size // 2
    gx = cfg["gx"]
    gy = cfg.get("gy", 0)
    full_trace = tuple(cfg["dirs"]) == (0, 1)

    def resp(dy: int, dx: int) -> float:
        xp = x.clone()
        xp[0, 0, c + dy, c + dx] += 10.0
        with torch.no_grad():
            return abs(float(net(xp)[0, 0, c, c]) - float(net(x)[0, 0, c, c]))

    hidden = [(0, 0), (0, gx), (0, -gx)]
    visible = [(0, gx + 1), (0, -gx - 1)]
    if full_trace:                       # the blind spot is the whole trace
        hidden += [(8, gx), (-24, 0), (31, -gx)]
    else:                                # a (2gy+1) x (2gx+1) box
        hidden += [(gy, gx), (-gy, -gx), (gy, -gx)]
        visible += [(gy + 1, 0), (-gy - 1, 0)]
    out = {"hidden_max": max(resp(*o) for o in hidden),
           "visible_min": min(resp(*o) for o in visible)}
    out["blind"] = out["hidden_max"] == 0.0 and out["visible_min"] > 1e-9
    return out


# --------------------------------------------------------------------------
# 3. training: ONE plane, predicting ITSELF through the blind spot
# --------------------------------------------------------------------------


@dataclass
class TrainCfg:
    steps: int = 3000
    batch: int = 16
    patch: int = 64
    lr: float = 1e-3
    seed: int = 0


def train(planes: list[np.ndarray], cfg_arm: dict, cfg: TrainCfg,
          device: torch.device, verbose: bool = False
          ) -> tuple[BlindSpotNet, list[float]]:
    """Self-supervised fit. The target IS the input -- no repeat, no pair, no
    clean image anywhere: one plane, its own blind-spot prediction of itself.

    Shape copied from n2n.py so the two routes are comparable -- batch 16,
    64 px patches, Adam at 1e-3 with the same x0.3 decay at 60% and 85%, same
    LOSS_MARGIN. Nothing here was tuned on the arbiter. The STEP COUNT is the
    one place the runs below fall short of n2n.py's 3000; see the module's
    RESULT section for what was actually run and why.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = BlindSpotNet(**cfg_arm).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(0.6 * cfg.steps), int(0.85 * cfg.steps)], gamma=0.3)
    m = n2n_mod.LOSS_MARGIN
    tens = [torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))
            for p in planes]
    h, w = tens[0].shape
    hist: list[float] = []
    for step in range(cfg.steps):
        xs = []
        for b in rng.integers(0, len(tens), cfg.batch):
            r0 = int(rng.integers(0, h - cfg.patch))
            c0 = int(rng.integers(0, w - cfg.patch))
            xp = tens[b][r0:r0 + cfg.patch, c0:c0 + cfg.patch]
            # Axis-preserving flips only, n2n.py's reason: the streak noise is
            # column-constant and a transpose would invent a noise model that
            # is not on the record -- and here it would also transpose the
            # blind spot away from the trace it was measured to cover.
            if rng.random() < 0.5:
                xp = torch.flip(xp, (0,))
            if rng.random() < 0.5:
                xp = torch.flip(xp, (1,))
            xs.append(xp)
        X = torch.stack(xs)[:, None].to(device)
        opt.zero_grad(set_to_none=True)
        out = net(X)
        loss = torch.mean((out - X)[..., m:-m, m:-m] ** 2)
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss.item()))
        if verbose and (step % 500 == 0 or step == cfg.steps - 1):
            print(f"    step {step:5d}  blind-spot loss {hist[-1]:.5f}")
    net.eval()
    return net, hist


# --------------------------------------------------------------------------
# 4. evaluation: n2n.py's arbiter, plus the controls it does not carry
# --------------------------------------------------------------------------


def matched_blur(plane: np.ndarray, target_rms: float, horizontal: bool = False
                 ) -> tuple[np.ndarray, float]:
    """The Gaussian that moves this plane by the same rms the network did.

    Bisection on sigma. `horizontal` blurs across traces only -- the
    shape-matched control for a whole-trace blind spot, which is the cheap
    thing such a network could do instead of denoising.
    """
    p = plane.astype(np.float64)
    ref = p.std()
    lo, hi = 0.05, 12.0
    f = lambda s: gaussian_filter(p, (0.0, s) if horizontal else s)
    for _ in range(18):          # 12/2^18: sigma resolved far past what matters
        mid = 0.5 * (lo + hi)
        if np.sqrt(np.mean((f(mid) - p) ** 2)) / ref < target_rms:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return f(s), s


def arm_mses(reg: np.ndarray, arms: dict[str, list[np.ndarray]]) -> dict:
    """Score extra predictors on the arbiter's own machinery.

    n2n.evaluate_triplet knows about five arms and cannot be extended without
    editing a file this task forbids editing, so the loop is re-assembled here
    -- but out of n2n.py's OWN pieces: its EVAL_CROP, its HI/LO bands, its
    band_mse, its tone_fields, its rotation over the three withheld planes and
    its averaging. `score` re-scores the network through both paths and
    asserts they agree to 1e-12, so if this ever drifts from the arbiter it
    fails loudly instead of quietly reporting a different measurement.
    """
    sl = np.s_[n2n_mod.EVAL_CROP:-n2n_mod.EVAL_CROP,
               n2n_mod.EVAL_CROP:-n2n_mod.EVAL_CROP]
    bands = {"full": None, "hi": n2n_mod.HI_BAND, "lo": n2n_mod.LO_BAND}
    acc = {a: {b: [] for b in bands} for a in arms}
    for k in range(3):
        i, j = [p for p in range(3) if p != k]
        maps = {p: n2n_mod.tone_fields(reg[p].astype(np.float64),
                                       reg[k].astype(np.float64)) for p in (i, j)}
        tgt = reg[k].astype(np.float64)
        for a, src in arms.items():
            preds = [maps[p][0] * src[p] + maps[p][1] for p in (i, j)]
            for bn, band in bands.items():
                acc[a][bn].append(float(np.mean(
                    [n2n_mod.band_mse((q - tgt)[sl], band) for q in preds])))
    return {f"m_{a}_{b}": float(np.mean(v[b])) for a, v in acc.items() for b in bands}


def oracle_blend(reg: np.ndarray, pred: list[np.ndarray]) -> dict:
    """The BEST blend of the plane with a prediction, chosen using the answer.

    Tier 3 by construction and labelled ORACLE everywhere it is printed: alpha
    is picked to minimise the held-out error itself, which no shippable method
    may do. It is here for one reason -- to separate "the circle's sigma was
    wrong" from "no blend of this prediction with the plane helps at all". If
    the oracle alpha cannot beat the identity either, the route is dead
    independently of how well the calibration transfers, and that is a much
    stronger statement than a single mis-set constant.

    Exact, not swept: with e(alpha) = alpha*e_x + (1-alpha)*e_f the band error
    is the quadratic alpha^2 A + 2 alpha(1-alpha) B + (1-alpha)^2 C, and the
    cross term comes from n2n.band_mse itself via
    B = (band_mse(e_x + e_f) - A - C)/2, so no new metric is introduced.
    """
    sl = np.s_[n2n_mod.EVAL_CROP:-n2n_mod.EVAL_CROP,
               n2n_mod.EVAL_CROP:-n2n_mod.EVAL_CROP]
    bands = {"full": None, "hi": n2n_mod.HI_BAND, "lo": n2n_mod.LO_BAND}
    acc = {b: [0.0, 0.0, 0.0] for b in bands}
    cnt = 0
    for k in range(3):
        i, j = [p for p in range(3) if p != k]
        maps = {p: n2n_mod.tone_fields(reg[p].astype(np.float64),
                                       reg[k].astype(np.float64)) for p in (i, j)}
        tgt = reg[k].astype(np.float64)
        for p in (i, j):
            a, b = maps[p]
            ex = ((a * reg[p].astype(np.float64) + b) - tgt)[sl]
            ef = ((a * pred[p] + b) - tgt)[sl]
            for bn, band in bands.items():
                A = n2n_mod.band_mse(ex, band)
                C = n2n_mod.band_mse(ef, band)
                B = 0.5 * (n2n_mod.band_mse(ex + ef, band) - A - C)
                acc[bn][0] += A
                acc[bn][1] += B
                acc[bn][2] += C
            cnt += 1
    A, B, C = (v / cnt for v in acc["full"])
    den = A - 2 * B + C
    al = float(np.clip((C - B) / den, 0.0, 1.0)) if den > 1e-30 else 1.0
    out = {"oracle_alpha": al}
    for bn in bands:
        a, b, c = (v / cnt for v in acc[bn])
        out[f"m_oracle_{bn}"] = float(al * al * a + 2 * al * (1 - al) * b
                                      + (1 - al) ** 2 * c)
    return out


def shrink(reg_i: np.ndarray, den_i: np.ndarray, sigma_n: float
           ) -> tuple[np.ndarray, float]:
    """Blend the blind prediction back toward the measurement, by the amount
    the CIRCLE says the measurement is worth. This is the step the blind spot
    cannot take on its own, and the one place the calibration target is not
    replaceable by anything else on the record.

    A blind-spot network returns f(x), the best estimate of the pixel from
    context that BY CONSTRUCTION ignores the pixel itself. It therefore throws
    away a real measurement -- the pixel -- whose only defect is its noise. Let
    e_f = f - s have variance sc^2 and n = x - s have variance sn^2. The blind
    spot makes them independent (that is exactly what it is for), so the best
    linear combination is

        y = a*x + (1-a)*f,      a = sc^2 / (sn^2 + sc^2)

    with error variance sn^2 sc^2 / (sn^2 + sc^2), strictly below both. The
    network can measure sc^2 + sn^2 = E[(f - x)^2] from its own output, but it
    cannot split that sum: sn^2 is exactly the quantity a single noisy image
    does not contain. The circle does contain it -- inside the ring the true
    value is known, so the noise is observed directly -- and, per arm, the part
    of it the blind spot actually hides (1 - leakage R^2, since a leaked
    component is not independent of f and must not be counted twice).

    ASSUMPTION, named: sn is measured on L000 and applied to every frame in
    the frame's own contrast units, i.e. the chain's noise is taken to be
    stationary across the record. That is what a chain calibration means and
    it is the reason this route covers all 156 frames; it is also the biggest
    thing the circle cannot check, because there is only one circle.
    """
    d = den_i - reg_i
    total = float(np.mean(d * d))            # sc^2 + sn^2, from the net alone
    sc2 = max(total - sigma_n ** 2, 0.0)
    a = sc2 / (sigma_n ** 2 + sc2) if (sigma_n ** 2 + sc2) > 0 else 1.0
    return a * reg_i + (1.0 - a) * den_i, float(a)


def linear_blind(plane: np.ndarray, hidden: set[tuple[int, int]],
                 rf: int = LEAK_RF, sub: int = 2) -> np.ndarray:
    """The BEST LINEAR blind-spot predictor of a plane from itself.

    Same J-invariance as the network -- every pixel predicted from a 17x17
    context with the blind rectangle removed, fitted on this plane alone with
    no repeat and no clean target -- but solved in closed form instead of by
    gradient descent. It exists to remove one objection from the network's
    results: if a trained net does no better than this, the net is not
    undertrained, the INFORMATION is not there. It is also a ceiling of sorts
    for the linear part of the problem, and it costs a second per plane.

    Fitted on every other pixel (`sub`) and applied everywhere; the border of
    width rf keeps its input, and n2n's EVAL_CROP of 24 discards it anyway.
    """
    p = np.asarray(plane, dtype=np.float64)
    H, W = p.shape
    offs = [(dy, dx) for dy in range(-rf, rf + 1) for dx in range(-rf, rf + 1)
            if (dy, dx) not in hidden]
    ys, xs = np.mgrid[rf:H - rf:sub, rf:W - rf:sub]
    ys, xs = ys.ravel(), xs.ravel()
    A = np.stack([p[ys + dy, xs + dx] for dy, dx in offs]
                 + [np.ones(len(ys))], axis=1)
    coef, *_ = np.linalg.lstsq(A, p[ys, xs], rcond=None)
    out = np.full_like(p, coef[-1])
    for c, (dy, dx) in zip(coef[:-1], offs):
        out[rf:H - rf, rf:W - rf] += c * p[rf + dy:H - rf + dy,
                                           rf + dx:W - rf + dx]
    out[:rf] = p[:rf]
    out[H - rf:] = p[H - rf:]
    out[:, :rf] = p[:, :rf]
    out[:, W - rf:] = p[:, W - rf:]
    return out


#: Extra trace-widening for the band-limited arms. The detail image d = p -
#: G_sigma(p) at a neighbouring trace still contains the centre trace through
#: the smoothing kernel, so a blind spot on d is only a blind spot on p if it
#: is wider than the kernel's reach. 3*sigma, rounded up.
HP_GUARD = 3


def linear_blind_hp(plane: np.ndarray, hidden: set[tuple[int, int]],
                    sigma: float = 1.0, rf: int = LEAK_RF) -> np.ndarray:
    """Blind-spot prediction of the DETAIL only; the smooth part is passed
    through untouched.

    This is the version of the route the circle's own table argues for, and it
    would be unfair to answer the question without trying it. The leakage table
    says the mask can only block the FINE grain (0.09 at hp1 for a whole trace
    +-1, but 0.81 at hp16): everything coarser is predictable from context, so
    a blind-spot network is bound to reproduce it whatever it does. Fine --
    then only ask it to handle the band where its mask works. p is split into
    G_sigma(p) + d, only d goes through the blind predictor, and the smooth
    part is added back unchanged, so nothing below ~1/(2 pi sigma) cyc/px can
    be damaged at all. The price is that the per-trace streak, which is the
    dominant defect and is broadband, mostly survives in the smooth part.

    The blind spot is DILATED by HP_GUARD in both axes, because d already mixes
    the centre pixel into its neighbours through the kernel; without that the
    arm would not be J-invariant and its result would be meaningless. (For the
    whole-trace shapes the dilation only widens them in traces, since they
    already span every row.)
    """
    p = np.asarray(plane, dtype=np.float64)
    s = gaussian_filter(p, sigma)
    g = HP_GUARD
    wide = {(dy + a, dx + b) for (dy, dx) in hidden
            for a in range(-g, g + 1) for b in range(-g, g + 1)}
    return s + linear_blind(p - s, wide, rf=rf)


class _Const(torch.nn.Module):
    """Wraps a precomputed prediction so the arbiter can score it unchanged."""

    def __init__(self, arr: np.ndarray):
        super().__init__()
        self.register_buffer("a", torch.from_numpy(
            np.ascontiguousarray(arr, dtype=np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a[None, None].to(x.device).expand_as(x)


class _Dispatch(torch.nn.Module):
    """Hands n2n.evaluate_triplet a per-plane model without touching it.

    Only needed by --single, where each plane has its own network fitted to
    itself; the fold runs pass the one model directly, exactly as n2n does.
    """

    #: Fingerprint pixels. NOT the plane's sum or variance: the cached planes
    #: are z-scored, so those are 0 and 1 for every one of them and the lookup
    #: would silently return the wrong network.
    PROBE = ((37, 61), (150, 250), (300, 99), (211, 430), (88, 501))

    def __init__(self, planes: np.ndarray, nets: list[torch.nn.Module]):
        super().__init__()
        self.keys = [np.array([float(p[r, c]) for r, c in self.PROBE])
                     for p in planes]
        self.nets = torch.nn.ModuleList(nets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = np.array([float(x[..., r, c]) for r, c in self.PROBE])
        d = [float(np.abs(k - q).sum()) for q in self.keys]
        assert min(d) < 1e-4, "plane fingerprint did not match any fitted net"
        return self.nets[int(np.argmin(d))](x)


def score(model: torch.nn.Module, reg: np.ndarray, device: torch.device,
          sigma_n: list[float] | None = None) -> dict:
    """n2n.py's held-out separation test, unmodified, plus the extra controls.

    n2n.evaluate_triplet builds every predictor of a withheld plane from the
    other two, applies the SAME tone map to an arm and to its baseline, and
    returns MSEs in the three bands plus chroma retention. It is imported and
    called; this file only adds arms it does not know about, through the same
    pieces, and checks the agreement.
    """
    r = n2n_mod.evaluate_triplet(model, reg, device)
    den = [n2n_mod.denoise_plane(model, reg[i], device) for i in range(3)]
    ref = [reg[i].astype(np.float64) for i in range(3)]
    drms = [float(np.sqrt(np.mean((den[i] - ref[i]) ** 2)) / ref[i].std())
            for i in range(3)]

    arms: dict[str, list[np.ndarray]] = {"chk": den}
    mb = [matched_blur(ref[i], drms[i]) for i in range(3)]
    hb = [matched_blur(ref[i], drms[i], horizontal=True) for i in range(3)]
    arms["mblur"] = [p for p, _ in mb]
    arms["hblur"] = [p for p, _ in hb]
    alpha = [1.0] * 3
    if sigma_n is not None:
        sh = [shrink(ref[i], den[i], sigma_n[i]) for i in range(3)]
        arms["shrink"] = [p for p, _ in sh]
        alpha = [a for _, a in sh]
        srms = [float(np.sqrt(np.mean((arms["shrink"][i] - ref[i]) ** 2))
                      / ref[i].std()) for i in range(3)]
        sb = [matched_blur(ref[i], srms[i]) for i in range(3)]
        arms["sblur"] = [p for p, _ in sb]
    r.update(arm_mses(reg, arms))
    r.update(oracle_blend(reg, den))
    # The same oracle applied to the plain blur control, so the ORACLE row is
    # compared against an equally-oracular classical estimator and not against
    # a handicapped one.
    r.update({k.replace("oracle", "oracleblur"): v for k, v in oracle_blend(
        reg, [gaussian_filter(ref[i], 1.0) for i in range(3)]).items()})

    # The re-assembled loop must reproduce the arbiter's own number for the
    # network exactly; if it ever does not, every extra arm is untrustworthy.
    for b in ("full", "hi", "lo"):
        assert abs(r[f"m_chk_{b}"] - r[f"m_n2n1_{b}"]) < 1e-12, \
            f"extra-arm scoring diverged from n2n.evaluate_triplet ({b})"

    r.update({f"hi_{k}": v for k, v in n2n_mod.algebra(r, "hi").items()})
    r["id_gap"] = r["m_n2n1_full"] - r["m_id_full"]
    r["delta_rms"] = float(np.mean(drms))
    r["mblur_sigma"] = float(np.mean([s for _, s in mb]))
    r["hblur_sigma"] = float(np.mean([s for _, s in hb]))
    r["alpha"] = float(np.mean(alpha))
    if sigma_n is not None:
        r["shrink_rms"] = float(np.mean(srms))
        r["sblur_sigma"] = float(np.mean([s for _, s in sb]))
    return r


# --------------------------------------------------------------------------
# carrying the circle's ABSOLUTE noise level onto z-scored planes
# --------------------------------------------------------------------------

CONTRAST = REPO / "data" / "cache" / "blindspot_contrast.json"


def plane_sigmas(image_n: int, sigma_abs: float) -> list[float]:
    """The circle's independent-noise rms, expressed in each plane's z-units.

    n2n.py's cached planes are z-scored, which discards the frame's contrast;
    the circle's noise is an ABSOLUTE quantity in the decoder's black-to-white
    units. The bridge is each frame's own std in those units, decoded once by
    the same fuse.decode_frame the cache was built with (data/cache/
    blindspot_contrast.json). A low-contrast frame is noisier in z-units, which
    is exactly right: the same chain noise buys more damage on a flat picture.
    """
    if not CONTRAST.exists():
        raise FileNotFoundError(
            f"{CONTRAST} missing; it is built once by decoding the colour "
            f"frames and taking each plane's std in decoder units")
    c = json.loads(CONTRAST.read_text())
    return [sigma_abs / c[f"{image_n}:{i}"] for i in range(3)]


def _hi_gain(m: dict, arm: str) -> float:
    n = 2.0 * (m["m_id_hi"] - m["m_avg_hi"])
    res = m[f"m_{arm}_hi"] - m["m_id_hi"] + n
    return 10.0 * math.log10(n / res) if n > 0 and res > 0 else float("nan")


# --------------------------------------------------------------------------


def run(arms: list[str], folds: int = 4, steps: int = 3000, seed: int = 0,
        quick: bool = False, json_out: Path | None = None) -> dict:
    device = _device()
    planes = n2n_mod.load_triplets()
    ns = sorted(planes)
    if not CAL_JSON.exists():
        calibrate(verbose=False)
    sig_abs = json.loads(CAL_JSON.read_text())["sigma_indep_abs"]
    print(f"device: {device};  {len(ns)} triplets (n2n.py's cache and its "
          f"image-8 exclusion)")
    print(f"folds: {folds}  steps: {steps}  arms: {', '.join(arms)}")
    print("circle-measured independent noise each blind spot hides "
          "(grey levels of 255): "
          + "  ".join(f"{a} {sig_abs[a]*255:.2f}" for a in arms) + "\n")

    print("BLIND-SPOT VERIFICATION (finite difference; the centre MUST NOT move "
          "the output)")
    for a in arms:
        p = verify_blind(ARMS[a])
        print(f"  {a:8s} response to hidden pixels {p['hidden_max']:.2e}  "
              f"to the nearest visible one {p['visible_min']:.3e}   "
              f"{'BLIND' if p['blind'] else '*** NOT BLIND ***'}")
        if not p["blind"]:
            raise RuntimeError(f"arm {a}: the blind spot is not blind; no "
                               f"conclusion may be drawn from it")
    print()

    results: dict[str, list[dict]] = {a: [] for a in arms}
    n_folds = 1 if quick else folds
    for f in range(n_folds):
        eval_ns = ns[f::folds]
        train_ns = [n for n in ns if n not in eval_ns]
        # Single planes only. Never a pair, never a triplet: 45 independent
        # noisy images, each its own target through the blind spot. This is the
        # whole point -- the identical recipe runs on the 96 single-scan frames.
        corpus = [planes[n][i] for n in train_ns for i in range(3)]
        print(f"fold {f}: {len(corpus)} single planes to train on, "
              f"evaluate on unseen {eval_ns}")
        for a in arms:
            net, hist = train(corpus, ARMS[a], TrainCfg(steps=steps, seed=seed + f),
                              device, verbose=False)
            q = [int(len(hist) * f) for f in (0.0, 0.25, 0.5, 0.75)] + [-1]
            sm = [float(np.mean(hist[max(0, i - 50):i + 50])) for i in q[:-1]]
            sm.append(float(np.mean(hist[-100:])))
            print(f"  arm {a:8s} loss " + " -> ".join(f"{v:.4f}" for v in sm))
            for n in eval_ns:
                r = {"n": n, "fold": f, "arm": a}
                r.update(score(net, planes[n], device, plane_sigmas(n, sig_abs[a])))
                results[a].append(r)
                coll = r["delta_rms"] < IDENTITY_RMS
                print(f"    img {n:3d}  d_rms {r['delta_rms']:.4f}"
                      f"{' IDENTITY' if coll else '        '}  "
                      f"id {r['m_id_full']:.5f}  bs {r['m_n2n1_full']:.5f}  "
                      f"mblur(s={r['mblur_sigma']:.2f}) {r['m_mblur_full']:.5f}  "
                      f"hblur {r['m_hblur_full']:.5f}  |  shrink a="
                      f"{r['alpha']:.3f} d_rms {r['shrink_rms']:.4f} "
                      f"{r['m_shrink_full']:.5f} vs blur {r['m_sblur_full']:.5f}")
        print()

    summary = {}
    nrow = len(results[arms[0]])
    print("=" * 100)
    print(f"SUMMARY -- n2n.py's arbiter: predict a withheld separation, "
          f"{nrow} unseen scenes")
    print("\nA. the blind-spot network's raw output (what N2V / StructN2V "
          "actually produce)")
    print(f"  {'arm':9s}{'d_rms':>8s}{'full MSE':>10s}{'vs id':>9s}"
          f"{'lo wins':>9s}{'hi dB':>8s}{'mblur dB':>10s}{'hblur dB':>10s}"
          f"{'chroma':>8s}")
    for a in arms:
        rows = results[a]
        mean = lambda k: float(np.mean([r[k] for r in rows]))
        w = lambda arm, b: sum(r[f"m_{arm}_{b}"] < r[f"m_id_{b}"] for r in rows)
        g = lambda arm: float(np.nanmean([_hi_gain(r, arm) for r in rows]))
        summary[a] = {
            "n": nrow, "m_id_full": mean("m_id_full"), "m_id_lo": mean("m_id_lo"),
            "m_avg_full": mean("m_avg_full"), "chroma": mean("chroma_retention"),
            "raw": {"delta_rms": mean("delta_rms"), "m_full": mean("m_n2n1_full"),
                    "m_lo": mean("m_n2n1_lo"), "m_hi": mean("m_n2n1_hi"),
                    "wins_vs_id": w("n2n1", "full"), "wins_lo": w("n2n1", "lo"),
                    "hi_gain_db": g("n2n1"), "m_mblur_full": mean("m_mblur_full"),
                    "mblur_gain_db": g("mblur"), "hblur_gain_db": g("hblur"),
                    "m_hblur_full": mean("m_hblur_full"),
                    "mblur_sigma": mean("mblur_sigma"),
                    "identity_collapsed": mean("delta_rms") < IDENTITY_RMS},
            "shrunk": {"alpha": mean("alpha"), "delta_rms": mean("shrink_rms"),
                       "m_full": mean("m_shrink_full"), "m_lo": mean("m_shrink_lo"),
                       "m_hi": mean("m_shrink_hi"),
                       "wins_vs_id": w("shrink", "full"), "wins_lo": w("shrink", "lo"),
                       "hi_gain_db": g("shrink"), "m_sblur_full": mean("m_sblur_full"),
                       "sblur_gain_db": g("sblur"), "sblur_sigma": mean("sblur_sigma"),
                       "beats_matched_blur": sum(
                           r["m_shrink_full"] < r["m_sblur_full"] for r in rows),
                       "identity_collapsed": mean("shrink_rms") < IDENTITY_RMS}}
        s = summary[a]["raw"]
        print(f"  {a:9s}{s['delta_rms']:8.4f}{s['m_full']:10.5f}"
              f"{s['wins_vs_id']:5d}/{nrow:<3d}{s['wins_lo']:6d}/{nrow:<3d}"
              f"{s['hi_gain_db']:8.2f}{s['mblur_gain_db']:10.2f}"
              f"{s['hblur_gain_db']:10.2f}{summary[a]['chroma']:8.2f}")
    print(f"  {'identity':9s}{0.0:8.4f}{summary[arms[0]]['m_id_full']:10.5f}")

    print("\nB. the same network SHRUNK toward the measurement by the circle's "
          "noise variance")
    print("   (alpha = weight on the raw pixel; alpha -> 1 means the circle "
          "says the blind\n    prediction is worth almost nothing and the "
          "denoiser should stand aside)")
    print(f"  {'arm':9s}{'alpha':>8s}{'d_rms':>8s}{'full MSE':>10s}{'vs id':>9s}"
          f"{'lo wins':>9s}{'hi dB':>8s}{'m.blur':>10s}{'blur dB':>9s}"
          f"{'beat blur':>11s}")
    for a in arms:
        s = summary[a]["shrunk"]
        print(f"  {a:9s}{s['alpha']:8.3f}{s['delta_rms']:8.4f}{s['m_full']:10.5f}"
              f"{s['wins_vs_id']:5d}/{nrow:<3d}{s['wins_lo']:6d}/{nrow:<3d}"
              f"{s['hi_gain_db']:8.2f}{s['m_sblur_full']:10.5f}"
              f"{s['sblur_gain_db']:9.2f}{s['beats_matched_blur']:8d}/{nrow:<3d}")
    print()
    for a in arms:
        for tag in ("raw", "shrunk"):
            s = summary[a][tag]
            if s["identity_collapsed"]:
                print(f"  {a} ({tag}): IDENTITY COLLAPSE -- the output moves the "
                      f"plane by {s['delta_rms']*100:.2f}% of its rms and the "
                      f"held-out MSE sits on the raw plane's "
                      f"({s['m_full']:.5f} vs {summary[a]['m_id_full']:.5f}). "
                      f"That is not a small win, it is nothing -- which is what "
                      f"the circle's leakage table predicted for this mask.")
    print("\nn2n.py, same arbiter, same 19 scenes, for reference:")
    print("  identity 0.03410   blur(s=1) 0.03928   n2n 0.03217 (19/19)   "
          "2-plane mean 0.02546")
    print("  hi band +7.73 dB (blur +5.60)   low band 15/19   chroma 0.93")

    out = {"summary": summary, "rows": {a: results[a] for a in arms}}
    if json_out:
        json_out.write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwrote {json_out}")
    return out


def run_single(arm: str = "trace2", steps: int = 1500, seed: int = 0,
               n_scenes: int = 6, json_out: Path | None = None) -> dict:
    """The version with NO repeats at all: one network per plane, fitted to
    that plane alone, then judged on the same held-out separation.

    This is the configuration that would run on the 96 single-scan images, so
    it is the one that answers the coverage question. Its cost: each network
    sees 193k pixels instead of 8.7M, at a reduced step budget.
    """
    device = _device()
    planes = n2n_mod.load_triplets()
    ns = sorted(planes)[:n_scenes]
    sig_abs = json.loads(CAL_JSON.read_text())["sigma_indep_abs"][arm]
    print(f"SINGLE-FRAME MODE (arm {arm}): every network is fitted to ONE plane "
          f"and nothing else.\n{len(ns)} scenes x 3 planes = {3*len(ns)} "
          f"independent fits, {steps} steps each.\n")
    rows = []
    for n in ns:
        reg = planes[n]
        nets = []
        for i in range(3):
            net, hist = train([reg[i]], ARMS[arm], TrainCfg(steps=steps, seed=seed),
                              device)
            nets.append(net)
        r = {"n": n, "arm": arm}
        r.update(score(_Dispatch(reg, nets), reg, device, plane_sigmas(n, sig_abs)))
        rows.append(r)
        print(f"  img {n:3d}  d_rms {r['delta_rms']:.4f}  id {r['m_id_full']:.5f}  "
              f"bs {r['m_n2n1_full']:.5f}  mblur {r['m_mblur_full']:.5f}  |  "
              f"shrink a={r['alpha']:.3f} {r['m_shrink_full']:.5f} vs blur "
              f"{r['m_sblur_full']:.5f}  lo "
              f"{'better' if r['m_shrink_lo'] < r['m_id_lo'] else 'WORSE '}")
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    out = {"arm": arm, "steps": steps, "n": len(rows),
           "delta_rms": mean("delta_rms"), "m_full": mean("m_n2n1_full"),
           "m_id_full": mean("m_id_full"), "m_mblur_full": mean("m_mblur_full"),
           "m_shrink_full": mean("m_shrink_full"),
           "m_sblur_full": mean("m_sblur_full"), "alpha": mean("alpha"),
           "wins": sum(r["m_n2n1_full"] < r["m_id_full"] for r in rows),
           "wins_shrink": sum(r["m_shrink_full"] < r["m_id_full"] for r in rows),
           "wins_shrink_blur": sum(r["m_shrink_full"] < r["m_sblur_full"]
                                   for r in rows),
           "wins_lo": sum(r["m_shrink_lo"] < r["m_id_lo"] for r in rows),
           "hi_gain_db": float(np.nanmean([_hi_gain(r, "shrink") for r in rows])),
           "chroma": mean("chroma_retention"), "rows": rows}
    print(f"\n  single-frame {arm}: raw {out['m_full']:.5f}, shrunk "
          f"{out['m_shrink_full']:.5f} vs identity {out['m_id_full']:.5f} "
          f"(shrunk beats it {out['wins_shrink']}/{out['n']}, beats its matched "
          f"blur {out['wins_shrink_blur']}/{out['n']}, low band "
          f"{out['wins_lo']}/{out['n']}, hi {out['hi_gain_db']:+.2f} dB)")
    if json_out:
        json_out.write_text(json.dumps(out, indent=1, default=float))
    return out


def run_linear(arms: list[str], hp: float = 0.0,
               json_out: Path | None = None) -> dict:
    """The closed-form blind-spot predictor on every scene, same arbiter.

    No training, no GPU, no fold structure needed: each plane's predictor is
    fitted to that plane and to nothing else, so every scene is already an
    unseen scene by construction. This is the control that separates "the
    network was not trained enough" from "the blind spot removes the
    information".
    """
    device = torch.device("cpu")
    planes = n2n_mod.load_triplets()
    cal = json.loads(CAL_JSON.read_text())
    # Band-limited arms are calibrated against the noise IN THEIR OWN BAND:
    # the sigma that matters is the independent noise of the residual they
    # actually see, not of the whole field.
    sig_abs = (cal["sigma_indep_abs_by_scale"][f"hp{hp:g}"] if hp
               else cal["sigma_indep_abs"])
    shape = {"n2v": ("point",), "line3": ("line", 0, 3), "trace0": ("trace", 0),
             "trace1": ("trace", 1), "trace2": ("trace", MASK_W),
             "trace4": ("trace", 4)}
    print("CLOSED-FORM BLIND-SPOT PREDICTOR -- least squares, fitted per plane "
          "to that\nplane alone. Same arbiter, same controls, no training "
          "budget to blame."
          + (f"\nBAND-LIMITED: only the detail above sigma={hp:g} px is "
             f"predicted; the smooth\npart passes through untouched and the "
             f"blind spot is widened by {HP_GUARD} traces.\n" if hp else "\n"))
    out: dict[str, dict] = {}
    for a in arms:
        rows = []
        hid = blind_offsets(*shape[a])
        for n in sorted(planes):
            reg = planes[n]
            den = [linear_blind_hp(reg[i], hid, hp) if hp
                   else linear_blind(reg[i], hid) for i in range(3)]
            r = {"n": n, "arm": a}
            r.update(score(_Dispatch(reg, [_Const(d) for d in den]), reg,
                           device, plane_sigmas(n, sig_abs[a])))
            rows.append(r)
        mean = lambda k: float(np.mean([r[k] for r in rows]))
        w = lambda arm, b: sum(r[f"m_{arm}_{b}"] < r[f"m_id_{b}"] for r in rows)
        out[a] = {"n": len(rows), "delta_rms": mean("delta_rms"),
                  "alpha": mean("alpha"), "m_id_full": mean("m_id_full"),
                  "m_full": mean("m_n2n1_full"), "m_shrink_full": mean("m_shrink_full"),
                  "m_mblur_full": mean("m_mblur_full"),
                  "m_sblur_full": mean("m_sblur_full"),
                  "wins_raw": w("n2n1", "full"), "wins_shrink": w("shrink", "full"),
                  "wins_shrink_lo": w("shrink", "lo"),
                  "beats_matched_blur": sum(r["m_shrink_full"] < r["m_sblur_full"]
                                            for r in rows),
                  "hi_gain_db": float(np.nanmean([_hi_gain(r, "shrink")
                                                  for r in rows])),
                  "chroma": mean("chroma_retention"),
                  "oracle_alpha": mean("oracle_alpha"),
                  "m_oracle_full": mean("m_oracle_full"),
                  "m_oracle_lo": mean("m_oracle_lo"),
                  "m_oracleblur_full": mean("m_oracleblur_full"),
                  "oracle_wins": sum(r["m_oracle_full"] < r["m_id_full"]
                                     for r in rows),
                  "oracle_beats_oracleblur": sum(
                      r["m_oracle_full"] < r["m_oracleblur_full"] for r in rows),
                  "rows": rows}
        s = out[a]
        if s["delta_rms"] < IDENTITY_RMS:
            print(f"  {a:8s} IDENTITY COLLAPSE: the optimal blind-spot "
                  f"predictor of this plane IS this plane, to "
                  f"{s['delta_rms']*100:.2f}% of its rms. Its 'win' is not one.")
        print(f"  {a:8s} raw d_rms {s['delta_rms']:.4f} MSE {s['m_full']:.5f} "
              f"({s['wins_raw']}/{s['n']} vs id {s['m_id_full']:.5f})  |  "
              f"shrunk a={s['alpha']:.3f} {s['m_shrink_full']:.5f} "
              f"({s['wins_shrink']}/{s['n']}), beats its matched blur "
              f"{s['beats_matched_blur']}/{s['n']}, lo {s['wins_shrink_lo']}"
              f"/{s['n']}, hi {s['hi_gain_db']:+.2f} dB, chroma {s['chroma']:.2f}")
        print(f"           ORACLE (tier 3, alpha chosen with the answer): "
              f"a={s['oracle_alpha']:.3f} {s['m_oracle_full']:.5f}, beats "
              f"identity {s['oracle_wins']}/{s['n']}, beats the SAME oracle on "
              f"a sigma=1 blur ({s['m_oracleblur_full']:.5f}) "
              f"{s['oracle_beats_oracleblur']}/{s['n']}")
    if json_out:
        json_out.write_text(json.dumps(out, indent=1, default=float))
    return out


def run_circle(arm: str = "trace2", steps: int = 1500, seed: int = 0) -> dict:
    """Constraint 4: the ring's geometry must not degrade under the denoiser.

    The network is fitted to L000 ALONE and applied to it; the ring is refitted
    from its output by the same estimator (circle.fit_ring -> quality) that
    produced 1.0053 / 0.837 on the float decode.
    """
    from . import circle as circle_mod
    from . import catalog as catalog_mod
    from . import fuse as fuse_mod

    device = _device()
    img = fuse_mod.decode_frame(catalog_mod.build().by_id(CAL_FRAME))
    z = (img - img.mean()) / img.std()
    if steps:
        net, _ = train([z.astype(np.float32)], ARMS[arm],
                       TrainCfg(steps=steps, seed=seed), device)
        out = n2n_mod.denoise_plane(net, z.astype(np.float32), device)
    else:
        # steps=0: the closed-form predictor instead of a trained net. Same
        # blind spot, no GPU, and it is the optimum of the linear family
        # rather than one draw from an optimiser, so the geometry check does
        # not depend on a training run.
        shape = {"n2v": ("point",), "line3": ("line", 0, 3),
                 "trace0": ("trace", 0), "trace1": ("trace", 1),
                 "trace2": ("trace", MASK_W), "trace4": ("trace", 4)}[arm]
        out = linear_blind(z, blind_offsets(*shape))
    before = circle_mod.fit_ring(img)
    after = circle_mod.fit_ring(out * img.std() + img.mean())
    d = float(np.sqrt(np.mean((out - z) ** 2)))
    print(f"CIRCLE INVARIANT (arm {arm}, fitted to {CAL_FRAME} alone, "
          f"d_rms {d:.4f}):")
    for tag, r in (("decode", before), ("denoised", after)):
        if r is None:
            print(f"  {tag:9s} RING FIT FAILED")
        else:
            print(f"  {tag:9s} axis ratio {r.axis_ratio:.4f}  radial rms "
                  f"{r.radial_rms:.3f} px  width {r.width:.2f}  "
                  f"coverage {r.coverage_deg:.0f} deg")
    return {"before": None if before is None else before.__dict__,
            "after": None if after is None else after.__dict__,
            "delta_rms": d}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true",
                    help="the circle measurement: ACF and blind-spot leakage")
    ap.add_argument("--report", action="store_true",
                    help="train the arms and run n2n.py's held-out arbiter")
    ap.add_argument("--single", action="store_true",
                    help="one network per plane, no repeats anywhere")
    ap.add_argument("--linear", action="store_true",
                    help="closed-form blind-spot predictor, per plane, no GPU")
    ap.add_argument("--hp", type=float, default=0.0,
                    help="band-limited: predict only the detail above this "
                         "sigma, pass the smooth part through (0 = off)")
    ap.add_argument("--circle-check", action="store_true",
                    help="constraint 4: ring geometry under the denoiser")
    ap.add_argument("--arms", default="n2v,line3,trace0,trace2")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--single-steps", type=int, default=1500)
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    did = False
    if args.calibrate:
        calibrate(); did = True
    if args.report:
        run(args.arms.split(","), folds=args.folds, steps=args.steps,
            seed=args.seed, quick=args.quick, json_out=args.json); did = True
    if args.linear:
        run_linear(args.arms.split(","), hp=args.hp, json_out=args.json)
        did = True
    if args.single:
        run_single(steps=args.single_steps, seed=args.seed,
                   n_scenes=args.scenes); did = True
    if args.circle_check:
        run_circle(steps=args.single_steps, seed=args.seed); did = True
    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
