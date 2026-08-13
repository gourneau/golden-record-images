# Machine learning on the Golden Record decode — state of play

Written as a handoff. Everything below is committed and pushed on `main`; the gallery is
live at https://gourneau.github.io/golden-record-images/.

**Repository: `/Users/josh/code/golden-record-images`.** (Separate from `golden-map` /
goldenrecord.voyage, which another session is working on and which does not touch these files.)

---

## The rule everything is judged by

**The arbiter is prediction of data the method never saw**, never an image metric. This project
measured that its own quality composite *rewards blur*, and that every method's metrics keep
improving past the correct setting. A prettier picture is not a truer one, and only a hold-out
tells them apart. Every result below either has such a hold-out or is labelled as not having one.

**Provenance tiers** (`pipeline/provenance.py`, enforced in code, not promised in prose):

| tier | may use | |
|---|---|---|
| 0 · record | the audio and the engraved cover | what an alien has |
| 1 · universal | priors any observer could hold | no Earth knowledge, but prior-based |
| 2 · earth | the original slides and captions | cheating; evaluation and presentation only |
| 3 · oracle | settings chosen by looking at the answer | a headroom measurement, not a decoder |

`decode.py` imports only `numpy`, `dotclock` and `sync`; its sole data input is the WAV. A check
walks the import graph and fails if a tier 0 module can reach reference material.

---

## What shipped

### 1. The accumulating-error correction — the big one

**Not machine learning, but it is the largest quality win and it reshaped the ML.** The dominant
defect is *one* mechanism producing *both* the vertical streaking and the brightness droop: an
error that **accumulates down each trace in proportion to the light already sent**, reset once per
trace by the porch clamp.

The discriminator is a tier-0 region the record supplies on every frame and nobody had used: the
**slide mount appears twice**, before the picture and after it. One physical object, one value, so
both readings must agree — and the hypotheses disagree about how. Shading is multiplicative and
the mount is dark, so it moves both ends alike; a per-trace gain or offset moves both together;
**only a causal accumulating error darkens the end alone.**

| | median slope vs trace level | negative on |
|---|---|---|
| bottom mount | **−0.284** | **12 of 12 frames** |
| top mount | +0.054 | 3 of 12 |

Scale, which reframes the whole "streaking" request: **the droop is 96.5% of the error and the
streaking is 0.2–0.7% of it.** Same mechanism, so fixing the droop is how the streaking goes.

`decode.UNCUMULATE_K = 9.4e-4`, fitted on the mount regions of 68 frames with L000 excluded,
chosen at 9.4e-4 rather than the value that best fits L000's own field because it minimises the
mount-convergence test *on frames it never saw*.

    L000 field rms     0.140 → 0.085
    white clipping     5.99% → 0.65%
    mount gap          1.478 → 0.388   (−74%, ten held-out frames)
    circle             axis ratio and radial rms both unmoved

**Control it passes** — one global parameter each, fitted on the k-set, scored on 13 held-out
mounts: cumulative **0.0587**, additive ramp 0.1030, multiplicative ramp 0.1072, nothing 0.3217.
With the rider that matters: on L000's *field alone* a multiplicative ramp nearly matches it
(0.0390 vs 0.0367), so **the field does not identify the mechanism — only the mount does.**

It shipped off for a while because in-path it delivered a third of the post-hoc result. That was
**three faults, two of them mine**: the accumulator summed from the porch rather than from black
(injecting an error the same size and sign as the droop), `k` was applied on the ~230-row dot grid
when it was fitted per 377-row (87% dose), and the rails were measured against an uncorrected
picture. All fixed; it is on by default.

### 2. Noise2Noise on the colour repeats — the shipped denoiser

`pipeline/n2n.py`, `pipeline/reconstruct.py`. Twenty images were scanned three times through
colour filters, which is **twenty sets of three independent noisy observations of one scene**.
Noise2Noise proves a network trained to map one noisy observation to another converges on the
clean-target estimator. The literature's stated obstacle is that independent noisy pairs are hard
to get; this record carries sixty frames of them.

**Tier 1 and an alien could run it**: no Earth photograph, no external corpus, not even which
separation is red — the method is symmetric in the three planes.

Measured on the **old** decodes (see *Outstanding*): denoised plane beats raw 19/19 unseen scenes,
beats a blur control 19/19, full-band MSE 0.03410 → 0.03217. It also **beats an oracle-tuned
Gaussian sweep** (best σ=0.50 gives 0.03268), which is the claim that does the real work.

Shipped as a **second tier**, never blended, with a switch on the page. Called *decoded* and
*denoised* — "reconstructed" was the wrong word, since nothing is invented.

### 3. Colour registration

`pipeline/colourfix.py`. The colour composites were built from **unregistered** separations. Plane
disagreement 0.553 → 0.477 (−13.9%), improved on 19/20, largest real correction 2.46 px. Image 8
(solar spectrum) is shipped unregistered by a plausibility cap rather than by name — its content
moves with wavelength, so the three separations genuinely are three different pictures.

---

## What was measured and rejected — the nulls

These matter as much as the wins; several look like successes from the outside.

| method | verdict |
|---|---|
| **Neural field** (`neuralfield.py`) | Qualified null — 1 of 6, and only on the calibration circle at the lowest bandwidth, which is what a smoothness prior wins. The bandwidth trend *reverses* between the circle and a photograph. |
| **Deep image prior** (`dip.py`) | Works — 3/3, +11% to +43% over neighbour-fill on withheld dots — but ~10 min/frame, 26 h for the record. Not shipped on cost. Its hold-out measures *interpolation*, not denoising. |
| **Plate-split N2N** (`platesplit.py`) | **Dead.** My idea: split the ~12 samples in each sample-and-hold plateau for two independent looks per dot, giving pairs on all 156 frames. Needs white noise; measured lag-1 autocorrelation **+0.93 and +0.99**. The failure mode is the dangerous kind — such a network trains fine, converges near the *identity*, and the hold-out **rewards keeping the noise**. |
| **Circle-calibrated Wiener** (`calnoise.py`) | Real (19/19, no training data, all 156 frames) but **indistinguishable from a well-chosen mild blur** on this arbiter — it lands exactly on the blur sweep's optimum where N2N beats it. And **the circle contributes ~8% of even that**: ablating it entirely gives 0.03293 vs 0.03283. |
| **Ring PSF deconvolution** (`ringpsf.py`) | **Null on new blur removal.** The ring gives the LSF at every angle, so terms separate by symmetry: camera 0.383 px² isotropic, along-trace 0.224 px², **across-trace gate +0.036 ± 0.057 — zero.** The removable blur is exactly what `aperture.py` already builds. Two firsts worth keeping: the camera *is* isotropic, and trace pitch adds no smearing. |
| **Spectral destripe** (`destripe.py`) | Rejected — generic analytic spectra carrying no circle information matched or beat the calibrated version on every test, so its tier-0 calibration was not load-bearing. |
| **Additive-pedestal droop** (`dedroop.py`) | Rejected — three independent record-wide instruments put its amplitude at 0.64, not the 1.0 it calibrated on L000. |
| **Per-frame picture gate** (`picgate.py`) | Null — the gate is blindly measurable on 156/156 and does not move (spread 0.18 of one row in 377). |
| **Line-art gating** (`lineart_eval.py`) | No rule needed — the denoiser treats line art as it treats photographs. |

---

## Corrections to our own published numbers

- **The +7.73 dB vs +5.60 dB comparison was not like-for-like.** +7.73 is the *mean of per-scene
  dB*; +5.60 is the *dB of mean residuals*. Consistently: 6.94 vs 5.60, or 7.73 vs 7.48. The
  advantage is 1.34 dB or 0.25 dB, not 2.13. The full-band result against the oracle-tuned sweep
  is untouched and was always the load-bearing part.
- **A "first run" that never existed.** Figures I compared against were placeholders written into
  a docstring under a "MEASURED RESULT" heading before anything ran. One run exists.
- **A confident mechanism explanation that was wrong.** A circle regression I attributed to the
  denoiser's receptive field was 8-bit quantisation; it vanished entirely on the float decode.
- **The oracle's 41% gap is not all headroom.** Its picture-shift search is 7–14× larger than the
  gate's entire measured variation, so it is buying alignment against the *scanned reference*,
  which is tier 2.
- **`PICTURE_END` 3040 → 3036.75 was recommended unanimously and I rejected it** — it cleans the
  bottom row but degrades the circle, and one row in 377 is not worth 4.4% of the geometry metric.

---

## Outstanding

1. **The Noise2Noise re-evaluation is running and is the blocking item.** Retraining on the
   corrected decodes **invalidated the 19/19 result and every dB figure in `n2n.py`** — those were
   earned on the old decodes and do not describe the shipped model. Command:
   `python -m pipeline.n2n --report --steps 3000 --json <out>`. Until it lands, **do not quote
   quality figures for the denoiser.**
2. **The (k, origin) degeneracy.** The mount and the flat field disagree about where the
   accumulator's zero sits — sweeping origin improves the mount (17.6 → 13.7 grey) and worsens the
   field (0.0367 → 0.0441). The two are algebraically coupled (`k·origin` is a linear ramp), so a
   *joint* fit rather than a sweep should resolve it. Not done.
3. **An 8σ unexplained cross term** in the ring PSF (−0.136 ± 0.016 against a +0.008 ± 0.007
   synthetic control, −8.3° tilt) — and the same diagonal signature appears in the circle's
   residual ellipticity. May be coincidence.
4. **The circle's 0.5% ellipticity is probably not ours.** No row count removes it (minimum sits
   at the shipped 377), and it appears tilted, which our axis-aligned constants cannot produce.
   **The 1977 slide may simply not have been round.** Forcing it would inject an error.
5. **Blind-spot denoiser** (`blindspot.py`) — unverified; its verifier died on an API error.
6. **The 96 mono images have no held-out measurement.** A colour separation *is* a single scan, so
   mono-ness is not the gap — the gap is *content type*: ~19 of them are diagrams and line art,
   and no scored scene was line art.

---

## Files

**Shipped decode path (tier 0):** `decode.py` `sync.py` `dotclock.py` `geometry.py` `wav.py`
`catalog.py` `build.py` `colourfix.py` `presentation.py`

**ML:** `forward.py` (differentiable chain + verified adjoint) `neuralfield.py` `dip.py` `n2n.py`
`reconstruct.py` `calnoise.py` `blindspot.py` `platesplit.py`

**Circle calibration:** `caltarget.py` `ringpsf.py` `levelsfix.py` `dedroop.py` `destripe.py`
`circle.py`

**Measurement / audit:** `provenance.py` `evalset.py` `oracle.py` `picgate.py` `lineart_eval.py`
`aperture_fit.py` `orient_blind.py` `quality.py` `testset.py`

**Reference handling (tier 2, evaluation only):** `align_refs.py` `refcrop.py` `trimcrop.py`

**Reports:** `docs/corrections.md` (findings and retractions) · `docs/reconstruct_report.json` ·
`docs/n2n_confirmatory.json` · `docs/colour_registration.json` · `docs/orientation_blind.json` ·
`docs/presentation.json` (tier-2 display metadata; **`build.py` drops it, run
`python -m pipeline.presentation` after every build**)

**Verify anything:** `python -m pipeline.provenance` · `python -m pipeline.forward` ·
`python -m pipeline.orient_blind` · `python -m pipeline.circle`
