# What didn't work

Nine ideas that were built, measured, and rejected. They are here rather than in the
[README](../README.md) because a homepage should say what the thing does — but they are not
deleted, because **a project that publishes only its successes cannot be checked**, and because
several of these are the most useful things learned.

If you are picking this up to improve it, read this first. It is a map of the holes people fall
into on this particular signal.

Related: [what did work](findings.md) · [the machine learning](ML_HANDOFF.md) ·
[corrections and retractions](corrections.md)

---

## The dangerous one: a method that "works" by doing nothing

**Plate-split Noise2Noise.** The scan converter sampled once per television line and held, so each
dot is a plateau of about twelve samples. Split them — even against odd — and you would have two
independent noisy looks at the same dot: Noise2Noise pairs for **all 156 frames**, not just the 20
colour ones, plus the held-out arbiter the mono frames lack.

It requires the noise to be white at 384 kHz. Measured on the content-free porch:

| lag | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| autocorrelation | **+0.93 to +0.99** | +0.87 to +0.97 | +0.80 to +0.95 | +0.75 to +0.93 |

Half the chain's noise power sits below 3.4 kHz — an order of magnitude under the sample rate. No
geometry inside a plateau rescues it: the widest possible split still correlates **+0.51**.

**Why this one matters most.** With input and target noise correlated at 0.9+, such a network
trains perfectly well and converges near the **identity**, while a hold-out on the other half
**rewards keeping the noise**. It would have looked like a success, and the blur control could not
have caught it, because the target is correlated with the input. *The arbiter would have been
broken in the method's favour.*

---

## Corrections that made things worse

**`_reclamp_after`** (in `decode.py`, kept and switched off). The chain correction amplifies each
trace's residual clamp error into a ramp — a streak. The slide mount is known-constant across
traces, so redistributing each trace's mount deviation back along its own accumulation should
cancel it. It made streaks **a third worse** (sd 0.0815 → 0.1109): the mount is only ~8 dots deep,
so the deviation read from it is mostly its own noise, injected as a ramp. A wider known-constant
region might still work.

**Moving `PICTURE_END` from 3040 to 3036.75.** Recommended unanimously across 156 frames — it
cleans a contamination signature in the bottom row. Rejected: the calibration circle degrades
(radial rms 0.837 → 0.847 shortening the window, 0.874 shifting it). One row in 377 is not worth
4.4% of the geometry metric.

**A trap worth knowing:** `decode.PICTURE_END` is bound into `Settings` at class-definition time,
so patching the module constant does nothing. The first test produced *identical* numbers for both
arms, which is what gave it away.

---

## Priors that turned out to be blur

**Circle-calibrated Wiener filter.** Measure the noise spectrum where the calibration slide's true
value is known, build the optimal linear filter. Real — 19/19 unseen scenes, no training data, all
156 frames. But a matched-strength Gaussian suppresses the high band exactly as hard and is
marginally *better* in the low band. **Indistinguishable from a well-chosen mild blur.**

And the calibration contributed **~8%** of even that: ablating the circle entirely gives 0.03293
against 0.03283.

**Spectral destriping.** Rejected because generic analytic spectra carrying *no* circle information
matched or beat the calibrated version on every test — so its tier-0 calibration was not
load-bearing.

**Neural field** (coordinate MLP through the forward operator). 1 of 6 configurations beat
neighbour-fill, and only on the calibration circle at the lowest bandwidth — which is what a
smoothness prior wins. The bandwidth trend *reverses* between the circle and a photograph.

---

## Things that are simply already right

**Per-frame picture gate.** The window is blindly measurable on 156/156 frames and **does not
move**: noise-corrected spread 1.36 bins = 0.18 of one output row in 377. The single global
constant was already correct.

*Consequence:* `oracle.py` searches picture shifts 7–14× larger than the entire measured
variation, so those shifts are not recovering a gate — they are buying alignment against the
scanned reference, which is tier 2. **That part of the 41% "oracle gap" is not headroom.**

**Ring PSF deconvolution.** The ring presents a known-thin line at every orientation, so blur terms
separate by symmetry: camera **0.383 px² isotropic**, along-trace 0.224 px², across-trace sampling
gate **+0.036 ± 0.057 — zero**. The removable blur is exactly what `aperture.py` already builds.
Deconvolving further degrades the circle and raises noise ×1.34 against a predicted ×1.21.

Two firsts worth keeping regardless: **the camera is isotropic** (assumed for years, never
checked), and **trace pitch adds no measurable smearing**.

**Line-art gating.** The denoiser treats diagrams as it treats photographs; no special rule needed.

**The additive-pedestal droop model.** Rejected — three independent record-wide instruments put its
amplitude at 0.64, not the 1.0 it had calibrated on the calibration frame.

**The aperture constant.** `TRANSITION_1090 = 0.49` was re-derived as 0.479 — until it was run on
eight frames the method was not developed on and gave **0.464 (0.449–0.471), which excludes the
shipped value**. A per-channel systematic the fit does not model.

---

## The lesson that cost the most

**A correction validated on the calibration frame is validated *for* the calibration frame.**

The chain correction removes 20–42% of the droop and *raises* the streak amplitude 24–96%. Every
instrument aimed at the flat field said it was working. The side effects lived in a band the flat
field cannot see, and only a photograph — or a person looking at one — showed them.
