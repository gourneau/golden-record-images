# Integration spec — four recoveries, refuted, ranked, and staged

Author: integration pass, 2026-08-12. Every number in this file was produced by
code I ran myself on `data/master/384kHzStereo.wav` during this pass, not
copied from the four source reports. Where my number differs from theirs, mine
is stated and the difference is explained.

Owners: this file is new and mine. `pipeline/decode.py`, `pipeline/build.py`,
`index.html` and `docs/reference/**` were **not** touched. Everything below is
written so that whoever holds those files next can apply it directly.

Raw data from this pass:
`/private/tmp/claude-501/-Users-josh-code-golden-record-images/484b41e6-9824-495a-97cb-a0b8bfe1c684/scratchpad/spec/{scan,ab,dc,parity}.json`

---

## 0. Verdicts, in one table

| # | claim | verdict | what my re-measurement found |
|---|---|---|---|
| A1 | `drift_span` ranks frames by scene diagonality, not decoder health | **SURVIVES** | corr(mean column shift, structure-tensor slope) = **0.972** over all 156; corr(log composite, log f_drift) = **0.967**; corr(composite, sharpness) = **−0.169** |
| A2 | `residual_drift` is the honest replacement | **SURVIVES, WITH LIMITS I FOUND** | catches injected staircase and jitter, is not inflated by sharpening — but is blind to a pure linear shear and is unstable (L011 unsharp 7.9 → 83.9) |
| A3 | the 59 droop "regressions" are an f_parity artefact | **SURVIVES** | 59/156 reproduced exactly; f_parity share of the loss median **1.05**, >0.5 on **45/59**; the coherent stripe it punishes never exceeds **0.17 %** of black–white anywhere on the record |
| A4 | dot clock: matched filter + narrow band, 17 false locks, 64 recoveries | **SURVIVES MY HARDEST TEST** | 262.4926 ± 0.0580 reproduced to 4 decimals; **widening the search to ±3 dots returns the identical peak on 156/156** |
| A4b | the dot-clock fix has no visible payoff | **STRENGTHENED (worse than reported)** | forcing a deliberately **wrong** rate (+0.6 dots) changes sharpness by <1 % and shift_rms by <0.01 — `track_phase` absorbs rate error. The payoff is zero, not small |
| A5 | L032 guard: `\|parity\| > 1 dot` → re-anchor | **SURVIVES** | parity −0.531 ± 0.455 dots record-wide, L032 alone at **−5.84**; guard gives accept_frac 0.244 → 1.000, composite 9.1 → 62.3 |
| A5b | apply only when triggered | **SURVIVES, AND I FOUND A HARD CASE** | unconditional re-anchoring moves L000's grid 3.5 samples and **regresses the circle axis ratio 1.0060 → 1.0079** |
| F1 | full-band phase correlation is biased to zero along-trace | **SURVIVES** | full band gives dy −0.02/−0.02/−0.01; band-limited +0.32/+0.31/+0.20; closure \|err\| **0.059 mean** band-limited vs **0.357** full band over 19 triplets |
| F2 | the along-trace creep is real, +0.33 px/frame, 19/19 | **SURVIVES MY BAND TEST** | creep = **+0.329 px** identically for bands 0.00–0.25, 0.02–0.25 and 0.05–0.25 cyc/px, positive on **19/19** in every band — it is not low-frequency shading |
| F3 | registration is unbiased | **SURVIVES** | self-registration exactly 0.0000; antisymmetry exactly 0.0000; injected (+0.37, −0.62) recovered exactly; post-registration residual exactly 0.000 |
| F4 | fusion buys 4.78 dB on independent noise, 1.96 dB per colour plane | **SURVIVES** (algorithmic control only) | reproduced on 8 triplets; 4.78 dB is a control of the *algorithm*, the material figure is noise 6.12 % → 3.53 % of picture rms |
| F5 | no super-resolution | **SURVIVES** | alias/power 0.056 mean, unfold penalty +1.48 dB |
| D1 | camera ESF p10 = 1.31 dots along / 1.59 traces across | **REPRODUCED (1.339 / 1.576) BUT THE STATISTIC IS BIASED** | p10 of a noisy width estimator is biased **−3 % to −22 %** at the real noise/amplitude; the true sharp end is **1.3–1.7 dots / 1.6–2.0 traces** |
| D2 | across-trace deconvolution must stay off | **SURVIVES** | Hann-windowed across-trace PSD at Nyquist / mid-band = 1.16 (L055), **7.35 (R040)**, 0.95 (L002); along-trace **falls** to 0.29–0.62 as a real optical spectrum must |
| D3 | ship no deconvolution yet | **SURVIVES** | plus D1: less headroom than claimed |
| S1 | the "15 % shelf" is an artefact of the estimator's normalisation band | **SURVIVES** | \|S(180 Hz)\| moves **0.558 → 1.089** with the flat band, shape flat in every band |
| S2 | `UNCOUPLE_TAU_384` is confirmed model-free | **SURVIVES** | H_L/H_R follows pole(530)/pole(295) to **2.8 % max, 1.0 % mean**, phase included |
| S3 | ship nothing from `shelf.py`, retract the docstrings | **SURVIVES** | — |

**Nothing was refuted outright.** Three claims were *weakened* by my testing (A4b, A5b, D1) and one metric weakness was found that none of the four reported (see §1.2). That is the unusual outcome; I looked hard for the project's known failure modes and they are not present in these four.

---

## 1. What I tried to break, and what broke

### 1.1 The audit's metric replacement (A1/A2)

Reproduced independently: for content `I(x,y) = f(y − mx)` adjacent columns
genuinely line up at shift `m`, `drift_span` accumulates that over 496 pairs,
so it *is* the picture's diagonal times the frame width. Measured
corr(mean shift, slope) = **0.972**; the regression slope is 1.22 (the shift
runs slightly steeper than the structure tensor, which is why `residual_drift`
must subtract the per-frame median of the residual as it does, not just `m`).

Consequences I re-measured on all 156: softest sharpness quartile scores
**25.4**, sharpest quartile **19.5**. The composite pays you to blur.

**What I found that the audit did not:**

1. **`residual_drift` is blind to a pure linear shear.** Injecting a 24 px
   end-to-end *linear* timing ramp on L055 moves `drift_span` 11.2 → 22.4 but
   `res_drift` only 4.7 → 5.9. This is by construction: a global shear turns
   every edge diagonal, so the structure tensor absorbs it. Acceptable — a
   linear timebase error is a period error and shows up as skew in
   `circle_metrics` on L000 — but it must be written down, because it means
   `res_drift` alone cannot police the timebase.
2. **`res_drift` is unstable under cosmetic filtering, in the safe
   direction.** L011 unsharp ×1.5: 7.9 → **83.9**. L057 unsharp: 15.3 → 12.8.
   It swings by 10× under transforms that should not matter. It cannot be
   *gamed upward* by a sharpener (which is what the house rule cares about),
   but it is not a precision instrument. Use it as a screen, not a ruler.
3. **The composite still rewards blur after the fix, and did far worse
   before.** Gaussian σ=1.0 on the current composite: L055 60.7 → **66.9**,
   R074 70.0 → **90.1**, R073 62.7 → **90.5**. Under the proposed composite:
   89.7 → 88.3, 89.6 → 97.8, 92.6 → 98.7. The fix cuts the blur reward from
   +28 points to +6, but does not remove it. Every metric in the composite is
   a registration or coherence statistic and every one of them gets cleaner
   when you smooth. **This is the standing hazard and no parameter choice
   fixes it.**

### 1.2 The parity factor (A3)

Reproduced: 59 frames score lower with the droop fix on than off; f_parity's
share of the loss is median 1.05 (i.e. every other factor *improved*).

My own instrument, not theirs: the coherent odd/even amplitude, projected as
`mean(hp(column profile) · (−1)^i)` where `hp` is a 5-tap high-pass. Calibrated
by injection on L055: injected 0.002/0.005/0.010/0.020/0.050 read back
0.00163/0.00403/0.00803/0.01603/0.04003, i.e. **measured = 0.80 × true**,
linear over the whole range.

- Record-wide, current decode: median **0.00026**, p90 0.00071, **max 0.00171**
  (L051) in units of black-to-white. **Zero frames exceed 1 %.**
- corr(`parity_db`, actual stripe amplitude) = **0.106**. `parity_db` is a
  band ratio; on a frame with little content in the reference band it reads
  high with no stripe present. It is not measuring what it claims.
- The median frame's `f_parity` is **0.695** — the composite is throwing away
  **30 % of every frame's score** for a stripe of 0.026 % of black–white.

That is decisive independently of the droop question, and it is why f_parity
must be re-based on amplitude, not dB.

### 1.3 The dot clock (A4)

The obvious attack is that narrowing the search band from ±3 % (±7.9 dots) to
±0.6 dots *manufactures* the answer, and the second is that 262.5 ± 0.6 spans
the integers 262 and 263, which are the 262nd and 263rd harmonics of the line
rate — a gated-picture artefact would land exactly there. Both tested on all
156 frames:

- recovered rate **262.4926 ± 0.0580**, min 262.360, max 262.655 — the ±0.6
  bounds are **never approached** (max excursion 0.155). No railing.
- **zero** frames within 0.02 of an integer. No line-harmonic leakage.
- re-running the matched filter over **±3.0 dots** returns a peak within
  0.05 dots of the narrow-band peak on **156 of 156 frames**. The narrow band
  is not doing any work; it is a sanity rail, not the estimator.
- split-half |h1 − h2| median 0.012, p90 0.135, max 0.285 dots; peak/floor
  excess min 8.9.
- gate (halves < 0.15 **and** excess > 6): **147/156 pass**; of the 66 frames
  the FFT detector fails to lock, **64 pass**; **17 of the 90 current locks
  disagree with the matched filter by > 0.15 dots** — the same 17 the audit
  named, at the same values.

**Where I weakened the claim.** I forced the decoder to sample at three rates
on five frames — the false FFT rate, the matched-filter rate, and a
deliberately wrong rate 0.6 dots away — and compared. Example, R065:
sharpness 0.0183 / 0.0182 / 0.0182; shift_rms 0.3896 / 0.3986 / 0.3894;
near/far along-trace gradient ratio 0.605 / 0.642 / 0.618 (non-monotone, i.e.
noise). Across all five frames nothing moves by more than ~1 %. The reason is
in `dotclock.track_phase`: it re-measures the plateau phase every 64 traces
and interpolates, so a rate error is absorbed as phase. **The dot-clock change
buys correctness, not pixels.** Rank it accordingly.

### 1.4 The L032 guard (A5)

Parity alternation across all 156, in dots (`parity_offset / (period/262.5)`):
mean **−0.531**, sd 0.455, p1 −0.54, p99 +0.27. The physical field-ID
alternation is half a dot and that is exactly what is measured.
**Only L032 exceeds 0.8 dots, at −5.84.** Two more tripwires isolate the same
single frame: `accept_frac` (L032 = 0.244, next lowest L038 = 0.994) and
`meas_noise` (L032 = 5.57, next highest L014 = 0.830, median 0.53).

Guard applied (`audit.reanchored_timebase`): parity −17.78 → +1.30,
accept_frac 0.244 → **1.000**, composite 9.1 → **62.3**, res_drift 11.5 → 7.8,
shift_rms 0.311 → 0.267, grid moved 6.18 samples.

**Where I hardened the "only when triggered" rule.** The audit reported
controls moving 0.6–0.7 samples. That is true for L031/L055/R040/L020
(0.64–0.70), but **L000 moves 3.54 samples and L075 moves 4.07** — precisely
the two frames whose own parity estimate is ~0 instead of the record-wide
−1.6. On L000 the consequence is an invariant regression:

```
L000 circle, current      axis_ratio 1.0060   radial_rms 0.8752 px
L000 circle, re-anchored  axis_ratio 1.0079   radial_rms 0.7250 px
```

The radial rms improves 17 % but **the axis ratio regresses**, which §4 rejects
outright. So: trigger-only, and the trigger must be the parity test, which
never fires on L000.

(Filed as a lead, not a change: a landmark discriminator that cuts L000's
radial rms from 0.875 to 0.725 px deserves its own evaluation with the circle
as arbiter. It is not part of this integration.)

### 1.5 Fusion (F1–F5)

The attack here is that the band-limited registrar is measuring a shared
low-frequency shading difference between separations (the droop residual tilt
of §5 is exactly such a thing), and that the "creep" is that tilt, not the
picture. Tested by moving the *low* edge of the correlation band:

| band (cyc/px) | 0.00–0.25 | 0.02–0.25 | 0.05–0.25 | 0.08–0.30 | 0.12–0.35 | 0.00–0.50 |
|---|---|---|---|---|---|---|
| mean creep px | +0.329 | +0.329 | +0.329 | +0.311 | +0.279 | +0.203 |
| sd | 0.145 | 0.145 | 0.145 | 0.145 | 0.136 | 0.163 |
| positive on | 19/19 | 19/19 | 19/19 | 19/19 | 19/19 | 15/19 |

Excluding everything below 0.05 cyc/px changes the answer by **0.000 px**. The
creep is picture, not shading. It decays only when the band is opened to
Nyquist, which is the contamination fuse.py identified.

Estimator checks, all exact: `phase_offset(a, a)` = (0.0000, 0.0000);
`phase_offset(a,b) + phase_offset(b,a)` = (0.0000, 0.0000) on every triplet
tried; an injected (+0.37, −0.62) is recovered to the 0.01 px grid; and after
`register()` the residual offset re-measures **+0.000, +0.000**.

Closure over all 19 non-spectrum triplets: band-limited mean **0.059** px,
max 0.230; full band mean **0.357**, max 0.910. (My closure numbers are
slightly worse than fuse.py's quoted 0.05/0.28 because I take the max of the
two axes rather than the along-trace component; the 6× ratio is the same.)

The decisive independent evidence remains fuse.py's own fringe instrument —
regressing (R−G) on ∇L in image space, which is not the estimator that did the
registering: residual misregistration **0.40 → 0.09 px (red)** and
**0.23 → 0.06 (blue)** replacing colour.py's full-band + spline path, *with*
luminance HF energy going up 14.5 %. Reproduced on 8 triplets (mean R 0.091,
B 0.060).

### 1.6 The camera ESF (D1)

I reproduced the survey on 10 frames (slant-gated |s| < 0.2, dot-native
decode): along-trace p10 = **1.339 dots** (deconv: 1.31), across-trace p10 =
**1.576 traces** (deconv: 1.59), medians 1.83 / 2.03.

**The failure the deconv agent did not test: p10 of a noisy estimator is
biased low even when the estimator is unbiased.** Monte Carlo through
`deconv.fit_edge_best` with the module's own survey gates, erf truth:

| true W | amp | noise | median error | **p10 error** | p5 error |
|---|---|---|---|---|---|
| 1.4 | 1.00 | 0.004 | −0.0 % | −3.3 % | −5.1 % |
| 1.4 | 1.00 | 0.012 | +0.5 % | **−9.0 %** | −13.0 % |
| 1.4 | 0.35 | 0.012 | +3.9 % | **−20.3 %** | — |
| 2.4 | 1.00 | 0.012 | +1.6 % | −5.9 % | −7.7 % |
| 2.4 | 0.35 | 0.012 | +2.6 % | **−20.4 %** | — |

The real surveyed population has amplitude p50 = 0.44 of black–white at a
measured noise of ~0.012, i.e. squarely in the −10 % to −20 % rows. **The
camera's sharp-end response is therefore ~1.3–1.7 dots along-trace and
~1.6–2.0 traces across, not 1.31/1.59.** The headline conclusion is unchanged
(the median is scene softness, the sharp end is the instrument) but the
instrument is wider than reported, and there is correspondingly *less* room
for deconvolution, not more.

---

## 2. Ranking

Ordered by (expected improvement per affected frame) × (frames affected).

| rank | change | frames touched | expected effect |
|---|---|---|---|
| **1** | `quality.py`: `drift_span` → `residual_drift`, `f_parity` re-based on stripe amplitude | **156** (and every future decision) | no pixels change; the instrument stops rewarding blur (softest/sharpest quartile 25.4/19.5 → 75.5/75.3) and stops vetoing correct low-frequency work |
| **2** | `colour.py`: replace registration with `fuse.phase_offset` / `register` | **60 frames → 20 colour images** | residual misregistration 0.40 → 0.09 px (R), 0.23 → 0.06 (B); +14.5 % luminance HF; edge chroma rms 0.672 → 0.518 |
| **3** | `colour.py`: `fuse.fuse_colour` (fused luminance + measured Wiener chroma) | **60 frames → 20 images** | +1.96 dB per output plane; noise 6.12 % → 3.53 % of picture rms; column-streak energy ÷3 |
| **4** | L032 sync guard (trigger-only re-anchor) | **1** | the worst frame on the record: composite 9.1 → 62.3, accept_frac 0.244 → 1.000 |
| **5** | `dotclock.py`: matched filter + split-half gate | **81** (17 wrong rates corrected, 64 locks recovered) | **no measurable image change** (§1.3). Correctness and the removal of physically impossible rates |
| **6** | `testset.py` label corrections | 6 labels | stops the frozen set asserting facts that are no longer true |
| **7** | docstring retractions in `decode.py` and `droop_blind.py` | 0 | removes a "known residual" that does not exist |
| — | camera deconvolution | 0 (stays off) | see §3.8 |

---

## 3. The changes

### 3.1 `quality.py` — replace `drift_span` with `residual_drift` (RANK 1)

**Add** (lift verbatim from `audit.column_slope` and `audit.residual_drift`;
they have no dependencies outside numpy and `quality.column_shifts`):

```python
def column_slope(a, w: int = 9) -> np.ndarray      # structure tensor, dy/dx per column
def residual_drift(img, trim: int = 8) -> dict     # {res_drift, res_rms, res_jump_frac}
```

`drift_metrics` keeps returning `drift_span` (it is a legitimate description of
the picture, just not of the decoder) but **the composite must stop using it.**

**Modify** `composite_score`:

```python
f_drift  = exp(-max(res_drift - 12.0, 0.0) / 40.0)          # was exp(-drift_span/40)
f_parity = 1.0 / (1.0 + max(stripe_amp - 0.010, 0.0) / 0.010)  # was 1/(1+max(parity_db,0)/6)
```

**Add** the stripe amplitude metric (mine, calibrated above):

```python
def stripe_amp(img, trim: int = 8) -> float:
    """Coherent odd/even trace amplitude, in units of black-to-white.
    Injection-calibrated on L055: reads 0.80 x the true amplitude, linear
    from 0.002 to 0.050. Record-wide the current decode measures
    median 0.00026, max 0.00171 -- i.e. nothing on this record is visible."""
    a = np.asarray(img, float)[:, trim:-trim]
    prof = a.mean(axis=0)
    hp = prof - np.convolve(np.pad(prof, (2, 2), "edge"), np.ones(5) / 5, "valid")
    s = ((np.arange(a.shape[1]) % 2) * 2 - 1).astype(float)
    return float(abs(np.dot(hp, s)) / len(s))
```

Keep `parity_db` in the metrics dict as a diagnostic; just stop scoring on it.

**Parameter justification.**
- knee 12.0 px on `res_drift`: measured distribution over 156 frames is
  p10 4.8, **p50 10.6**, p75 14.7, p90 19.2, max 62.5. A knee at 12 leaves the
  clean half unpenalised and still bites on the damaged tail (L001 62.5,
  L021 41.7, R060 38.3, R066 34.0, L075 32.6).
- I also tested knee 20 / scale 30. It correlates marginally less with
  sharpness (+0.042 vs +0.054) but **rewards blur more** (L055 Gaussian σ=1:
  96.8 with knee 20 vs 88.3 with knee 12, against 89.7 clean). Knee 12 wins.
- threshold 0.010 on stripe amplitude: 1 % of black–white, **~6× the worst
  value present anywhere on the record** (max 0.00171 on L051) and ~38× the
  median. A 2 % injected stripe costs 38 % of the score; the record's own
  worst frame costs 0 %. The visibility threshold for an alternating column
  pattern was *not* measured — 1 % is chosen as a wide margin above what the
  record actually contains, and should be revisited if any future change
  raises measured stripe amplitude above ~0.003.

**Tests that prove it worked** (all run in this pass, reproduce them):

1. `corr(composite, sharpness)` over 156 frames: **−0.169 → +0.054**.
   Softest/sharpest sharpness quartile means: **25.4 / 19.5 → 75.5 / 75.3**.
2. Injected damage still detected, on L055 / R074 / R073 (clean → injected):
   staircase 48 px/16 steps **89.7 → 52.3**, 89.6 → 35.0, 92.6 → 40.0;
   per-column jitter σ=1 px **89.7 → 35.2**, 89.6 → 32.0, 92.6 → 32.5;
   injected 2 % parity **89.7 → 54.2**, 89.6 → 51.3, 92.6 → 57.4.
3. Not inflated by sharpening: unsharp ×1.5 gives 85.7 / 84.2 / 89.8, i.e.
   **below** the clean 89.7 / 89.6 / 92.6 on all three.
4. Ranking sanity: worst frames become L001 13.9, **L032 20.2**, L021 29.6,
   L033 31.1, L075 32.9 — L032 moves from 29th-worst to **1st or 2nd worst**,
   and Mars (L011), the fish schools (R000–R002) and the sand dunes (L057)
   leave the bottom ten entirely.
5. Regression guard: the number of frames where the droop fix appears to make
   things worse falls from 59 to 0 *on the parity factor*; ~17 frames still
   lose >5 points through `res_drift`/`shift_rms` (median res_drift +2.5), which
   is the integrator's genuine low-frequency noise gain. That residue is
   **expected and must not be chased** — see §4.

**Warning that must go in the docstring.** The composite is a *timing-damage
detector*, not a picture-quality score. Gaussian σ=1 blur still *raises* it on
2 of 3 frames tested (+8.2, +6.1) even after this change. Never use it to
compare decodes of different sharpness; use it to detect that a change broke
the timebase, and use §4's ungameable criteria for everything else.

### 3.2 `colour.py` — registration (RANK 2)

**Replace** the full-band phase correlation and `ndimage.shift(order=1)` with:

```python
from . import fuse
reg, offsets = fuse.register(seps)      # band-limited + DFT-refined + Fourier shift
dy, dx = fuse.closure(seps)             # triangle-closure self-check
```

Parameters, all already in `fuse.py` and all measured:
- `REG_FMAX = 0.25` cyc/px. Verified insensitive: the answer is identical for
  low-edges 0.00, 0.02 and 0.05 cyc/px (§1.5), and degrades only when the band
  is opened past 0.35.
- `upsample = 100` (0.01 px grid) for the local inverse-DFT refinement;
  replaces parabolic interpolation, which carries up to 0.12 px of bias.
- `fourier_shift`, not `ndimage.shift(order=1)`: the triangle kernel costs
  14.5 % of the 0.15–0.30 cyc/px luminance energy.

**Default: ON** for all 20 colour images. It strictly reduces a residual
measured by an instrument that is not the estimator doing the work.

**Gate on closure.** `abs(closure) > 1.0 px` on either axis ⇒ do not register,
log it. This is what keeps image 8 (the solar spectrum, whose content really
does move ±90 px with wavelength) out of the registration path; its closure is
+0.41, **+13.63**, against a mean of 0.059 for real triplets.

**Tests.** (a) closure |err| ≤ 0.25 px on all 19 non-spectrum triplets;
(b) post-registration re-measured offset = 0.000 ± 0.001 px;
(c) fringe residual (R−G on ∇L) ≤ 0.15 px red and blue, mean ≤ 0.10 —
    currently 0.091 / 0.060 against colour.py's 0.398 / 0.227;
(d) luminance energy in 0.15–0.30 cyc/px must **rise** ≥ 10 % versus the
    order-1-spline path (measured +14.5 %). If it falls, a smoothing shift
    has crept back in.

**Correct the published claim.** The record currently states that along-trace
colour registration is already sub-0.1-sample. It is not: the picture creeps
**+0.329 px = +0.20 dots = +2.4 samples at 384 kHz per frame**, in the same
direction on 19 of 19 triplets (sign test p = 4e−6).

### 3.3 `colour.py` — fusion (RANK 3)

```python
planes, offsets, lum = fuse.fuse_colour(seps)
```

`fuse.chroma_gain` measures the Wiener gain `g(f) = 1 − N/I` per image from the
data; there is **no fixed chroma bandwidth to choose**, which matters because
incoherent (chroma) density is still 39×/21×/12×/6.8×/3.9× the noise density in
bands 0.14–0.18 … 0.30–0.35 cyc/px: a conventional narrow chroma low-pass would
throw away real colour.

**Default: ON** for the 20 colour images. **Do not apply to the 136 monochrome
frames** — there is nothing to fuse.

**Tests.** (a) `injected_noise_gain_db` = 4.78 ± 0.02 on every triplet (this is
the algorithm control; 4.77 is the sqrt(3) ideal); (b) `plane_noise_gain_db`
≥ 1.5 (measured 1.70–2.52, mean 1.96); (c) `nsr_fused_pct` ≈ 0.577 ×
`nsr_single_pct`; (d) prediction and end-to-end injection must agree to
0.05 dB.

**Do not ship super-resolution.** Recorded negative: the sub-dot phase
diversity is real (mean 0.287 of a dot, 8/19 triplets at the 1/3 optimum) but
the aliased energy is not — alias/power 0.056 ± 0.056 over a band that is
itself 0.37 % of picture power — and unfolding costs +1.48 dB mean, +11.9 dB
worst, of luminance noise.

### 3.4 `sync.py` — the L032 guard (RANK 4)

**Add** to `sync.py` (moved from `audit.reanchored_timebase`, which is a
measurement harness, not a shipping path):

```python
MAX_PARITY_DOTS = 0.8   # see below; audit proposed 1.0

def _reanchor(x, tb, half_window: float = 0.02):
    """Re-detect every landmark on the already-anchored zero-crossing grid
    using the per-trace steepest-fall + downward-zero-crossing discriminator,
    then re-fit line / wow / parity from those detections."""
```

**Trigger, in `recover()`, after the parity estimate is formed:**

```python
if abs(tb.parity_offset) > MAX_PARITY_DOTS * tb.period / dotclock.DOTS_PER_TRACE:
    tb = _reanchor(x, tb);  tb.reanchored = True     # and log it
```

**Threshold justification.** Measured over all 156 frames, the parity
alternation is **−0.531 ± 0.455 dots** (p1 −0.54, p99 +0.27) — the field-ID
alternation is physically half a dot and that is what is there. L032 sits at
**−5.84 dots**. I specify **0.8**, not the audit's 1.0: it still fires on L032
alone across the whole record (verified), with 1.5× more margin above the
population and 7× below L032.

**Default: ON, trigger-only.** It must never run unconditionally: on L000 it
moves the grid 3.54 samples and regresses the calibration circle's axis ratio
1.0060 → 1.0079 (§1.4), which §4 forbids.

**Also log, unconditionally, without acting on them**, both measured over all
156 this pass: `accept_frac < 0.95` (L032 = 0.244; next lowest **L038 at
0.994**, a 4× separation) and `meas_noise > 2.0` (L032 = 5.57; next highest
**L014 at 0.830**, a 6.7× separation). Three independent tripwires, each
isolating exactly the same single frame.

**Tests.** (a) L032: `parity_offset` −17.78 → +1.30 samples, `accept_frac`
0.244 → 1.000, composite 9.1 → 62.3, `shift_rms` 0.311 → 0.267. (b) The
trigger fires on **exactly one** of 156 frames. (c) L000's circle is
bit-identical to the current build: axis ratio 1.0060, radial rms 0.8752 px.
(d) L031, L055, R040, L020 decode byte-identical (the guard never fires).

### 3.5 `dotclock.py` — matched filter (RANK 5)

**Add**, replacing the FFT-bin detector as the primary path:

```python
SEARCH_DOTS = 0.6        # sanity rail only; see the +-3 dot control below
MAX_HALF_DISAGREE = 0.15 # dots, between independent trace halves
MIN_EXCESS = 6.0         # peak / median-far-field

def fold_coherence(dx_abs, starts, period, lo_f, hi_f, spds, sl=slice(None)) -> np.ndarray
def measure_mf(x, tb, *, lo_f, hi_f) -> DotClock   # cross-validated on trace halves
```

Algorithm as in `audit._fold_coherence`: gate |dx| to the picture band of every
trace, take one zero-padded (2^22) DFT of the whole frame, read the candidate
rates off it. One FFT per frame; 156 frames in 69 s wall clock in this pass —
**no O(n²) correlation anywhere.**

**Accept only if** `|h1 − h2| < 0.15` dots **and** `excess > 6`. Otherwise fall
back to the predicted `period / 262.5` exactly as today. Measured: 147/156 pass;
excess min 8.9, p10 13.6, median 22.8; split-half median 0.012, p90 0.135.

**Default: ON.** It removes 17 physically impossible rates from the build (the
worst, L011 at 263.101, is 13σ from the hardware constant) and recovers the
clock on 64 of the 66 frames that currently fall back to interpolation.

**Tests.** (a) rate = 262.4926 ± 0.058 over 156, patent value 262.500;
(b) **no frame within 0.02 of an integer** (line-harmonic control);
(c) **no frame within 0.05 of the ±0.6 rail** (railing control);
(d) **re-running at ±3.0 dots reproduces the ±0.6 peak on 156/156** — this is
the test that proves the narrow band is not manufacturing the answer, and it
must be kept as a regression test;
(e) the 17 named frames change rate: L006 L007 L010 L011 L012 L033 L050 L060
R006 R012 R019 R022 R030 R056 R065 R067 R072.

**Set expectations in the changelog:** forcing a rate 0.6 dots wrong changes
sharpness by <1 % and `shift_rms` by <0.01, because `track_phase` re-measures
phase every 64 traces and absorbs rate error. **Nobody will see this change.**
Ship it because the current detector silently reports rates the hardware cannot
produce, not because the pictures improve.

### 3.6 `testset.py` — stale labels (RANK 6)

The frozen 16-frame set stays frozen; only the expectation column is wrong.
`L040` and `L075` are no longer "destroyed by rank-1 smear" — post-droop-fix
L040 ("Birth") is fully resolved. `R000`, `R022`, `R070`, `R077` labelled bad
decode cleanly. Add a dated note rather than rewriting history.

### 3.7 Docstring retractions (RANK 7)

- `decode.py`, "KNOWN RESIDUAL … an extra ~15 % shelving loss between ~200 Hz
  and ~2.5 kHz … mid-scale structure stays ~15 % under-restored" → **retract.**
  The number is set by the estimator's normalisation band: refitting the same
  frames with the flat band moved gives |S(180 Hz)| anywhere from **0.558 to
  1.089** with the shape flat in every band. No mid-scale under-restoration was
  ever demonstrated.
- `droop_blind.py` points 2, 3 and 5 carry the same retraction; point 4's
  "normalisation band moves |H(180)| by ~0.03" understates the sensitivity by
  ~10× (true swing 0.56 → 1.09).
- **Add** the positive result: `UNCOUPLE_TAU_384` is now confirmed a second
  time and model-free — `H_L/H_R = (D_L/D_R)·const` needs no source model
  because the same converter generated both channels' sync patterns, and it
  follows pole(530)/pole(295) to **2.8 % max, 1.0 % mean, phase included**,
  over 180 Hz–2.5 kHz.

### 3.8 Camera deconvolution — NOT SHIPPED

Keep `Settings.deconv = 0.0`. Do not wire `deconv.wiener_axis` into the build.
Three reasons, in order:

1. `wiener_axis`'s `Pobs` is an unwindowed periodogram. On L000 — a **uniform
   white slide** — it reports S/N ≈ 10 at along-trace Nyquist; with a Hann
   window the same data give 0.5–1.6. That is spectral leakage from the
   field's own shading being read as recoverable detail. Any resolution claim
   built on it is self-flattering.
2. Across-trace, the band a sharpener would boost is contaminated. Hann-
   windowed across-trace power at Nyquist over the 0.35–0.45 mid-band: 1.16
   (L055), **7.35 (R040)**, 0.95 (L002) — while the same ratio *along* the
   trace is 0.29–0.62, i.e. falling, as a post-camera optical spectrum must.
   A PSD-driven Wiener reads that rise as signal and applies ~6× gain; the
   first sweep did exactly that (noise 0.0038 → 0.0164, coherence 0.976 →
   0.640, composite 31 → 1).
3. The headroom is smaller than reported (§1.6).

If it is ever exposed, the terms are: along-trace axis only, **off by default**,
labelled *"camera deconvolution — restores the slide, not the recording"*, with
the kernel width and its provenance printed with the image, and gated on the
§4 invariants.

---

## 4. Invariants — automatic rejection

Any change that does one of these is rejected regardless of how good it looks.

1. **The calibration circle.** `quality.circle_metrics` on the L000 decode must
   hold `axis_ratio ≤ 1.0060` and `radial_rms ≤ 0.88 px`, with `inliers ≥ 180`
   and `coverage_deg ≥ 300`. Current build, re-measured this pass: **1.0060 /
   0.8752 px / 190 inliers / 320°**. (Pre-droop for reference: 1.0075 /
   0.9477.) *This rule already killed one candidate in this pass:
   unconditional sync re-anchoring, at 1.0079.*
2. **The L000 flat field.** The frame is a uniform white slide. Field mean must
   stay ≥ 0.84 (current 0.8429; pre-droop 0.4695 — the droop fix is what buys
   this) and the field must not acquire a monotone ramp.
3. **The scene-free tau probe.** `UNCOUPLE_TAU_384 = {"L": 530, "R": 295}` is
   fixed. It may only change if the change is demonstrated on the parity
   sync-width probe on content-free gap lines *and* survives the model-free
   channel-ratio test (H_L/H_R must track pole(530)/pole(295) to better than
   the current 2.8 % max / 1.0 % mean). No refit from picture statistics, ever.
4. **No reference imagery in the decode path.** `decode.py` may continue to
   import only numpy + `dotclock` + `sync`, and its only data input is the WAV.
   Everything specified above obeys this: `residual_drift` and `stripe_amp` are
   image statistics of the decode itself; `fuse` registers the record's own
   three separations against each other; the dot clock and the L032 guard are
   pure signal measurements.
5. **Gap lines stay flat, bright objects trail nothing.** Gap-line residual and
   the level behind Mars (L011) / L055's bright objects are the arbiters for
   anything touching the low-frequency response. The pole currently takes the
   trailing-shadow hold ratio from 0.05 to 0.87 (over-correction past 1.0 is a
   failure, not a win — the shelf's magnitude reached 1.11).
6. **`build.py` must not tune on the frames it reports.** The 16-frame frozen
   set is a report, not a training set; every threshold in §3 was set from the
   full 156-frame distribution or from injection, not from the frozen set.

---

## 5. What remains unrecovered, and how close we are to the ceiling

### 5.1 The measured ceiling

Along a trace the decoder samples **230 dots** (one per NTSC source line) and
across the frame **512 traces**. The camera's sharp-end 10–90 rise, after
correcting the p10 selection bias of §1.6:

| axis | sampled | 10–90 rise (sharp end) | resolvable elements | oversampling |
|---|---|---|---|---|
| along-trace | 230 dots | **1.3 – 1.7 dots** | **138 – 172** | 1.34 – 1.67× |
| across-trace | 512 traces | **1.6 – 2.0 traces** | **260 – 324** | 1.58 – 1.97× |

The recording channel itself is nowhere near this: a single-trace edge fit on
the recording channel gives 1.75–2.37 samples = 0.14–0.19 dots, a capacity of
over 1000 elements per trace. **The 1977 camera is ~8× worse than the tape,
and the decoder is already sampling ~1.4–1.9× above the camera's Nyquist in
both axes.**

So: **we are at the ceiling.** No change to sampling, timing, interpolation or
filtering can add resolution, because the resolution is not in the signal. The
138–172 elements per trace is consistent with the project's established
130–190 and is now measured two independent ways. Everything left is either
(a) noise, (b) low-frequency response, or (c) inference about the slide.

### 5.2 The one real open defect: the sub-120 Hz tilt

After the pole, 85–90 % of what is left on content-free gap lines is a
**straight line**: rms 0.0033–0.0080 falls to 0.00044–0.0025 under linear
detrending, leaving **+0.011 to +0.023 per trace, i.e. 4–9 % of black-to-white
across a trace**. It is already present in the raw signal (L010 raw +0.0132 vs
corrected +0.0113) and it lives **below 120 Hz**, beneath the parity probe's
lowest clean harmonic, where the probe is hum-degenerate and measures nothing.
A second inverse pole reduces it on 6 of 8 frames but always adds curvature and
makes R020/R065 worse — it is not one extra pole.

This is the defect the "shelf" was mistaken for, and it is genuinely
unidentified. L000's field carries a 27 %-of-contrast version of the same ramp,
but **L000 cannot arbitrate**: a 1977 vidicon's vertical shading prints an
identical monotone ramp along a trace.

### 5.3 What I would attack next, in order

1. **A scene-free probe below 120 Hz.** Everything about the low-frequency
   response is unresolved because the only content-free excitation on this
   record — the parity sync-width alternation — has no energy there. The
   inter-frame gaps are the candidate: long runs of known-constant input,
   several per channel, at the right timescale. If a 10–100 Hz response can be
   measured on gap runs the way `droop.py` measured the pole, the tilt becomes
   correctable; if it cannot, say so and stop.
2. **Vertical banding.** Still the dominant visible artefact on all 156 frames
   and still unattributed. The per-trace back-porch clamp moves 8–24 % of
   black–white peak-to-peak across a frame with 0.6–1.7 % trace-to-trace
   scatter; whether that scatter is the cause was **not** established. The
   experiment is a clamp-off / clamp-smoothed A/B judged on gap flatness and
   the L000 field, not on the composite.
3. **The porch window on bright-left-edge frames.** `PORCH_START..PORCH_END =
   0.020..0.072` is content-free only while the slide's left edge is dark.
   L032 (picture step 0.72 of sync amplitude inside the window), R063 (0.43)
   and R066 (0.42) violate that. Narrowing the window to 0.020..0.045 was
   tested and is **worse** (L000 black clipping 4.2 % → 34.0 %). Unresolved; a
   per-frame adaptive window gated on the measured step is the obvious next
   try, and it must be checked against the L000 clipping figure.
4. **The re-anchoring discriminator as a general landmark detector** (§1.4). It
   cut L000's circle radial rms from 0.875 to 0.725 px — a 17 % improvement in
   the one ground-truth timing metric on the record — while regressing the axis
   ratio. Worth one careful pass with the circle as arbiter; worth nothing at
   all if it cannot hold both numbers at once.
5. **Fusion beyond colour.** 20 images get √3 of noise averaging because they
   were scanned three times. The other 96 were not, and nothing here helps
   them. That asymmetry is now the largest remaining difference in output
   quality across the record, and it is not fixable — it is what was recorded.
