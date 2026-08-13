# breakthrough_spec.md — adversarial review of the four attacks, and what to integrate

Reviewer's remit: refute before believing. Every number below was produced by code I
ran myself against `data/master/384kHzStereo.wav` in this session. Where I merely
reproduce another agent's number I say so; where I could not reproduce it, or could
not reproduce it at the size claimed, I say that too. Nothing here was tuned on one
set of frames and reported on another: the arms are the frozen `testset.TEST_FRAMES`
16, and where I use a subset it is named and the reason given.

Scratch harnesses live in the session scratchpad (`gt_control.py`, `gt_control2.py`,
`ap_width.py`, `ap_calib.py`, `decim_mtf.py`, `ap_holdout.py`, `integrate.py`,
`flat.py`, `esf.py`, `coverage.py`, `strength.py`, `hd.py`, `wb_hp2.py`). They are
disposable; the reproduction recipes are given inline below.

---

## VERDICT IN ONE TABLE

| attack | claim | my verdict | frames touched |
|---|---|---|---|
| **globaltime `wideband`** | timebase follows flutter sync's smoother drops | **SURVIVES — integrate, default ON** | **156 / 156** |
| **aperture `apply_dots`** | exact, known dot-domain operator worth +3.16 dB at dot Nyquist | **SURVIVES with conditions — integrate, default ON only where the dot clock is measured** | **90 / 156 (58%)** |
| **halfdot** | quincunx is real but already exploited; no gain | **NULL CONFIRMED — no change** | n/a |
| **mapsolve** | MAP solve does not beat the sequential chain | **NULL CONFIRMED — no change** | n/a |

Two real wins, both small and both reference-free. Two rigorous nulls. Nothing here
is a new picture; together they are worth about **+10% of effective resolution along
the trace and about a quarter of a pixel of trace placement**.

---

## 1. globaltime `wideband` — SURVIVES, and it survives the gate its own author left open

### 1.1 The missing control, run

globaltime shipped `scrambled()` written but never wired into `report_images`, and
said plainly that until the scrambled arm was shown to *raise* the picture-side
residual, the −54% hold-out "could in principle mean only 'the grid moved'". That was
the one gate. I ran it, and then four more controls it did not propose.

Picture-side residual (correction built from the SYNC alone; residual measured from
the PICTURE by same-parity column pairs, 5–25 Hz), all 16 test frames:

```
            base    wideband   scram(0)  scram(17)
MEAN       2.104     0.962      2.138     2.683
lower than base on:  16/16       4/16       2/16
wideband lower than scram(0) on 16/16
```

I reproduce globaltime's headline exactly: **2.104 → 0.962, −54.3%, lower on 16 of 16.**

Stronger controls, on 6 frames (these preserve more structure than phase-scrambling —
`rev` has the identical amplitude spectrum *and* the identical sample distribution,
only the alignment destroyed; `neg` is the correct correction with the wrong sign):

```
frame    base    wide     rev  roll+37  roll-91     neg  scram(mean of 6 seeds)
 L000   1.900   0.794   2.720    2.248    2.428   2.646   2.285
 L020   1.864   0.746   2.667    1.927    2.659   2.570   2.181
 L055   2.072   0.800   2.645    2.380    2.650   2.710   2.440
 R040   1.894   0.841   2.161    2.217    2.541   2.815   2.340
 R056   1.860   0.782   2.728    2.291    2.266   2.488   2.306
 R010   2.051   0.840   2.162    2.444    2.523   2.700   2.500
```

**Every control raises the residual; `wideband` is the unique minimum on every frame.**
The sign-flip control (`neg`, +33% to +49%) is the decisive one: a correction that
merely "moved the grid" would be symmetric in sign. This one is not.

### 1.2 It also improves an invariant it cannot game

L000 calibration circle, decoded through the full shipping chain:

```
arm      axis_ratio   radial_rms   inliers
base       1.0060       0.875 px     190     <- matches the frozen baseline exactly
wideband   1.0050       0.863 px     189     <- BETTER on both
aperture   1.0063       0.877 px     190     <- within noise
both       1.0057       0.867 px     189     <- better than base
```

`quality.circle_metrics` documents radial_rms as "residual timing error that
sharpening cannot fake". `wideband` is a pure timing correction and it lowers it.
Axis ratio moves *toward* 1.000. **No regression, and a small genuine improvement.**

### 1.3 Image metrics, sync-side factors held at baseline

Mean over the 16 frozen frames (I hold `quality.frame_report`'s timebase at the
*baseline* `tb` so a timebase that tracks its own landmarks more closely cannot
inflate `accept_frac` — globaltime's own precaution, kept):

```
arm       composite  shift_rms  res_drift  parity_db  sharpness
base        68.89      0.398      15.25      2.83      0.0191
wideband    72.30      0.366      14.71      3.03      0.0190
aperture    69.86      0.389      15.9*      3.04      0.0204
both        73.31      0.356        —        3.28      0.0203
```

`shift_rms` 0.398 → 0.366 reproduces globaltime's number to three decimals, better on
**15 of 16** frames. `sharpness` is unchanged (0.0191 → 0.0190): **this is not a
cosmetic sharpener**, which is exactly what you want from a timing fix.

### 1.4 Reported negatives

* **3 of 16 frames regress on `composite`**: L075 (32.9 → 22.2), L077 (49.2 → 45.3),
  R010 (51.6 → 48.0). In all three the *only* factor that moves the wrong way is
  `res_drift` (L075 32.6 → 43.8). `shift_rms` improves on L077 and R010 even as
  composite falls, so this is partly the metric disagreeing with itself.
* **L075 is the one frame where the correction is genuinely wrong**: `shift_rms`
  0.810 → 0.948. L075 is also the only test frame with **located < 99%** (92.4%,
  39 coasted traces). `wideband` sets the residual of coasted traces to exactly 0
  before filtering, which puts a step into the correction at every coast. That is a
  mechanism and it gives a gate (§5.1).
* **`parity_db` regresses**, 2.83 → 3.03 mean. Small but systematic.
* **A refinement I proposed and then had to withdraw.** I hypothesised that the
  `res_drift` regressions came from `wideband` adding back the very-low band that
  `sync.recover`'s Savitzky-Golay smoother already covers (~0–4 Hz = bins 0–2 of the
  64-bin gain). I high-passed the correction zero-phase at 127/63/31 traces and
  measured: hold-out 0.962 → 0.956/0.957/0.966, composite 72.30 → 72.30/72.25/72.23,
  **the same 3 frames still regress**. The idea is a **null**. Report it as such.
* **A latent bug found while testing that**: `globaltime._fir_from_gain` ends with
  `return h / h.sum()`, which pins the filter's DC gain to 1 regardless of
  `WIENER_GAIN[0]`. `WIENER_GAIN`'s DC bin is inert, and zeroing low bins makes the
  normaliser ~0.0015 (and at k=3, **−0.001**, which inverts the entire filter and
  destroys every frame: composite → 0.007). Anyone who later edits the low bins will
  get silence or catastrophe with no warning. Fix or document before integration.

---

## 2. aperture `apply_dots` — SURVIVES, but three things in its report do not

### 2.1 First, a delivery defect that must be recorded

**`pipeline/aperture.py` imports only `numpy` and `dataclasses`. It never opens the
master.** The 323 single-trace step fits, the 0.49-dot width, the 16-frame
cross-trace half-window test, the 6/6 hold-out, the fixed-edge-set MTF numbers and
the circle/flat-field checks exist in that file **as prose in the docstring only**.
None of it is reproducible from the delivered artifact. That is precisely the
condition under which this project has previously shipped a number that turned out
to measure its own error.

So I re-measured the one constant the entire operator depends on.

### 2.2 The width: independently re-measured, and it holds

Single-trace fits only, never stacked (so our own dot-clock jitter cannot enter),
on undrooped 384 kHz data, isolated plateau steps with two flat dots either side:

```
frame   n   median 10-90 (dots, raw estimator)
L000   17          0.382
L020   41          0.378
L034    7          0.383
R010   34          0.369
L055   89          0.395
R040   91          0.359
```

That raw estimator is **biased low**, because the plateau values it normalises
against are themselves full-dot integrals that already contain part of the
transition. I calibrated it on a synthetic staircase of known width through the same
code path:

```
TRUE 10-90   estimator reads
   0.40           0.330
   0.45           0.360
   0.49           0.382
   0.55           0.416
```

Inverting: **my measured 0.359–0.395 corresponds to a TRUE 10-90 of 0.449–0.512
dots, centre 0.482.** aperture ships `TRANSITION_1090 = 0.49` with a stated honest
range of 0.464–0.523. **Confirmed, independently, by a different estimator with a
different bias.** This is the strongest evidence for the module and it is the reason
to accept it.

### 2.3 Its negative result about `/4` decimation is also confirmed — properly

aperture argued the `/4` decimation in `build.py` is not a hidden aperture because
the width "reads the same in dots at both rates". That argument is not sound — at
96 kHz a 0.49-dot rise is 1.4 samples and my estimator saturates at ~0.41–0.43 dots
regardless of the truth, so it cannot resolve the question either way. But the
conclusion is right, for a better reason. Measured directly from the filter
`resample_poly(x, 1, 4)` actually uses:

```
|H| at dot Nyquist (f = 0.04106 cyc/sample) = 1.00086
|H| at 2x dot Nyquist                       = 1.00169
equivalent Gaussian 10-90 of the decimation filter = 0.0000 dots
```

**Flat to 0.09% across the whole dot band.** `build.py`'s decimation adds nothing.
Negative result stands.

### 2.4 The hold-out: real, but smaller than reported, and it does not confirm 1.0

Re-implemented independently: each dot estimated from the settled **central 40%** of
its own plateau, corrected for that window, then used to predict the mean of the raw
384 kHz samples in the **outer 60%** of every dot — samples that contributed nothing
to the estimate.

```
strength    L000    L020    L034    L055    R056    L002     (ratio to strength 0)
   0.0     1.0000  1.0000  1.0000  1.0000  1.0000  1.0000
   0.5     0.9967  0.9947  0.9970  0.9974  0.9979  0.9942
   1.0     0.9943  0.9912  0.9952  0.9960  0.9965  0.9900   <- shipped
   1.5     0.9929  0.9896  0.9947  0.9959  0.9959  0.9876   <- hold-out optimum
   2.0     0.9926  0.9902  0.9954  0.9971  0.9961  0.9869
   3.0     0.9950  0.9978  1.0011  1.0038  0.9989  0.9914
```

* **The direction is real and the test is bounded.** Strength `−1.0` is worse at every
  width (1.006–1.34); strength 3.0 turns back and is worse than no correction at all
  on L034/L055. A test that simply rewarded sharpening would not do that.
* **The effect size is 0.4–1.3%**, not the 0.5–4.9% aperture reported. I am at the
  bottom of its range. Different window definitions, so not a contradiction, but I
  cannot reproduce the top of it.
* **The hold-out prefers strength 1.5–2.0, not 1.0.** aperture's own explanation is
  correct — the predictor uses each trace's own imperfect lattice phase, so a larger
  correction absorbs timing error — but the consequence must be stated: **the
  hold-out does not confirm the shipped value. It confirms only the sign and the
  boundedness.** The value 1.0 rests entirely on the directly measured width (§2.2),
  and shipping 1.0 rather than the hold-out's 1.5 is the correct conservative choice,
  because correcting to the hold-out optimum would be deconvolving our own dot-clock
  jitter — the retracted-MTF mistake.
* A width sweep at fixed strength 1.0 gives a **broad flat minimum from 0.49 to 1.00
  dots**, turning over past 1.3. It does not rail. My first pass looked like a rail
  and was not; I record that because the ratio-to-uncorrected framing exaggerates it.

### 2.5 The metrics CANNOT support it, and this is the finding to carry forward

At strength 1.0 the correction's impulse response is
`[..., -0.1088, +1.1996, -0.1088, ...]` — **to three decimals the canonical 3-tap
Laplacian sharpener.** Its *shape* therefore proves nothing whatsoever. Only the
*amount* is a physical claim. So I swept the amount, on the 12 dot-locked test frames:

```
strength  composite  shift_rms  coh_ratio  sharpness
   0.0      68.58      0.3927     0.9871     0.0190
   0.5      68.82      0.3879     0.9875     0.0196
   1.0      69.46      0.3844     0.9879     0.0203
   1.5      69.81      0.3831     0.9885     0.0212
   2.0      69.57      0.3818     0.9893     0.0222
   3.0      68.76      0.3928     0.9914     0.0248
```

**`shift_rms` improves monotonically to strength 2.0 and `coh_ratio` to 3.0**, for a
filter that acts along the trace and does nothing at all across traces. `composite`
peaks at 1.5. **These metrics are rewarding the quantity of sharpening, not the
correctness of the operator.** The `+1.3%` composite gain at strength 1.0 must NOT be
cited as evidence for the aperture. It is not evidence. Only §2.2 (measured width),
§2.4 (bounded hold-out) and §2.6/2.7 (ESF agreement and invariants) are.

### 2.6 The ESF does what the operator says it does, and no more

Single-profile `deconv.fit_edge` (erf), edge set selected once on the **baseline**
image and refitted in the same windows afterwards, slant-gated, dot domain:

```
frame    axis    n    base    aper    change   operator's OWN prediction
L020    along  102   1.631   1.399   -14.2%          -13.0%
L034    along   47   1.816   1.575   -13.2%          -11.2%
L000    along   82   1.930   1.729   -10.4%          -10.4%
R056    along   78   2.114   1.911    -9.6%           -9.2%

L034   across   24   1.707   1.696    -0.7%   (control: must be ~0)
L055   across  167   1.752   1.748    -0.3%   (control)
R056   across   38   1.837   1.856    +1.0%   (control)
```

The "operator's own prediction" column is the same correction applied to a synthetic
erf edge of the measured base width and refitted with the *same* fitter. **Measured
narrowing matches prediction to within 0 to +2 percentage points on all four frames.**
That is the check a ringing-driven over-narrowing fails, and it is tighter than the
+3%/+13% aperture claimed for itself.

**Across-trace control passes on the three frames with enough axis-aligned edges.**
I exclude L020's across-trace reading (base 0.860 dots, n=21): a sub-dot "edge" width
is not a scene edge, it is L020's known 90-px staircase artifact, and the wideband arm
returns 0.415 dots there, which is nonsense. Excluded and stated, not hidden.

**R040 measured +0.0% change** — because R040's dot clock is not measured, so decode
takes the Lanczos fallback and the plateau path never runs. That is the operator's own
precondition asserting itself, and it is the gate in §5.2.

### 2.7 Invariants: all hold

```
L000, ring interior         base      aperture
mean                       0.8476     0.8476     <- DC exactly preserved
sd                         0.0280     0.0281
p99 - p1                   0.1071     0.1086
row-to-row noise           0.00296    0.00379    <- x1.28
```

Flat field, circle and DC all reproduce aperture's stated numbers **exactly**
(0.0280 → 0.0281; 0.1071 → 0.1086; x1.21 predicted / x1.28 measured noise). So the
report is honest even though the code is absent. That materially raises my confidence
in the parts I could not re-derive — but it does not remove the obligation in §5.4.

### 2.8 Coverage: 58%, not 100%

`dotclock.measure` over **all 156 frames** (`MIN_STRENGTH = 1.8`):

```
dot clock measured:  90 / 156 frames (58%);  median strength 1.99
dots/trace over those 90:  262.457 +/- 0.123
sync landmarks located:  99.9% of traces; 156/156 frames above 50%
```

The 66 frames without a measured dot clock include **R000, R010, R040, R070** from the
frozen test set. On those, `decode` bins with a stretched Lanczos kernel at the
predicted pitch and `aperture.transfer` does not describe what happened. My
`integrate.py` run applied the correction to them anyway and their composites all went
*up* — which is exactly §2.5's point restated: the metric rewards sharpening even where
the physics is absent. **The gate is mandatory, not advisory.**

---

## 3. halfdot — NULL CONFIRMED

Re-ran `halfdot.holdout` on 4 frames across 3 estimator configurations (the failure
mode for a null is a weak estimator, so the estimator is swept):

```
config          true/naive   true/shuffled   true/separable
Wx3 Wy3 deg3      0.8999        0.8454          0.9969
Wx4 Wy4 deg4      0.8813        0.8334          0.9985
Wx2 Wy5 deg3      0.9253        0.8931          0.9983
```

* The offsets are real: using them beats ignoring them by 7–12%.
* The shuffled-offset control fails as it must (wrong offsets are worse than none).
* **`true/separable` is 0.995–1.001 in every cell.** `decode._rows_from_dots(delta)` —
  per-column band-limited resampling — already recovers 99.7% of what a joint 2-D
  non-uniform gridder recovers.

halfdot's conclusion is correct and its recommendation ("no change to decode.py") is
correct. **Do not integrate a gridder.** The one thing worth acting on is its
suggestion that a wider fractional-delay kernel in `_rows_from_dots` (Lanczos-8 vs
Lanczos-3) would collect the last 0.3% — and it is below the noise, so it is optional.

---

## 4. mapsolve — NULL CONFIRMED

`python -m pipeline.mapsolve --frame {L055,R040} --holdout` reproduces exactly:

```
                                  L055      R040
neighbour mean, UNREGISTERED    0.00487   0.00892
neighbour mean, REGISTERED      0.00386   0.00707     <- best
MAP Tikhonov lam=0.03           0.00554   0.00782
```

* MAP loses to a plain two-neighbour Lanczos-shifted average on both frames.
  **Do not integrate.**
* Its finding 4 is confirmed and is worth keeping: **registering the sub-plateau
  offsets beats ignoring them by 20.7% on both frames** — an independent, ungameable
  confirmation of `decode.py`'s `delta`, from predicting traces the estimate never saw.
  It agrees with halfdot's `true/naive` from a completely different estimator.
* Its run also independently reports `R040 ... dot_locked False (strength 1.12)`,
  corroborating §2.8 from a third code path.
* **Keep `mapsolve.trace_holdout` as a permanent test.** It is cheap, reference-free,
  and any future registration or super-resolution claim should have to pass it.

---

## 5. INTEGRATION SPEC

### 5.1 `sync` — the wideband timebase

**Function.** Move `wideband`, `_fir_from_gain`, `_filtfilt_reflect`, `_parity_signed`
and `WIENER_GAIN` from `globaltime.py` into **`sync.py`**, and call it at the end of
`sync.recover`. Do **not** make `decode.py` import `globaltime`: `globaltime` imports
`catalog`, which reads `data/frame_map.json`, and that would end `decode.py`'s
property of importing only `numpy` + `dotclock` + `sync` with the WAV as its only data
input. `wideband` itself needs nothing but a `Timebase`.

```python
def wideband(tb, gain=WIENER_GAIN, clip_mad=6.0) -> Timebase
```

**Parameters, measured values.**

| parameter | value | provenance |
|---|---|---|
| `gain` | `WIENER_GAIN`, 33 bins, unity below 0.33 cyc/trace, rolling to 0.01 at Nyquist | gap-PSD / picture-PSD over 3839 picture and 33 content-free segments on the master |
| `clip_mad` | 6.0 | outlier guard; unchanged |
| `FIR_TAPS` | 65 | unchanged |

**Default: ON, for every frame** (156/156 have >99% of landmarks located).

**Gate.** Skip when `located.mean() < 0.99`. L075 is the only frame in the frozen 16
below that line (92.4%) and is the only frame whose `shift_rms` regresses. Mechanism:
`wideband` zeroes the residual of coasted traces before filtering, so every coast
injects a step. The cheaper alternative to a gate is to **interpolate the residual
across coasted runs instead of zeroing it**; that is untested and should be measured
before it is preferred.

**Test that proves it (all must pass):**
1. `picture_tracks` residual, 5–25 Hz, must fall on 16/16 frames — **and the four
   controls (`scrambled` two seeds, time-reversed, circular-rolled, negated) must all
   RAISE it.** Wire `scrambled` into `report_images`; add `rev` and `neg`. This is the
   gate globaltime left open and it is now mandatory in the test, not optional.
2. L000 circle: `axis_ratio <= 1.0060` and `radial_rms <= 0.875 px`. (Measured
   1.0050 / 0.863 — it improves them; a regression means something else broke.)
3. `shift_rms` must improve on >= 14 of 16.
4. `sharpness` must NOT change by more than 1% — this is a timing fix, and any
   sharpness movement means it has become something else.

**Known cost:** `res_drift` regresses on L075/L077/R010, `parity_db` 2.83 → 3.03.
Accepted; documented; not fixed by low-band high-passing (measured null, §1.4).

### 5.2 `decode` — the aperture correction

**Function.** Add to `aperture.py` nothing; call from `decode.decode` at exactly one
point — on the `(n_traces, n_dots)` matrix `pic`, **after** `dc_restore`, `dehum` and
dropout repair, **immediately before** `_rows_from_dots`:

```python
if cfg.aperture > 0 and dot_locked:
    pic = aperture.apply_dots(pic, strength=cfg.aperture,
                              width=aperture.TRANSITION_1090, axis=1)
```

Order matters and is not arbitrary: after `dehum` so the correction does not amplify
the fixed-pattern profile's high-dot-frequency content; before `_rows_from_dots` so it
is not correcting the row interpolator as well. It commutes exactly with `dc_restore`
(unity gain at DC, verified: L000 ring-interior mean unchanged to 4 decimals) and with
dropout repair (both linear across traces).

**Parameters, measured values.**

| parameter | value | provenance |
|---|---|---|
| `TRANSITION_1090` | **0.49 dots** | aperture's 323 single-trace fits; **independently re-derived here as 0.449–0.512, centre 0.482**, by a differently-biased estimator calibrated on a known synthetic |
| `strength` | **1.0** | = full inverse of the exactly-known `A(f)`; `A(0) = 1`, `A(0.5) = 0.695`, no null, inverse bounded by 1.44 |
| `gain_cap` | 2.0 | fail-safe only; never binds at these widths |
| `window` | `(0.0, 1.0)` | decode integrates the whole plateau |

**Default: ON at strength 1.0, and ONLY when `dot_locked` is true.** Off otherwise.
This is not a stylistic preference: on the 66 non-dot-locked frames `decode` bins with
a stretched Lanczos kernel at the predicted pitch, `A(f)` does not describe what
happened, and the metrics will still improve (§2.8) — which is exactly why the gate has
to be in the code and not in the operator's judgement.

**Do NOT choose `strength` by any image metric.** §2.5 shows `shift_rms`,
`coh_ratio`, `sharpness` and `composite` all keep improving past 1.0. The value 1.0 is
fixed by the measured width and by nothing else. If the width is ever re-measured,
`strength` stays at 1.0 and `width` changes.

**Test that proves it (all must pass):**
1. **Hold-out**: estimate dots from the settled central 40% of each plateau, correct
   for that window, predict the outer 60%. `strength +1.0` must beat `0.0`, and
   `strength -1.0` must be worse than `0.0`, on every frame tested. (Measured:
   0.4–1.3% better, and the sign control fails by 0.6–34%.)
2. **ESF, fixed edge set chosen on the uncorrected image**: along-trace 10-90 must
   narrow by within 3 percentage points of the operator's own prediction on a
   synthetic edge of the same starting width. (Measured: 0–2 pp on 4/4 frames.)
   More narrowing than predicted means ringing, and fails.
3. **Across-trace control**: |change| < 2% on frames with >= 20 axis-aligned
   across-trace edges. (Measured: −0.7%, −0.3%, +1.0%.)
4. L000 circle: `axis_ratio <= 1.0065`, `radial_rms <= 0.88 px`. (Measured 1.0063 /
   0.877.)
5. L000 flat field: ring-interior **mean unchanged to 4 decimals** (this is the DC
   invariance the operator claims by construction) and `sd` within +0.0002.
6. Noise: dot-domain row-to-row noise must rise by no more than x1.30. (Measured
   x1.28 against x1.21 predicted for white noise.)

### 5.3 What must NOT be combined with what

* **`aperture.apply_dots` and `deconv.py` must never both run on the same data without
  a re-fit.** `deconv.py` fits its ESF on edges of the *decoded* dot matrix, which
  already carry `A(f)`; that fitted width IS the composite, and deconvolving it removes
  `A` as a side effect. Exactly one of: (a) `apply_dots` first, then **re-fit**
  `deconv`'s ESF on the corrected matrix; or (b) `apply_dots` off. Recommend (a).
  This collision is currently latent, not active: **`Settings.deconv` defaults to
  `0.0`**, and even when enabled `decode._wiener_rows` uses `measure_psf(tb)` — the
  sync-edge response, which is a near-no-op — not `deconv.py`'s camera fit. The
  collision arms itself the moment `deconv.py` is wired in. Put the warning in
  `Settings.deconv`'s docstring now.
* **`aperture` and `undroop` do not collide.** `undroop` is a one-pole inverse with
  tau 530/295 samples = 43/24 dots; it is a low-frequency boost and its gain at the dot
  Nyquist is unity. `aperture` acts only near the dot Nyquist. Different bands, no
  double-correction. **The scene-free tau probe is untouched**: `undroop` runs on the
  raw signal with a hard-coded per-channel tau, before any dot exists; neither
  candidate modifies the raw signal or refits tau. Do not let either become a reason
  to refit tau from picture statistics.
* **`aperture` and `wideband` are independent and may both be on.** Measured: base
  68.89, aperture alone 69.86, wideband alone 72.30, both 73.31 — additive to 0.04.
  Circle with both: 1.0057 / 0.867 px, better than base.
* **Do not integrate `mapsolve`'s MAP solve, and do not integrate a half-dot gridder.**
  Both are measured nulls (§3, §4), both cost 80–100x, and both produce a crisper
  image with no held-out support.
* **No reference imagery enters either survivor.** `aperture.py` imports `numpy` alone.
  `wideband` takes only a `Timebase`; `WIENER_GAIN` came from the gap-vs-picture PSD
  ratio on the master. Both are derivable from the signal alone. Verified by import
  inspection, not by assertion.

### 5.4 Blocking condition on aperture

**`aperture.py` must ship the code that produced `TRANSITION_1090` before this is
merged**, or the constant must be re-sourced to a module that does. A 520-line file
whose entire empirical content is a docstring is not auditable, and this project's
retraction history is specifically about numbers nobody could re-run. My §2.2
re-measurement is a substitute for that audit, not a replacement for it. Everything
else in the module is closed-form and verifiable by inspection: I checked the
aliasing identity `SUM_m sinc^2(f+m) = 1`, and `A(0.5) = 0.694` from the stated
Gaussian, by hand.

---

## 6. HOW CLOSE THE DECODE IS TO THE PHYSICAL CEILING

Measured, in the dot domain, on `dot_native` decodes (**230 active dots per trace**),
single-profile erf fits on a fixed edge set.

**Along-trace 10-90 today: 1.82 dots (median over L000/L020/L034/R056).**
**With aperture: 1.65 dots.**

Effective elements per trace (active dots / 10-90):

```
                                    10-90       effective elements
today (shipping)                  1.82 dots            127
+ aperture                        1.65 dots            140      (+10%)
+ a perfect dot-clock phase       ~1.52 dots           ~152     (+9% more)
sampling ceiling (dot lattice)    1.00 dot             230      unreachable
```

**Composition of the remaining gap**, in blur variance at the decoded output today
(quadrature, Gaussian-equivalent):

```
  1977 camera PSF        1.52 dots   ~70%   IRREDUCIBLE. Not ours. The record's ceiling.
  our plateau boxcar     0.70 dots   ~15%   EXACTLY KNOWN. This is what aperture removes.
  our dot-clock jitter   0.64 dots   ~12%   OURS, and NOT removed by anything shipping.
  rounding / other                    ~3%
```

Three statements about that table, in decreasing order of confidence:

1. **The 1.82 → 1.65 step is measured, held-out, and confirmed against the operator's
   own prediction.** After aperture, the decode sits at about **140 of a possible ~152
   effective elements per trace** — roughly **92% of what the 1977 camera left on the
   record**, on the 58% of frames where the dot clock is visible.
2. **The largest remaining term that belongs to us is dot-clock lattice jitter**, and
   aperture is right that it is worth roughly twice what aperture itself is worth.
   `dotclock.track_phase` claims ~0.1 dots for itself; aperture's cross-trace
   half-window test implies 0.15–0.35 dots. **I did not independently measure it** —
   the width/jitter degeneracy in that fit is real and I had no un-degenerate
   instrument to hand. Treat 0.25 dots as aperture's estimate, not as established, and
   treat the 12% share and the "~152" figure as resting on it. That measurement is the
   single highest-value piece of work left, and it belongs to whoever owns
   `dotclock.py`.
3. **The 1.52-dot camera figure is inferred by subtraction, not measured directly**,
   and it disagrees with the ~15.3-sample (1.26-dot) full-chain ESF on file. If the
   camera really is 1.26 dots, then the residual jitter is ~0.44 dots rms — *worse*
   than aperture's range, which would make the remaining prize larger still. Either
   way the ordering of the table does not change. It should be settled by a direct
   raw-waveform ESF on a picture edge, on the same frames, before anyone quotes a
   camera number.

**What this is not.** None of this is a new picture. Two real, reference-free,
held-out corrections worth about 10% of along-trace sharpness on 58% of frames and a
quarter of a pixel of trace placement on 100% of them, plus two rigorous nulls that
stop three plausible-looking super-resolution schemes from being built. The resolution
ceiling of the Voyager image record is a television camera in 1977 and it still is.
