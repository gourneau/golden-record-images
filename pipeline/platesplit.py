"""Within-plateau splitting: does the sample-and-hold give every frame its own
Noise2Noise pairs?

THE HYPOTHESIS UNDER TEST
-------------------------
The 1977 scan converter sampled the source picture once per NTSC line and HELD
the value (dotclock.py): each dot is a plateau ~12.18 samples wide at 384 kHz.
Within a plateau the underlying signal is constant by construction, so the
samples inside it are repeated measurements of one value, and splitting them
(even/odd, or first half vs second half) would yield two independent noisy
observations of every dot -- Noise2Noise pairs for all 156 frames, mono
included, plus the missing hold-out arbiter for the 96 frames that were
scanned only once.

n2n.py proved the pair-based method works when the pairs are real (8.28 dB on
the colour triplets). The whole question here is whether split-plateau pairs
are real, i.e. whether the noise in the two halves is INDEPENDENT. Three ways
it can fail, all measured below before anything is built:

  1. The noise is not white. The tape chain and digitiser are band-limited
     (sync-edge 10-90 rise is ~2 samples at 384 kHz, decode.measure_psf), so
     adjacent samples may share their noise. Then even/odd halves are two
     copies of ONE observation, and a network trained on such pairs learns the
     identity while appearing to train fine.
  2. The plateau is not flat: the transition edge settles into the first
     samples, band-limited pre-shoot leaks the NEXT plateau into the last
     samples, and AC-coupling droop tilts the hold. Any systematic shape is
     shared structure, not noise, and must be excluded, not averaged over.
  3. Slow noise -- per-trace streaks, hum, tape noise below the dot rate -- is
     constant across one plateau by definition, so it appears IDENTICALLY in
     both halves. The split cannot expose it, and worse: a hold-out arbiter
     built on one half is not independent of the input built from the other
     half wherever they share this common-mode noise.

WHAT IS MEASURED (every number from data/master/384kHzStereo.wav; the only
data input is the WAV, tier 0/1 throughout)
-------------------------------------------------------------------------
  `acf`      The decisive measurement. Autocorrelation of the within-plateau
             residual (each plateau's own mean removed) at lags 1..8 samples,
             pooled over ~10^5 plateaus per frame, against a white-noise
             reference pushed through the IDENTICAL estimator (mean removal
             biases the ACF by ~-1/(m-1); the reference carries that bias so
             the comparison is honest). Also measured on the content-free
             BACK PORCH (~150 flat samples per trace, no plateau structure,
             no scene), which separates channel noise colour from plateau
             settling. Reported raw and undrooped.
  `settle`   Mean residual by position within the plateau, and the regression
             of the residual at each position onto the step from the previous
             plateau and the step to the next: the settling transient and the
             pre-shoot, in fraction-of-step units, position by position.
  `budget`   How much of the total per-dot noise the split can even see:
             Var(firstHalf - secondHalf)/4 (the between-half-independent
             part reaching a full-plateau mean) against the total per-dot
             noise from adjacent-trace differences in the flattest scene
             windows. Slow noise is common to both halves and invisible to
             the split by construction; this puts a number on how much that
             is.
  `halfcov`  Direct covariance of the two half-mean noises on the porch
             (content-free, so the half means ARE noise), for the even/odd,
             half/half and widest (first-3 vs last-3) splits, against the
             same quantities predicted from the measured ACF -- a consistency
             check that the ACF is the whole story.

MEASURED RESULT (2026-08-13, 8 mono frames -- L002 L023 L040 L076 R005 R024
R051 R076, both channels, ~111k plateaus per frame; reproduce with
`python -m pipeline.platesplit --measure`): THE HYPOTHESIS IS DEAD.

  Autocorrelation of the within-plateau residual, mean over frames
  (flat plateaus only, so settling cannot masquerade as correlation):
      lag          1       2       3       4       5       6       7       8
      picture   +0.705  +0.284  -0.044  -0.189  -0.207  -0.200  -0.210  -0.219
      white ref -0.083  -0.075  -0.068  -0.062  -0.052  -0.045  -0.038  -0.030
      porch     +0.947  +0.848  +0.746  +0.671  +0.625  +0.587  +0.542  +0.492
      (porch lags 9-12: +0.443 +0.403 +0.367 +0.332; per-frame lag-1 spread
       0.690..0.754 picture, 0.926..0.953 porch -- every frame, both channels)
  Excess lag-1 over the white reference: +0.79. The negative picture tail at
  lags 4+ is the mean-removal of a residual dominated by sub-dot-rate noise,
  not anticorrelated noise. Identical without undroop (L002 lag 1: +0.719 raw
  vs +0.718 undrooped), so no correction of ours creates the correlation.

  Where the noise lives (content-free porch, quadratic-detrended): half the
  power below 3.4 kHz, 90% below 23.4 kHz, 8% above the dot rate (31.5 kHz),
  0.0% above 96 kHz (Nyquist 192). The chain's noise is tape/electronics
  noise band-limited far below the sample rate: at 384 kHz every sample
  shares its noise with its neighbours by an order of magnitude more than
  the sqrt(n) bookkeeping of "12 independent looks" assumes.

  Half-mean noise correlation, measured DIRECTLY on the porch (signal-free)
  and predicted from the ACF (agreement validates that the ACF is the whole
  story); identical on both channels to +-0.02:
      even/odd            +0.991    (ACF predicts +0.987)
      half/half           +0.79     (ACF predicts +0.766)
      first3/last3        +0.71     (ACF predicts +0.669)
      widest, 0-2 vs 9-11 +0.51     (ACF predicts +0.475)
  Even the widest geometrically possible split leaves the two "independent
  observations" sharing half their noise, and that is the porch best case --
  in the picture, samples 0-2 and 9-11 sit inside the edge transitions.

  The plateau is not flat either: the incoming step is still ~30% unsettled
  at sample 0 and ~12% at sample 2, and the outgoing edge's band-limited
  pre-shoot reaches ~29% of the step by sample 11 (settle regression below;
  the transition smears ~4 samples into each end, leaving ~4 clean interior
  samples of the 12). Any split therefore also shares deterministic scene
  structure through the edges, on top of the shared noise.

  Noise budget: Var(firstHalf - secondHalf)/4 -- everything the split could
  possibly address -- is 0.18..0.85 (mean 0.39) of the total per-dot noise
  from adjacent-trace differences. The remainder is slower than one plateau
  (streaks, hum-residual, low-frequency tape noise) and enters both halves
  IDENTICALLY: no within-plateau scheme can even see it.

VERDICT: the sample-and-hold does NOT give every frame its own Noise2Noise
pairs. The two halves of a plateau are not independent observations of the
dot -- they are one observation of band-limited noise read twice (even/odd:
rho = 0.99). A network trained on such pairs can predict most of the
target's noise from its input, so the Noise2Noise fixed point is close to
the identity on the shared component: it would "train fine" and denoise
almost nothing, while a hold-out arbiter built on the other half would
REWARD keeping the shared noise -- predictors that preserve the input's
noise score better against a correlated target than predictors that remove
it, and the mandated blur control cannot rescue a target that is itself
correlated with the input. The split-N2N stage and the mono-frame arbiter
are therefore NOT built; building them would have produced numbers with the
right shape and no meaning. The 96 mono frames still have no independent
repeats; the colour triplets (n2n.py) remain the only real ones on the
record. A real null, in the project's tradition of publishing them.

PROVENANCE: tier 0 measurements / tier 1 method. Inputs: the WAV via
halfdot.load_frame (sync.py timebase, dotclock.py clock and phase). No
reference image, no Earth knowledge, no catalog field beyond frame ids and
channel. Image metrics choose nothing here; the only arbiter contemplated is
prediction of held-out measurements, and it is only built if the ACF says the
pairs are real.

Usage:
    python -m pipeline.platesplit --measure              # the kill tests
    python -m pipeline.platesplit --measure --frames L002,R040
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from . import catalog as catalog_mod
from . import decode as decode_mod
from . import dotclock as dot_mod
from . import halfdot as halfdot_mod
from . import sync as sync_mod

#: Frames used for the measurement: mono (single-scan) frames, both channels,
#: spread across the record. Chosen by position in the catalog, not by content.
DEFAULT_FRAMES = ("L002", "L023", "L040", "L076", "R005", "R024", "R051", "R076")

#: Samples taken per plateau. spd ~= 12.18, and ceil(edge)+12 always lands
#: inside [edge, edge+spd), so a fixed 12 gives a rectangular matrix with no
#: sample ever crossing a plateau boundary.
PLATEAU_W = 12

#: ACF lags reported, in samples.
LAGS = tuple(range(1, 9))


# --------------------------------------------------------------------------
# extraction: the raw samples inside each plateau
# --------------------------------------------------------------------------


def plateau_samples(fid: str, cat=None, *, uncouple: bool = True):
    """(vals, frac, step_prev, step_next, meta) for one frame.

    vals[p, m]  raw signal sample m of plateau p (12 per plateau, all strictly
                inside the plateau on the tracked dot grid);
    frac[p]     position of sample 0 within its plateau, in samples (0..1) --
                the sub-sample jitter the non-integer spd forces;
    step_prev   this plateau's mean minus the previous plateau's mean (the
                step the transition edge had to traverse INTO this plateau);
    step_next   next minus this (the step OUT).

    Mirrors decode.decode()'s dot-locked geometry (same clock, same tracked
    phase, same dropout mask); no porch clamp or dehum -- those are per-trace
    constants, invisible to a within-plateau residual, and skipping them keeps
    the samples raw. `uncouple` matters: droop tilts every plateau by
    ~(level/tau) per sample, a deterministic within-plateau slope that would
    masquerade as correlated noise; undroop removes it exactly and its own
    noise colouring lives below ~120 Hz, invisible at 1..8-sample lags.
    """
    cat = cat or catalog_mod.build()
    fr, x, tb = halfdot_mod.load_frame(fid, cat)
    period = tb.period
    n_tr = min(sync_mod.TRACES_PER_FRAME, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])

    if uncouple:
        tau = decode_mod.UNCOUPLE_TAU_384[fr.id[0]] * period / sync_mod.NOMINAL_PERIOD
        x = decode_mod.undroop(x, tau)
    x = np.asarray(x, dtype=np.float64)

    ps, span_f = decode_mod.PICTURE_START, decode_mod.PICTURE_SPAN
    lo_f, hi_f = ps + 0.01, ps + span_f - 0.01
    clock = dot_mod.measure(x, tb, lo_f=lo_f, hi_f=hi_f)
    spd = clock.samples_per_dot
    psi, coherence = dot_mod.track_phase(x, starts, period, spd, ps, ps + span_f)

    porch, porch_ok = decode_mod._region(
        x, starts, period, decode_mod.PORCH_START, decode_mod.PORCH_END)
    pl = np.median(porch, axis=1)
    med = np.median(pl)
    mad = 1.4826 * np.median(np.abs(pl - med)) + 1e-9
    dropouts = (np.abs(pl - med) > 6 * mad) | (~porch_ok)

    ref = decode_mod.measure_levels(x, starts, period)
    amp = float(ref[1]) if ref is not None else float(np.std(x))

    vals_l, frac_l = [], []
    for i in range(10, n_tr - 2):
        if dropouts[i]:
            continue
        a = starts[i] + lo_f * period
        b = starts[i] + hi_f * period
        k0 = int(math.ceil((a - psi[i]) / spd)) + 1
        k1 = int(math.floor((b - psi[i]) / spd)) - 1
        if k1 <= k0:
            continue
        e = psi[i] + np.arange(k0, k1) * spd
        j0 = np.ceil(e).astype(np.int64)
        if j0[0] < 0 or j0[-1] + PLATEAU_W >= len(x):
            continue
        idx = j0[:, None] + np.arange(PLATEAU_W)[None, :]
        vals_l.append(x[idx])
        frac_l.append(j0 - e)

    vals = np.concatenate(vals_l, axis=0)
    frac = np.concatenate(frac_l, axis=0)
    mean = vals.mean(axis=1)
    step_prev = np.empty_like(mean)
    step_next = np.empty_like(mean)
    step_prev[1:] = mean[1:] - mean[:-1]
    step_prev[0] = 0.0
    step_next[:-1] = mean[1:] - mean[:-1]
    step_next[-1] = 0.0
    meta = {"spd": float(spd), "coherence": float(coherence),
            "strength": float(clock.strength), "measured": bool(clock.measured),
            "amp": amp, "n_plateaus": int(len(vals)), "period": float(period)}
    return vals, frac, step_prev, step_next, meta


def porch_segments(fid: str, cat=None, *, uncouple: bool = False) -> tuple[np.ndarray, float]:
    """(segments, amp): detrended content-free porch runs, one per good trace.

    The back porch is ~150 samples of blanking level per trace: no scene, no
    plateau structure, so its fluctuation IS the channel noise. Each segment
    has a per-trace quadratic removed (porch level drifts with hum and AC
    recovery; a 3-parameter fit over ~150 samples absorbs a negligible share
    of the noise and no structure at 1..12-sample lags).
    """
    cat = cat or catalog_mod.build()
    fr, x, tb = halfdot_mod.load_frame(fid, cat)
    period = tb.period
    n_tr = min(sync_mod.TRACES_PER_FRAME, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])
    if uncouple:
        tau = decode_mod.UNCOUPLE_TAU_384[fr.id[0]] * period / sync_mod.NOMINAL_PERIOD
        x = decode_mod.undroop(x, tau)
    x = np.asarray(x, dtype=np.float64)

    ref = decode_mod.measure_levels(x, starts, period)
    amp = float(ref[1]) if ref is not None else float(np.std(x))

    lo = int(round((decode_mod.PORCH_START + 0.002) * period))
    hi = int(round((decode_mod.PORCH_END - 0.002) * period))
    w = hi - lo
    t = np.arange(w)
    V = np.stack([t ** 2, t, np.ones(w)], axis=1)
    segs = []
    for i in range(10, n_tr - 2):
        a = int(round(starts[i])) + lo
        if a < 0 or a + w > len(x):
            continue
        seg = x[a:a + w]
        coef, *_ = np.linalg.lstsq(V, seg, rcond=None)
        segs.append(seg - V @ coef)
    return np.asarray(segs), amp


# --------------------------------------------------------------------------
# the decisive measurement: residual autocorrelation
# --------------------------------------------------------------------------


def pooled_acf(rows: np.ndarray, lags=LAGS, demean: bool = True) -> np.ndarray:
    """ACF of row residuals, pooled: rho_l = sum r_j r_{j+l} / sum r_j^2.

    With `demean` each row's own mean is removed first -- exactly what "the
    plateau's own mean" means -- which biases every lag by ~-1/(m-1) even for
    white noise. Compare against white_reference(), never against zero.
    """
    r = rows - rows.mean(axis=1, keepdims=True) if demean else rows
    denom = float(np.sum(r * r))
    out = []
    for l in lags:
        out.append(float(np.sum(r[:, :-l] * r[:, l:])) / denom)
    return np.asarray(out)


def white_reference(n_rows: int, width: int, lags=LAGS, seed: int = 0) -> np.ndarray:
    """The same estimator applied to iid noise of the same shape: the honest
    zero point (mean removal makes it ~-1/(width-1), not 0)."""
    rng = np.random.default_rng(seed)
    return pooled_acf(rng.standard_normal((min(n_rows, 200000), width)), lags)


def settle_profile(vals: np.ndarray, step_prev: np.ndarray, step_next: np.ndarray):
    """Per-position regression r(m) ~ alpha(m)*step_prev + beta(m)*step_next.

    alpha(m) is the fraction of the INCOMING step still unsettled at sample m
    (a band-limited edge decays over the first samples); beta(m) is pre-shoot
    of the NEXT edge leaking backward (symmetric-filter smear). Both are
    shared deterministic structure: whatever is nonzero must be excluded from
    any split, because it correlates the halves through the scene itself.
    """
    r = vals - vals.mean(axis=1, keepdims=True)
    X = np.stack([step_prev, step_next, np.ones(len(vals))], axis=1)
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)  # (3, W)
    return coef[0], coef[1], r.mean(axis=0)


# --------------------------------------------------------------------------
# the noise budget: what the split can and cannot see
# --------------------------------------------------------------------------


def split_budget(fid: str, cat=None, guard_lo: int = 3, guard_hi: int = 1) -> dict:
    """Between-half-visible noise vs total per-dot noise, one frame.

    The split difference d = mean(first half) - mean(second half) of the
    guarded interior cancels everything constant across the plateau: scene,
    streaks, hum, all noise slower than one dot. Var(d)/4 is the noise power
    the split could remove from a full-plateau mean IF the halves were
    independent -- an upper bound on what split-N2N can address. Total
    per-dot noise comes from adjacent-trace dot differences in the flattest
    3% of 16x16 windows (scene gradients only inflate it, so the ratio
    visible/total is if anything an overestimate).
    """
    vals, frac, sp, sn, meta = plateau_samples(fid, cat)
    a, b = guard_lo, PLATEAU_W - guard_hi
    interior = vals[:, a:b]
    w = interior.shape[1]
    h1 = interior[:, : w // 2].mean(axis=1)
    h2 = interior[:, w - w // 2:].mean(axis=1)
    d = h1 - h2
    # flat plateaus only, so residual settling cannot leak scene into d
    flat = (np.abs(sp) < 2 * d.std()) & (np.abs(sn) < 2 * d.std())
    var_d = float(np.var(d[flat]))

    df = halfdot_mod.dot_field(fid, cat, dehum=True)
    v = df.v[~df.dropouts]
    dv = np.diff(v, axis=0)
    B = 16
    wins = []
    for i in range(0, dv.shape[0] - B, B):
        for j in range(0, dv.shape[1] - B, B):
            wins.append(np.var(dv[i:i + B, j:j + B]))
    wins = np.sort(np.asarray(wins))
    n_tot = float(np.mean(wins[: max(1, len(wins) // 33)])) / 2.0

    n_vis = var_d / 4.0
    return {"fid": fid, "var_half_diff": var_d, "n_visible": n_vis,
            "n_total": n_tot, "fraction": n_vis / max(n_tot, 1e-30),
            "amp": meta["amp"], "n_flat": int(flat.sum())}


def half_cov_porch(segs: np.ndarray, guard_lo: int = 3, guard_hi: int = 1) -> dict:
    """Direct half-mean noise covariance on the porch (signal-free ground truth).

    Windows of PLATEAU_W samples are cut from the detrended porch and split
    exactly as a plateau would be. Because the porch is content-free the half
    means are pure noise, so Corr(h1, h2) is measured directly -- no algebra,
    no assumption. This is the number that decides whether a hold-out half is
    independent of the input half.
    """
    n, w = segs.shape
    k = w // PLATEAU_W
    win = segs[:, : k * PLATEAU_W].reshape(n * k, PLATEAU_W)
    a, b = guard_lo, PLATEAU_W - guard_hi
    interior = win[:, a:b]
    m = interior.shape[1]
    out = {}
    splits = {
        "even/odd": (np.arange(m) % 2 == 0, np.arange(m) % 2 == 1),
        "half/half": (np.arange(m) < m // 2, np.arange(m) >= m - m // 2),
        "first3/last3": (np.arange(m) < 3, np.arange(m) >= m - 3),
    }
    for name, (ia, ib) in splits.items():
        h1 = interior[:, ia].mean(axis=1)
        h2 = interior[:, ib].mean(axis=1)
        c = np.corrcoef(h1, h2)
        out[name] = {"corr": float(c[0, 1]),
                     "var1": float(h1.var()), "var2": float(h2.var())}
    # The widest split geometrically possible: samples 0-2 vs 9-11 of the full
    # plateau. On the porch there is no settling to guard against, so this is
    # the best case any split could ever reach (in the picture those samples
    # are inside the transition and it would be worse).
    h1 = win[:, 0:3].mean(axis=1)
    h2 = win[:, 9:12].mean(axis=1)
    out["widest 0-2/9-11"] = {"corr": float(np.corrcoef(h1, h2)[0, 1]),
                              "var1": float(h1.var()), "var2": float(h2.var())}
    return out


def porch_spectrum(segs: np.ndarray, fs: float = 384000.0) -> dict:
    """Where the channel noise lives, from the content-free porch.

    Median and 90th-percentile frequency of the noise power, and the fraction
    of power above the dot rate (fs/spd ~ 31.5 kHz) -- the part of the noise
    that can differ at all between two halves of one plateau. Detrending
    removes a little power below ~2/window ~ 5 kHz; that only makes the
    high-frequency fraction reported here an OVERestimate, i.e. it biases
    toward keeping the hypothesis alive, not toward killing it.
    """
    n, w = segs.shape
    win = np.hanning(w)
    P = np.mean(np.abs(np.fft.rfft(segs * win, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(w) * fs
    P = P[1:]
    f = f[1:]
    c = np.cumsum(P) / P.sum()
    f50 = float(np.interp(0.5, c, f))
    f90 = float(np.interp(0.9, c, f))
    dot_rate = fs / 12.18
    above_dot = float(P[f > dot_rate].sum() / P.sum())
    above_q = float(P[f > fs / 4].sum() / P.sum())
    return {"f50_khz": f50 / 1e3, "f90_khz": f90 / 1e3,
            "frac_above_dot_rate": above_dot, "frac_above_96k": above_q}


def cov_from_acf(acf_full: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> float:
    """Corr(mean_A, mean_B) predicted from a sample ACF (lag 0 = 1)."""
    pa, pb = np.where(ia)[0], np.where(ib)[0]
    def s(p, q):
        tot = 0.0
        for i in p:
            for j in q:
                l = abs(int(i - j))
                tot += acf_full[l] if l < len(acf_full) else 0.0
        return tot / (len(p) * len(q))
    cab = s(pa, pb)
    return cab / math.sqrt(s(pa, pa) * s(pb, pb))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def measure(frames=DEFAULT_FRAMES) -> None:
    cat = catalog_mod.build()
    print("PLATEAU-SPLIT KILL TESTS -- is the within-plateau noise independent?\n")

    long_lags = tuple(range(1, 13))
    acc_pic, acc_porch, acc_alpha, acc_beta, acc_meanr = [], [], [], [], []
    acc_budget, acc_spec = [], []
    segs_by_ch: dict[str, list] = {"L": [], "R": []}
    for fid in frames:
        vals, frac, sp, sn, meta = plateau_samples(fid, cat)
        sigma = float((vals - vals.mean(axis=1, keepdims=True)).std())
        # flat plateaus: settling cannot masquerade as correlation there
        thr = 2.0 * sigma
        flat = (np.abs(sp) < thr) & (np.abs(sn) < thr)
        acf_flat = pooled_acf(vals[flat])
        alpha, beta, meanr = settle_profile(vals, sp, sn)

        segs, amp = porch_segments(fid, cat)
        acf_porch = pooled_acf(segs, lags=long_lags, demean=False)
        segs_by_ch[fid[0]].append(segs)
        spec = porch_spectrum(segs)

        acc_pic.append(acf_flat)
        acc_porch.append(acf_porch)
        acc_alpha.append(alpha)
        acc_beta.append(beta)
        acc_meanr.append(meanr / max(sigma, 1e-30))
        acc_spec.append(spec)

        bud = split_budget(fid, cat)
        acc_budget.append(bud)

        print(f"  {fid}  spd {meta['spd']:.3f}  clock strength {meta['strength']:.1f} "
              f"({'measured' if meta['measured'] else 'PREDICTED'})  "
              f"plateaus {meta['n_plateaus']}  flat {int(flat.sum())}  "
              f"resid sigma/amp {sigma / meta['amp']:.4f}")
        print(f"        acf(pic,flat) lag1..4  " +
              " ".join(f"{v:+.3f}" for v in acf_flat[:4]) +
              f"   acf(porch) lag1..4  " +
              " ".join(f"{v:+.3f}" for v in acf_porch[:4]))
        print(f"        noise spectrum: f50 {spec['f50_khz']:.1f} kHz  f90 "
              f"{spec['f90_khz']:.1f} kHz  above dot rate "
              f"{spec['frac_above_dot_rate']:.2f}")
        print(f"        split budget: visible {bud['n_visible']:.3e}  total "
              f"{bud['n_total']:.3e}  fraction {bud['fraction']:.2f}")

    n_ref = 100000
    ref = white_reference(n_ref, PLATEAU_W)

    print("\nAUTOCORRELATION OF THE WITHIN-PLATEAU RESIDUAL (mean over frames)")
    print("  lag        " + "".join(f"{l:>8d}" for l in LAGS))
    m_pic = np.mean(acc_pic, axis=0)
    m_porch = np.mean(acc_porch, axis=0)
    print("  picture    " + "".join(f"{v:+8.3f}" for v in m_pic) +
          "   (flat plateaus, own mean removed)")
    print("  white ref  " + "".join(f"{v:+8.3f}" for v in ref) +
          "   (iid noise through the same estimator)")
    print("  porch      " + "".join(f"{v:+8.3f}" for v in m_porch[:len(LAGS)]) +
          "   (content-free, detrended, NOT mean-removed)")
    print("  porch l9-12" + "".join(f"{v:+8.3f}" for v in m_porch[len(LAGS):]))

    m_spec = {k: float(np.mean([s[k] for s in acc_spec])) for k in acc_spec[0]}
    print(f"\nPORCH NOISE SPECTRUM (mean over frames): half of the noise power "
          f"is below {m_spec['f50_khz']:.1f} kHz, 90% below "
          f"{m_spec['f90_khz']:.1f} kHz;")
    print(f"  fraction above the dot rate (31.5 kHz, the only part that can "
          f"differ between halves of one plateau): "
          f"{m_spec['frac_above_dot_rate']:.2f};  above 96 kHz: "
          f"{m_spec['frac_above_96k']:.2f}  (Nyquist is 192 kHz)")

    print("\nSETTLING (fraction of the adjacent step present at each position)")
    m_a = np.mean(acc_alpha, axis=0)
    m_b = np.mean(acc_beta, axis=0)
    m_r = np.mean(acc_meanr, axis=0)
    print("  pos          " + "".join(f"{m:>8d}" for m in range(PLATEAU_W)))
    print("  alpha(in)    " + "".join(f"{v:+8.3f}" for v in m_a))
    print("  beta(out)    " + "".join(f"{v:+8.3f}" for v in m_b))
    print("  mean resid   " + "".join(f"{v:+8.3f}" for v in m_r) + "  (sigma units)")

    print("\nHALF-MEAN NOISE CORRELATION ON THE PORCH (direct, signal-free)")
    acf_full = np.concatenate([[1.0], m_porch])
    m = PLATEAU_W - 3 - 1
    W = PLATEAU_W
    pred = {
        "even/odd": cov_from_acf(acf_full, np.arange(m) % 2 == 0, np.arange(m) % 2 == 1),
        "half/half": cov_from_acf(acf_full, np.arange(m) < m // 2, np.arange(m) >= m - m // 2),
        "first3/last3": cov_from_acf(acf_full, np.arange(m) < 3, np.arange(m) >= m - 3),
        "widest 0-2/9-11": cov_from_acf(acf_full, np.arange(W) < 3, np.arange(W) >= 9),
    }
    hc_by_ch = {}
    for ch in ("L", "R"):
        if not segs_by_ch[ch]:
            continue
        hc = half_cov_porch(np.concatenate(segs_by_ch[ch], axis=0))
        hc_by_ch[ch] = hc
        for name, d in hc.items():
            print(f"  {ch}  {name:<16s} corr {d['corr']:+.3f}   "
                  f"(predicted from ACF {pred[name]:+.3f})")
    hc = hc_by_ch.get("L") or hc_by_ch.get("R")

    print("\nNOISE BUDGET (what fraction of per-dot noise the split can even see)")
    fr_mean = float(np.mean([b['fraction'] for b in acc_budget]))
    for b in acc_budget:
        print(f"  {b['fid']}  visible {b['n_visible']:.3e}  total {b['n_total']:.3e}"
              f"  fraction {b['fraction']:.2f}")
    print(f"  mean fraction {fr_mean:.2f}")

    verdict(m_pic, ref, m_porch, hc, fr_mean)


def verdict(m_pic, ref, m_porch, hc, budget_fraction) -> None:
    print("\nVERDICT")
    lag1 = m_pic[0] - ref[0]
    print(f"  excess lag-1 correlation over white reference: {lag1:+.3f}")
    alive = abs(lag1) < 0.1 and abs(hc["half/half"]["corr"]) < 0.1
    if alive:
        print("  -> within-plateau noise is near-independent between halves; "
              "split pairs are plausibly real. Proceed to split-N2N evaluation.")
    else:
        print("  -> the halves share their noise; even/odd or half/half split "
              "pairs are NOT independent observations.")
        wc = hc["widest 0-2/9-11"]["corr"]
        print(f"  widest geometrically possible split (0-2 vs 9-11): noise "
              f"corr {wc:+.3f}"
              + ("  -> widening the split does NOT recover independence; no "
                 "split geometry inside one plateau does." if wc > 0.2 else
                 "  -> a wide split may partially recover independence."))
    print(f"  fraction of per-dot noise visible to the split: {budget_fraction:.2f}")
    if budget_fraction < 0.5:
        print("  -> most of the per-dot noise is slower than one plateau "
              "(streaks, tape noise) and appears identically in both halves; "
              "a split-based arbiter is blind to it, and a network trained on "
              "such pairs is trained to KEEP the shared component -- it can "
              "predict the target's noise from its input, which is exactly "
              "the failure mode Noise2Noise's premise excludes.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--frames", default=",".join(DEFAULT_FRAMES))
    args = ap.parse_args(argv)
    if args.measure:
        measure(tuple(args.frames.split(",")))
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
