"""Blind (non-parametric) estimate of the decay-bias transfer function.

pipeline/droop.py characterised the distortion PARAMETRICALLY (one-pole
AC-coupling high-pass, tau_L = 530 / tau_R = 295 samples at 384 kHz). This
module estimates the chain's low-frequency transfer WITHOUT assuming a pole
model, then checks whether the parametric model survives.

    python -m pipeline.droop_blind
    python -m pipeline.droop_blind --section transfer,invert,porch,flat,step,porchfit
    python -m pipeline.droop_blind --section images --frames L000,L055,R040

THE INSTRUMENT
--------------
The sync pulse width alternates with trace parity, so the source signal in the
inter-frame gap is periodic with period exactly TWO traces: the wide-vs-narrow
pulse difference is a deterministic, piecewise-constant excitation with edges
at known bins (3048 / 3071 / 3155 of 3200; geometry.py) and everything else
identical between parities. The even-minus-odd median gap profile is therefore
the chain's steady-state response to a known periodic input, and

    H(f_k) = FFT[response] / FFT[input]   at the ODD harmonics f_k = k/(2P)

is a model-free measurement of the transfer function, ~60 Hz spacing, with
phase. The input's levels (not shapes) are re-measured on the inverse-filtered
signal and iterated, so the estimate does not inherit the droop bias of
reading amplitudes off the drooped signal. k = 1 (~60 Hz) is contaminated by
the scan-locked hum (period exactly 2 traces -- the same degeneracy droop.py
documented) and is flagged, never silently used.

RESULTS (this module, run against data/master/384kHzStereo.wav, 2026-08)
------------------------------------------------------------------------
1. ESTIMATOR VALIDATED ON SYNTHETIC TRUTH first: a known one-pole chain,
   hum and noise pushed through the identical code path is recovered to
   <=0.02 in |H| and ~1 deg in phase at every clean harmonic. (The first
   version of this estimator -- window-median levels + iteration -- was
   BIASED and the synthetic test caught it; see fit_input's docstring.)

2. THE CHAIN IS NOT A SINGLE POLE. Measured H per channel (6 gap frames
   each, sem 0.004-0.02): the single-pole hypothesis is rejected at
   chi2/dof 74 (L) / 131 (R) over 20 clean harmonics. On top of a pole-like
   low-frequency cut there is an extra SHELVING LOSS of ~15% (L) / ~18% (R)
   between ~200 Hz and ~2.5 kHz, recovering to unity by ~2.7 kHz, with
   matching excess phase (+10..+20 deg):
       |H| meas (L): 0.72@180Hz 0.77@300 0.82@540 0.84@1260 0.88@1860 1(imposed)>=2.7k
       |H| pole 530: 0.84      0.93     0.98     1.00      1.00
       |H| meas (R): 0.58@180  0.69@300 0.78@540 0.84@1260 0.87@1860
       |H| pole 295: 0.66      0.82     0.94     0.99      1.00
   In image terms the shelf sits at along-trace periods of ~18-250 px: the
   mid-scale band where the emboss lives. The pole alone under-restores
   every structure in that band by 15-20%.

3. WHY droop.py MISSED IT: its flattening metric (and the exp-decay fits)
   look at the tail SHAPE with a free constant. Measured directly: the
   pole-corrected parity profile is flat but sits at a constant ~-0.012
   with its pulse plateau ~30% short of the source level -- a pure
   amplitude deficit that a shape metric with a free constant cannot see.
   Time-domain inverse_pole and frequency-domain division agree exactly
   (LTI sanity held), so this is a metric blind spot, not a code fault.
   droop.py's tail time constants themselves are CONFIRMED: the decay tail
   measured here decays at tau ~550 (L), and the best low-f pole factor is
   droop's tau within its systematic band.

4. ARTIFACT HUNTING (all measured, all negative):
   * Scan-locked hum HARMONICS (buzz at 60k Hz falls on every odd probe
     harmonic): the L000 pulse-free control bounds the induced H error at
     0.5-3%, vs the 16-29% discrepancy. Additive per-harmonic offsets,
     fixed or sign-flipping, fitted across frames change |H| by <0.01.
   * Normalisation band: refitting the input levels in 5.5-12 kHz instead
     of 2.7-9.7 kHz moves |H(180)| by ~0.03. (Bands >=12 kHz break down --
     edge fine structure; that is the model's stated validity limit.)
   * k=1 (~60 Hz) is hum-degenerate and never used; below 180 Hz the
     inverse extends the measured shelf with droop.py's pole ("np-hybrid").

5. VALIDATION of W = 1/H (leave-one-out on the gap probe; independent data
   elsewhere):
   * gap probe, honest full metric (constant kept): raw 0.0054-0.0083 ->
     np-hybrid 0.0007-0.0013 (6-12x); pole manages only 2.2x (0.0023-0.0038)
     because of the amplitude deficit it leaves. Under droop.py's
     shape-only metric the pole still looks better (0.0004 vs 0.0007) --
     that comparison is the blind spot of point 3, printed for the record.
   * L000 flat field (independent): along-trace profile of the uniform
     surround, std 0.0268 raw -> 0.0225 pole -> 0.0148 np-hybrid (left of
     ring; right side 0.0240 -> 0.0191 -> 0.0113). The shelf correction
     roughly halves the residual non-flatness on the one field known flat.
   * porch-transfer slope: CONFOUNDED, reported as such -- any low-f boost
     amplifies the scan-locked hum envelope, which couples the porch to the
     adjacent picture window through the regression; the slope grows with
     boost (raw -0.9..-4.9; pole and np both move it unpredictably). Not
     usable to rank inverses without a hum-envelope-immune formulation.
   * frozen test set through untouched decode.py (384 kHz signal filtered
     before decimation): with the shipped reference levels the corrected
     frames CLIP WHITE (the +/-0.75 x amp anchors were calibrated on
     drooped statistics; mean composite 18.9 -> 14.6, L000's field blown
     out) -- the level anchors must be re-derived after correction. With
     percentile levels on both sides the composite is 18.5 -> 16.2 (7
     frames up, 9 down; L000 circle fit passes both, axis ratio 1.0074 ->
     1.0051). The residual losses concentrate in parity_db (worst L034
     4.4 -> 14.7 dB): boosting 60 Hz before decode.py's dehum amplifies
     the parity-locked hum envelope. The composite measures timing and
     striping, not emboss; visually (see the section's PNG dump) the
     emboss, the field shading and the trailing anti-shadows are largely
     gone (R056, L000, L055), which no printed metric here fully captures.
     INTEGRATION NOTE for decode.py: apply the inverse AFTER dehum (LTI
     commutes with the linear decode steps), or re-estimate the hum on the
     corrected signal, and re-derive the level anchors; both are decode.py
     changes outside this module's remit.

6. METHOD B, porch-constancy inverse fit: NEGATIVE RESULT, kept printed.
   Solving x_hat = y + g*y so that corrected porches are constant fixes
   the errors-in-variables problem of droop.py's failed forward
   regressions, but the hum envelope still dominates the porch variance:
   the fitted kernel comes out ~10x below any candidate model with
   sign flips (implied |H|~1 everywhere), and porch variance drops only
   1.5-2x. Content-based porch identification remains dead.

HOUSE RULES
-----------
* The low/high energy ratio of decoded images rises under ANY low-frequency
  boost; it is printed as description, never used as a success criterion.
* Validation of the inverse built from H is leave-one-out on the gap probe
  (the aggregate excludes the frame being validated) and otherwise uses data
  the estimate never saw (picture-frame porches, L000's field, step edges,
  the frozen test set).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import droop, geometry

FS = 384000.0
BINS = geometry.BINS  # 3200
TWO = 2 * BINS

# Source parity-difference edges (bins; geometry.py): long burst rises at
# 3048, short-parity shelf at 3071, short burst at 3155, both fall at 3200.
SEGS = ((3048, 3071), (3071, 3155), (3155, 3200))
LEVEL_WINS = ((3052, 3068), (3076, 3150), (3160, 3196))  # interior margins

KMAX = 41          # highest odd harmonic reported (~2.5 kHz)
K_HUM = 1          # the hum-degenerate harmonic, flagged throughout
K_MIN_CLEAN = 3    # lowest harmonic treated as clean (~180 Hz)
# Band where the chain is treated as FLAT (|H| = 1): 3.7-9.7 kHz. This is the
# one imposed assumption (any deconvolution needs a normalisation); it is
# physically anchored -- the measured pole leaves <0.1% there, and droop.py's
# plateau-slope regression bounds any separate midband tilt. The input levels
# are fitted HERE, in the frequency domain, so they carry no droop bias, no
# baseline bias and no hum bias (hum lives at k=1).
FIT_K = (45, 161)

GAP_FIDS = droop.GAP_FIDS
GAP_DAMAGED = {"L040", "R077"}  # droop.py: damaged gaps, excluded from means
FRAME_FIDS = droop.FRAME_FIDS
PORCHFIT_FIDS = {"L": ["L034", "L055", "L002", "L020"],
                 "R": ["R040", "R056", "R022", "R068"]}


# --------------------------------------------------------------------------
# the non-parametric transfer measurement
# --------------------------------------------------------------------------


def _pattern_levels(d: np.ndarray) -> tuple[list[float], float]:
    """Piecewise-constant input levels read off a parity profile, relative to
    its out-of-pulse baseline."""
    base = float(np.median(d[300:3000]))
    return [float(np.median(d[a:b]) - base) for a, b in LEVEL_WINS], base


def _build_input(levels: list[float]) -> np.ndarray:
    """Two-trace source parity-difference waveform: +pattern at the end of the
    first trace, -pattern at the end of the second (odd harmonics only)."""
    x = np.zeros(TWO)
    for (a, b), lv in zip(SEGS, levels):
        x[a:b] = lv
        x[BINS + a:BINS + b] = -lv
    return x


def _odd_harmonics() -> np.ndarray:
    return np.arange(1, KMAX + 1, 2)


def h_pole(f_hz: np.ndarray, tau: float) -> np.ndarray:
    """Discrete one-pole high-pass H(z)=(1-z^-1)/(1-a z^-1) at f (Hz, 384k)."""
    a = np.exp(-1.0 / tau)
    z1 = np.exp(-2j * np.pi * np.asarray(f_hz, dtype=np.float64) / FS)
    return (1.0 - z1) / (1.0 - a * z1)


def make_W(f_pts: np.ndarray, H_pts: np.ndarray, low_mode: str = "flat",
           tau_ext: float | None = None, f_cap: float = 20.0):
    """Inverse-filter frequency response W(f) from measured points.

    Interpolates log|1/H| and unwrapped arg(1/H) linearly in log f between the
    measured harmonics; blends to W=1 over one octave above the last point.
    Below the lowest point:
      low_mode='flat': hold |W|, taper phase linearly in f to 0 at DC
      low_mode='pole': follow 1/h_pole(f; tau_ext), |W| capped at its value
                       at f_cap (the per-trace porch clamp owns slower scales)
    """
    ok = np.isfinite(H_pts) & (np.abs(H_pts) > 1e-6)
    f = np.asarray(f_pts, dtype=np.float64)[ok]
    Wp = 1.0 / np.asarray(H_pts)[ok]
    lg = np.log(np.abs(Wp))
    ph = np.unwrap(np.angle(Wp))
    lf = np.log(f)
    f_min, f_max = f[0], f[-1]
    f_hi = 2.0 * f_max

    def W(freq: np.ndarray) -> np.ndarray:
        freq = np.abs(np.asarray(freq, dtype=np.float64))
        out_lg = np.zeros_like(freq)
        out_ph = np.zeros_like(freq)
        mid = (freq >= f_min) & (freq <= f_max)
        ff = np.log(np.maximum(freq[mid], 1e-12))
        out_lg[mid] = np.interp(ff, lf, lg)
        out_ph[mid] = np.interp(ff, lf, ph)
        hi = freq > f_max
        if hi.any():
            s = np.clip((np.log(f_hi) - np.log(freq[hi]))
                        / (np.log(f_hi) - np.log(f_max)), 0.0, 1.0)
            out_lg[hi] = lg[-1] * s
            out_ph[hi] = ph[-1] * s
        lo = freq < f_min
        if lo.any():
            if low_mode in ("pole", "hybrid") and tau_ext is not None:
                Wl = 1.0 / h_pole(np.maximum(freq[lo], 1e-9), tau_ext)
                cap = float(np.abs(1.0 / h_pole(np.array([f_cap]), tau_ext))[0])
                if low_mode == "hybrid":
                    # keep the measured shelf factor at f_min constant below
                    # it: W = S(f_min)^-1 / h_pole, S = H(f_min)/h_pole(f_min)
                    s_inv = (np.exp(lg[0] + 1j * ph[0])
                             * h_pole(np.array([f_min]), tau_ext)[0])
                    Wl = Wl * s_inv
                    cap = cap * abs(s_inv)
                mag = np.minimum(np.abs(Wl), cap)
                phl = np.angle(Wl) * np.clip(freq[lo] / f_cap, 0.0, 1.0)
                out_lg[lo] = np.log(mag)
                out_ph[lo] = phl
            else:
                out_lg[lo] = lg[0]
                out_ph[lo] = ph[0] * (freq[lo] / f_min)
        w = np.exp(out_lg + 1j * out_ph)
        w[freq == 0.0] = np.exp(out_lg[freq == 0.0].real)  # DC must be real
        return w

    return W


def apply_W(x: np.ndarray, W) -> np.ndarray:
    """Filter x (384 kHz samples) with frequency response W(f in Hz)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mu = float(x.mean())
    nfft = 1 << int(np.ceil(np.log2(n + 65536)))
    X = np.fft.rfft(x - mu, nfft)
    f = np.fft.rfftfreq(nfft) * FS
    y = np.fft.irfft(X * W(f), nfft)[:n]
    return y + mu * float(np.real(W(np.array([0.0]))[0]))


def _seg_bases() -> np.ndarray:
    """(3, nfreq) FFTs of the unit-level two-trace basis patterns."""
    rows = []
    for i in range(3):
        lv = [0.0, 0.0, 0.0]
        lv[i] = 1.0
        rows.append(np.fft.rfft(_build_input(lv)))
    return np.asarray(rows)


def fit_input(D: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit the source pattern's three real levels plus one global sub-bin
    timing offset to the measured response, using ONLY the flat band FIT_K.

    An earlier version of this module measured the levels as window medians on
    the (inverse-filtered) profile and iterated. A synthetic end-to-end test
    (known one-pole truth) showed that estimator is BIASED and the iteration
    preserves the bias: droop, hum and the decay tail under the baseline
    window all leak into the levels, and a pure gain error in H is a fixed
    point of the iteration (H = g*H_true with levels/g reproduces itself).
    The flat-band frequency-domain fit recovers the synthetic truth to <1%.
    """
    B = _seg_bases()
    ks = np.arange(FIT_K[0], FIT_K[1] + 1, 2)
    best = None
    for delta in np.arange(-3.0, 3.001, 0.1):
        ph = np.exp(-2j * np.pi * ks * delta / TWO)
        G = (B[:, ks] * ph).T  # (nk, 3)
        Gr = np.vstack([G.real, G.imag])
        Dr = np.concatenate([D[ks].real, D[ks].imag])
        A, res, *_ = np.linalg.lstsq(Gr, Dr, rcond=None)
        r = float(((Dr - Gr @ A) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, delta, A)
    return best[2], best[1], best[0]


def transfer_one(fid: str) -> dict | None:
    """H at the odd harmonics of 1/(2P) for one frame's gap: complex division
    of the measured parity response by the fitted source pattern."""
    try:
        x, m, _ = droop.load(fid, 195.0, 60)
        lo, hi = droop.gap_slots(m)
        d = droop.parity_profile(x, m, lo, hi)
    except Exception as exc:
        print(f"  {fid}: FAILED ({exc})")
        return None
    P = float(np.median(np.diff(m[max(lo, 1):hi])))
    ks = _odd_harmonics()
    f_hz = ks * FS / (2.0 * P)

    D = np.fft.rfft(np.concatenate([d, -d]))
    A, delta, _ = fit_input(D)
    B = _seg_bases()
    Xk = (A @ B[:, ks]) * np.exp(-2j * np.pi * ks * delta / TWO)
    good = np.abs(Xk) > 0.10 * np.abs(Xk).max()
    H = np.where(good, D[ks] / np.where(good, Xk, 1.0), np.nan + 0j)
    return {"fid": fid, "P": P, "ks": ks, "f_hz": f_hz, "H": H,
            "levels": [float(v) for v in A], "delta": float(delta),
            "amp": float(np.abs(A).max()), "D": D[ks], "Xk": Xk}


def aggregate(results: list[dict], exclude: set[str] = frozenset()) -> dict:
    """Robust per-harmonic centre and scatter of H across frames."""
    use = [r for r in results
           if r is not None and r["fid"] not in exclude
           and r["fid"] not in GAP_DAMAGED]
    Z = np.array([r["H"] for r in use])  # (n, k)
    n = len(use)
    cr = np.nanmedian(Z.real, axis=0)
    ci = np.nanmedian(Z.imag, axis=0)
    sr = 1.4826 * np.nanmedian(np.abs(Z.real - cr), axis=0)
    si = 1.4826 * np.nanmedian(np.abs(Z.imag - ci), axis=0)
    f_hz = np.mean([r["f_hz"] for r in use], axis=0)
    sem = np.sqrt((sr**2 + si**2) / 2.0) / np.sqrt(max(n, 1))
    return {"n": n, "fids": [r["fid"] for r in use], "ks": use[0]["ks"],
            "f_hz": f_hz, "H": cr + 1j * ci, "sd_r": sr, "sd_i": si,
            "sem": sem}


def fit_tau(agg: dict, k_min: int = K_MIN_CLEAN, k_max: int = KMAX) -> dict:
    """Best single pole through the measured harmonics (complex residuals,
    weighted by across-frame scatter). Also reports chi2/dof: whether ONE POLE
    IS AN ADEQUATE MODEL of the non-parametric points at all."""
    ks, f, H = agg["ks"], agg["f_hz"], agg["H"]
    sr = np.maximum(agg["sd_r"] / np.sqrt(agg["n"]), 1e-4)
    si = np.maximum(agg["sd_i"] / np.sqrt(agg["n"]), 1e-4)
    sel = (ks >= k_min) & (ks <= k_max) & np.isfinite(H)
    taus = np.geomspace(60, 4000, 600)
    chi = np.array([
        np.sum(((H[sel] - h_pole(f[sel], t)).real / sr[sel]) ** 2
               + ((H[sel] - h_pole(f[sel], t)).imag / si[sel]) ** 2)
        for t in taus])
    i = int(np.argmin(chi))
    lo_i = np.where(chi[:i + 1] <= chi[i] + 1.0)[0]
    hi_i = i + np.where(chi[i:] <= chi[i] + 1.0)[0]
    dof = 2 * int(sel.sum()) - 1
    return {"tau": float(taus[i]), "tau_lo": float(taus[lo_i[0]]),
            "tau_hi": float(taus[hi_i[-1]]),
            "chi2_dof": float(chi[i] / max(dof, 1)), "n_pts": int(sel.sum())}


def measure_transfer(results_out: dict | None = None) -> dict:
    print("== NON-PARAMETRIC TRANSFER (gap parity probe, complex division) ==")
    print("H(f) at odd harmonics of 1/(2P); input = piecewise-constant pulse")
    print("difference (edges 3048/3071/3155), levels + sub-bin timing fitted in")
    print(f"the flat band k={FIT_K[0]}..{FIT_K[1]} (3.7-9.7 kHz) where |H|=1 is")
    print(f"imposed (stated normalisation). k={K_HUM} (~60 Hz) hum-degenerate: FLAGGED.")
    res = {"L": [], "R": []}
    for fid in GAP_FIDS:
        r = transfer_one(fid)
        if r is None:
            continue
        res[fid[0]].append(r)
        tag = "  [damaged gap, excluded from aggregate]" if fid in GAP_DAMAGED else ""
        h3 = r["H"][list(r["ks"]).index(3)]
        h1 = r["H"][list(r["ks"]).index(1)]
        print(f"  {fid}: |H(60)| {np.abs(h1):.3f}  |H(180)| {np.abs(h3):.3f} "
              f"arg {np.degrees(np.angle(h3)):+.1f} deg  "
              f"levels {['%+.4f' % v for v in r['levels']]} "
              f"dt {r['delta']:+.1f}{tag}")
    out = {}
    for ch in "LR":
        agg = aggregate(res[ch])
        out[ch] = agg
        tau_ref = droop.TAU_384[ch]
        print(f"\n  channel {ch} (n={agg['n']} frames: {' '.join(agg['fids'])})")
        print(f"  {'k':>3} {'f Hz':>7} {'|H|':>7} {'+-':>6} {'arg deg':>8} "
              f"{'|H|pole':>7} {'arg pole':>8}   (pole tau={tau_ref:.0f})")
        Hp = h_pole(agg["f_hz"], tau_ref)
        for j, k in enumerate(agg["ks"]):
            if k > 21 and k not in (25, 31, 41):
                continue
            H = agg["H"][j]
            if not np.isfinite(H):
                continue
            flag = "  << HUM-DEGENERATE" if k == K_HUM else ""
            print(f"  {k:>3} {agg['f_hz'][j]:>7.0f} {np.abs(H):>7.3f} "
                  f"{agg['sem'][j]:>6.3f} {np.degrees(np.angle(H)):>+8.1f} "
                  f"{np.abs(Hp[j]):>7.3f} {np.degrees(np.angle(Hp[j])):>+8.1f}{flag}")
        ft = fit_tau(out[ch])
        ft1 = fit_tau(out[ch], k_min=1)
        hi_band = np.abs(out[ch]["H"][(out[ch]["ks"] >= 25)])
        hi_band = hi_band[np.isfinite(hi_band)]
        print(f"  best single pole (k>={K_MIN_CLEAN}): tau {ft['tau']:.0f} "
              f"[{ft['tau_lo']:.0f}..{ft['tau_hi']:.0f}] chi2/dof {ft['chi2_dof']:.2f} "
              f"over {ft['n_pts']} harmonics; incl. hum-flagged k=1: tau {ft1['tau']:.0f}")
        print(f"  high-band |H| (k>=25, should be 1): "
              f"{hi_band.mean():.3f} +- {hi_band.std():.3f}")
        out[ch]["fit"] = ft
    if results_out is not None:
        results_out.update({"per_frame": res, "agg": out})
    return {"per_frame": res, "agg": out}


# --------------------------------------------------------------------------
# inverse variants
# --------------------------------------------------------------------------


def build_inverses(agg: dict, exclude: set[str] = frozenset(),
                   per_frame: dict | None = None) -> dict:
    """Per-channel inverse-filter builders. When `exclude` is given and
    per_frame results are supplied, the aggregate is recomputed leaving those
    frames out (leave-one-out honesty for gap validation)."""
    inv = {}
    for ch in "LR":
        a = agg[ch]
        if exclude and per_frame is not None:
            a = aggregate(per_frame[ch], exclude=exclude)
            a["fit"] = fit_tau(a)
        ks, f, H = a["ks"], a["f_hz"], a["H"]
        clean = (ks >= K_MIN_CLEAN) & np.isfinite(H)
        withk1 = (ks >= 1) & np.isfinite(H)
        tau_np = a["fit"]["tau"] if "fit" in a else fit_tau(a)["tau"]
        inv[ch] = {
            "np-flat": make_W(f[clean], H[clean], "flat"),
            "np-k1": make_W(f[withk1], H[withk1], "flat"),
            "np-pole-ext": make_W(f[clean], H[clean], "pole", tau_ext=tau_np),
            # measured shelf >=180 Hz; below it droop.py's pole times the
            # shelf factor at 180 Hz (shelves flatten toward DC; the k=1
            # harmonic that could decide is hum-degenerate)
            "np-hybrid": make_W(f[clean], H[clean], "hybrid",
                                tau_ext=droop.TAU_384[ch]),
            "tau_np": tau_np,
        }
    return inv


def _pole_filter(ch: str):
    tau = droop.TAU_384[ch]
    return lambda x: droop.inverse_pole(x, tau)


def _np_filter(inv: dict, ch: str, variant: str):
    W = inv[ch][variant]
    return lambda x: apply_W(x, W)


# --------------------------------------------------------------------------
# validation 1: gap-probe flattening, leave-one-out
# --------------------------------------------------------------------------


def _dehum_rms(h: np.ndarray, keep_const: bool = False) -> float:
    """Residual rms of the (smoothed) profile over bins 30..3030 after
    removing the k=1 hum (sin+cos of the half-cycle).

    keep_const=False also removes a CONSTANT -- that is droop.py's metric,
    and this module discovered it is BLIND to an overall amplitude deficit:
    the pole-corrected profile is flat at ~-0.012, not at 0, and the metric
    cannot tell. The source parity difference outside the pulses is exactly
    zero (the blanking is parity-common), so keep_const=True is the honest
    version: any offset is uncorrected response.
    """
    hs = droop._smooth(h, 25)
    bb = np.arange(30, 3030, dtype=np.float64)
    th = np.pi * bb / float(BINS)
    cols = [np.sin(th), np.cos(th)]
    if not keep_const:
        cols.append(np.ones(len(bb)))
    X = np.column_stack(cols)
    seg = hs[30:3030]
    c, *_ = np.linalg.lstsq(X, seg, rcond=None)
    return float(np.sqrt(((seg - X @ c) ** 2).mean()))


def validate_gap(meas: dict) -> None:
    print("\n== VALIDATION 1: gap parity profile flattening (LEAVE-ONE-OUT) ==")
    print("aggregate H excludes the validated frame; pole = droop.py constants.")
    print("Two metrics: 'shape' removes a constant (droop.py's, blind to any")
    print("amplitude deficit); 'full' keeps it (source is exactly 0 there).")
    print(f"  {'frame':6} {'metric':6} {'raw':>8} {'pole':>8} {'np-flat':>8} "
          f"{'np-hybr':>8} {'np-pole':>8}")
    for fid in ("L030", "L055", "R020", "R040"):
        ch = fid[0]
        x, m, _ = droop.load(fid, 195.0, 60)
        lo, hi = droop.gap_slots(m)
        inv = build_inverses(meas["agg"], exclude={fid},
                             per_frame=meas["per_frame"])
        profs = [droop.parity_profile(x, m, lo, hi)]
        for filt in (_pole_filter(ch),
                     _np_filter(inv, ch, "np-flat"),
                     _np_filter(inv, ch, "np-hybrid"),
                     _np_filter(inv, ch, "np-pole-ext")):
            profs.append(droop.parity_profile(filt(x), m, lo, hi))
        for label, keep in (("shape", False), ("full", True)):
            vals = [_dehum_rms(p, keep_const=keep) for p in profs]
            print(f"  {fid:6} {label:6} " + " ".join(f"{v:8.5f}" for v in vals))


# --------------------------------------------------------------------------
# validation 2: porch transfer (trailing shadows on a known-constant signal)
# --------------------------------------------------------------------------


def _porch_slope(x: np.ndarray, m: np.ndarray) -> tuple[float, float, float]:
    """Slope/correlation of porch level vs previous-trace recorded picture
    mean (droop.py Q3). A correct inverse drives both to ~0."""
    n = min(len(m) - 1, 512)
    porch, pm, par = [], [], []
    for k in range(6, n):
        p = m[k + 1] - m[k]
        pa, pb = int(m[k] + 100 / 3200 * p), int(m[k] + 225 / 3200 * p)
        a, b = int(m[k - 1] + 232 / 3200 * p), int(m[k - 1] + 3040 / 3200 * p)
        if pa < 0 or pb > len(x) or a < 0 or b > len(x):
            continue
        porch.append(float(np.median(x[pa:pb])))
        pm.append(float(x[a:b].mean()))
        par.append(k % 2)
    porch, pm, par = np.asarray(porch), np.asarray(pm), np.asarray(par)
    for pp in (0, 1):
        porch[par == pp] -= np.median(porch[par == pp])
    pm -= pm.mean()
    med = np.median(porch)
    mad = 1.4826 * np.median(np.abs(porch - med)) + 1e-12
    keep = np.abs(porch - med) < 8 * mad
    porch, pm = porch[keep], pm[keep]
    b = float((pm * porch).sum() / (pm * pm).sum())
    r = float((pm * porch).sum()
              / np.sqrt((pm * pm).sum() * (porch * porch).sum()))
    return b, r, float(porch.max() - porch.min())


def validate_porch(meas: dict) -> None:
    print("\n== VALIDATION 2: porch transfer (shadow/anti-shadow), picture frames ==")
    print("porch is constant at source; slope vs prev-trace picture mean and its")
    print("wander must go to ~0 under a correct inverse. Data the estimate never saw.")
    inv = build_inverses(meas["agg"])
    print(f"  {'frame':6} {'raw slope':>10} {'r':>6} {'p-p':>7} | "
          f"{'pole slope':>10} {'r':>6} {'p-p':>7} | "
          f"{'np-hybrid':>10} {'r':>6} {'p-p':>7}")
    for fid in FRAME_FIDS:
        ch = fid[0]
        x, m, _ = droop.load(fid, 3.0, 515)
        cells = []
        for filt in (None, _pole_filter(ch), _np_filter(inv, ch, "np-hybrid")):
            xs = x if filt is None else filt(x)
            b, r, pp = _porch_slope(xs, m)
            cells.append(f"{b:+10.2f} {r:+6.2f} {pp:7.4f}")
        print(f"  {fid:6} " + " | ".join(cells))


# --------------------------------------------------------------------------
# validation 3: L000 flat calibration field
# --------------------------------------------------------------------------


def validate_flat(meas: dict) -> None:
    print("\n== VALIDATION 3: L000 calibration field flatness ==")
    print("uniform grey surround left (traces 15..95) and right (430..500) of")
    print("the ring, rows 35..330 (between the black bars): the along-trace")
    print("profile must be flat. std/p-p of the median column profile, raw")
    print("signal units (frame black-white range is ~0.24).")
    x, m, _ = droop.load("L000", 3.0, 515)
    inv = build_inverses(meas["agg"])
    for name, filt in (("raw", None), ("pole", _pole_filter("L")),
                       ("np-hybrid", _np_filter(inv, "L", "np-hybrid")),
                       ("np-pole-ext", _np_filter(inv, "L", "np-pole-ext"))):
        xs = x if filt is None else filt(x)
        img = geometry.decode_marks(xs, m)
        outs = []
        for c0, c1 in ((15, 95), (430, 500)):
            prof = np.median(img[35:330, c0:c1], axis=1)
            outs.append(f"std {prof.std():.4f} p-p {prof.ptp():.4f}")
        print(f"  {name:12} left: {outs[0]}   right: {outs[1]}")


# --------------------------------------------------------------------------
# validation 4: picture step edges hold their level (qualitative)
# --------------------------------------------------------------------------


def validate_step(meas: dict) -> None:
    print("\n== VALIDATION 4: picture step-edge hold (qualitative; scene-")
    print("contaminated -- droop.py's warning applies, do not fit) ==")
    inv = build_inverses(meas["agg"])
    for fid in ("L055", "R040"):
        ch = fid[0]
        x, m, _ = droop.load(fid, 3.0, 515)
        for name, filt in (("raw", None), ("pole", _pole_filter(ch)),
                           ("np", _np_filter(inv, ch, "np-hybrid"))):
            xs = x if filt is None else filt(x)
            R, _ = droop._step_ensemble(droop._dot_matrix(xs, m))
            if len(R) < 25:
                print(f"  {fid} {name}: only {len(R)} edges")
                continue
            med = np.median(R, axis=0)
            print(f"  {fid} {name:4} ({len(R):3d} edges): @2 {med[1]:+.2f} "
                  f"@5 {med[4]:+.2f} @10 {med[9]:+.2f} @20 {med[19]:+.2f} "
                  f"@40 {med[39]:+.2f}")


# --------------------------------------------------------------------------
# validation 5: the frozen test set through the real decoder
# --------------------------------------------------------------------------


def validate_images(meas: dict, frames: list[str] | None = None) -> None:
    """Decode test frames with decode.py (untouched), the 384 kHz signal
    inverse-filtered BEFORE decimation -- the exact correction path droop.py
    prescribed, prototyped without editing the decoder."""
    import math

    from scipy.signal import resample_poly

    from . import catalog as catalog_mod
    from . import decode as decode_mod
    from . import quality
    from . import sync as sync_mod
    from . import testset
    from . import wav

    print("\n== VALIDATION 5: frozen test set, decode.py untouched ==")
    print("384 kHz signal filtered before decimation; composite from quality.py.")
    print("Also printed: along-trace low/high spectral ratio -- DESCRIPTION ONLY,")
    print("it rises under any low-frequency boost and is not a success criterion.")
    inv = build_inverses(meas["agg"])
    cat = catalog_mod.build()
    fids = frames or [f[0] for f in testset.TEST_FRAMES]
    info = wav.probe(testset.MASTER)

    def lowhigh(img):
        a = np.asarray(img, dtype=np.float64)
        a = a - a.mean(axis=0, keepdims=True)
        P = np.abs(np.fft.rfft(a, axis=0)) ** 2
        return float(P[1:9].mean() / P[47:150].mean())

    def one(fid, filt):
        frame = cat.by_id(fid)
        head = testset.PRE_384
        span = int(math.ceil((sync_mod.TRACES_PER_FRAME + 6)
                             * sync_mod.NOMINAL_PERIOD))
        x = np.asarray(wav.read(info, frame.channel, frame.seed_sample - head,
                                head + span), dtype=np.float64)
        if filt is not None:
            x = filt(x)
        x96 = resample_poly(x, 1, testset.DECIM)
        tb = sync_mod.recover(x96, period_guess=testset.NOMINAL_96,
                              n_traces=sync_mod.TRACES_PER_FRAME)
        # Percentile levels for BOTH raw and corrected: decode.py's reference
        # anchors (porch +/- 0.75 x sync amplitude) were calibrated on the
        # DROOPED picture statistics; the restored picture legitimately spans
        # a wider range and clips white under them (first run of this section
        # measured mean composite 18.9 -> 14.6 with L000's field clipped to
        # white; the anchors need re-deriving after correction, outside this
        # module's remit).
        dec = decode_mod.decode(np.asarray(x96, dtype=np.float32),
                                decode_mod.Settings(levels="percentile"), tb)
        fs = quality.frame_report(dec.image, tb, frame_id=fid)
        return dec, fs.metrics

    rows = []
    print(f"  {'frame':6} {'comp raw':>8} {'comp np':>8} {'d':>6} | "
          f"{'par raw':>7} {'par np':>7} | {'shift raw':>9} {'np':>5} | "
          f"{'L/H raw':>8} {'np':>8}")
    for fid in fids:
        ch = fid[0]
        try:
            dec0, m0 = one(fid, None)
            dec1, m1 = one(fid, _np_filter(inv, ch, "np-hybrid"))
        except Exception as exc:
            print(f"  {fid:6} FAILED: {exc}")
            continue
        rows.append((fid, m0, m1, lowhigh(dec0.image), lowhigh(dec1.image),
                     dec0.image, dec1.image))
        print(f"  {fid:6} {m0['composite']:8.1f} {m1['composite']:8.1f} "
              f"{m1['composite'] - m0['composite']:+6.1f} | "
              f"{m0['parity_db']:7.1f} {m1['parity_db']:7.1f} | "
              f"{m0['shift_rms']:9.2f} {m1['shift_rms']:5.2f} | "
              f"{rows[-1][3]:8.2f} {rows[-1][4]:8.2f}")
    if rows:
        c0 = np.mean([r[1]["composite"] for r in rows])
        c1 = np.mean([r[2]["composite"] for r in rows])
        print(f"  mean composite raw {c0:.1f} -> np {c1:.1f} ({c1 - c0:+.1f}) "
              f"over {len(rows)} frames")
        if "L000" in [r[0] for r in rows]:
            from . import quality as q
            i = [r[0] for r in rows].index("L000")
            for tag, img in (("raw", rows[i][5]), ("np", rows[i][6])):
                cm = q.circle_metrics(img)
                print(f"  CIRCLE L000 {tag}: ok={cm['ok']} ratio "
                      f"{cm['axis_ratio']:.4f} rms {cm['radial_rms']:.2f}")
    # save PNG pairs for eyeballing (scratch output, not a deliverable)
    try:
        from pathlib import Path

        from PIL import Image
        out = Path("/private/tmp/claude-501/-Users-josh-code-golden-record-images"
                   "/484b41e6-9824-495a-97cb-a0b8bfe1c684/scratchpad/blind_png")
        out.mkdir(parents=True, exist_ok=True)
        for fid, _, _, _, _, i0, i1 in rows:
            pair = np.concatenate([i0, np.ones((i0.shape[0], 4)), i1], axis=1)
            Image.fromarray((np.clip(pair, 0, 1) * 255 + 0.5).astype(np.uint8),
                            "L").save(out / f"{fid}_raw_np.png")
        print(f"  raw|np PNG pairs -> {out}")
    except Exception as exc:
        print(f"  (png dump skipped: {exc})")


# --------------------------------------------------------------------------
# method B: porch-constancy inverse fit (independent, content-driven)
# --------------------------------------------------------------------------

LAG_EDGES = [322, 700, 1400, 2600, 4800, 9600]


def porchfit_channel(ch: str, ridge: float = 1e-4) -> dict | None:
    """Solve for the inverse correction kernel directly from the constraint
    that the CORRECTED porch must be constant.

    Model: x_hat = y + g*y with g piecewise-constant over LAG_EDGES windows
    (lags < 322 samples never reach picture content from the porch and are
    unidentifiable -- dropped). Porch of x_hat is linear in the window
    weights theta_m = g_m * width_m, so least squares applies with the
    RECORDED signal as regressor -- no errors-in-variables, unlike the
    forward regressions droop.py documented as failures. Per-parity running
    medians are removed from target and features alike (hum + drift).
    Pole prediction: g_m = 1 - exp(-1/tau), flat in lag.
    """
    rows_T, rows_F = [], []
    for fid in PORCHFIT_FIDS[ch]:
        try:
            x, m, _ = droop.load(fid, 3.0, 515)
        except Exception as exc:
            print(f"    {fid}: skipped ({exc})")
            continue
        cs = np.concatenate([[0.0], np.cumsum(x)])
        nwin = len(LAG_EDGES) - 1
        n = min(len(m) - 1, 512)
        P, F = [], []
        for k in range(6, n):
            p = m[k + 1] - m[k]
            pa = int(m[k] + 100 / 3200 * p)
            pb = int(m[k] + 225 / 3200 * p)
            if pa - LAG_EDGES[-1] < 0 or pb > len(x):
                continue
            P.append(float(np.median(x[pa:pb])))
            feats = []
            idx = np.arange(pa, pb)
            for w in range(nwin):
                l0, l1 = LAG_EDGES[w], LAG_EDGES[w + 1]
                # mean over porch samples of the lag-window mean of y
                s = (cs[idx - l0 + 1] - cs[idx - l1 + 1]).mean() / (l1 - l0)
                feats.append(s)
            F.append(feats)
        P = np.asarray(P)
        F = np.asarray(F)
        if len(P) < 64:
            continue
        # per-parity running-median detrend of target AND features
        def detr(v):
            out = v.astype(np.float64).copy()
            for par in (0, 1):
                s = out[par::2]
                if len(s) < 9:
                    continue
                pad = np.pad(s, 4, mode="edge")
                win = np.lib.stride_tricks.sliding_window_view(pad, 9)
                out[par::2] = s - np.median(win, axis=1)
            return out
        Pd = detr(P)
        Fd = np.column_stack([detr(F[:, j]) for j in range(F.shape[1])])
        med = np.median(Pd)
        mad = 1.4826 * np.median(np.abs(Pd - med)) + 1e-12
        keep = np.abs(Pd - med) < 8 * mad
        rows_T.append(Pd[keep])
        rows_F.append(Fd[keep])
    if not rows_T:
        return None
    T = np.concatenate(rows_T)
    F = np.concatenate(rows_F)
    # minimize |T + F theta|^2  (corrected porch deviation -> 0), IRLS + ridge
    scale = np.sqrt((F ** 2).mean(axis=0)) + 1e-12
    Fn = F / scale
    w = np.ones(len(T))
    theta_n = np.zeros(F.shape[1])
    for _ in range(6):
        A = Fn * np.sqrt(w)[:, None]
        b = -T * np.sqrt(w)
        theta_n, *_ = np.linalg.lstsq(
            np.vstack([A, np.sqrt(ridge * len(T)) * np.eye(F.shape[1])]),
            np.concatenate([b, np.zeros(F.shape[1])]), rcond=None)
        r = T + Fn @ theta_n
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = np.clip(r / (4.0 * s), -1, 1)
        w = (1 - u ** 2) ** 2
    theta = theta_n / scale
    r = T + F @ theta
    resvar = float((w * r ** 2).sum() / max(w.sum() - len(theta), 1))
    cov = resvar * np.linalg.pinv((F * w[:, None]).T @ F)
    se = np.sqrt(np.diag(cov))
    var_raw = float(np.var(T))
    var_fit = float(np.var(r))
    g = np.array([theta[m_] / (LAG_EDGES[m_ + 1] - LAG_EDGES[m_])
                  for m_ in range(len(theta))])
    g_se = np.array([se[m_] / (LAG_EDGES[m_ + 1] - LAG_EDGES[m_])
                     for m_ in range(len(theta))])
    return {"theta": theta, "se": se, "g": g, "g_se": g_se,
            "var_raw": var_raw, "var_fit": var_fit, "n": len(T)}


def measure_porchfit() -> None:
    print("\n== METHOD B: porch-constancy inverse fit (independent of the gap probe) ==")
    print("solve x_hat = y + g*y such that corrected porches are constant;")
    print("g piecewise-constant over lag windows (samples). Pole predicts")
    print("g = 1-exp(-1/tau), FLAT in lag: L 1.89e-3, R 3.39e-3.")
    for ch in "LR":
        r = porchfit_channel(ch)
        if r is None:
            print(f"  channel {ch}: no usable data")
            continue
        pred = 1.0 - np.exp(-1.0 / droop.TAU_384[ch])
        print(f"  channel {ch} (n={r['n']} porches, frames "
              f"{' '.join(PORCHFIT_FIDS[ch])}):")
        print(f"    {'lag window':>14} {'g fitted':>10} {'+-':>9} "
              f"{'pole pred':>10}")
        for m_ in range(len(r["g"])):
            print(f"    {LAG_EDGES[m_]:>6}..{LAG_EDGES[m_ + 1]:<6} "
                  f"{r['g'][m_]:>10.2e} {r['g_se'][m_]:>9.1e} {pred:>10.2e}")
        print(f"    porch variance: raw {r['var_raw']:.3e} -> fitted "
              f"{r['var_fit']:.3e} ({r['var_raw'] / max(r['var_fit'], 1e-30):.2f}x)")
        # implied frequency response at a few checkpoints
        f_chk = np.array([100.0, 180.0, 300.0, 600.0])
        om = 2 * np.pi * f_chk / FS
        Wb = np.ones(len(f_chk), dtype=complex)
        for m_ in range(len(r["g"])):
            l = np.arange(LAG_EDGES[m_], LAG_EDGES[m_ + 1])
            Wb += r["g"][m_] * np.exp(-1j * np.outer(om, l)).sum(axis=1)
        Hb = 1.0 / Wb
        print("    implied |H| at 100/180/300/600 Hz: "
              + " ".join(f"{np.abs(v):.3f}" for v in Hb)
              + "   (pole: "
              + " ".join(f"{np.abs(h_pole(np.array([f]), droop.TAU_384[ch])[0]):.3f}"
                         for f in f_chk) + ")")


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="non-parametric decay-bias estimate")
    ap.add_argument("--section", default="transfer,invert,porch,flat,step,porchfit",
                    help="comma list of transfer,invert,porch,flat,step,porchfit,images")
    ap.add_argument("--frames", default="", help="subset for --section images")
    args = ap.parse_args(argv)
    wanted = [s.strip() for s in args.section.split(",") if s.strip()]
    meas = measure_transfer()
    for name in wanted:
        if name == "transfer":
            continue  # always runs first
        elif name == "invert":
            validate_gap(meas)
        elif name == "porch":
            validate_porch(meas)
        elif name == "flat":
            validate_flat(meas)
        elif name == "step":
            validate_step(meas)
        elif name == "porchfit":
            measure_porchfit()
        elif name == "images":
            frames = [s.strip() for s in args.frames.split(",") if s.strip()] or None
            validate_images(meas, frames)
        else:
            print(f"unknown section {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
