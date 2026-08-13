"""The low-frequency "shelf": measured, explained, and REFUTED.

    python -m pipeline.shelf                       # everything (~60 s)
    python -m pipeline.shelf --section band,synth   # pick sections

WHAT THIS RESOLVES
------------------
pipeline/droop_blind.py estimated the chain's transfer non-parametrically from
the sync-parity gap probe and found, on top of droop.py's one-pole model, an
extra SHELVING LOSS of |H| ~ 0.72-0.88 between 180 Hz and 2.5 kHz, plateau
~0.87-0.90 on both channels, one-pole rejected at chi2/dof 74 (L) / 131 (R).
Its inverse ("np-hybrid") halved L000's flat-field ripple in signal space but
regressed the decoder end to end, so it was documented as an open question.

VERDICT: THE SHELF IS AN ARTEFACT OF THE ESTIMATOR, NOT A PROPERTY OF THE
CHAIN.  It is manufactured by high-frequency content of the excitation that the
estimator's source model does not contain.  There is nothing there to recover.
Five independent lines, each measured below on the real master:

1. IT IS NOT IDENTIFIED BY THE DATA.  The estimator has to declare some band
   flat to fix the source amplitudes; the plateau depth is whatever that
   choice makes it.  Refitting the same frames in five different bands moves
   |S| = |H|/|H_pole| at 180 Hz-2.5 kHz over 1.09 / 0.86 / 0.78 / 0.78 / 0.56
   (bands 1.4-2.7k, 2.7-9.7k = shipped, 3.7-7.3k, 9.7-15.7k, 15.7-24k) --
   while the SHAPE inside 180 Hz-2.5 kHz stays flat in every one of them.  A
   quantity that flat in shape and that unstable in scale is a gain, and gain
   is exactly what the probe cannot measure and what the decoder's level
   anchors remove anyway.  (`band`)

2. THE SOURCE MODEL FAILS WHERE THE NORMALISATION IS IMPOSED.  The recorded
   parity profile carries 1.1x the ideal-rectangle model's energy at 5-7.5
   kHz, 1.3x at 10-14 kHz and 1.6-2.0x at 14-26 kHz, and its pulse edges
   overshoot ~2x (bin 3054 reads -0.23 against a -0.115 plateau on L030).
   The levels are fitted at 2.7-9.7 kHz, i.e. inside the region where the
   model is already 10-30% wrong.  (`probe`)

3. A KNOWN ONE-POLE CHAIN REPRODUCES THE WHOLE EFFECT.  Push a synthetic
   tau=530 chain -- truth S == 1 by construction -- with an excitation that
   carries the same edge peaking through the identical estimator: it reports
   a flat shelf |S| = 0.90 (symmetric peaking, HF excess 1.9x) or 0.61-0.75
   (causal peaking), flat below ~2 kHz and "recovering" toward unity at the
   fit-band edge, with excess phase +5..+13 deg when the peaking is causal.
   That is the measured signature -- |S| 0.82-0.88, arg +6..+20 deg -- out of
   a chain that has no shelf at all.  (`synth`)

4. IT IS NOT MINIMUM PHASE AND IT IS NOT PER CHANNEL.  |S| is flat to +-3%
   over a decade yet carries +6..+20 deg of phase; a minimum-phase network
   with flat magnitude has none.  And S_L = S_R to ~1-2% at every harmonic
   although the two channels' poles differ by 1.8x -- so whatever S is, it is
   not the per-channel AC-coupling the shelf was supposed to extend.  The
   channel RATIO H_L/H_R, which needs no source model at all (the same
   converter generated both patterns, so its amplitude cancels), agrees with
   pole_530/pole_295 to 1-3% over 180 Hz-2.5 kHz: the two POLES are
   confirmed model-free, the shelf is common-mode.  (`models`)

5. THE ONE INSTRUMENT WITH A KNOWN INPUT CANNOT SEE IT.  Content-free gap
   lines must come out flat.  Measured noise-unbiased (two independent halves
   of the gap traces, cross-covariance, so a filter that merely amplifies
   noise cannot win) and gain-invariant (residual divided by the sync
   amplitude read off the same corrected profile, so a filter that merely
   attenuates cannot win either), the objective RESOLVES A POLE -- interior
   optimum at tau 400-530 on the L frames and 530-900 on the R, 25-35% below
   the uncorrected value -- and is FLAT against the shelf: sweeping the depth
   over 0.8 <= g <= 1.4 at corners 400 Hz and 2.5 kHz moves it by 7-25%
   monotonically, with no optimum anywhere.  A shelf whose corner sits above
   the band this probe excites is a gain, and a gain is exactly what neither
   the probe nor the decoder can see.  Without the gain-invariant
   normalisation the same scan is monotone the other way -- the np-hybrid's
   direction makes the known-flat line worse on 10 frames of 10 -- but that
   comparison is itself only measuring gain, and is reported as such.  (`gap`)

WHY THE PROTOTYPE LOOKED LIKE A WIN, AND WHY IT REGRESSED
---------------------------------------------------------
The flat-field number that justified np-hybrid (L000 field std 0.0225 -> 0.0148)
does not come from the shelf.  Split the np-hybrid inverse into its magnitude
and its phase and run each alone through the same test (`flat`):

    pole                     std 0.0225   contrast-normalised 0.1235
    np-hybrid                std 0.0148                       0.0819
    np MAGNITUDE only        std 0.0260                       0.1194   <- worse
    np PHASE only            std 0.0130                       0.0852   <- all of it
    pole x constant 20 deg   std 0.0141                       0.0937
    pole x constant 30 deg   std 0.0103                       0.0616   <- and more
    pole x gain 0.66         std 0.0148                       0.1235   <- gain only

The shelf's magnitude makes L000's field WORSE, by exactly its own gain factor
(0.0225 x 1.166 = 0.0262 measured 0.0260).  The improvement is entirely the
CONSTANT -19.5 deg phase rotation that make_W applies below its lowest
measured harmonic, and that knob is unbounded: 30 degrees "improves" the field
further, 40 further still.  A constant phase rotation is a partial Hilbert
transform -- it turns the field's monotone ramp into a symmetric bump, which
lowers its standard deviation without removing anything.  And because the two
arms had different overall gains, the std comparison in raw signal units was
measuring gain as well: a pure 0.66x gain reproduces the reported 0.0148
exactly, while the contrast-normalised number does not move at all (0.1235).

PARITY STRIPING: THE SHELF IS NOT THE CULPRIT (`images`).  droop_blind read
parity_db 4.4 -> 14.7 dB on L034, but that compared NO correction against
np-hybrid, which contains the pole.  Isolated against the shipped decoder --
every arm keeping decode.py's own pole and its undrooped rails, only the extra
filter changing -- the shelf does not raise striping at all:

    frame L034 parity_db:  no correction 4.5 | shipped pole 10.9 |
                           pole + np excess 8.3 | pole + shelf magnitude 11.4
    16-frame mean composite: 18.9 | 19.1 | 17.5 | 17.2
    circle L000 axis ratio/rms: 1.0075/0.95 | 1.0060/0.88 | 1.0042/1.00 | 1.0064/0.91

The +6.4 dB on L034 belongs to the POLE, which is already shipped: inverting a
high-pass boosts 60 Hz by 2.16x, and the parity-locked pattern lives at exactly
60 Hz with an envelope that a single median profile cannot follow.  De-humming
in signal space BEFORE the correction (subtract the parity-median profile from
the raw signal inside the picture window, then undroop) was tried and is not a
fix: L034 parity_db 10.9 -> 9.4 dB but L000 3.3 -> 4.0 dB and its composite
30.1 -> 20.4.  Reported as a negative result, not shipped.

THE SAMPLE-AND-HOLD APERTURE (`aperture`): CHECKED, EXCLUDED, WRONG SIGN
------------------------------------------------------------------------
The converter holds each dot for spd = P/262.5 = 12.17 samples, an aperture
whose first null is the dot rate, 31.5 kHz.  |sinc| over the shelf band is
0.99998 (180 Hz) to 0.9896 (2.5 kHz): 1.0% of shape where the shelf claims
12-28%.  Over the normalisation band it is 0.977 (3.7 kHz) to 0.851 (9.7 kHz),
mean 0.930 -- so if the aperture WERE in the probe path, imposing unity there
would push the low-frequency estimate UP by 1/0.930 = +7.5%, the opposite sign
to the reported deficit.  It is not in the probe path in any case: the sync
pattern is generated by the converter's own timing, not sampled from the
source; and in the picture path decode.py already integrates each plateau over
its own dwell, which is the matched filter for a hold and returns the held
value exactly.  The aperture's real cost is on the SOURCE side -- it band-limits
what the converter saw -- and that is not invertible from the audio.

WHAT IS ACTUALLY LEFT (open, and NOT a shelf)
---------------------------------------------
After the shipped pole the known-flat regions are flat to a residual that is
85-90% a straight LINE across the trace: the gap line's residual drops from
0.0033-0.0055 to 0.0005-0.0007 rms under linear detrending, and the leftover
tilt is +0.011..+0.023 signal units per trace (4-9% of the black-white range).
It is present in the RAW signal too (+0.0132 on L010 uncorrected vs +0.0113
corrected), it scales with 1/tau only weakly, and it sits BELOW the probe's
lowest clean harmonic (180 Hz), where the parity probe is degenerate with the
hum and measures nothing.  A second inverse pole reduces the tilt on 6 frames
of 8 but always adds curvature and never nulls both, and on the R channel it
makes the tilt worse -- so it is not a single extra pole either.  L000's field
carries a much larger version of the same ramp (0.066 p-p, 27% of contrast),
but that frame cannot arbitrate: a 1977 vidicon's vertical shading would print
exactly the same monotone ramp along a trace, and nothing in the signal
separates the two.  This is the honest open question the shelf was mistaken
for; it lives at 10-100 Hz, not at 180 Hz-2.5 kHz.

INTEGRATION SPEC
----------------
* Ship nothing from this module.  The pole alone remains the validated,
  artefact-free correction; UNCOUPLE_TAU_384 is untouched and independently
  confirmed here (the channel ratio, model-free, and the L-channel gap
  flatness, which minimises at tau = 530).
* decode.py's docstring paragraph "KNOWN RESIDUAL, deliberately not corrected:
  ... an extra ~15% shelving loss between ~200 Hz and ~2.5 kHz ... mid-scale
  structure stays ~15% under-restored" should be REPLACED: the estimate is not
  identified, the number is set by the estimator's normalisation band, and no
  mid-scale under-restoration has been demonstrated.
* droop_blind.py's docstring points 2, 3 and 5 (the shelf, the "amplitude
  deficit" and the flat-field validation) carry the same retraction.
* Any future transfer estimate from this probe must fit the SOURCE's high
  frequency content, not assume sharp rectangles, or must normalise in a band
  where the source model has been shown to hold -- and must state the gain it
  cannot measure rather than absorb it into a shelf.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import droop, droop_blind as db, geometry

FS = 384000.0
BINS = geometry.BINS
TWO = 2 * BINS

# The known-flat window of a gap trace: inside the picture gate (232..3040)
# with margin at both ends.
FLAT_LO, FLAT_HI = 300, 3000

GAP_FIDS = ["L010", "L020", "L030", "L055", "L065",
            "R020", "R030", "R040", "R056", "R065"]

# Odd harmonics reported by the transfer sections, k -> f = k*FS/(2P) ~ 60k Hz.
KS_REPORT = np.array([3, 5, 9, 15, 21, 31, 41, 51, 71, 101, 141, 201, 281, 361])

_GAP_CACHE: dict = {}


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------


def shelf(f, g: float, fc: float) -> np.ndarray:
    """First-order minimum-phase shelf: |S| = g at DC, 1 well above fc.

    g < 1 is the hypothesised extra low-frequency LOSS; its inverse 1/S is the
    low-frequency BOOST the np-hybrid applies.
    """
    f = np.abs(np.asarray(f, dtype=np.float64))
    return (g + 1j * f / fc) / (1.0 + 1j * f / fc)


def apply_inverse(x: np.ndarray, W) -> np.ndarray:
    """Filter x (384 kHz samples) with the frequency response W(f in Hz)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n + 65536)))
    mu = float(x.mean())
    X = np.fft.rfft(x - mu, nfft)
    f = np.fft.rfftfreq(nfft) * FS
    y = np.fft.irfft(X * W(f), nfft)[:n]
    return y + mu * float(np.real(W(np.array([0.0]))[0]))


def inv_shelf_W(g: float, fc: float):
    def W(f):
        f = np.abs(np.asarray(f, dtype=np.float64))
        out = 1.0 / shelf(f, g, fc)
        return np.where(f == 0, np.real(out), out)
    return W


def np_excess_W(inv: dict, ch: str, mode: str = "full"):
    """np-hybrid's inverse DIVIDED BY the pole inverse: exactly what the shelf
    correction adds on top of what decode.py already does.  mode 'mag' keeps
    only its magnitude, 'phase' only its phase."""
    Wnp, tau = inv[ch]["np-hybrid"], droop.TAU_384[ch]

    def W(f):
        f = np.abs(np.asarray(f, dtype=np.float64))
        E = Wnp(f) * db.h_pole(np.maximum(f, 1e-9), tau)
        if mode == "mag":
            E = np.abs(E) + 0j
        elif mode == "phase":
            E = np.exp(1j * np.angle(E))
        return np.where(f == 0, np.real(E), E)
    return W


def pole_rotate_W(tau: float, theta_deg: float = 0.0, gain: float = 1.0):
    """The pole inverse times a constant complex factor -- the operation that
    make_W performs below its lowest measured harmonic."""
    def W(f):
        f = np.abs(np.asarray(f, dtype=np.float64))
        out = gain * np.exp(-1j * np.radians(theta_deg)) / db.h_pole(np.maximum(f, 1e-9), tau)
        return np.where(f == 0, np.real(out), out)
    return W


# --------------------------------------------------------------------------
# the gap instrument: content-free lines that must come out flat
# --------------------------------------------------------------------------


def load_gap(fid: str, n_traces: int = 190):
    """(signal, marks, gap trace indices).  Gap traces are selected on
    HIGH-frequency content (adjacent-bin scatter), which picture detail has
    and the droop response does not -- so the selection cannot prefer one
    sync parity over the other."""
    if fid in _GAP_CACHE:
        return _GAP_CACHE[fid]
    x, m, _ = droop.load(fid, 195.0, n_traces)
    M = geometry.norm_traces(x, m)
    hf = np.nanstd(np.diff(M[:, FLAT_LO:FLAT_HI], axis=1), axis=1)
    good = np.isfinite(hf) & (hf < 2.0 * np.nanpercentile(hf, 20))
    idx = np.where(good)[0]
    if len(idx) > 40:
        idx = idx[idx <= max(idx.max() - 5, 40)]
    _GAP_CACHE[fid] = (x, m, idx)
    return _GAP_CACHE[fid]


def half_profiles(x: np.ndarray, m: np.ndarray, idx: np.ndarray):
    """Two INDEPENDENT common-mode gap profiles (traces interleaved by 2, so
    each half keeps both sync parities).

    Common mode, not parity difference: the scan-locked hum is exactly
    2-trace periodic, so it cancels in the mean and contaminates nothing here.
    """
    M = geometry.norm_traces(x, m)
    out = []
    for h in (0, 1):
        sub = idx[(idx // 2) % 2 == h]
        ps = []
        for p in (0, 1):
            s = sub[sub % 2 == p]
            s = s[np.isfinite(M[s, 0])]
            ps.append(np.median(M[s], axis=0))
        out.append(0.5 * (ps[0] + ps[1]))
    return out


_BB = (np.arange(FLAT_LO, FLAT_HI, dtype=np.float64)
       - (FLAT_LO + FLAT_HI) / 2) / (FLAT_HI - FLAT_LO)


def _detrend(p: np.ndarray, order: int) -> np.ndarray:
    seg = p[FLAT_LO:FLAT_HI]
    X = np.column_stack([_BB ** k for k in range(order + 1)])
    c, *_ = np.linalg.lstsq(X, seg, rcond=None)
    return seg - X @ c


def gap_residual(pa: np.ndarray, pb: np.ndarray, order: int = 0) -> float:
    """Noise-unbiased systematic residual of the known-flat window.

    Cross-covariance of two independent half-profiles: tape noise is
    independent between halves and averages to zero, so a filter cannot win
    this by trading systematic error for noise.  Returns a signed rms
    (negative when the cross-covariance is negative, i.e. residual < noise).
    """
    a, b = _detrend(pa, order), _detrend(pb, order)
    c = float(np.mean(a * b))
    return float(np.sign(c) * np.sqrt(abs(c)))


def gap_amplitude(p: np.ndarray) -> float:
    """Sync amplitude (short-burst plateau minus back porch) read off the SAME
    corrected profile.  Dividing the residual by this makes the objective
    GAIN-INVARIANT: without it, 'flattest' is won by whatever attenuates the
    low frequencies most, which is the very degeneracy under test.
    """
    return float(np.median(p[3165:3195]) - np.median(p[100:225]))


def gap_shape(p: np.ndarray):
    """(tilt, curvature, rms after a quadratic) of the known-flat window."""
    seg = p[FLAT_LO:FLAT_HI]
    X = np.column_stack([np.ones(len(_BB)), _BB, _BB ** 2])
    c, *_ = np.linalg.lstsq(X, seg, rcond=None)
    return float(c[1]), float(c[2]), float(np.std(seg - X @ c))


# --------------------------------------------------------------------------
# section: reproduce the transfer estimate, then move the normalisation band
# --------------------------------------------------------------------------


def _fit_input_band(D: np.ndarray, k0: int, k1: int):
    """droop_blind.fit_input with the flat band as an argument."""
    B = db._seg_bases()
    ks = np.arange(k0, k1 + 1, 2)
    best = None
    for delta in np.arange(-3.0, 3.001, 0.05):
        ph = np.exp(-2j * np.pi * ks * delta / TWO)
        G = (B[:, ks] * ph).T
        Gr = np.vstack([G.real, G.imag])
        Dr = np.concatenate([D[ks].real, D[ks].imag])
        A, *_ = np.linalg.lstsq(Gr, Dr, rcond=None)
        r = float(((Dr - Gr @ A) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, delta, A)
    return best[2], best[1]


def _gap_spectrum(fid: str):
    x, m, _ = droop.load(fid, 195.0, 60)
    lo, hi = droop.gap_slots(m)
    d = droop.parity_profile(x, m, lo, hi)
    P = float(np.median(np.diff(m[max(lo, 1):hi])))
    return np.fft.rfft(np.concatenate([d, -d])), P


BANDS = [
    ("1.4-2.7 kHz", 23, 45),
    ("2.7-9.7 kHz  <- shipped", 45, 161),
    ("3.7-7.3 kHz", 61, 121),
    ("9.7-15.7 kHz", 161, 261),
    ("15.7-24 kHz", 261, 401),
]

CH_FIDS = {"L": ["L010", "L020", "L030", "L055", "L065", "L077"],
           "R": ["R010", "R020", "R030", "R040", "R056", "R065"]}


def section_band() -> None:
    print("== 1. THE PLATEAU IS WHATEVER BAND YOU DECLARE FLAT ==")
    print("S(f) = |H| / |H_pole(shipped tau)|.  A real shelf is a property of the")
    print("chain; this one is a property of the estimator's normalisation choice.")
    for ch in "LR":
        tau = droop.TAU_384[ch]
        Ds, Ps = zip(*[_gap_spectrum(f) for f in CH_FIDS[ch]])
        P = float(np.mean(Ps))
        f = KS_REPORT * FS / (2 * P)
        print(f"\n  channel {ch} (n={len(Ds)} gap frames, pole tau={tau:.0f})")
        print("    normalisation band     " + "".join(f"{v:8.0f}" for v in f[:8]) + "  Hz")
        for name, k0, k1 in BANDS:
            Hs = []
            for D in Ds:
                A, delta = _fit_input_band(D, k0, k1)
                Xk = (A @ db._seg_bases()[:, KS_REPORT]) * np.exp(
                    -2j * np.pi * KS_REPORT * delta / TWO)
                Hs.append(D[KS_REPORT] / Xk)
            H = np.median(np.real(Hs), axis=0) + 1j * np.median(np.imag(Hs), axis=0)
            S = H / db.h_pole(f, tau)
            sd = np.std(np.abs(np.asarray(Hs) / db.h_pole(f, tau)[None, :]), axis=0)
            print(f"    |S| {name:22}" + "".join(f"{abs(v):8.3f}" for v in S[:8]))
            print(f"        {'':22}" + "".join(f"{v:8.3f}" for v in sd[:8] / np.sqrt(len(Ds)))
                  + "   +- (sem across frames)")
        print("    the SHAPE is flat in every band; only the SCALE moves, by 2x.")


def section_probe() -> None:
    print("\n== 2. WHERE THE SOURCE MODEL FAILS ==")
    print("|D| / |X_model| by band.  The estimator imposes |H| = 1 over 2.7-9.7 kHz")
    print("and reads the shelf off everything below; if the recorded pattern is not")
    print("the model's sharp rectangle up there, the fitted levels -- and therefore")
    print("the whole low-frequency scale -- are wrong by that much.")
    for fid in ("L030", "R040"):
        D, P = _gap_spectrum(fid)
        A, delta = _fit_input_band(D, 45, 161)
        ks = np.arange(1, 500, 2)
        Xk = (A @ db._seg_bases()[:, ks]) * np.exp(-2j * np.pi * ks * delta / TWO)
        f = ks * FS / (2 * P)
        ratio = np.abs(D[ks]) / np.abs(Xk)
        print(f"  {fid}: levels {['%+.4f' % v for v in A]} dt {delta:+.2f} bins")
        for f0, f1 in ((0.15, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0),
                       (5.0, 7.5), (7.5, 10.0), (10, 14), (14, 20), (20, 26)):
            s = (f >= f0 * 1000) & (f < f1 * 1000)
            tag = "   <- levels fitted here" if 2.7 <= f0 < 9.7 else ""
            if s.any():
                print(f"    {f0:5.1f}-{f1:5.1f} kHz  n={int(s.sum()):3d}  "
                      f"|D|/|X| {np.median(ratio[s]):6.3f}{tag}")
        x, m, _ = droop.load(fid, 195.0, 60)
        lo, hi = droop.gap_slots(m)
        d = droop.parity_profile(x, m, lo, hi)
        print(f"    pulse edge overshoot: bin 3054 {d[3054]:+.4f} against the "
              f"3070-3086 plateau {np.median(d[3070:3086]):+.4f} "
              f"({d[3054]/np.median(d[3070:3086]):.2f}x)")


# --------------------------------------------------------------------------
# section: synthetic truth -- manufacture the shelf from a shelf-free chain
# --------------------------------------------------------------------------


def _synth_source(levels, alpha: float, w: int, causal: bool) -> np.ndarray:
    """The two-trace parity pattern with edge peaking added: the source is NOT
    the estimator's ideal rectangle, which is the whole point."""
    x = db._build_input(levels)
    if alpha == 0:
        return x
    k = np.ones(w) / w
    y = np.concatenate([x, x, x])
    sm = (np.convolve(y, k, mode="full")[:len(y)] if causal
          else np.convolve(y, k, mode="same"))
    return x + alpha * (x - sm[TWO:2 * TWO])


def section_synth() -> None:
    print("\n== 3. SYNTHETIC TRUTH: a shelf-free chain that measures a shelf ==")
    print("chain = one pole tau=530 EXACTLY, so the true S is 1.000 at every")
    print("frequency.  Only the excitation carries unmodelled edge peaking.")
    P, tau = 3195.0, 530.0
    f = KS_REPORT * FS / (2 * P)
    Ht = db.h_pole(f, tau)
    lev = [-0.140, -0.132, -0.017]
    ideal = np.fft.rfft(db._build_input(lev))
    ks_hi = np.arange(233, 434, 2)
    print(f"  {'excitation':28}{'HFexc':>7}" + "".join(f"{v:8.0f}" for v in f[:7]) + "  Hz")
    print(f"  {'true S':28}{'':>7}" + "".join(f"{1.0:8.3f}" for _ in f[:7]))
    for alpha, w, causal in ((0.0, 9, False), (1.0, 17, False), (2.0, 17, False),
                             (1.0, 17, True), (2.0, 17, True), (1.5, 25, True)):
        s = _synth_source(lev, alpha, w, causal)
        X = np.fft.rfft(s)
        D = X * db.h_pole(np.arange(len(X)) * FS / TWO, tau)
        A, delta = _fit_input_band(D, 45, 161)
        Xk = (A @ db._seg_bases()[:, KS_REPORT]) * np.exp(
            -2j * np.pi * KS_REPORT * delta / TWO)
        S = (D[KS_REPORT] / Xk) / Ht
        hf = float(np.median(np.abs(X[ks_hi]) / np.abs(ideal[ks_hi])))
        name = f"peak a={alpha} w={w} {'causal' if causal else 'symmetric'}"
        print(f"  |S| {name:24}{hf:7.2f}" + "".join(f"{abs(v):8.3f}" for v in S[:7]))
        print(f"  arg {'':24}{'':7}" + "".join(
            f"{np.degrees(np.angle(v)):8.1f}" for v in S[:7]) + "  deg")
    print("  MEASURED on the master, channel L, for comparison:")
    Ds = [_gap_spectrum(fid)[0] for fid in CH_FIDS["L"]]
    Hs = []
    for D in Ds:
        A, delta = _fit_input_band(D, 45, 161)
        Xk = (A @ db._seg_bases()[:, KS_REPORT]) * np.exp(
            -2j * np.pi * KS_REPORT * delta / TWO)
        Hs.append(D[KS_REPORT] / Xk)
    H = np.median(np.real(Hs), axis=0) + 1j * np.median(np.imag(Hs), axis=0)
    S = H / Ht
    print(f"  |S| {'real data':24}{'1.5-2.0':>7}" + "".join(f"{abs(v):8.3f}" for v in S[:7]))
    print(f"  arg {'':24}{'':7}" + "".join(
        f"{np.degrees(np.angle(v)):8.1f}" for v in S[:7]) + "  deg")
    print("  A chain with no shelf, probed with a source model that is wrong at")
    print("  high frequency, returns the measured shelf in depth, flatness and sign.")


# --------------------------------------------------------------------------
# section: model fits and the model-free channel ratio
# --------------------------------------------------------------------------


def section_models() -> None:
    print("\n== 4. MODEL FITS, AND THE CHANNEL RATIO THAT NEEDS NO SOURCE MODEL ==")
    res = {"L": [], "R": []}
    for fid in db.GAP_FIDS:
        r = db.transfer_one(fid)
        if r is not None:
            res[fid[0]].append(r)
    agg = {ch: db.aggregate(res[ch]) for ch in "LR"}
    for ch in "LR":
        a = agg[ch]
        ks, f, H = a["ks"], a["f_hz"], a["H"]
        sr = np.maximum(a["sd_r"] / np.sqrt(a["n"]), 1e-4)
        si = np.maximum(a["sd_i"] / np.sqrt(a["n"]), 1e-4)
        sel = (ks >= 3) & np.isfinite(H)
        fs, Hs, srs, sis = f[sel], H[sel], sr[sel], si[sel]
        n2 = 2 * int(sel.sum())

        def chi2(model):
            return float(np.sum(((Hs - model).real / srs) ** 2
                                + ((Hs - model).imag / sis) ** 2))

        tau0 = droop.TAU_384[ch]
        taus = np.geomspace(60, 4000, 800)
        Hp = db.h_pole(fs, tau0)
        g = float(np.sum((Hs * np.conj(Hp)).real / srs ** 2)
                  / np.sum((np.abs(Hp) ** 2) / srs ** 2))
        best = min(((chi2(gg * db.h_pole(fs, t)), t, gg) for t, gg in
                    ((t, float(np.sum((Hs * np.conj(db.h_pole(fs, t))).real / srs ** 2)
                               / np.sum((np.abs(db.h_pole(fs, t)) ** 2) / srs ** 2)))
                     for t in taus)))
        print(f"\n  channel {ch}: {int(sel.sum())} harmonics 180 Hz-2.46 kHz, "
              f"n={a['n']} frames")
        print(f"    shipped pole tau={tau0:.0f}, gain 1        chi2/dof "
              f"{chi2(db.h_pole(fs, tau0)) / n2:8.1f}")
        c = [chi2(db.h_pole(fs, t)) for t in taus]
        i = int(np.argmin(c))
        print(f"    pole tau free -> {taus[i]:5.0f}, gain 1     chi2/dof "
              f"{c[i] / (n2 - 1):8.1f}   <- droop_blind's fit; tau railed low")
        print(f"    shipped tau, FREE GAIN {g:.3f}       chi2/dof "
              f"{chi2(g * Hp) / (n2 - 1):8.1f}")
        print(f"    tau {best[1]:5.0f} + free gain {best[2]:.3f}     chi2/dof "
              f"{best[0] / (n2 - 2):8.1f}")
        print("    a free gain -- the one thing the probe cannot measure -- absorbs"
              " most of the misfit.")

    print("\n  MODEL-FREE: the ratio H_L/H_R = (D_L/D_R) x const.  The same converter")
    print("  generated both channels' sync patterns, so the source cancels and no")
    print("  amplitude model is needed.  If the two poles are right, the ratio must")
    print("  follow pole(530)/pole(295) up to one constant.")
    ksel = np.arange(3, 42, 2)
    aL, aR = agg["L"], agg["R"]
    iL = [list(aL["ks"]).index(k) for k in ksel]
    iR = [list(aR["ks"]).index(k) for k in ksel]
    fL = aL["f_hz"][iL]
    ratio = aL["H"][iL] / aR["H"][iR]
    pred = db.h_pole(fL, droop.TAU_384["L"]) / db.h_pole(fL, droop.TAU_384["R"])
    scale = float(np.median(np.abs(ratio) / np.abs(pred)))
    print(f"    scale factor (channel gains, expected ~1): {scale:.3f}")
    print(f"    {'f Hz':>7} {'|H_L/H_R|':>10} {'pole pred':>10} {'ratio':>7} "
          f"{'arg meas':>9} {'arg pred':>9}")
    for j, k in enumerate(ksel):
        if k not in (3, 5, 7, 9, 13, 21, 31, 41):
            continue
        print(f"    {fL[j]:7.0f} {abs(ratio[j]):10.3f} {abs(pred[j]) * scale:10.3f} "
              f"{abs(ratio[j]) / (abs(pred[j]) * scale):7.3f} "
              f"{np.degrees(np.angle(ratio[j])):9.1f} "
              f"{np.degrees(np.angle(pred[j])):9.1f}")
    dev = np.abs(np.abs(ratio) / (np.abs(pred) * scale) - 1.0)
    print(f"    max deviation from the two-pole prediction over 180 Hz-2.5 kHz: "
          f"{dev.max() * 100:.1f}%  (mean {dev.mean() * 100:.1f}%)")
    print("    => the POLES are confirmed without any source model, and whatever the")
    print("       common-mode S is, it is identical on two channels whose poles")
    print("       differ by 1.8x -- so it is not the AC coupling.")
    return agg


# --------------------------------------------------------------------------
# section: the sample-and-hold aperture
# --------------------------------------------------------------------------


def section_aperture() -> None:
    print("\n== 5. THE SCAN CONVERTER'S SAMPLE-AND-HOLD APERTURE ==")
    P = 3195.0
    spd = P / 262.5
    f_dot = FS / spd
    print(f"  dwell spd = P/262.5 = {spd:.3f} samples at 384 kHz -> dot rate "
          f"{f_dot / 1000:.2f} kHz, first null there")
    print(f"  {'f Hz':>8} {'|sinc(f/f_dot)|':>16}")
    for f in (60, 180, 540, 1260, 2464, 3700, 6000, 9700, 15000, 31549):
        print(f"  {f:8.0f} {abs(np.sinc(f / f_dot)):16.4f}")
    lo = np.abs(np.sinc(np.array([180.0, 2464.0]) / f_dot))
    band = np.linspace(3700, 9700, 200)
    mb = float(np.mean(np.abs(np.sinc(band / f_dot))))
    print(f"  shape across the SHELF band 180 Hz-2.5 kHz: {lo[0]:.4f} -> {lo[1]:.4f} "
          f"= {100 * (1 - lo[1] / lo[0]):.1f}% (the shelf claims 12-28%)")
    print(f"  mean over the NORMALISATION band 3.7-9.7 kHz: {mb:.3f}")
    print(f"  => if the aperture were in the probe path, imposing |H|=1 there would")
    print(f"     bias the low-frequency estimate by 1/{mb:.3f} = {1 / mb:+.3f}, i.e. "
          f"{100 * (1 / mb - 1):+.1f}% -- an EXCESS, the opposite sign to the shelf.")
    print("  It is not in the probe path anyway (the sync pattern is the converter's")
    print("  own timing, not a held sample), and in the picture path decode.py")
    print("  integrates each plateau over its own dwell -- the matched filter for a")
    print("  hold, which returns the held value with no aperture loss to invert.")


# --------------------------------------------------------------------------
# section: the ungameable gap-flatness test
# --------------------------------------------------------------------------


def section_gap() -> None:
    print("\n== 6. THE UNGAMEABLE TEST: content-free gap lines must come out flat ==")
    print("Residual is noise-unbiased (cross-covariance of two independent halves");
    print("of the gap traces), so amplifying noise cannot win.  Signal units; the")
    print("frame black-white range is ~0.24.")
    for fid in GAP_FIDS:
        x, m, idx = load_gap(fid)
        print(f"  {fid}: {len(idx)} gap traces", end="")
        if fid == GAP_FIDS[-1]:
            print()
        else:
            print(",", end=" " if GAP_FIDS.index(fid) % 5 else "\n")

    print("\n  (a) tau scan, pole only (1e9 = no correction at all)")
    taus = [200, 295, 400, 530, 700, 900, 1200, 1800, 3000, 1e9]
    print("     frame  " + "".join(f"{t:>9.0f}" for t in taus) + f"{'noise':>9}")
    for fid in GAP_FIDS:
        x, m, idx = load_gap(fid)
        vals = []
        for t in taus:
            pa, pb = half_profiles(droop.inverse_pole(x - x.mean(), t), m, idx)
            vals.append(gap_residual(pa, pb))
        pa, pb = half_profiles(x, m, idx)
        nz = float(np.std((pa - pb)[FLAT_LO:FLAT_HI]) / 2)
        print(f"     {fid} " + "".join(f"{v:9.5f}" for v in vals) + f"{nz:9.5f}"
              + f"   min at tau={taus[int(np.argmin(vals))]:.0f}")
    print("     noise floor is 20-50x below the systematic residual: this is signal.")
    print("     the L channel minimises at the shipped tau=530 -- an independent,")
    print("     input-model-free confirmation of droop.py's L constant.")

    print("\n  (b) shelf depth scan on top of the shipped pole, RAW residual")
    print("      (g < 1 = the hypothesised loss, so its inverse boosts;")
    print("       g > 1 = the opposite)")
    gs = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.40])
    for fc in (2500.0, 400.0):
        print(f"     fc={fc:.0f} Hz")
        print("     frame  " + "".join(f"{g:>9.2f}" for g in gs))
        for fid in GAP_FIDS:
            x, m, idx = load_gap(fid)
            xp = droop.inverse_pole(x - x.mean(), droop.TAU_384[fid[0]])
            vals = []
            for g in gs:
                xs = xp if g == 1.0 else apply_inverse(xp, inv_shelf_W(g, fc))
                pa, pb = half_profiles(xs, m, idx)
                vals.append(gap_residual(pa, pb))
            print(f"     {fid} " + "".join(f"{v:9.5f}" for v in vals))
    print("     monotone, no optimum, on 10 frames of 10: the np-hybrid's direction")
    print("     (g<1) makes the known-flat line worse at every depth and corner.")
    print("     But note WHY it is monotone -- see (b2).")

    print("\n  (b2) the same scan, GAIN-INVARIANT (residual / sync amplitude read")
    print("       off the same corrected profile).  A pure gain now scores zero")
    print("       change, so only SHAPE errors can move this number.")
    for fc in (2500.0, 400.0):
        print(f"     fc={fc:.0f} Hz")
        print("     frame  " + "".join(f"{g:>9.2f}" for g in gs))
        for fid in GAP_FIDS:
            x, m, idx = load_gap(fid)
            xp = droop.inverse_pole(x - x.mean(), droop.TAU_384[fid[0]])
            vals = []
            for g in gs:
                xs = xp if g == 1.0 else apply_inverse(xp, inv_shelf_W(g, fc))
                pa, pb = half_profiles(xs, m, idx)
                vals.append(gap_residual(pa, pb) / gap_amplitude(0.5 * (pa + pb)))
            print(f"     {fid} " + "".join(f"{v:9.5f}" for v in vals))
    print("     FLAT: 0.8 <= g <= 1.4 moves it 7% (fc 2.5 kHz) or 25% (fc 400 Hz),")
    print("     monotonically, with no optimum anywhere -- the objective is merely")
    print("     preferring less low-frequency gain.  The instrument CANNOT SEE the")
    print("     shelf as a defect -- because over the band this probe excites,")
    print("     the shelf IS a gain, which is precisely the quantity nothing in the")
    print("     signal can determine and the level anchors discard.")

    print("\n  (b3) the same gain-invariant objective against tau -- proof that it")
    print("       has power against a real shape error")
    taus2 = [200, 295, 400, 530, 700, 900, 1200, 1800, 3000, 1e9]
    print("     frame  " + "".join(f"{t:>9.0f}" for t in taus2))
    for fid in GAP_FIDS:
        x, m, idx = load_gap(fid)
        vals = []
        for t in taus2:
            pa, pb = half_profiles(droop.inverse_pole(x - x.mean(), t), m, idx)
            vals.append(gap_residual(pa, pb) / gap_amplitude(0.5 * (pa + pb)))
        print(f"     {fid} " + "".join(f"{v:9.5f}" for v in vals)
              + f"   min at tau={taus2[int(np.argmin(vals))]:.0f}")
    print("     interior optimum on every frame, 25-35% below the uncorrected value:")
    print("     the objective resolves a pole and is blind to a shelf, which is the")
    print("     whole point -- one is a shape, the other is a gain.")

    print("\n  (c) what IS left after the pole: shape of the residual")
    print(f"     {'frame':6} {'tilt/trace':>11} {'curvature':>10} {'rms after quad':>15} "
          f"{'rms flat':>9} {'raw tilt':>9}")
    for fid in GAP_FIDS:
        x, m, idx = load_gap(fid)
        tau = droop.TAU_384[fid[0]]
        pa, pb = half_profiles(droop.inverse_pole(x - x.mean(), tau), m, idx)
        t, q, r = gap_shape(0.5 * (pa + pb))
        flat = gap_residual(pa, pb, 0)
        pa0, pb0 = half_profiles(x - x.mean(), m, idx)
        t0, _, _ = gap_shape(0.5 * (pa0 + pb0))
        print(f"     {fid:6} {t:+11.5f} {q:+10.5f} {r:15.5f} {flat:9.5f} {t0:+9.5f}")
    print("     85-90% of what the pole leaves is a straight LINE across the trace,")
    print("     it is already there in the raw signal, and it lives below 120 Hz --")
    print("     nowhere near the 180 Hz-2.5 kHz band the shelf claimed.")


# --------------------------------------------------------------------------
# section: L000's flat field, decomposed
# --------------------------------------------------------------------------


def section_flat(agg=None) -> None:
    print("\n== 7. L000's UNIFORM FIELD: where the prototype's win came from ==")
    print("along-trace profile of the surround, left (traces 15-95) and right")
    print("(430-500), rows 35-330.  'contrast' is the frame's own p1-p99 range:")
    print("std in raw signal units is NOT comparable between arms with different")
    print("gains, and the shelf changes gain -- so std/contrast is the honest column.")
    if agg is None:
        res = {"L": [], "R": []}
        for fid in db.GAP_FIDS:
            r = db.transfer_one(fid)
            if r is not None:
                res[fid[0]].append(r)
        agg = {ch: db.aggregate(res[ch]) for ch in "LR"}
    for ch in "LR":
        agg[ch]["fit"] = db.fit_tau(agg[ch])
    inv = db.build_inverses(agg)
    tau = droop.TAU_384["L"]
    x, m, _ = droop.load("L000", 3.0, 515)

    arms = [
        ("raw", None),
        ("pole (shipped)", pole_rotate_W(tau)),
        ("np-hybrid", inv["L"]["np-hybrid"]),
        ("np MAGNITUDE only", lambda f: np_excess_W(inv, "L", "mag")(f)
         * pole_rotate_W(tau)(f)),
        ("np PHASE only", lambda f: np_excess_W(inv, "L", "phase")(f)
         * pole_rotate_W(tau)(f)),
        ("pole x minphase shelf 0.85", lambda f: inv_shelf_W(0.85, 2500.0)(f)
         * pole_rotate_W(tau)(f)),
        ("pole x const rot 10 deg", pole_rotate_W(tau, 10.0)),
        ("pole x const rot 20 deg", pole_rotate_W(tau, 20.0)),
        ("pole x const rot 30 deg", pole_rotate_W(tau, 30.0)),
        ("pole x const gain 0.66", pole_rotate_W(tau, 0.0, 0.66)),
    ]
    print(f"  {'arm':28} {'left std':>9} {'left p-p':>9} {'right std':>10} "
          f"{'contrast':>9} {'std/contrast':>13}")
    for name, W in arms:
        xs = x if W is None else apply_inverse(x, W)
        img = geometry.decode_marks(xs, m)
        l = np.median(img[35:330, 15:95], axis=1)
        r = np.median(img[35:330, 430:500], axis=1)
        rng = float(np.percentile(img, 99) - np.percentile(img, 1))
        print(f"  {name:28} {l.std():9.4f} {l.ptp():9.4f} {r.std():10.4f} "
              f"{rng:9.4f} {l.std() / rng:13.4f}")
    print("  the shelf's MAGNITUDE makes the field worse by exactly its own gain;")
    print("  the win is a constant phase rotation, and more of it 'wins' more.")
    print("  NOTE the field's residual is a monotone ramp of ~27% of contrast, and")
    print("  a 1977 vidicon's vertical shading prints the same ramp -- L000 cannot")
    print("  arbitrate a low-frequency response on its own.")


# --------------------------------------------------------------------------
# section: trailing shadows
# --------------------------------------------------------------------------


def section_shadows(agg=None) -> None:
    print("\n== 8. TRAILING SHADOWS: does the level behind an edge hold? ==")
    print("median normalised level N dots after a flat-run picture edge.  Scene-")
    print("contaminated (droop.py's warning applies: do NOT fit to it); the useful")
    print("statistic is the RATIO @40/@5, which is 1.0 when the level holds.")
    if agg is None:
        res = {"L": [], "R": []}
        for fid in db.GAP_FIDS:
            r = db.transfer_one(fid)
            if r is not None:
                res[fid[0]].append(r)
        agg = {ch: db.aggregate(res[ch]) for ch in "LR"}
    for ch in "LR":
        agg[ch].setdefault("fit", db.fit_tau(agg[ch]))
    inv = db.build_inverses(agg)
    for fid in ("L055", "R040", "R056"):
        ch = fid[0]
        x, m, _ = droop.load(fid, 3.0, 515)
        tau = droop.TAU_384[ch]
        xp = droop.inverse_pole(x - x.mean(), tau)
        arms = [("raw", x), ("pole", xp),
                ("pole + np excess", apply_inverse(xp, np_excess_W(inv, ch))),
                ("pole + shelf 0.85", apply_inverse(xp, inv_shelf_W(0.85, 2500.0)))]
        for name, xs in arms:
            R, _ = droop._step_ensemble(droop._dot_matrix(xs, m))
            if len(R) < 25:
                print(f"  {fid} {name:18}: only {len(R)} edges, skipped")
                continue
            med = np.median(R, axis=0)
            print(f"  {fid} {name:18} ({len(R):3d} edges) @5 {med[4]:+.3f} "
                  f"@10 {med[9]:+.3f} @20 {med[19]:+.3f} @40 {med[39]:+.3f}  "
                  f"hold @40/@5 {med[39] / med[4]:+.2f}")
    print("  the pole restores the hold (L055 0.05 -> 0.87); the shelf pushes it")
    print("  PAST unity (1.11) -- over-correction, not restoration.")


# --------------------------------------------------------------------------
# section: the frozen test set through the real decoder
# --------------------------------------------------------------------------


def section_images(agg=None, frames=None) -> None:
    import math

    from scipy.signal import resample_poly

    from . import catalog as catalog_mod
    from . import decode as decode_mod
    from . import quality
    from . import sync as sync_mod
    from . import testset
    from . import wav

    print("\n== 9. THE FROZEN TEST SET THROUGH decode.py, ARMS DECOMPOSED ==")
    print("every arm keeps decode.py's OWN pole (channel=) and its undrooped level")
    print("rails; only the extra pre-filter changes.  That is what droop_blind's")
    print("comparison lacked: it scored 'no correction' against 'pole + shelf' and")
    print("charged the difference to the shelf.")
    if agg is None:
        res = {"L": [], "R": []}
        for fid in db.GAP_FIDS:
            r = db.transfer_one(fid)
            if r is not None:
                res[fid[0]].append(r)
        agg = {ch: db.aggregate(res[ch]) for ch in "LR"}
    for ch in "LR":
        agg[ch].setdefault("fit", db.fit_tau(agg[ch]))
    inv = db.build_inverses(agg)

    cat = catalog_mod.build()
    info = wav.probe(testset.MASTER)
    fids = frames or [f[0] for f in testset.TEST_FRAMES]

    def dehum_signal(x):
        """Remove the parity-locked fixed pattern from the RAW signal, inside
        the picture window only -- the sync pulses genuinely alternate and must
        not be touched."""
        m, _ = geometry.find_marks(x)
        M = geometry.norm_traces(x, m)
        ok = np.isfinite(M[:, 0])
        ks = np.arange(len(M))
        e = np.median(M[ok & (ks % 2 == 0)], axis=0)
        o = np.median(M[ok & (ks % 2 == 1)], axis=0)
        prof = 0.5 * (e - o)
        prof[:240] = 0.0
        prof[3030:] = 0.0
        y = np.asarray(x, dtype=np.float64).copy()
        grid = np.arange(BINS)
        for k in range(len(m) - 1):
            a, b = m[k], m[k + 1]
            i0 = int(np.ceil(a))
            n = int(np.floor(b)) - i0
            if n <= 0 or i0 < 0 or i0 + n > len(y):
                continue
            t = (np.arange(i0, i0 + n) - a) / (b - a) * BINS
            y[i0:i0 + n] += (-1.0 if k % 2 == 0 else 1.0) * np.interp(t, grid, prof)
        return y

    def run(fid, mode, uncouple, prehum=False):
        frame = cat.by_id(fid)
        ch = frame.id[0]
        head = testset.PRE_384
        span = int(math.ceil((sync_mod.TRACES_PER_FRAME + 6) * sync_mod.NOMINAL_PERIOD))
        x = np.asarray(wav.read(info, frame.channel, frame.seed_sample - head,
                                head + span), dtype=np.float64)
        if prehum:
            x = dehum_signal(x)
        if mode in ("full", "mag", "phase"):
            x = apply_inverse(x, np_excess_W(inv, ch, mode))
        elif mode == "mps":
            x = apply_inverse(x, inv_shelf_W(0.85, 2500.0))
        x96 = resample_poly(x, 1, testset.DECIM)
        tb = sync_mod.recover(x96, period_guess=testset.NOMINAL_96,
                              n_traces=sync_mod.TRACES_PER_FRAME)
        cfg = decode_mod.Settings(channel=ch if uncouple else "", uncouple=uncouple)
        dec = decode_mod.decode(np.asarray(x96, dtype=np.float32), cfg, tb)
        return dec, quality.frame_report(dec.image, tb, frame_id=fid).metrics

    arms = [("no correction", None, False),
            ("shipped pole", None, True),
            ("pole + np excess", "full", True),
            ("pole + np |E| only", "mag", True),
            ("pole + np phase only", "phase", True),
            ("pole + minphase shelf 0.85", "mps", True)]
    out = {}
    for name, mode, unc in arms:
        rows, circ = {}, None
        for fid in fids:
            try:
                dec, m = run(fid, mode, unc)
            except Exception as exc:
                print(f"  {name} {fid} FAILED: {exc}")
                continue
            rows[fid] = m
            if fid == "L000":
                circ = quality.circle_metrics(dec.image)
        out[name] = rows
        comp = float(np.mean([m["composite"] for m in rows.values()]))
        par = np.array([m["parity_db"] for m in rows.values()])
        cs = (f"{circ['axis_ratio']:.4f}/{circ['radial_rms']:.2f} ok={circ['ok']}"
              if circ else "n/a")
        print(f"  {name:28} composite {comp:5.1f}  parity_db mean {par.mean():5.1f} "
              f"max {par.max():5.1f}  circle {cs}")
    print("\n  the +6.4 dB on L034 (4.5 -> 10.9) belongs to the POLE, which is")
    print("  already shipped: inverting a high-pass boosts 60 Hz by 2.16x and the")
    print("  parity-locked pattern lives at exactly 60 Hz.  The shelf on top of it")
    print("  changes striping by -2.6 dB (np excess) or +0.5 dB (magnitude only).")

    print("\n  DE-HUM BEFORE THE CORRECTION (the fix the brief proposed): subtract")
    print("  the parity-median profile from the RAW signal inside the picture")
    print("  window, then undroop.  Tested, and it is not a fix:")
    print(f"    {'frame':6} {'arm':16} {'hum':>7} {'parity_db':>10} {'composite':>10}")
    for fid in ("L034", "R040", "R056", "L000"):
        for label, mode, unc, pre in (("none", None, False, False),
                                      ("shipped pole", None, True, False),
                                      ("dehum -> pole", None, True, True)):
            try:
                dec, m = run(fid, mode, unc, prehum=pre)
            except Exception as exc:
                print(f"    {fid:6} {label:16} FAILED: {exc}")
                continue
            print(f"    {fid:6} {label:16} {dec.quality.hum_amplitude:7.4f} "
                  f"{m['parity_db']:10.1f} {m['composite']:10.1f}")

    names = [a[0] for a in arms]
    for key, label in (("parity_db", "parity_db (striping, dB)"),
                       ("composite", "composite")):
        print(f"\n  per-frame {label}")
        print(f"    {'frame':6}" + "".join(f"{n[:13]:>15}" for n in names))
        for fid in fids:
            print(f"    {fid:6}" + "".join(
                f"{out[n].get(fid, {}).get(key, float('nan')):15.1f}" for n in names))


# --------------------------------------------------------------------------

SECTIONS = ["band", "probe", "synth", "models", "aperture", "gap", "flat",
            "shadows", "images"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="resolve the low-frequency shelf")
    ap.add_argument("--section", default=",".join(SECTIONS),
                    help="comma list of " + ",".join(SECTIONS))
    ap.add_argument("--frames", default="", help="subset for --section images")
    args = ap.parse_args(argv)
    wanted = [s.strip() for s in args.section.split(",") if s.strip()]
    agg = None
    for name in wanted:
        if name == "band":
            section_band()
        elif name == "probe":
            section_probe()
        elif name == "synth":
            section_synth()
        elif name == "models":
            agg = section_models()
        elif name == "aperture":
            section_aperture()
        elif name == "gap":
            section_gap()
        elif name == "flat":
            section_flat(agg)
        elif name == "shadows":
            section_shadows(agg)
        elif name == "images":
            frames = [s.strip() for s in args.frames.split(",") if s.strip()] or None
            section_images(agg, frames)
        else:
            print(f"unknown section {name}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
