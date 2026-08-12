"""Physical attribution of the decay bias: WHICH mechanism in the real chain
made it, tested against the master. Companion to pipeline/droop.py, which
characterised the distortion (one-pole AC-coupling high-pass per channel,
tau_L ~530 samples / tau_R ~295 at 384 kHz, acting in raster time).

    python -m pipeline.droop_physical
    python -m pipeline.droop_physical --section provenance,shape,invert,porch,dcgain,step

THE CHAIN, AS RESEARCHED (citations in the section prints / final report)
------------------------------------------------------------------------
1977: slide -> vidicon television camera -> Colorado Video scan converter
(one sample per TV line, sample-and-hold; US3950607 / US4802008) -> analog
audio -> RECORDED ON MAGNETIC TAPE (the image-data master reel). The reel was
recorded at 2x the speed written on its box (confirmed by Barry with
Pescovitz: "the master tapes had been recorded at an unusual tape speed ...
they should have been 2x faster").

2016/17 (Ozma Records 40th-anniversary project): the ORIGINAL MASTER TAPE was
played back -- at the labelled speed, i.e. HALF the true speed -- and
digitised. The circulating 1.4 GB float32 stereo WAV is that capture,
presented at 384 kHz so it plays at the true 1977 rate.

THERE IS NO DISC IN THIS SIGNAL'S CHAIN. The images were never cut to lacquer
for this capture: no stylus, no phono preamp, hence NO RIAA stage anywhere.
(The copper record on the spacecraft has the disc chain; our master does
not.) RIAA is nevertheless tested below as a signal hypothesis, because the
task asked and because the refutation is cheap and quantitative.

SPEED MAPPING (matters for every physical time constant)
--------------------------------------------------------
File time = true 1977 signal time. The 2016 transfer ran at half speed, so an
analog time constant T seconds in the TRANSFER chain occupies T/2 seconds of
file time = T * 192000 samples-at-384k. A 1977 RECORD-side constant maps 1:1
= T * 384000 samples. Hence:

    constant        transfer side (2016)   record side (1977)
    NAB LF 3180 us     610.6 samples          1221.1 samples
    RIAA   3180 us     610.6                  1221.1
    RIAA    318 us      61.1                   122.1
    RIAA     75 us      14.4                    28.8
    IEC ph. 7950 us    1526.4                  3052.8

    measured:  tau_L = 530 (syst 460..740)   tau_R = 295 (syst 275..430)

i.e. in transfer-chain wall-clock terms the measured poles are
    tau_L = 2.76 ms (fc 57.7 Hz)    tau_R = 1.54 ms (fc 103.6 Hz)
-- both textbook AC-coupling corners for a tape reproduce chain; the L value
brackets the NAB 3180 us (50 Hz) low-frequency turnover of a 15 in/s
reproducer, the R value does not match any published standard.

WHAT THIS MODULE MEASURES
-------------------------
shape   Fit the content-free gap parity response (droop.py's deterministic
        probe) with the SIMULATED steady-state response of each candidate
        transfer function: free one-pole HP; one-pole fixed at each standard
        time constant; NAB-pole + free-second-pole cascade; unremoved RIAA
        pre-emphasis P(z); wrongly-applied RIAA de-emphasis N(z). All
        candidates get the same free parameters (scale + locked hum + const).
invert  Apply each candidate's CORRECTION to the raw signal and measure how
        flat the gap parity probe becomes (droop.py's validation metric).
porch   Porch-vs-previous-picture regression (Barry's shadow, measured on the
        one signal constant at the source) on raw vs corrected signals.
dcgain  The DC discriminator: the porch's step across the scan stop. Every
        RIAA/NAB *mis-equalisation* candidate has finite, unity-order DC
        gain and predicts NO step of a source-constant level; a DC-blocking
        pole predicts a step equal to the change in local mean signal, and
        the correct inverse must null it.
step    Picture step-edge ensemble (qualitative): decays to ~0 raw, must
        hold ~1 corrected; RIAA correction must fail to fix it.

Sections print their own verdicts; run --section summary for the roll-up.

NEW FINDING (porch section): the porch also couples to scene brightness
ACAUSALLY -- to its own trace's picture, which is scanned AFTER it -- as
strongly as to the previous trace (b_own -1.3..-2.5 everywhere, raw and
corrected). No causal LTI filter can do that; it is a source-side effect
(converter/camera brightness modulation), it survives the pole inverse as it
must, and it means droop.py Q3's single-regressor slope overstated the
pole's porch transfer. The pole's clean fingerprint is the CAUSAL EXCESS
b_past - b_own: -0.4..-1.3 raw, -0.04..-0.25 after the per-channel inverse.

RESEARCH SOURCES
----------------
* Ron Barry, "How to decode the images on the Voyager Golden Record",
  Boing Boing, 2017-09-05: audio obtained from David Pescovitz's master-tape
  access ("David not only had gotten access to the master tapes for his
  anniversary vinyl edition, but happily provided me with a copy of the
  image data audio"); the 2x tape-speed anomaly ("the master tapes had been
  recorded at an unusual tape speed, and ... they should have been 2x
  faster"); the LED square-wave/exponential-decay anecdote; shadow and
  anti-shadow. https://boingboing.net/2017/09/05/how-to-decode-the-images-on-th.html
* aizquier/voyagerimb README (the 384 kHz WAV's circulation provenance:
  Pescovitz team, original master tapes, 32-bit 384 kHz).
  https://github.com/aizquier/voyagerimb
* RIAA replay time constants 3180/318/75 us (50.05/500.5/2122 Hz) and exact
  curve: S. P. Lipshitz, "On RIAA Equalization Networks", JAES 1979;
  https://pearl-hifi.com/06_Lit_Archive/14_Books_Tech_Papers/Lipschitz_Stanley/Lipshitz_on_RIAA_JAES.pdf
  and https://www.stereophile.com/features/cut_and_thrust_riaa_lp_equalization/
* NAB reproduce EQ 3180 us + 50 us at 15/7.5 in/s; IEC 35 us at 15 in/s;
  30 in/s AES/IEC 17.5 us with NO LF time constant: Magnetic Reference
  Laboratory, "Tape Recording Equalization Fundamentals and 15 in/s
  Equalizations", https://mrltapes.com/pages/equalization-fundamentals.html
  and R. L. Hess, "Equalization", richardhess.com.
* Colorado Video scan converter (the 1977 encoder): US3950607 (field-code
  sync), US4802008 (one sample per TV line, sample-and-hold) -- cited from
  pipeline/decode.py and pipeline/dotclock.py where they were established.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import signal as sig

from . import droop, geometry

BINS = geometry.BINS  # 3200
FS = 384000.0

# file-domain samples per second of analog wall clock, by chain side
TRANSFER_SIDE = 192000.0  # 2016 playback at half speed
RECORD_SIDE = 384000.0  # 1977 record side ran at true speed

# standard time constants, seconds
US = 1e-6
NAB_LF = 3180 * US
NAB_HF_15 = 50 * US
RIAA_T1, RIAA_T2, RIAA_T3 = 3180 * US, 318 * US, 75 * US
RIAA_T4 = 3.18 * US  # 'Neumann' HF pole, needed to make P(s) proper
IEC_PHONO_LF = 7950 * US

GAP_FIDS = ["L010", "L020", "L030", "L055", "R010", "R020", "R030", "R056"]
CHAN_TAU = dict(droop.TAU_384)  # {'L': 530, 'R': 295}


# --------------------------------------------------------------------------
# candidate filters (all normalised to unity gain at 2 kHz file-frequency,
# well above every corner under test and below the dot structure)
# --------------------------------------------------------------------------


def _norm_2k(b, a):
    w, h = sig.freqz(b, a, worN=[2 * np.pi * 2000.0 / FS])
    g = abs(h[0])
    return b / g, a


def one_pole_hp(tau: float):
    """H(z) = (1-z^-1)/(1-alpha z^-1), the forward (damaging) filter."""
    al = float(np.exp(-1.0 / tau))
    return np.array([1.0, -1.0]), np.array([1.0, -al])


def one_pole_inv(tau: float):
    al = float(np.exp(-1.0 / tau))
    return np.array([1.0, -al]), np.array([1.0, -1.0])


def _bilinear_tf(zeros_T, poles_T):
    """Analog TF prod(1+s T_z)/prod(1+s T_p) -> digital via bilinear at FS."""
    num = np.poly1d([1.0])
    for T in zeros_T:
        num = num * np.poly1d([T, 1.0])
    den = np.poly1d([1.0])
    for T in poles_T:
        den = den * np.poly1d([T, 1.0])
    b, a = sig.bilinear(num.coeffs, den.coeffs, fs=FS)
    return _norm_2k(b, a)


def riaa_P(side: float):
    """Unremoved RIAA record pre-emphasis, mapped into file time.

    Analog constant T seconds occupies T*side samples of file = T*side/FS
    seconds of file time. P(s) is improper (2 zeros / 1 pole), so the
    standard 'Neumann' 3.18 us HF pole is added to make it realisable; it is
    3+ octaves above every corner under test and irrelevant at LF."""
    return _bilinear_tf([T * side / FS for T in (RIAA_T1, RIAA_T3)],
                        [T * side / FS for T in (RIAA_T2, RIAA_T4)])


def riaa_N(side: float):
    """Wrongly-applied RIAA playback de-emphasis, mapped into file time."""
    return _bilinear_tf([RIAA_T2 * side / FS],
                        [T * side / FS for T in [RIAA_T1, RIAA_T3]])


def apply_ba(x, ba):
    return sig.lfilter(ba[0], ba[1], np.asarray(x, dtype=np.float64))


# --------------------------------------------------------------------------
# shape: fit the gap parity probe with each candidate's simulated response
# --------------------------------------------------------------------------


def _model_profile(ba, n_settle: int = 10) -> np.ndarray:
    """Steady-state even-minus-odd profile of the candidate filter driven by
    the sync pulse-width difference: +rect on even traces, -rect on odd
    (leading edges 3048 wide / 3155 narrow, shared falling edge; droop.py,
    geometry.py). Returns 3200 bins, unit excitation amplitude."""
    T = BINS
    v = np.zeros(2 * T)
    v[3048:3155] = +1.0
    v[T + 3048: T + 3155] = -1.0
    y = apply_ba(np.tile(v, n_settle), ba)
    return y[(2 * n_settle - 2) * T: (2 * n_settle - 1) * T]


def _fit_profile(h_meas: np.ndarray, model: np.ndarray,
                 lo: int = 30, hi: int = 3030) -> tuple[float, float]:
    """LSQ fit scale*model + hum(sin,cos over the 2-trace period) + const to
    the smoothed measured profile over [lo,hi); returns (rms residual,
    fitted scale = implied excitation amplitude, signal units)."""
    hs = droop._smooth(h_meas)
    bb = np.arange(lo, hi, dtype=np.float64)
    th = np.pi * bb / float(BINS)
    seg = hs[lo:hi]
    X = np.column_stack([droop._smooth(model)[lo:hi],
                         np.sin(th), np.cos(th), np.ones(len(bb))])
    c, *_ = np.linalg.lstsq(X, seg, rcond=None)
    return float(np.sqrt(((seg - X @ c) ** 2).mean())), float(c[0])


def _pulse_amplitude(h_meas: np.ndarray) -> float:
    """The excitation amplitude actually recorded: the parity profile's own
    direct pulse (bins 3060..3145), which the chain passes ~unmodified at
    mid/high frequency, minus the mid-trace baseline."""
    hs = droop._smooth(h_meas)
    return float(np.median(hs[3060:3145]) - np.median(hs[1000:3000]))


def _fit_null(h_meas: np.ndarray, lo: int = 30, hi: int = 3030) -> float:
    hs = droop._smooth(h_meas)
    bb = np.arange(lo, hi, dtype=np.float64)
    th = np.pi * bb / float(BINS)
    seg = hs[lo:hi]
    X = np.column_stack([np.sin(th), np.cos(th), np.ones(len(bb))])
    c, *_ = np.linalg.lstsq(X, seg, rcond=None)
    return float(np.sqrt(((seg - X @ c) ** 2).mean()))


def measure_shape() -> None:
    print("SHAPE == candidate transfer functions vs the gap parity response " + "=" * 10)
    print("model rms residual (x1e-4), hum+const always free; 'null' = hum+const only")
    taus = np.geomspace(80, 4000, 120)
    cand_fixed = {
        "NAB3180@xfer(611)": one_pole_hp(NAB_LF * TRANSFER_SIDE),
        "NAB3180@1977(1221)": one_pole_hp(NAB_LF * RECORD_SIDE),
        "RIAA318@xfer(61)": one_pole_hp(RIAA_T2 * TRANSFER_SIDE),
        "RIAA318@1977(122)": one_pole_hp(RIAA_T2 * RECORD_SIDE),
        "riaaP@xfer": riaa_P(TRANSFER_SIDE),
        "riaaP@1977": riaa_P(RECORD_SIDE),
        "riaaN@xfer": riaa_N(TRANSFER_SIDE),
        "riaaN@1977": riaa_N(RECORD_SIDE),
    }
    hdr = f"{'frame':6} {'null':>6} {'free pole':>14} {'cascade 611+':>14} "
    hdr += " ".join(f"{k:>18}" for k in cand_fixed)
    print(hdr)
    prof_fixed = {k: _model_profile(ba) for k, ba in cand_fixed.items()}
    for fid in GAP_FIDS:
        try:
            x, m, _ = droop.load(fid, 195.0, 60)
            lo_s, hi_s = droop.gap_slots(m)
            h = droop.parity_profile(x, m, lo_s, hi_s)
        except Exception as exc:
            print(f"{fid:6} FAILED: {exc}")
            continue
        null = _fit_null(h)
        amp = _pulse_amplitude(h)
        best = (np.inf, 0.0, 0.0)
        for tau in taus:
            r, sc = _fit_profile(h, _model_profile(one_pole_hp(tau)))
            if r < best[0]:
                best = (r, tau, sc)
        bestc = (np.inf, 0.0)
        pre = one_pole_hp(NAB_LF * TRANSFER_SIDE)
        for tau in taus:
            b2, a2 = one_pole_hp(tau)
            ba = (np.convolve(pre[0], b2), np.convolve(pre[1], a2))
            r, _ = _fit_profile(h, _model_profile(ba))
            if r < bestc[0]:
                bestc = (r, tau)
        row = (f"{fid:6} {1e4 * null:6.2f} {1e4 * best[0]:6.2f}@tau{best[1]:5.0f} "
               f"{1e4 * bestc[0]:6.2f}@tau{bestc[1]:5.0f}")
        scales = {}
        for k in cand_fixed:
            r, sc = _fit_profile(h, prof_fixed[k])
            scales[k] = sc
            row += f" {1e4 * r:18.2f}"
        print(row)
        print(f"       implied excitation / measured pulse amp ({amp:+.4f}): "
              f"free pole {best[2] / amp:5.2f}  riaaN@xfer "
              f"{scales['riaaN@xfer'] / amp:7.2f}  riaaN@1977 "
              f"{scales['riaaN@1977'] / amp:7.2f}  riaaP@xfer "
              f"{scales['riaaP@xfer'] / amp:7.2f}")
    print("""  reading. (1) TIME CONSTANT: with scale free, a mis-EQ residual is
  degenerate with a one-pole tail at its own pole (riaaN@xfer ~ pole 611,
  riaaP@xfer ~ pole 61), so the rms column tests the time constant only: the
  RIAA 318us corner (61/122) is far too fast on every frame; 3180us-class
  poles fit L but are 3-8x worse than the free pole on R. (2) DC GAIN, the
  sign line: the implied-excitation ratio must be positive and order 1.
  The free pole gives +1.4..+1.9 (order 1; the residual excess is the
  smoothed-window/hum-degeneracy bias, cf. droop.py Q2's 3-20% boundary
  check). riaaN implies a NEGATIVE excitation (-0.3..-1.1): a de-emphasis
  residual boosts LF and predicts a SAME-sign overhang after the pulse,
  while the master shows an OPPOSITE-sign decay -- the signature of a
  DC-blocking pole, impossible for any unity-DC EQ-curve mismatch.
  (3) The cascade 'NAB 3180us shelf at the half-speed transfer deck + one
  free coupling-cap pole per channel' fits within 0.02-0.31e-4 of the free
  single pole on every frame: shape alone cannot distinguish one physical
  cap from NAB-plus-cap, but either way it is AC coupling in the tape
  electronics, not a disc or EQ-curve error.""")


# --------------------------------------------------------------------------
# invert: does the candidate's correction actually flatten the probe?
# --------------------------------------------------------------------------


def _dehum_rms(h: np.ndarray) -> float:
    hs = droop._smooth(h, 25)
    bb = np.arange(30, 3030, dtype=np.float64)
    th = np.pi * bb / float(BINS)
    X = np.column_stack([np.sin(th), np.cos(th), np.ones(len(bb))])
    seg = hs[30:3030]
    c, *_ = np.linalg.lstsq(X, seg, rcond=None)
    return float(np.sqrt(((seg - X @ c) ** 2).mean()))


def measure_invert() -> None:
    print("\nINVERT == candidate corrections vs gap-probe flatness " + "=" * 20)
    print("dehummed parity rms (x1e-4); lower = flatter; 'pole(ch)' = droop.py taus")
    cands = [
        ("pole(ch)", None),  # per-channel adopted tau, exact inverse
        ("pole611", ("inv", NAB_LF * TRANSFER_SIDE)),
        ("pole1221", ("inv", NAB_LF * RECORD_SIDE)),
        ("riaaN@xfer", ("ba", riaa_N(TRANSFER_SIDE))),
        ("riaaN@1977", ("ba", riaa_N(RECORD_SIDE))),
        ("riaaP@xfer", ("ba", riaa_P(TRANSFER_SIDE))),
    ]
    print(f"{'frame':6} {'raw':>7} " + " ".join(f"{k:>11}" for k, _ in cands))
    for fid in ("L030", "L055", "R020", "R056"):
        x, m, _ = droop.load(fid, 195.0, 60)
        lo_s, hi_s = droop.gap_slots(m)
        row = f"{fid:6} {1e4 * _dehum_rms(droop.parity_profile(x, m, lo_s, hi_s)):7.2f} "
        for name, spec in cands:
            if spec is None:
                y = droop.inverse_pole(x, CHAN_TAU[fid[0]])
            elif spec[0] == "inv":
                y = apply_ba(x, one_pole_inv(spec[1]))
            else:
                y = apply_ba(x, spec[1])
            row += f" {1e4 * _dehum_rms(droop.parity_profile(y, m, lo_s, hi_s)):11.2f}"
        print(row)
    print("  only an inverse whose pole matches the measured tau can flatten the")
    print("  probe; RIAA corrections must leave it essentially unchanged.")


# --------------------------------------------------------------------------
# porch: shadow transfer, raw vs corrected
# --------------------------------------------------------------------------


def _porch_two_sided(y: np.ndarray, m: np.ndarray):
    """Regress the porch on BOTH its neighbours: the previous trace's picture
    mean (before it in raster time -- a causal filter may couple here) and its
    OWN trace's picture mean (after it in time -- no causal filter can).
    Returns (b_past, b_own, causal excess b_past-b_own, se of the excess).

    FINDING (this module, 2026-08): the porch couples to the FUTURE nearly as
    strongly as to the past on every frame tried (b_own -1.3..-2.5, raw and
    corrected). That component is scene-symmetric in time, so it cannot be
    the pole -- or any causal LTI distortion -- and no causal inverse should
    (or does) remove it; it looks like source-side porch modulation by local
    scene brightness (converter/camera side, e.g. supply or thermal sag).
    droop.py's Q3 single-regressor slope conflated the two components; the
    pole's real fingerprint is the CAUSAL EXCESS."""
    n = min(len(m) - 1, 511)
    porch, past, fut, par = [], [], [], []
    for k in range(6, n):
        p = m[k + 1] - m[k]
        pa, pb = int(m[k] + 100 / 3200 * p), int(m[k] + 225 / 3200 * p)
        a0, b0 = int(m[k - 1] + 232 / 3200 * p), int(m[k - 1] + 3040 / 3200 * p)
        a1, b1 = int(m[k] + 232 / 3200 * p), int(m[k] + 3040 / 3200 * p)
        if a0 < 0 or b1 > len(y):
            continue
        porch.append(float(np.median(y[pa:pb])))
        past.append(float(y[a0:b0].mean()))
        fut.append(float(y[a1:b1].mean()))
        par.append(k % 2)
    porch, past, fut, par = map(np.asarray, (porch, past, fut, par))
    for pp in (0, 1):
        porch[par == pp] -= np.median(porch[par == pp])
    past -= past.mean()
    fut -= fut.mean()
    X = np.column_stack([past, fut])
    c, *_ = np.linalg.lstsq(X, porch, rcond=None)
    res = porch - X @ c
    cov = float((res ** 2).sum() / (len(porch) - 2)) * np.linalg.pinv(X.T @ X)
    se_ex = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    return float(c[0]), float(c[1]), float(c[0] - c[1]), se_ex


def measure_porch() -> None:
    print("\nPORCH == shadow transfer, split into causal and acausal parts " + "=" * 12)
    print("porch regressed on prev-trace mean (b_past) AND own-trace picture mean")
    print("(b_own, later in raster time). Causal excess = b_past - b_own is the")
    print("pole's fingerprint; the common part is acausal and NOT the pole.")
    for fid in droop.FRAME_FIDS:
        x, m, _ = droop.load(fid, 3.0, 515)
        for name, y in (
            ("raw", np.asarray(x, dtype=np.float64)),
            ("pole-corr", droop.inverse_pole(x, CHAN_TAU[fid[0]])),
            ("riaaN@xfer", apply_ba(x, riaa_N(TRANSFER_SIDE))),
        ):
            bp, bo, ex, se = _porch_two_sided(y, m)
            print(f"  {fid} {name:10}: b_past {bp:+6.2f}  b_own {bo:+6.2f}  "
                  f"causal excess {ex:+6.2f} +- {se:.2f}")
    print("""  the pole inverse must (and does) take the causal excess to ~0 while the
  acausal common component survives untouched, as a source-side effect must.
  riaaN's apparent excess shrinkage comes with b_past GROWING -- it inflates
  every low-frequency variance rather than removing the causal coupling.
  CORRECTION TO droop.py Q3: its single-regressor slope (-0.9..-4.9) mixed
  this acausal scene-symmetric component with the pole; only the causal
  excess (-0.4..-1.3 raw) belongs to the recording chain.""")


# --------------------------------------------------------------------------
# dcgain: the DC discriminator at the scan stop
# --------------------------------------------------------------------------


def measure_dcgain() -> None:
    print("\nDCGAIN == porch step across the scan stop " + "=" * 32)
    print("""sync and porch keep running in the gap, so to the extent the SOURCE porch
is constant through the scan stop, a unity-DC-gain candidate (any EQ-curve
mismatch: RIAA/NAB shelving -- all pass DC at order-1 gain) predicts NO porch
step, while a DC-blocking pole predicts step = -(change in the tau-leaky mean
of the signal just before the porch). Both the step and that leaky-mean
change are measured below. CAVEATS, printed so nobody over-reads this: the
converter's state genuinely changes at the stop (a true source step cannot be
excluded), and on the corrected signal the integrator's free DC drifts; this
section is corroboration only -- the load-bearing DC discriminators are the
parity-tail SIGN (shape) and the causal excess (porch).""")
    for fid in ("L055", "R040"):
        x, m, _ = droop.load(fid, 5.0, 620)
        k0 = int(np.argmin(np.abs(m - 5.0 * geometry.NOMINAL_PERIOD)))
        tau = CHAN_TAU[fid[0]]
        for name, y in (
            ("raw", np.asarray(x, dtype=np.float64)),
            ("pole-corr", droop.inverse_pole(x, tau)),
            ("riaaN@xfer", apply_ba(x, riaa_N(TRANSFER_SIDE))),
        ):
            porch, loc = {}, {}
            for k in range(k0 + 495, min(k0 + 605, len(m) - 1)):
                p = m[k + 1] - m[k]
                pa, pb = int(m[k] + 100 / 3200 * p), int(m[k] + 225 / 3200 * p)
                a = int(m[k]) - int(2.0 * tau)
                if a < 0 or pb > len(y):
                    continue
                porch[k - k0] = float(np.median(y[pa:pb]))
                w = np.exp((np.arange(a, int(m[k])) - m[k]) / tau)
                loc[k - k0] = float((y[a:int(m[k])] * w).sum() / w.sum())
            pre = np.median([v for t, v in porch.items() if 500 <= t <= 530 and t % 2 == 1])
            post = np.median([v for t, v in porch.items() if 545 <= t <= 600 and t % 2 == 1])
            lpre = np.median([v for t, v in loc.items() if 500 <= t <= 530])
            lpost = np.median([v for t, v in loc.items() if 545 <= t <= 600])
            extra = ""
            if name == "raw":
                extra = (f"  DC-block predicts {-(lpost - lpre):+.4f} "
                         f"(= -leaky-mean change); unity-DC predicts +0.0000")
            print(f"  {fid} {name:10}: porch step {post - pre:+.4f}{extra}")
    print("  raw steps match the DC-block prediction in SIGN and order (0.5-1x),")
    print("  not the unity-DC zero; the residual step on the corrected signal is")
    print("  within the source-step/drift caveat above.")


# --------------------------------------------------------------------------
# step: picture step-edge ensemble, raw vs corrected (qualitative)
# --------------------------------------------------------------------------


def measure_step() -> None:
    print("\nSTEP == picture step-edge ensemble (qualitative; scene-contaminated) " + "=" * 5)
    for fid in ("L055", "R056"):
        x, m, _ = droop.load(fid, 3.0, 515)
        for name, y in (
            ("raw", np.asarray(x, dtype=np.float64)),
            ("pole-corr", droop.inverse_pole(x, CHAN_TAU[fid[0]])),
            ("riaaN@xfer", apply_ba(x, riaa_N(TRANSFER_SIDE))),
        ):
            R, _ = droop._step_ensemble(droop._dot_matrix(y, m))
            if len(R) < 25:
                print(f"  {fid} {name}: only {len(R)} edges")
                continue
            med = np.median(R, axis=0)
            print(f"  {fid} {name:10} ({len(R):3d} edges): @2 {med[1]:+.2f} @5 {med[4]:+.2f} "
                  f"@10 {med[9]:+.2f} @20 {med[19]:+.2f} @40 {med[39]:+.2f}")
    print("  raw must decay toward 0; the physical correction must hold ~1. riaaN")
    print("  yields no usable edges at all: its 20 dB LF boost swamps the flat-run")
    print("  detector -- failure by another route.")


# --------------------------------------------------------------------------


def print_provenance() -> None:
    print(__doc__.split("WHAT THIS MODULE MEASURES")[0])


def print_summary() -> None:
    tl, tr = CHAN_TAU["L"], CHAN_TAU["R"]
    print("\nSUMMARY == physical attribution of the decay bias " + "=" * 24)
    print(f"""  the measured pole in physical units:
    file domain:  tau_L {tl:.0f} samples (fc {FS / (2 * np.pi * tl):.0f} Hz),"""
          f""" tau_R {tr:.0f} (fc {FS / (2 * np.pi * tr):.0f} Hz)
    transfer-deck wall clock (half speed): tau_L {tl / TRANSFER_SIDE * 1e3:.2f} ms"""
          f""" (fc {TRANSFER_SIDE / (2 * np.pi * tl):.1f} Hz), tau_R {tr / TRANSFER_SIDE * 1e3:.2f} ms (fc {TRANSFER_SIDE / (2 * np.pi * tr):.1f} Hz)
    1977-deck wall clock (true speed):     tau_L {tl / RECORD_SIDE * 1e3:.2f} ms"""
          f""" (fc {RECORD_SIDE / (2 * np.pi * tl):.1f} Hz), tau_R {tr / RECORD_SIDE * 1e3:.2f} ms (fc {RECORD_SIDE / (2 * np.pi * tr):.1f} Hz)

  MECHANISM VERDICTS (each tested above on the master):
  * RIAA disc EQ ............ EXCLUDED. By provenance (the capture came from
    the original master TAPE via the Ozma project -- no disc, stylus, or
    phono preamp in this chain) AND by signal: RIAA residuals are unity-DC
    shelves; the 318us corner is 3-8x too fast for the parity tail, the
    de-emphasis residual needs a NEGATIVE excitation (wrong tail sign), its
    'correction' worsens gap flatness 4-8x, fails every criterion.
  * Tape/electronics AC coupling ... CONFIRMED, the decay bias. One
    DC-blocking pole per channel in the tape record/reproduce electronics.
    Which side is not decidable from shape: a single free pole and the
    'NAB 3180us LF turnover at the half-speed transfer deck + one coupling
    cap' cascade fit equally well. The 1.8x L/R asymmetry rules out any
    matched published curve as the sole cause: standards are implemented
    with matched precision parts, coupling caps are -20/+80% electrolytics.
  * NAB/IEC tape EQ mismatch (HF 50us/35us/17.5us corners) ... NOT the
    decay bias: those corners live at 3-9 kHz analog and shelve with
    unity-order gain; they cannot produce a DC-blocking 60-100 Hz pole.
    (Any HF mis-EQ would blur/ring dots -- a separate, bounded question.)
  * Converter sample-and-hold droop ... EXCLUDED as a separate mechanism:
    droop.py Q5 bounded any excess within-plateau decay at <4e-4/sample
    (<0.5%/dot); the within-plateau slope IS the chain pole.
  * Vidicon camera lag/shading ... NOT the decay bias (the decay lives in
    the camera-free gap and moves the converter-synthesised porch). BUT
    this module found a genuine source-side artefact: the porch couples to
    scene brightness ACAUSALLY (b_own -1.3..-2.5 on its own trace's later
    content), scene-symmetric in time -- converter/camera-side brightness
    modulation. That is part of the SOURCE image, on the far side of the
    restoration line: the pole inverse neither can nor should remove it,
    and we do not attempt to.

  CORRECTION PATH (unchanged from droop.py; decode.py untouched here):
    per-channel exact inverse y[n]=y[n-1]+x[n]-a x[n-1] on the raw signal in
    raster order, tau_L 530 / tau_R 295, porch clamp pins the free DC.""")


SECTIONS = {
    "provenance": print_provenance,
    "shape": measure_shape,
    "invert": measure_invert,
    "porch": measure_porch,
    "dcgain": measure_dcgain,
    "step": measure_step,
    "summary": print_summary,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="physical attribution of the decay bias")
    ap.add_argument("--section", default="",
                    help=f"comma list of {','.join(SECTIONS)} (default: all)")
    args = ap.parse_args(argv)
    wanted = [s.strip() for s in args.section.split(",") if s.strip()] or list(SECTIONS)
    for name in wanted:
        SECTIONS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
