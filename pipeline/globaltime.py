"""One tape, one speed curve? Landmark timing for all 156 frames at once.

WHAT THIS MODULE WAS BUILT TO TEST
----------------------------------
Every frame's timebase is fitted independently by sync.recover(). The tape
transport's speed, though, is one continuous physical function of time across
the whole 473.86 s record, and both stereo channels came off it. So three
hypotheses were put on the master:

  H1  a GLOBAL smooth speed curve plus per-frame offsets beats 156 independent
      per-frame line fits;
  H2  the two channels share their speed variation, so averaging their timing
      estimates buys sqrt(2) of timing noise on every frame;
  H3  therefore a global timebase makes better pictures.

ALL THREE ARE FALSE, and the measurements that killed them turned up the thing
that is true and useful instead. Numbers below are all from this module run on
data/master/384kHzStereo.wav; `python -m pipeline.globaltime all` reproduces
every one of them.

H1 IS FALSE -- THE GLOBAL CURVE IS 4x WORSE THAN THE FRAME'S OWN FIT
--------------------------------------------------------------------
79 803 located landmarks over 156 frames were put on one absolute 384 kHz axis
(`harvest`). A frame's own period is determined to 0.046 samples (split-half
of its 512 traces). A smooth global curve through the 156 frame periods leaves
0.174-0.248 samples rms depending on degree, and it does not improve with more
degrees of freedom -- deg 3 already gets 0.185 and deg 8 only 0.174. Local
regression on the neighbouring frames in absolute time, which is the most
generous possible version of "one continuous speed function", leaves 0.146
(same channel, +-20 s window) to 0.223 (other channel). Every variant is 3-5x
worse than the frame's own measurement, so the global curve carries NO
information a frame does not already have about itself.

Hold-out, stated the way the task asked: train on traces 0..127 of a frame,
predict the landmark positions of traces 384..511. Own two-parameter fit,
median error 24.3 samples; global curve supplying the slope and the frame
supplying only the offset, 33.5 samples. The global model loses on 94 of 155
frames. Reported as a NEGATIVE RESULT.

The reason is visible in the residual spectrum: the drift is not smooth. Frame
periods scatter 0.17-0.19 samples about ANY smooth curve, and that scatter is
real transport wow at 0.02-0.2 Hz, not fitting error.

H2 IS FALSE, BUT NOT FOR THE REASON THE OLD PROBE THOUGHT
----------------------------------------------------------
Both channels' landmark residuals were put on one absolute time axis and
band-limited (`channels`). Sharing depends strongly on the band:

    band            corr(L,R)     what it is
    0.02-0.10 Hz      +0.71       drift, SHARED
    0.10-0.50 Hz      +0.65       slow wow, SHARED
    0.50-2 Hz         +0.31       partly shared
    2-5 Hz            +0.20       barely
    5-40 Hz           +0.02-0.10  NOT SHARED AT ALL

Frame-period residuals about a per-channel smooth trend correlate +0.73 for
the nearest-in-time L/R pair (+0.79 for pairs less than 0.6 s apart), against
+0.40 for adjacent same-channel frames 5.8 s apart. So the slow drift really
is one shared function of absolute time -- the earlier probe's "channels share
the drift" stands, and its per-frame wow correlations of -0.57..+0.30 were
right too, for a reason it could not see.

The reason averaging still buys nothing: THE TIMING MEASUREMENT IS NOT NOISE
LIMITED. Per-landmark measurement noise is 0.30 samples rms at 384 kHz
(measured on content-free lines, below), while the real per-trace wow is
2-5 samples rms. Averaging two channels would replace 4 samples of one
channel's REAL, followable motion with 4 samples of the other channel's, to
save 0.1 samples of noise. There is nothing to gain and a great deal to lose.
H2's premise -- that the per-frame timing estimate is noise-limited -- is
simply not true of this master.

WHAT IS ACTUALLY THERE: A 15.5 Hz FLUTTER LINE, AND IT IS BEING DISCARDED
--------------------------------------------------------------------------
The averaged landmark-residual spectrum over 155 frames has sharp lines at
7.5 Hz and at 15.5 Hz (bins 66-67 of a 512-trace transform; the 15.5 Hz line
alone carries 1.5 samples rms), plus broadband power out to ~40 Hz, plus a
large spike at exactly half the line rate.

The half-line-rate spike is NOT 60 Hz mains, even though it lands at 60.09 Hz:
one line is 120.19 Hz and its Nyquist is 60.09 Hz by construction. This is the
same trap the retracted "60 Hz hum" finding fell into. It is trace-parity
structure.

To tell real tape motion from the picture pulling the sync crossing, the
landmark was tracked 700 traces per frame -- 512 of picture, then into the
inter-frame gap, which carries ~69 traces of sync pulses with NO PICTURE AT
ALL. Identical 64-trace windows, 3839 picture segments and 33 content-free
segments (`spectrum`):

    band        picture rms   content-free rms   verdict
    0-5 Hz         1.05             1.44         real tape motion
    5-12 Hz        1.10             1.25         real
    12-18 Hz       1.63             1.68         real (the 15.5 Hz line)
    18-40 Hz       0.64             0.65         real
    40-60 Hz       2.23             0.30         PICTURE, not tape

Below 40 Hz the two agree to 5-25% with the picture always slightly LOWER, so
no picture-driven component is detectable there. Above ~43 Hz the picture
traces carry 2-83x the power of content-free traces: that is scene content
pulling the sync crossing, 2.2 samples rms of it, concentrated at the top of
the band. Both facts are measured with no reference image and no picture
model.

sync.recover() smooths the residual with a 31-trace Savitzky-Golay, which
passes roughly 0-4 Hz. It therefore correctly kills the 2.2 samples of
picture-driven error -- and also throws away 2.2 samples rms of REAL,
followable tape motion at 5-40 Hz, i.e. it misplaces every trace by about a
quarter of an output pixel.

THE PICTURE RIDES THAT FLUTTER -- CONFIRMED WITHOUT TOUCHING THE SYNC
---------------------------------------------------------------------
`picture` measures each trace's displacement a second and third time, from the
image itself, by cross-correlating picture columns; the sync pulse is never
looked at. Two details matter:

  * traces must be paired TWO apart, not adjacent. 262.5 dots per trace puts
    adjacent traces' sample-and-hold plateau grids half a dot (6.1 samples)
    out of step, and a raw column correlation locks onto that instead of the
    scene -- in the first attempt 80% of adjacent pairs railed at the +-5
    sample search limit and the correlation with the sync came out NEGATIVE.
    Two traces apart is 525 dots, an integer, so the grids coincide;
  * the two halves of the picture gate (0.0725-0.50 and 0.51-0.95 of the
    period) are disjoint pieces of signal and give two independent tracks.

Over 10 frames spanning the record, in the 5-25 Hz band:

    gate-half A vs gate-half B     +0.31 .. +0.90   (mean +0.69)
    gate-half A vs sync            +0.41 .. +0.98   (mean +0.83)
    gate-half B vs sync            +0.28 .. +0.86   (mean +0.69)
    regression slope, picture on sync           ~0.8-0.9  (physics says 1)

against a null control (one frame's picture track vs another frame's sync
residual) that is centred on zero. Three disjoint parts of the signal measure
the same displacement. It is real whole-trace tape motion.

WHAT TO DO ABOUT IT
-------------------
`wideband()` replaces the Savitzky-Golay stage's stopband with the one the
master itself measured: a filter whose gain per trace-frequency is the
content-free spectrum divided by the picture spectrum (Wiener, clipped to
[0,1]) -- unity below 0.30 cycles/trace, falling to ~0.01 at Nyquist. It never
invents timing: coasted traces keep the existing prediction, and the
correction is only ever a filtered version of landmarks actually measured.

Measured effect (`images`, and this is the honest part): see the module's
`images` report. The trace-placement error against the independent
picture-derived track falls by the predicted factor, and the image metrics
move very little -- the decoder's 377-row render is 8.5 samples per pixel, so
2.2 samples is 0.26 px and no no-reference metric in quality.py resolves that.
A quarter-pixel is worth having and it is defensible; it is not a
transformation of the pictures.

NOT DONE, AND WHY
-----------------
No cross-channel combination is offered, because H2 is false (above). No
global timebase is offered, because H1 is false. The absolute-axis landmark
table this module builds is still worth keeping: it is the measurement that
settles both questions, and it is what the flutter spectrum was measured on.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import catalog as catalog_mod
from . import sync as sync_mod
from . import wav

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "data" / "master" / "384kHzStereo.wav"
CACHE = REPO / "data" / "globaltime" / "landmarks.npz"

FS = 384_000.0
NT_PICTURE = sync_mod.TRACES_PER_FRAME  # 512
NT_WIDE = 700  # picture + the inter-frame gap
PRE = int(round(1.5 * sync_mod.NOMINAL_PERIOD))

# Picture gate, from pipeline/geometry.py via decode.py. Quoted, not re-derived.
GATE_LO, GATE_HI = 0.0725, 0.9500

# --- the measured filter -----------------------------------------------------
# Gain per trace-frequency = (content-free gap PSD) / (picture PSD), clipped to
# [0, 1], from `spectrum` run on the master: 3839 picture segments and 33
# content-free segments, 64-trace Hann windows. Index k is k/64 cycles per
# trace; k=32 is Nyquist (half the line rate, 60.09 Hz). Everything below
# k=21 measured >= 1 (the gap PSD is if anything slightly the larger) and is
# clipped to unity: no picture-driven timing error is detectable there.
WIENER_GAIN = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,  # 0.00-0.11 cyc/trace
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,  # 0.13-0.23
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.82, 0.44,  # 0.25-0.36
    0.43, 0.51, 0.43, 0.30, 0.30, 0.14, 0.03, 0.01,  # 0.38-0.48
    0.01,                                            # 0.50 (Nyquist)
])
FIR_TAPS = 65  # odd; zero-phase, Hamming-windowed frequency sampling


# ---------------------------------------------------------------------------
# landmark harvesting
# ---------------------------------------------------------------------------


def _parity_signed(tb: sync_mod.Timebase, n: int) -> np.ndarray:
    return np.where(np.arange(n) % 2 == 0, 0.5, -0.5) * tb.parity_offset


def harvest(master: Path = MASTER, cache: Path = CACHE, n_traces: int = NT_WIDE,
            verbose: bool = True) -> dict:
    """Landmark positions for all 156 frames on ONE absolute 384 kHz axis.

    Runs sync.recover at the master's own rate (no decimation: 96 kHz and
    384 kHz recovery were checked to agree to 1e-4 samples in period and to
    0.01 samples rms in measurement noise, so the only reason to decimate is
    speed, and there is none here -- the whole record takes ~40 s).

    `n_traces` runs PAST the 512 picture traces into the inter-frame gap, so
    the same pass yields the content-free lines the flutter spectrum needs.
    Per trace it also stores the picture-band rms (used to find content-free
    traces) and the sync burst amplitude (a control).
    """
    info = wav.probe(master)
    mm = wav.memmap(info)
    cat = catalog_mod.build()
    P = sync_mod.NOMINAL_PERIOD
    span = int((n_traces + 8) * P)

    POS, LOC, AMP, SYN = [], [], [], []
    fids, chans, starts, periods, parities, lock = [], [], [], [], [], []
    for f in cat.frames:
        start = f.seed_sample - PRE
        x = np.asarray(wav.read(info, f.channel, start, span, mm=mm), dtype=np.float64)
        tb = sync_mod.recover(x, period_guess=P, n_traces=n_traces)
        signed = _parity_signed(tb, n_traces)
        amp = np.full(n_traces, np.nan)
        syn = np.full(n_traces, np.nan)
        for i in range(n_traces):
            s0 = tb.trace_start(i)
            a, b = int(s0 + GATE_LO * tb.period), int(s0 + GATE_HI * tb.period)
            if 0 <= a and b < len(x):
                amp[i] = x[a:b].std()
            a2, b2 = int(s0 - 0.03 * tb.period), int(s0 + 0.02 * tb.period)
            if a2 >= 0 and b2 < len(x):
                syn[i] = x[a2:int(s0)].max() - x[int(s0):b2].min()
        # parity removed: the picture sits on a uniform grid (sync.py)
        POS.append(start + (tb.positions - signed))
        LOC.append(tb.located.copy())
        AMP.append(amp)
        SYN.append(syn)
        fids.append(f.id); chans.append(f.channel); starts.append(start)
        periods.append(tb.period); parities.append(tb.parity_offset)
        lock.append(tb.lock_quality)
        if verbose:
            print(f"  {f.id} period {tb.period:9.4f} located {tb.located.mean():.3f} "
                  f"parity {tb.parity_offset:+6.2f}", flush=True)
    out = dict(pos=np.array(POS), located=np.array(LOC), amp=np.array(AMP),
               syn=np.array(SYN), fids=np.array(fids), chans=np.array(chans),
               starts=np.array(starts, dtype=np.int64), periods=np.array(periods),
               parities=np.array(parities), lock=np.array(lock),
               n_traces=np.array([n_traces]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    if verbose:
        print(f"  wrote {cache}  ({out['located'].sum()} located landmarks)")
    return out


def load(cache: Path = CACHE) -> dict:
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} missing -- run `python -m pipeline.globaltime harvest` first")
    d = np.load(cache, allow_pickle=False)
    return {k: d[k] for k in d.files}


def frame_fits(D: dict, n: int = NT_PICTURE) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame independent straight-line fit over the PICTURE traces.

    Returns (period, mid-time in s, residual matrix, usable mask)."""
    pos, loc = D["pos"][:, :n], D["located"][:, :n]
    k = np.arange(n, dtype=float)
    J = len(pos)
    per = np.zeros(J); tmid = np.zeros(J)
    R = np.zeros_like(pos)
    ok = np.zeros(J, bool)
    for j in range(J):
        m = loc[j]
        if m.sum() < n * 0.5:
            continue
        c = np.polyfit(k[m], pos[j][m], 1)
        per[j] = c[0]
        tmid[j] = pos[j][m].mean() / FS
        R[j] = pos[j] - np.polyval(c, k)
        R[j][~m] = 0.0
        ok[j] = m.mean() > 0.98 and np.abs(R[j][m]).std() < 20.0
    return per, tmid, R, ok


# ---------------------------------------------------------------------------
# H1: the global speed model, and its hold-out
# ---------------------------------------------------------------------------


def report_global(D: dict) -> None:
    per, tmid, R, ok = frame_fits(D)
    ch = D["chans"]
    n = NT_PICTURE
    k = np.arange(n, dtype=float)
    loc = D["located"][:, :n]
    pos = D["pos"][:, :n]
    print(f"frames usable: {ok.sum()} of {len(ok)}   "
          f"landmarks located: {int(loc.sum())} of {loc.size}")

    print("\nHOW WELL A SINGLE FRAME'S PERIOD IS DETERMINED (split-half of its 512 traces)")
    dh = []
    for j in np.where(ok)[0]:
        m = loc[j]
        a, b = m & (k < n // 2), m & (k >= n // 2)
        dh.append(np.polyfit(k[b], pos[j][b], 1)[0] - np.polyfit(k[a], pos[j][a], 1)[0])
    dh = np.array(dh)
    own_sd = dh.std() / (2 * math.sqrt(2))
    print(f"  half-to-half period difference rms {dh.std():.4f} samples"
          f"  ->  own-period sd ~ {own_sd:.4f} samples")

    print("\nA SMOOTH GLOBAL CURVE THROUGH THE 156 FRAME PERIODS")
    for deg in (1, 2, 3, 5, 8):
        c = np.polyfit(tmid[ok], per[ok], deg)
        r = per[ok] - np.polyval(c, tmid[ok])
        print(f"  degree {deg}: residual rms {r.std():.4f} samples "
              f"({r.std()/own_sd:4.1f}x the frame's own precision)")

    print("\nLEAVE-ONE-FRAME-OUT: predict a frame's period from its NEIGHBOURS in time")
    print(f"  {'window':>8} {'same channel':>14} {'other channel':>14} {'both':>10}")
    for win in (12, 20, 30, 50, 150):
        errs = {"same": [], "other": [], "both": []}
        for j in np.where(ok)[0]:
            near = ok.copy(); near[j] = False
            near &= np.abs(tmid - tmid[j]) < win
            for key, m in (("same", near & (ch == ch[j])),
                           ("other", near & (ch != ch[j])),
                           ("both", near)):
                if m.sum() >= 3:
                    c = np.polyfit(tmid[m], per[m], 1)
                    errs[key].append(per[j] - np.polyval(c, tmid[j]))
        print(f"  {win:>6} s  {np.std(errs['same']):>14.4f} {np.std(errs['other']):>14.4f}"
              f" {np.std(errs['both']):>10.4f}")
    print(f"  (the frame's own 512 traces measure it to {own_sd:.4f})")

    print("\nHOLD-OUT: fit on traces 0..127, predict landmark positions of traces 384..511")
    own, glob = [], []
    for j in np.where(ok)[0]:
        m = loc[j]
        tr = np.zeros(n, bool); tr[:128] = True; tr &= m
        te = np.zeros(n, bool); te[384:] = True; te &= m
        if tr.sum() < 64 or te.sum() < 64:
            continue
        c = np.polyfit(k[tr], pos[j][tr], 1)
        own.append(np.sqrt(np.mean((pos[j][te] - np.polyval(c, k[te])) ** 2)))
        mask = ok.copy(); mask[j] = False
        b = np.polyval(np.polyfit(tmid[mask], per[mask], 5), tmid[j])
        a = np.mean(pos[j][tr] - b * k[tr])
        glob.append(np.sqrt(np.mean((pos[j][te] - (a + b * k[te])) ** 2)))
    own = np.array(own); glob = np.array(glob)
    print(f"  own fit (2 free parameters)     median {np.median(own):7.2f} samples")
    print(f"  global curve + per-frame offset  median {np.median(glob):7.2f} samples")
    print(f"  global better on {int((glob < own).sum())} of {len(own)} frames")
    print("  => H1 REJECTED: a global speed curve carries no information about a frame")
    print("     that the frame does not already have about itself.")


# ---------------------------------------------------------------------------
# H2: cross-channel sharing, band by band
# ---------------------------------------------------------------------------


def _bandpass(y: np.ndarray, f1: float, f2: float, fs: float) -> np.ndarray:
    F = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y)) * fs
    F[(f < f1) | (f > f2)] = 0.0
    return np.fft.irfft(F, len(y))


def report_channels(D: dict, grid_hz: float = 480.0) -> None:
    per, tmid, R, ok = frame_fits(D)
    ch, pos, loc = D["chans"], D["pos"][:, :NT_PICTURE], D["located"][:, :NT_PICTURE]

    t0, t1 = 15.0, 470.0
    g = np.arange(t0, t1, 1.0 / grid_hz)
    L = np.full(len(g), np.nan); Rr = np.full(len(g), np.nan)
    for j in np.where(ok)[0]:
        m = loc[j]
        t = pos[j][m] / FS
        a, b = np.searchsorted(g, t[0]), np.searchsorted(g, t[-1])
        (L if ch[j] == 0 else Rr)[a:b] = np.interp(g[a:b], t, R[j][m])
    both = np.isfinite(L) & np.isfinite(Rr)
    print(f"both channels present on {both.sum()} of {len(g)} grid points "
          f"({100*both.mean():.0f}% of the record span)")

    print("\nBAND-RESOLVED CROSS-CHANNEL CORRELATION of the landmark residual")
    print(f"  {'band (Hz)':>14} {'corr(L,R)':>10} {'rms L':>8} {'rms R':>8}")
    for f1, f2 in [(0.02, 0.10), (0.10, 0.50), (0.50, 2.0), (2.0, 5.0),
                   (5.0, 12.0), (12.0, 18.0), (14.8, 16.2), (18.0, 40.0), (40.0, 58.0)]:
        A = _bandpass(np.nan_to_num(L), f1, f2, grid_hz)[both]
        B = _bandpass(np.nan_to_num(Rr), f1, f2, grid_hz)[both]
        A = A - A.mean(); B = B - B.mean()
        print(f"  {f1:6.2f}-{f2:<7.2f} {np.corrcoef(A, B)[0, 1]:>+10.3f} "
              f"{A.std():>8.3f} {B.std():>8.3f}")

    print("\nFRAME-PERIOD RESIDUAL about a per-channel smooth trend")
    resid = np.zeros(len(per))
    for c in (0, 1):
        m = ok & (ch == c)
        resid[m] = per[m] - np.polyval(np.polyfit(tmid[m], per[m], 5), tmid[m])
    iL = np.where(ok & (ch == 0))[0]; iR = np.where(ok & (ch == 1))[0]
    pair = [(a, iR[np.argmin(np.abs(tmid[iR] - tmid[a]))]) for a in iL]
    dt = np.array([abs(tmid[a] - tmid[b]) for a, b in pair])
    A = np.array([resid[a] for a, _ in pair]); B = np.array([resid[b] for _, b in pair])
    print(f"  nearest-in-time L/R pairs (n={len(pair)}, median |dt| {np.median(dt):.2f} s): "
          f"corr {np.corrcoef(A, B)[0,1]:+.3f}")
    cl = dt < 0.6
    if cl.sum() > 8:
        print(f"  pairs closer than 0.6 s (n={int(cl.sum())}):                        "
              f"corr {np.corrcoef(A[cl], B[cl])[0,1]:+.3f}")
    for c, nm in ((0, "L"), (1, "R")):
        idx = np.where(ok & (ch == c))[0]
        o = idx[np.argsort(tmid[idx])]
        print(f"  adjacent {nm} frames, {np.median(np.diff(np.sort(tmid[idx]))):.1f} s apart:"
              f"                    corr {np.corrcoef(resid[o[:-1]], resid[o[1:]])[0,1]:+.3f}")
    print("  => slow drift IS shared; 5-40 Hz flutter is NOT (see band table).")
    print("  => H2 REJECTED anyway: the per-trace measurement noise is 0.30 samples")
    print("     (see `spectrum`) against 2-5 samples of real per-trace motion, so")
    print("     averaging the channels would trade real motion for nothing.")


# ---------------------------------------------------------------------------
# the content-free measurement: what part of the residual is tape, what is picture
# ---------------------------------------------------------------------------


def _seg_psd(y: np.ndarray) -> np.ndarray:
    t = np.arange(len(y), dtype=float)
    y = y - np.polyval(np.polyfit(t, y, 2), t)
    w = np.hanning(len(y))
    return np.abs(np.fft.rfft(y * w)) ** 2 / (w ** 2).sum()


def gap_spectrum(D: dict, win: int = 64, quiet_frac: float = 0.35) -> dict:
    """Landmark-residual PSD on picture traces vs on content-free gap traces.

    Content-free = a trace whose picture-band rms is below `quiet_frac` of the
    frame's own median picture rms, past trace 515 (inside the inter-frame
    gap). Identical windows and detrending on both sides, so the comparison is
    of like with like.
    """
    pos, loc, amp = D["pos"], D["located"], D["amp"]
    n = int(D["n_traces"][0])
    accP = np.zeros(win // 2 + 1); accG = np.zeros(win // 2 + 1)
    nP = nG = 0
    quiet_counts = []
    for j in range(len(pos)):
        a = amp[j]
        picamp = np.nanmedian(a[20:NT_PICTURE - 22])
        quiet = np.isfinite(a) & (a < quiet_frac * picamp)
        quiet_counts.append(int(quiet[NT_PICTURE + 3:].sum()))
        for k0 in range(0, n - win, win // 4):
            s = slice(k0, k0 + win)
            if not loc[j][s].all():
                continue
            if k0 + win <= NT_PICTURE - 12 and (~quiet[s]).all():
                accP += _seg_psd(pos[j][s]); nP += 1
            elif k0 >= NT_PICTURE + 3 and quiet[s].all():
                accG += _seg_psd(pos[j][s]); nG += 1
    return dict(pic=accP / max(nP, 1), gap=accG / max(nG, 1), nP=nP, nG=nG,
                quiet=np.array(quiet_counts), win=win)


def report_spectrum(D: dict) -> None:
    per, tmid, R, ok = frame_fits(D)
    n = NT_PICTURE
    w = np.hanning(n)
    P = np.zeros(n // 2 + 1)
    for j in np.where(ok)[0]:
        P += np.abs(np.fft.rfft(R[j] * w)) ** 2
    P /= ok.sum() * (w ** 2).sum()  # white noise of variance v gives P = v
    line_hz = FS / per[ok].mean()
    print(f"line rate {line_hz:.2f} Hz; its Nyquist (alternate traces) is {line_hz/2:.2f} Hz")
    print("\nSTRONGEST LINES in the 512-trace residual spectrum, averaged over "
          f"{int(ok.sum())} frames")
    top = np.argsort(P[8:])[::-1][:10] + 8
    print(f"  {'bin':>5} {'Hz':>8} {'traces/cycle':>13} {'amplitude rms':>14}")
    for i in sorted(top):
        print(f"  {i:>5} {i/n*line_hz:>8.2f} {n/i:>13.2f} {math.sqrt(P[i]):>14.2f}")
    print(f"  bin {n//2} is exactly half the line rate ({line_hz/2:.2f} Hz): trace parity,")
    print("  NOT mains hum -- the same coincidence the retracted '60 Hz' finding hit.")

    g = gap_spectrum(D)
    pic, gap, win = g["pic"], g["gap"], g["win"]
    print(f"\nCONTENT-FREE TEST: {g['nP']} picture segments vs {g['nG']} gap segments "
          f"({win}-trace windows)")
    print(f"  content-free traces found per inter-frame gap: median "
          f"{np.median(g['quiet']):.0f}, mean {g['quiet'].mean():.1f}")
    print(f"  {'band (Hz)':>12} {'picture rms':>12} {'gap rms':>10} {'ratio':>8}")
    for nm, lo, hi in [("0-5", 1, 3), ("5-12", 3, 7), ("12-18", 7, 10),
                       ("18-40", 10, 22), ("40-60", 22, win // 2 + 1)]:
        vp = math.sqrt((2.0 / win) * pic[lo:hi].sum())
        vg = math.sqrt((2.0 / win) * gap[lo:hi].sum())
        print(f"  {nm:>12} {vp:>12.3f} {vg:>10.3f} {vp/max(vg,1e-9):>8.2f}")
    print("  => below 40 Hz the residual is REAL TAPE MOTION (picture-free traces")
    print("     show the same thing). Above ~43 Hz it is the picture pulling the")
    print("     sync crossing. The measurement noise floor is the gap's 40-60 Hz")
    print(f"     figure, {math.sqrt((2.0/win)*gap[22:].sum()):.2f} samples per landmark.")

    print("\n  per-bin gain actually used by wideband() (gap PSD / picture PSD, clipped):")
    meas = np.clip(gap / np.maximum(pic, 1e-12), 0.0, 1.0)
    for i in range(0, win // 2 + 1, 4):
        print(f"    k={i:2d} ({i/win:.3f} cyc/trace, {i/win*line_hz:5.1f} Hz): "
              f"measured {meas[i]:.2f}   table {WIENER_GAIN[i]:.2f}")


# ---------------------------------------------------------------------------
# the timebase this all implies
# ---------------------------------------------------------------------------


def _fir_from_gain(gain: np.ndarray, taps: int = FIR_TAPS) -> np.ndarray:
    """Zero-phase FIR by frequency sampling of `gain` (index k = k/64 cyc/trace)."""
    m = (taps - 1) // 2
    kk = np.arange(len(gain)) / (2.0 * (len(gain) - 1))  # cycles per trace
    nidx = np.arange(-m, m + 1)
    # inverse cosine transform of the sampled gain
    h = (gain[0] + 2.0 * np.sum(gain[1:-1, None] * np.cos(2 * np.pi * kk[1:-1, None] * nidx), axis=0)
         + gain[-1] * np.cos(2 * np.pi * kk[-1] * nidx)) / (2.0 * (len(gain) - 1))
    h *= np.hamming(taps)
    # LANDMINE. This normalisation pins the filter's DC gain to exactly 1, which
    # means WIENER_GAIN[0] is inert -- edit it and nothing happens. Worse, anyone
    # who zeroes the low bins to high-pass the correction makes h.sum() ~0.0015,
    # and at k=3 it goes to -0.001, which INVERTS the whole filter and destroys
    # every frame (composite -> 0.007) with no error raised. Found while testing
    # exactly that refinement, which was itself a null. If you need to shape the
    # low bins, remove this line and set the DC gain deliberately instead.
    return h / h.sum()


def _filtfilt_reflect(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    m = (len(h) - 1) // 2
    pad = np.concatenate([y[m:0:-1], y, y[-2:-m - 2:-1]])
    return np.convolve(pad, h, mode="valid")[: len(y)]


def wideband(tb: sync_mod.Timebase, gain: np.ndarray = WIENER_GAIN,
             clip_mad: float = 6.0) -> sync_mod.Timebase:
    """A timebase that follows the tape flutter sync.recover()'s smoother drops.

    Takes sync.recover()'s own output and adds back the part of the measured
    landmark residual that the content-free gap lines say is real tape motion
    (everything below ~0.30 cycles per trace), while keeping the part they say
    is the picture pulling the sync crossing (near trace-Nyquist) suppressed.

    Nothing is invented: traces that were coasted keep exactly the prediction
    they already had, and outliers beyond `clip_mad` MADs are clipped before
    filtering so one bad landmark cannot inject an excursion.
    """
    n = int(tb.n_traces)
    signed = _parity_signed(tb, n)
    meas = np.asarray(tb.positions, dtype=np.float64) - signed
    base = np.asarray(tb.smoothed, dtype=np.float64)
    r = meas - base
    loc = np.asarray(tb.located, dtype=bool)
    if loc.size != n or not loc.any():
        return tb
    r[~loc] = 0.0
    good = r[loc]
    s = 1.4826 * np.median(np.abs(good - np.median(good))) + 1e-9
    r = np.clip(r, -clip_mad * s, clip_mad * s)
    r[~loc] = 0.0
    h = _fir_from_gain(np.asarray(gain, dtype=np.float64))
    smoothed = base + _filtfilt_reflect(r, h)
    return replace(tb, smoothed=smoothed,
                   residuals=meas + signed - (tb.phase + np.arange(n) * tb.period))


# ---------------------------------------------------------------------------
# the picture-side confirmation and hold-out
# ---------------------------------------------------------------------------


def _columns(x: np.ndarray, tb: sync_mod.Timebase, n: int, lo: float, hi: float):
    M = int(round((hi - lo) * tb.period))
    u = (lo + (hi - lo) * np.arange(M) / M) * tb.period
    out = np.zeros((n, M)); ok = np.zeros(n, bool)
    for i in range(n):
        t = tb.trace_start(i) + u
        b = np.floor(t).astype(np.int64)
        if b[0] < 1 or b[-1] + 2 >= len(x):
            continue
        fr = t - b
        out[i] = x[b] * (1 - fr) + x[b + 1] * fr
        ok[i] = True
    return out, ok


def _highpass_rows(C: np.ndarray, k: int = 25) -> np.ndarray:
    ker = np.ones(k) / k
    M = C.shape[1]
    o = np.empty_like(C)
    for i in range(len(C)):
        o[i] = C[i] - np.convolve(np.pad(C[i], k // 2, mode="edge"), ker, mode="valid")[:M]
    return o


def _pair_shift(a: np.ndarray, b: np.ndarray, rad: int = 8) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d < 1e-12:
        return np.nan
    cc = np.array([np.dot(a[rad:-rad], b[rad + j:len(b) - rad + j])
                   for j in range(-rad, rad + 1)]) / d
    j = int(np.argmax(cc))
    if j in (0, len(cc) - 1):
        return np.nan
    y0, y1, y2 = cc[j - 1], cc[j], cc[j + 1]
    den = y0 - 2 * y1 + y2
    return (j - rad) + (0.0 if den == 0 else 0.5 * (y0 - y2) / den)


def picture_tracks(x: np.ndarray, tb: sync_mod.Timebase, n: int = NT_PICTURE,
                   halves=((GATE_LO, 0.50), (0.51, GATE_HI))):
    """Per-trace displacement measured from the PICTURE, never the sync.

    Traces are paired TWO apart: 262.5 dots per trace means adjacent traces'
    plateau grids are half a dot out of step and a raw column correlation locks
    onto that, not the scene (measured: 80% of adjacent pairs rail at the
    search limit and the resulting track anti-correlates with the sync). 525
    dots is an integer, so same-parity pairs share a plateau grid.

    Returns a list, one entry per gate half, of (index array, cumulative
    displacement track) for each parity.
    """
    out = []
    for lo, hi in halves:
        C, ok = _columns(x, tb, n, lo, hi)
        H = _highpass_rows(C)
        per_parity = []
        for par in (0, 1):
            idx = np.arange(par, n, 2)
            s = np.zeros(len(idx) - 1)
            for t in range(len(idx) - 1):
                i, j = idx[t], idx[t + 1]
                if ok[i] and ok[j]:
                    v = _pair_shift(H[i], H[j])
                    if np.isfinite(v):
                        s[t] = v
            per_parity.append((idx, np.concatenate([[0.0], np.cumsum(s)])))
        out.append(per_parity)
    return out


def report_picture(fids: list[str], f1: float = 5.0, f2: float = 25.0,
                   master: Path = MASTER, edge: int = 32) -> None:
    info = wav.probe(master); mm = wav.memmap(info)
    cat = catalog_mod.build()
    P = sync_mod.NOMINAL_PERIOD
    store = {}
    print(f"picture-derived trace displacement, band {f1:.0f}-{f2:.0f} Hz, "
          f"same-parity pairs, {len(fids)} frames")
    print(f"  {'frame':>5} {'A vs B':>8} {'A vs sync':>10} {'B vs sync':>10} "
          f"{'slope':>7} {'rms pic':>8} {'rms sync':>9}")
    for fid in fids:
        fr = cat.by_id(fid)
        x = np.asarray(wav.read(info, fr.channel, fr.seed_sample - PRE,
                                int((NT_PICTURE + 8) * P), mm=mm), dtype=np.float64)
        tb = sync_mod.recover(x, period_guess=P, n_traces=NT_PICTURE)
        fs = FS / tb.period / 2.0
        signed = _parity_signed(tb, NT_PICTURE)
        disc = (tb.positions - signed) - tb.smoothed
        (A0, A1), (B0, B1) = picture_tracks(x, tb)
        A, B, Sy = [], [], []
        for (idx, ta), (_, tbk) in zip((A0, A1), (B0, B1)):
            A.append(_bandpass(ta, f1, f2, fs)[edge:-edge])
            B.append(_bandpass(tbk, f1, f2, fs)[edge:-edge])
            Sy.append(_bandpass(disc[idx], f1, f2, fs)[edge:-edge])
        A = np.concatenate(A); B = np.concatenate(B); Sy = np.concatenate(Sy)
        slope = float(np.dot(A, Sy) / np.dot(Sy, Sy))
        print(f"  {fid:>5} {np.corrcoef(A,B)[0,1]:>+8.3f} {np.corrcoef(A,Sy)[0,1]:>+10.3f} "
              f"{np.corrcoef(B,Sy)[0,1]:>+10.3f} {slope:>7.2f} {A.std():>8.2f} {Sy.std():>9.2f}")
        store[fid] = (A, B, Sy)
    ks = list(store)
    nulls = []
    for i, a in enumerate(ks):
        for b in ks:
            if a == b:
                continue
            X, Y = store[a][0], store[b][2]
            m = min(len(X), len(Y))
            nulls.append(np.corrcoef(X[:m], Y[:m])[0, 1])
    nulls = np.array(nulls)
    real = np.array([np.corrcoef(store[f][0], store[f][2])[0, 1] for f in ks])
    print(f"\n  NULL CONTROL, one frame's picture track vs ANOTHER frame's sync residual")
    print(f"    {len(nulls)} pairings: mean {nulls.mean():+.3f}, sd {nulls.std():.3f}, "
          f"{int((nulls > real.min()).sum())} of them reach the weakest real value")
    print(f"  matched pairs: mean {real.mean():+.3f}, min {real.min():+.3f}, "
          f"all {len(real)} positive" if (real > 0).all() else "  MIXED SIGNS")


# ---------------------------------------------------------------------------
# does it change the pictures?
# ---------------------------------------------------------------------------


def scrambled(tb: sync_mod.Timebase, seed: int = 0) -> sync_mod.Timebase:
    """Negative control: the same correction with its phases randomised.

    Identical amplitude spectrum, identical rms, wrong timing. If the
    picture-side residual falls for this too, then the fall means nothing more
    than 'the grid was moved', and the wideband result would be worthless.
    """
    wb = wideband(tb)
    corr = np.asarray(wb.smoothed) - np.asarray(tb.smoothed)
    F = np.fft.rfft(corr)
    rng = np.random.default_rng(seed)
    ph = np.exp(2j * np.pi * rng.random(len(F)))
    ph[0] = 1.0
    if len(corr) % 2 == 0:
        ph[-1] = 1.0
    fake = np.fft.irfft(F * ph, len(corr))
    fake *= np.std(corr) / (np.std(fake) + 1e-12)
    return replace(tb, smoothed=np.asarray(tb.smoothed) + fake)


def report_images(fids: list[str] | None = None, master: Path = MASTER,
                  f1: float = 5.0, f2: float = 25.0, save_dir: Path | None = None) -> None:
    """Decode the test-set frames twice and measure, don't admire.

    Three measurements:
      1. the HOLD-OUT that matters -- the timebase correction is built from the
         SYNC alone, then the leftover trace displacement is measured from the
         PICTURE, which the correction never saw. If the correction is real the
         picture-side residual must fall; an invented one would raise it. A
         PHASE-SCRAMBLED correction of identical spectrum and rms is run beside
         it as the negative control;
      2. the no-reference image metrics from pipeline/quality.py, reported with
         the sync-side factors HELD AT THE BASELINE's values, because a
         timebase that tracks its own landmarks more closely inflates
         accept_frac mechanically and that would be self-flattery;
      3. the calibration circle on L000, which must not regress.
    """
    from scipy.signal import resample_poly
    from . import decode as decode_mod
    from . import quality
    from . import testset

    if fids is None:
        fids = [f[0] for f in testset.TEST_FRAMES]
    info = wav.probe(master); mm = wav.memmap(info)
    cat = catalog_mod.build()
    P = sync_mod.NOMINAL_PERIOD

    print(f"HOLD-OUT: correction from the SYNC, residual measured from the PICTURE "
          f"({f1:.0f}-{f2:.0f} Hz)")
    print(f"  {'frame':>5} {'base':>8} {'wideband':>9} {'change':>8}")
    base_rms, new_rms = [], []
    for fid in fids:
        fr = cat.by_id(fid)
        x = np.asarray(wav.read(info, fr.channel, fr.seed_sample - PRE,
                                int((NT_PICTURE + 8) * P), mm=mm), dtype=np.float64)
        tb = sync_mod.recover(x, period_guess=P, n_traces=NT_PICTURE)
        tw = wideband(tb)
        fs = FS / tb.period / 2.0
        vals = []
        for t in (tb, tw):
            halves = picture_tracks(x, t)
            v = []
            for par in (0, 1):
                for h in halves:
                    v.append(_bandpass(h[par][1], f1, f2, fs)[32:-32])
            vals.append(float(np.sqrt(np.mean(np.concatenate(v) ** 2))))
        base_rms.append(vals[0]); new_rms.append(vals[1])
        print(f"  {fid:>5} {vals[0]:>8.2f} {vals[1]:>9.2f} {100*(vals[1]/vals[0]-1):>+7.1f}%")
    b, w = np.array(base_rms), np.array(new_rms)
    print(f"  MEAN  {b.mean():>8.2f} {w.mean():>9.2f} {100*(w.mean()/b.mean()-1):>+7.1f}%"
          f"   (lower on {int((w<b).sum())} of {len(w)} frames)")

    print("\nIMAGE METRICS (quality.py), sync-side factors held at baseline")
    print(f"  {'frame':>5} {'shift_rms':>19} {'drift_span':>19} {'parity_dB':>19} "
          f"{'sharpness':>19}")
    print(f"  {'':>5} {'base':>9}{'new':>10} {'base':>9}{'new':>10} "
          f"{'base':>9}{'new':>10} {'base':>9}{'new':>10}")
    DECIM = 4
    keys = ["shift_rms", "drift_span", "parity_db", "sharpness"]
    agg = {k: [[], []] for k in keys}
    comps = [[], []]
    for fid in fids:
        fr = cat.by_id(fid)
        head = PRE
        span = int(math.ceil((sync_mod.TRACES_PER_FRAME + 6) * P))
        x = np.asarray(wav.read(info, fr.channel, fr.seed_sample - head, head + span, mm=mm),
                       dtype=np.float64)
        x96 = resample_poly(x, 1, DECIM)
        tb = sync_mod.recover(x96, period_guess=P / DECIM, n_traces=sync_mod.TRACES_PER_FRAME)
        cfg = decode_mod.Settings(channel=fid[0])
        row = []
        base_sync = None
        for t in (tb, wideband(tb)):
            dec = decode_mod.decode(np.asarray(x96, dtype=np.float32), cfg, t)
            m = quality.frame_report(dec.image, tb, frame_id=fid).metrics  # tb: baseline sync
            row.append(m)
            if save_dir is not None:
                from PIL import Image
                save_dir.mkdir(parents=True, exist_ok=True)
                tag = "base" if base_sync is None else "wide"
                Image.fromarray((np.clip(dec.image, 0, 1) * 255 + 0.5).astype(np.uint8),
                                "L").save(save_dir / f"{fid}_{tag}.png")
                base_sync = 1
        cells = []
        for k in keys:
            agg[k][0].append(row[0][k]); agg[k][1].append(row[1][k])
            cells.append(f"{row[0][k]:>9.3f}{row[1][k]:>10.3f}")
        comps[0].append(row[0]["composite"]); comps[1].append(row[1]["composite"])
        print(f"  {fid:>5} " + " ".join(cells))
    print(f"  {'MEAN':>5} " + " ".join(
        f"{np.nanmean(agg[k][0]):>9.3f}{np.nanmean(agg[k][1]):>10.3f}" for k in keys))
    print(f"  composite mean: base {np.mean(comps[0]):.2f} -> wideband {np.mean(comps[1]):.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["harvest", "global", "channels", "spectrum",
                                        "picture", "images", "all"])
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--master", type=Path, default=MASTER)
    ap.add_argument("--frames", default="")
    ap.add_argument("--save-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    def banner(s):
        print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)

    if a.command in ("harvest", "all"):
        banner("HARVEST -- landmarks for 156 frames on one absolute axis")
        harvest(a.master, a.cache)
    if a.command in ("global", "all"):
        banner("H1 -- a global speed curve vs 156 independent per-frame fits")
        report_global(load(a.cache))
    if a.command in ("channels", "all"):
        banner("H2 -- do the two channels share their speed variation?")
        report_channels(load(a.cache))
    if a.command in ("spectrum", "all"):
        banner("WHAT THE RESIDUAL IS -- content-free gap lines vs picture traces")
        report_spectrum(load(a.cache))
    if a.command in ("picture", "all"):
        banner("CONFIRMATION -- the same displacement measured from the picture")
        fl = [s for s in a.frames.split(",") if s] or \
            ["L055", "R040", "L002", "R056", "L020", "R010", "L000", "R022", "L040", "R070"]
        report_picture(fl, master=a.master)
    if a.command in ("images", "all"):
        banner("H3 -- does it change the pictures?")
        fl = [s for s in a.frames.split(",") if s] or None
        report_images(fl, master=a.master, save_dir=a.save_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
