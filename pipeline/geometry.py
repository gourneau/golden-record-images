"""Frame geometry of the Voyager image signal, measured from the 384 kHz master.

Every number here was measured by the functions below running against
data/master/384kHzStereo.wav; `python -m pipeline.geometry --all` reruns the
whole suite. Offsets are stated as fractions of the LOCAL line period (the
period drifts 3197.4 -> 3193.5 samples across the record, so fractions are the
only stable currency), on a 3200-bin grid: one bin = period/3200 = 0.999
samples at the nominal period.

THE LANDMARK (what every offset is relative to)
-----------------------------------------------
The DOWNWARD ZERO CROSSING OF THE SYNC TRAILING EDGE. The sync burst ends in a
fall from the plateau (~+0.15) through zero into an undershoot; the zero
crossing of that fall is the most repeatable feature of the trace:
consecutive-spacing std 5.6-5.9 samples on every frame tried (7 frames
spanning the record, photographs included), against ~100 for the burst peak
Barry used. `find_marks` locks 99.6-100% of traces per frame via
steepest-derivative-fall candidates -> longest-chain DP vote -> per-trace
zero-crossing refinement; the chain vote is what survives the calibration
circle's own edges, which defeat a global phase histogram.

The crossing alternates +-5.5 samples with trace parity (the two sync widths
of the Colorado Video 262; 11 samples ~ one dot, see pipeline/dotclock.py).
MEASURED: THE PICTURE MOVES WITH IT. Decoding each trace relative to its own
crossing leaves residual odd/even column shifts of only -0.08..+0.10 px
(L055/R040/R056), while imposing a +-5.5-sample parity correction CREATES
+-1.45 px. With this landmark there is NO parity fudge: Barry's "+-12 on even
traces" compensated his landmark, not the picture.

TRACE LAYOUT (bins of period/3200 after the crossing)
-----------------------------------------------------
      0          zero crossing; undershoot minimum ~+4
     ~15..230    back porch: content-free (across-trace variance at floor)
      232        PICTURE START (uncertainty ~[225, 240]). Within-parity
                 across-trace std half-rises at p10 225 / median 260 over 24
                 profiles; frames with featureless top rows rise late, so the
                 gate is the distribution's EARLY edge. A full-resolution
                 render of bins 100-700 on R056/R040 shows image structure
                 beginning at ~235-255 with featureless wash above.
      3040       PICTURE END: the front-porch dip sits at 3044-3063 in every
                 parity-median profile (p10 3044); content is gone by 3040.
      3048+-4    long-sync parity: burst leading edge (n=27, p10=p90=3048).
                 The video gate cannot extend past this. Any "picture end"
                 measured beyond 3048 is measuring blanking structure.
      3071+-1    short-sync parity: blanking shelf rise (sd 0.9 bins); the
                 short burst itself rises at ~3155.
      3200       next zero crossing.
The picture is ONE CONTIGUOUS RUN of 2808 bins = 87.75% of the period, and IT
DOES NOT WRAP THE LANDMARK. The old "2873 samples wrapping the lock point" was
an artefact of anchoring on the notch inside the picture. Barry's 2680-sample
window ([~225..2905] in these coordinates) starts 8 bins early in the porch
and stops 135 bins short: he cut ~4% off the picture bottom, which is why his
circle needed a hand aspect correction.

In dot units (pipeline/dotclock.py: the trace is 262.5 sample-and-hold dots of
~12.17 samples -- confirmed here, spd 12.16-12.18 on 8 frames): the gate is
dots 19.1 .. 249.8, i.e. ~231 active dots of 262.5.

ASPECT / ISOTROPY (from the calibration circle, which must be round)
--------------------------------------------------------------------
Fitted directly in signal coordinates (trace index, bin) -- no decode
resampling -- with ring points taken by BOTH scan directions (along-trace and
across-trace, 626/1195 points, radial rms 0.7%, the two fits agreeing to
0.1%):

    ISOTROPY = 7.440 +- 0.007 (stat) +- ~0.05 (syst) bins of trace-time
               per trace spacing        [= 0.6113 dots/trace]

A decode-then-fit span scan brackets the same value (round between spans
2840-2880 on 384 rows -> 7.40-7.50). One earlier tall-render fit gave 7.33;
recorded as a known-biased negative result (uneven angular point selection).

VERIFIED AGAINST THE RAW SIGNAL (second, independent pass): the ring's two
crossings per trace, read straight off porch-clamped, row-static-subtracted
single traces (find_marks + norm_traces, no fit in the loop), give vertical
chords of 1999 / 2249 / 1993 bins at traces 200 / 269 / 340 -- the fitted
ellipse predicts 2008 / 2256 / 1994, agreement 0.3%, centre bin 1577-1585 vs
fitted 1589. A dot-grid ellipse fit in the same pass initially gave 7.31
(centre bin 1512); the raw chords refute it -- same angular-selection bias as
the 7.33 above. 7.44 stands.

MULTI-FRAME ROUNDNESS CHECK (`--roundness`): render at the recommended
512 x 377 square-pixel geometry, track each circular feature's edge to
sub-pixel per column AND per row, fit a pure circle, and read the cos(2th)
residual term, which converts directly to an x/y stretch factor (round = 1;
the refuted "4:3 = 512 traces" isotropy of 7.31 would read 1.018 on every
row). Measured stretch:
    L000 ring        1.0006  (rms 1.05 px, 360 deg)
    L053 outer ring  0.9976  (rms 0.58 px -- cleanest line-art circle)
    L053 middle ring 0.9890  (rms 1.05 px)
    L013/14/15 Earth 0.9990 / 0.9966 / 0.9839  (limb, rms ~4 px)
    L011 Mars limb   0.9809  (partial 200 deg arc, weakest)
Seven circles across five frames and both channel-independent slide scans sit
at 0.981-1.001: round at the ~1% level, uniformly below the 1.018 the
512-trace-4:3 hypothesis demands. ISOTROPY 7.44 (+-1%, i.e. 4:3 width
503 +- 5 traces) is confirmed by every frame with circular content.

Consequences, with the gate height V = 2808 bins:
  * square-pixel render of the full gate: 512 traces x 377 rows (2808/7.440);
  * the gate is 377.4 trace-units tall, so an EXACTLY-4:3 frame is
    503 +- 4 traces wide, not 512: rendering 512 traces at true isotropy
    gives aspect 1.357, ~1.8% wider than 4:3;
  * a 512-trace-wide 4:3 frame would need 2857 bins of picture height, which
    overlaps the burst leading edge at 3048 -- physically impossible. The
    cover's "512" is the scan's trace count, not the width of the 4:3 area.
Ring: centre trace 269 of the decode window (267 from frame start), bin 1589;
radii 151.6 traces x 1127.8 bins. The ring is NOT exactly centred in the
frame (+15 traces, -48 bins from gate centre), so centring is deliberately
not used to pin the gate -- roundness is invariant, centring is an assumption.

TRACE COUNT / FRAME EXTENT
--------------------------
Between frames the converter keeps running: sync continues (find_marks locks
through the gap) and the gap carries a stationary gray texture whose
trace-to-trace correlation is 0.77-0.99 -- which is why a naive
"neighbour-correlation = content" detector overruns; use activity
stationarity instead. Each frame is announced by a MARKER: 21-22 traces of
high-contrast diagonal hatch ending exactly at Barry's seed (measured
[seed-22, seed-1] on 22 frames). Content then runs ~515-525 traces (activity
stationarity: median ~515, sd ~30 where measurable; visual inspection of
sharp frames: 520-535; it varies because the photographed originals vary).
So: 512 traces from the seed is a sound default; Barry's 540 = the marker
plus ~518 of content; the exactly-4:3 crop is the first ~503 traces.

INTENSITY
---------
The signal is AC-coupled; the only per-trace DC reference present on both
parities is the BACK PORCH, bins [100, 225]. After clamping there:
  * sync amplitude amp = short-burst plateau [3165..3195] - porch = 0.144 ..
    0.181 on this master, drifting smoothly along the record;
  * L000 is binary by design (white ring on black) and yields both references:
        WHITE = porch - 0.745 * amp      BLACK = porch + 0.741 * amp
    (symmetric about the porch to 0.5%);
  * across 16 frames spread over the record, picture p1/p99 span
    -0.17..-0.74 / +0.49..+0.78 of amp -- content-dependent, so fixed
    references at +-0.75 * amp clip essentially nothing;
  * more signal = darker (the picture is a negative), as established;
  * gamma: the record carries no gray-step target, so transfer curvature
    CANNOT be measured from the signal. Use linear (gamma 1.0); any cosine or
    gamma shaping is taste, not restoration.

RECOMMENDED DECODE CONSTANTS (fractions of the local period, relative to the
zero-crossing landmark)
----------------------------------------------------------------------------
    PICTURE_START = 232/3200  = 0.0725
    PICTURE_END   = 3040/3200 = 0.9500        (span 0.8775)
    PORCH         = [0.03125, 0.0703125]      (bins 100..225)
    ISOTROPY      = 7.440/3200 = 0.002325 periods per trace
    render: 512 traces x 377 rows (square pixels), no parity offset,
    black/white at porch +- 0.75 * amp, linear.

RECONCILIATION WITH THE 2026-08 "dot frame" geometry draft
----------------------------------------------------------
A concurrent draft of this module claimed: picture start ~355, ~219.5 active
dots (2672 samples), D/T ratio ~0.415, and a ~535-trace 4:3 frame. The draft's
own second pass retracted most of that before the two lines of work merged:
its "picture start ~355" was one frame's own content edge (L055's featureless
sky pushes the variance rise late; L020/L053/R010/R040 put content at
~250-290), and its per-frame offsets carried a hidden per-frame landmark
error, because sync.recover()'s fold-derived origin is NOT frame-invariant:
on L000 it sits 220 samples from the zero-crossing landmark (the frame's
trace-invariant dark bands survive the fold and hijack find_blanking), while
on L055 the two coincide to 0.2 samples. That is the strongest argument for
this module's independent landmark. The draft's dot-grid ring fit (D/T 0.600
dots/trace = 7.31 bins/trace) was refuted by its own raw-chord check -- see
VERIFIED AGAINST THE RAW SIGNAL above. What DOES survive from that line of
work, and is adopted here: the dot clock itself (dotclock.py, 12.17
samples/dot, confirmed on 8 more frames, spd 12.16-12.18), the sync-pulse
anatomy (short pulse whose width alternates ~145/~45 samples = ~12/~4 dots
with parity, falling edge shared -- matching the 3048/3155 burst rises
above), and scanning stopping at ~534-536 traces, measured two independent
ways: the AC-coupled blanking-shelf offset relaxes to the gap baseline at
trace ~534 on all 12 frames tried, and L000's sync pulses vanish outright
after trace 534 (its gap carries no per-period pulses; other frames' gaps
do, so pulse presence alone cannot delimit every frame). ~535 is where the
CONVERTER stops, not the width of the 4:3 area, which the circle fixes at
~503.

Validation on the frozen test set (--score, using the reference mini-decoder
below): the circle reference metric goes from FAILING (ratio 1.0195, 185 deg
coverage, centre 105 px low) to PASSING (ratio 1.006, 315 deg coverage,
centre within 14 px), and shift_rms improves on 14 of 16 frames (L077 1.55 ->
0.90, L020 1.41 -> 0.49). The mean composite (14.2 vs baseline 15.1) is NOT
better, reported honestly: the mini-decoder has no droop correction, so
column striping raises parity_db everywhere, and frames with genuine sync
damage (R000) need predictive tracking, which belongs in sync.py. The
constants and the landmark are the deliverable here, not the decoder around
them.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# the measured constants (see module docstring for the evidence)
# ---------------------------------------------------------------------------

BINS = 3200  # measurement grid: bins per line period

PICTURE_START_F = 232 / BINS  # 0.0725 of a period after the zero crossing
PICTURE_END_F = 3040 / BINS  # 0.9500
PICTURE_SPAN_F = PICTURE_END_F - PICTURE_START_F  # 0.8775

PORCH_F = (100 / BINS, 225 / BINS)  # per-trace DC reference window

ISOTROPY_F = 7.440 / BINS  # periods of trace-time per trace spacing
SQUARE_ROWS_512 = 377  # rows for a square-pixel render of 512 traces
ASPECT_43_TRACES = 503  # traces spanned by an exactly-4:3 frame (+-4)

MARKER_TRACES = 22  # diagonal-hatch marker immediately before each frame
CONTENT_TRACES = 515  # typical content extent from the seed (varies ~500-535)
SCAN_STOP_TRACES = 535  # where the converter's scan dies (dot-frame draft)

BLACK_REF_AMP = +0.75  # picture level of black, in sync amplitudes vs porch
WHITE_REF_AMP = -0.75

NOMINAL_PERIOD = 3197.4  # 384 kHz samples; drifts to ~3193.5 by record end


# ---------------------------------------------------------------------------
# landmark detector (independent of sync.py on purpose: this module's
# measurements must not move when sync.py is rewritten)
# ---------------------------------------------------------------------------


def find_marks(
    x: np.ndarray,
    period_guess: float = NOMINAL_PERIOD,
    sigma: float = 6.0,
    fall_frac: float = 0.6,
    max_skip: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trace sub-sample positions of the sync trailing-edge zero crossing.

    1. Candidates: local minima of the Gaussian-smoothed derivative below
       `fall_frac` x the median per-trace steepest fall (the sync edge is the
       steepest fall of nearly every trace, within +-5% frame-wide, so this is
       parity-blind and amplitude-robust).
    2. Longest chain of candidates spaced ~k x period (k <= max_skip), by DP.
       Immune to interlopers such as the calibration circle's own edges, which
       defeat a global phase vote.
    3. Missing slots interpolated, then every slot refined to the nearest raw
       downward zero crossing, linearly interpolated to sub-sample.

    `sigma` is in samples and assumes ~384 kHz; scale with the sample rate.
    Returns (marks, measured): one mark per trace slot; `measured` flags slots
    located from the signal rather than interpolated.
    """
    g = gaussian_filter1d(np.asarray(x, dtype=np.float64), sigma)
    d = np.diff(g)
    n_tr = int(len(x) / period_guess) - 1
    if n_tr < 4:
        raise ValueError("signal shorter than a few traces")
    tmins = np.array(
        [d[int(t * period_guess): int((t + 1) * period_guess)].min() for t in range(n_tr)]
    )
    thr = fall_frac * np.median(tmins)
    loc = np.where((d[1:-1] < d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < thr))[0] + 1
    c = loc.astype(np.float64)
    if len(c) < 8:
        raise ValueError(f"only {len(c)} sync-edge candidates")

    n = len(c)
    best = np.ones(n)
    prev = np.full(n, -1, dtype=int)
    j0 = 0
    for i in range(1, n):
        while c[i] - c[j0] > (max_skip + 0.5) * period_guess:
            j0 += 1
        for j in range(j0, i):
            gap = c[i] - c[j]
            k = round(gap / period_guess)
            if 1 <= k <= max_skip and abs(gap - k * period_guess) <= 45.0 * k:
                if best[j] + 1 > best[i]:
                    best[i] = best[j] + 1
                    prev[i] = j
    i = int(np.argmax(best))
    chain = []
    while i >= 0:
        chain.append(i)
        i = prev[i]
    chain = c[np.array(chain[::-1])]

    slots = np.concatenate([[0], np.cumsum(np.round(np.diff(chain) / period_guess))]).astype(int)
    n_slots = slots[-1] + 1
    pos = np.full(n_slots, np.nan)
    pos[slots] = chain
    idx = np.arange(n_slots)
    good = np.isfinite(pos)
    pos[~good] = np.interp(idx[~good], idx[good], pos[good])

    out = pos.copy()
    meas = good.copy()
    for s in range(n_slots):
        i0 = int(round(pos[s]))
        lo, hi = i0 - 40, i0 + 41
        if lo < 1 or hi > len(x) - 2:
            meas[s] = False
            continue
        seg = x[lo:hi]
        cr = np.where((seg[:-1] >= 0) & (seg[1:] < 0))[0]
        if len(cr) == 0:
            meas[s] = False
            continue
        j = cr[np.argmin(np.abs(cr - (i0 - lo)))] + lo
        out[s] = j + x[j] / (x[j] - x[j + 1])
    return out, meas


def norm_traces(x: np.ndarray, marks: np.ndarray, n: int | None = None) -> np.ndarray:
    """(n_traces, BINS) matrix; trace k is x over [marks[k], marks[k+1])
    resampled onto BINS points, so bin b is period-fraction b/BINS of trace k."""
    m = len(marks) - 1 if n is None else min(n, len(marks) - 1)
    out = np.full((m, BINS), np.nan)
    for k in range(m):
        a, b = marks[k], marks[k + 1]
        t = a + (b - a) * np.arange(BINS) / BINS
        i = np.floor(t).astype(int)
        if i[-1] + 1 >= len(x):
            break
        fr = t - i
        out[k] = (1 - fr) * x[i] + fr * x[i + 1]
    return out


# ---------------------------------------------------------------------------
# reference mini-decoder (validation only; decode.py is the real one)
# ---------------------------------------------------------------------------


def decode_marks(
    x: np.ndarray,
    marks: np.ndarray,
    start_f: float = PICTURE_START_F,
    end_f: float = PICTURE_END_F,
    height: int = SQUARE_ROWS_512,
    n: int = 512,
    dc_restore: bool = True,
) -> np.ndarray:
    """Box-filter decode locked to the zero-crossing marks. Returns (height, n)
    brightness (negative of signal), porch-clamped and de-hummed. No parity
    offset -- measured unnecessary with this landmark."""
    n = min(n, len(marks) - 1)
    pic = np.zeros((n, height))
    porch = np.zeros(n)
    cs = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
    for k in range(n):
        p = marks[k + 1] - marks[k]
        s = marks[k] + start_f * p
        e = marks[k] + end_f * p
        edges = np.linspace(s, e, height + 1)
        lo = np.floor(edges[:-1]).astype(int)
        hi = np.maximum(np.floor(edges[1:]).astype(int), lo + 1)
        lo = np.clip(lo, 0, len(x) - 1)
        hi = np.clip(hi, 1, len(x))
        pic[k] = (cs[hi] - cs[lo]) / (hi - lo)
        a = int(marks[k] + PORCH_F[0] * p)
        b = int(marks[k] + PORCH_F[1] * p)
        if 0 <= a < b <= len(x):
            porch[k] = np.median(x[a:b])
    if dc_restore:
        pic -= porch[:, None]
    odd = np.median(pic[1::2], axis=0)
    even = np.median(pic[0::2], axis=0)
    half = 0.5 * (odd - even)
    pic[1::2] -= half
    pic[0::2] += half
    return -pic.T


# ---------------------------------------------------------------------------
# measurements: each rederives one docstring section from the master
# ---------------------------------------------------------------------------


def _load(fid: str, pre_traces: float = 2.0, n_traces: int = 520):
    from pathlib import Path

    from . import catalog, wav

    master = Path(__file__).resolve().parent.parent / "data" / "master" / "384kHzStereo.wav"
    cat = catalog.build()
    info = wav.probe(master)
    f = cat.by_id(fid)
    s = f.seed_sample - int(pre_traces * NOMINAL_PERIOD)
    n = int((n_traces + 4) * NOMINAL_PERIOD)
    return np.asarray(wav.read(info, f.channel, s, n), dtype=np.float64), f


def measure_landmark_stability(fids: list[str]) -> None:
    for fid in fids:
        x, _ = _load(fid)
        m, meas = find_marks(x)
        idx = np.arange(len(m))
        A = np.polyfit(idx[meas], m[meas], 1)
        rm = (m - np.polyval(A, idx))[meas]
        im = idx[meas]
        oe = np.mean(rm[im % 2 == 0]) - np.mean(rm[im % 2 == 1])
        cons = np.diff(m[meas])[np.diff(im) == 1]
        print(
            f"{fid}: measured {100 * meas.mean():5.1f}%  period {A[0]:9.4f}  "
            f"resid rms {rm.std():5.2f}  odd-even {oe:+5.2f}  spacing std {cons.std():5.2f}"
        )


def measure_active_window(fids: list[str]) -> None:
    """Within-parity across-trace std: blanking is trace-invariant, picture is
    not. Content can only make a frame's rise LATE, so the gate start is the
    early edge of the distribution. The hum flips sign each trace, so the
    statistics MUST be within-parity or the contrast vanishes."""
    starts, dips, long_edges, shelves = [], [], [], []
    for fid in fids:
        try:
            x, _ = _load(fid)
            m, _ = find_marks(x)
            M = norm_traces(x, m)
            M = M[~np.isnan(M[:, 0])]
        except Exception as exc:
            print(f"{fid}: skipped ({exc})")
            continue
        for p in (0, 1):
            S = M[p::2]
            med = np.median(S, axis=0)
            sd = S.std(axis=0)
            floor = np.median(sd[60:200])
            plateau = np.median(sd[350:900])
            if plateau > 2.2 * floor:
                half = floor + 0.5 * (plateau - floor)
                j = 200
                while j < 400 and not np.all(sd[j: j + 12] > half):
                    j += 1
                if j < 400:
                    starts.append(j)
            g = np.diff(np.convolve(med, np.ones(9) / 9, mode="same"))
            if med[3090:3140].mean() < 0.5 * med[3160:3195].mean():  # short parity
                shelves.append(3040 + int(np.argmax(g[3040:3120])))
                dips.append(3000 + int(np.argmin(med[3000:3070])))
            else:
                long_edges.append(3020 + int(np.argmax(g[3020:3100])))
    for name, v in (
        ("picture-start half-rise", starts),
        ("front-porch dip (short parity)", dips),
        ("long-burst leading edge", long_edges),
        ("short-parity shelf rise", shelves),
    ):
        v = np.array(v)
        if len(v):
            print(
                f"{name}: n={len(v)}  median {np.median(v):.0f}  p10 {np.percentile(v, 10):.0f}  "
                f"p90 {np.percentile(v, 90):.0f}  sd {v.std():.1f}   (bins of {BINS})"
            )


def measure_parity_independence(fids: list[str]) -> None:
    """Adjacent-column vertical shift by pair parity: proves the picture rides
    the alternating zero crossing (shifts ~0) rather than a uniform grid."""
    for fid in fids:
        x, _ = _load(fid)
        m, _ = find_marks(x)
        img = decode_marks(x, m)
        a = img - np.median(img, axis=1, keepdims=True)
        h, w = a.shape
        res = {0: [], 1: []}
        for c in range(8, w - 9):
            A, B = a[:, c], a[:, c + 1]
            if A.std() < 1e-4 or B.std() < 1e-4:
                continue
            vals = []
            for s in range(-8, 9):
                aa, bb = (A[: h - s], B[s:]) if s >= 0 else (A[-s:], B[:h + s])
                aa = aa - aa.mean()
                bb = bb - bb.mean()
                den = np.sqrt((aa**2).sum() * (bb**2).sum())
                vals.append((aa * bb).sum() / den if den > 0 else -1.0)
            vals = np.array(vals)
            k = int(np.argmax(vals))
            if vals[k] < 0.5:
                continue
            sh = float(k - 8)
            if 0 < k < len(vals) - 1:
                den = vals[k - 1] - 2 * vals[k] + vals[k + 1]
                if den != 0:
                    sh += 0.5 * (vals[k - 1] - vals[k + 1]) / den
            res[c % 2].append(sh)
        print(
            f"{fid}: even-pair shift {np.mean(res[0]):+.3f} px (n={len(res[0])}), "
            f"odd-pair {np.mean(res[1]):+.3f} px (n={len(res[1])})"
        )


def measure_isotropy() -> None:
    """Fit the L000 calibration ring in raw (trace, bin) coordinates, with
    ring points from BOTH scan directions so oblique-crossing bias cancels."""
    x, _ = _load("L000")
    m, _ = find_marks(x)
    M = norm_traces(x, m)[:512]
    ok = ~np.isnan(M[:, 0])
    med0 = np.median(M[0::2][ok[0::2]], axis=0)
    med1 = np.median(M[1::2][ok[1::2]], axis=0)
    R = M - np.where((np.arange(len(M)) % 2 == 0)[:, None], med0, med1)
    R = R[:, int(PICTURE_START_F * BINS): int(PICTURE_END_F * BINS)]

    def fit(t, v):
        X = np.column_stack([v * v, t, v, np.ones_like(t)])
        A, B, C, D = np.linalg.lstsq(X, -(t * t), rcond=None)[0]
        t0, v0 = -B / 2, -C / (2 * A)
        Rt2 = t0 * t0 + A * v0 * v0 - D
        return t0, v0, np.sqrt(Rt2), np.sqrt(Rt2 / A)

    pts = []
    Gv = gaussian_filter1d(R, 8.0, axis=1)
    for t in range(len(R)):
        col = Gv[t]
        if np.isnan(col[0]):
            continue
        pk = np.where((col[1:-1] > col[:-2]) & (col[1:-1] > col[2:]))[0] + 1
        pts += [(t, v) for v in pk[col[pk] > 0.02]]
    Gt = gaussian_filter1d(np.nan_to_num(R), 1.1, axis=0)
    for v in range(0, R.shape[1], 4):
        row = Gt[:, v]
        pk = np.where((row[1:-1] > row[:-2]) & (row[1:-1] > row[2:]))[0] + 1
        pts += [(t, v) for t in pk[row[pk] > 0.02]]
    pts = np.array(pts, float)
    t, v = pts[:, 0], pts[:, 1]
    t0, v0, Rt, Rv = 270.0, 1350.0, 152.0, 1128.0
    sel = np.ones(len(t), bool)
    for it in range(8):
        rr = np.sqrt(((t - t0) / Rt) ** 2 + ((v - v0) / Rv) ** 2)
        sel = np.abs(rr - 1) < (0.10 if it < 2 else 0.04)
        t0, v0, Rt, Rv = fit(t[sel], v[sel])
    iso = Rv / Rt
    rng = np.random.default_rng(0)
    tt, vv = t[sel], v[sel]
    boot = []
    for _ in range(300):
        ii = rng.integers(0, len(tt), len(tt))
        try:
            _, _, a, b = fit(tt[ii], vv[ii])
            boot.append(b / a)
        except Exception:
            pass
    boot = np.array(boot)
    print(
        f"ring: centre trace {t0:.1f}, bin {v0 + PICTURE_START_F * BINS:.0f} rel landmark; "
        f"radii {Rt:.2f} traces x {Rv:.1f} bins; n={int(sel.sum())}"
    )
    print(
        f"ISOTROPY {iso:.4f} bins/trace  bootstrap sd {boot.std():.4f}  "
        f"CI95 [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]"
    )
    print(
        f"=> gate height {BINS * PICTURE_SPAN_F / iso:.1f} trace-units; "
        f"4:3 frame width {BINS * PICTURE_SPAN_F / iso * 4 / 3:.1f} traces; "
        f"square-pixel rows for 512 traces {BINS * PICTURE_SPAN_F / iso:.0f}"
    )


def measure_roundness() -> None:
    """Multi-frame aspect check: circles must be round in the recommended
    square-pixel render (512 traces x SQUARE_ROWS_512).

    For each circular feature: track its edge to sub-pixel per COLUMN and per
    ROW (both directions, so oblique-crossing bias cancels), robust-fit a pure
    circle, then regress the radial residuals on cos/sin(2 theta). The cos(2t)
    coefficient converts to an x/y stretch: k = 1 + 2*a2/r. Round = 1.000;
    an isotropy error of e reads as k = 1 + e on every feature (7.31 bins/tr,
    the value a 512-trace 4:3 frame needs, would read 1.018).

    Photo limbs (Earth, Mars) carry ~4 px edge noise from clouds/terrain and
    droop streaks; the line-art rings are the precise ones.
    """
    features = [
        # fid, seed cx, cy, r, dark-line ring (True) vs bright-disk limb
        ("L000", 270.0, 182.0, 151.0, True),
        ("L053", 364.0, 176.0, 145.0, True),   # outer dashed ring
        ("L053", 364.0, 176.0, 112.0, True),   # middle solid ring
        ("L013", 300.0, 162.0, 142.0, False),  # Earth, red separation
        ("L014", 300.0, 162.0, 142.0, False),  # Earth, green
        ("L015", 300.0, 162.0, 142.0, False),  # Earth, blue
        ("L011", 259.0, 22.0, 210.0, False),   # Mars limb (centre above frame)
    ]
    imgs = {}
    for fid, cx, cy, r0, dark in features:
        if fid not in imgs:
            x, _ = _load(fid)
            m, _ = find_marks(x)
            img = decode_marks(x, m)
            lo, hi = np.percentile(img, [1, 99])
            imgs[fid] = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)
        a = imgs[fid]
        s = a - gaussian_filter1d(gaussian_filter1d(a, 6, axis=0), 6, axis=1)
        v = -s if dark else s
        H, W = a.shape
        gate, sd = 8.0, 3.0
        keep = x_ = y_ = None
        for _ in range(6):
            pts = []
            for axis in (0, 1):  # 0: per column, 1: per row
                n_outer = W if axis == 0 else H
                for u in range(n_outer):
                    d2 = r0 * r0 - ((u - cx) ** 2 if axis == 0 else (u - cy) ** 2)
                    if d2 <= 4:
                        continue
                    for sgn in (-1, 1):
                        w0 = (cy if axis == 0 else cx) + sgn * np.sqrt(d2)
                        lo_, hi_ = int(w0 - gate), int(w0 + gate) + 1
                        if lo_ < 1 or hi_ > (H if axis == 0 else W) - 1:
                            continue
                        w = v[lo_:hi_, u] if axis == 0 else v[u, lo_:hi_]
                        k = int(np.argmax(w))
                        if w[k] < 0.02:
                            continue
                        kk = float(k)
                        if 0 < k < len(w) - 1:
                            den = w[k - 1] - 2 * w[k] + w[k + 1]
                            if den != 0:
                                kk += 0.5 * (w[k - 1] - w[k + 1]) / den
                        pts.append((u, lo_ + kk) if axis == 0 else (lo_ + kk, u))
            P = np.array(pts, float)
            if len(P) < 60:
                keep = None
                break
            x_, y_ = P[:, 0], P[:, 1]
            keep = np.ones(len(P), bool)
            for _ in range(4):  # trimmed Kasa circle fit
                A = np.column_stack([x_[keep], y_[keep], np.ones(int(keep.sum()))])
                c, *_ = np.linalg.lstsq(A, x_[keep] ** 2 + y_[keep] ** 2, rcond=None)
                cx, cy = c[0] / 2, c[1] / 2
                r0 = np.sqrt(c[2] + cx * cx + cy * cy)
                res = np.hypot(x_ - cx, y_ - cy) - r0
                sd = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-9
                keep = np.abs(res) < 3.0 * sd
            gate = max(4.0, min(gate, 3 * sd + 2))
        if keep is None:
            print(f"{fid} r~{r0:.0f}: no fit")
            continue
        th = np.arctan2(y_[keep] - cy, x_[keep] - cx)
        res = np.hypot(x_[keep] - cx, y_[keep] - cy) - r0
        A = np.column_stack([np.cos(2 * th), np.sin(2 * th),
                             np.cos(th), np.sin(th), np.ones(len(th))])
        coef, *_ = np.linalg.lstsq(A, res, rcond=None)
        stretch = 1 + 2 * coef[0] / r0
        cov = len(np.unique(((th + np.pi) / (2 * np.pi) * 72).astype(int))) * 5
        print(f"{fid} r {r0:6.1f} centre ({cx:6.1f},{cy:6.1f}) n {int(keep.sum()):4d} "
              f"rms {res.std():5.2f} cov {cov:3.0f} deg  x/y stretch {stretch:.4f}")


def measure_levels(fids: list[str]) -> None:
    rows = []
    for fid in fids:
        try:
            x, _ = _load(fid)
            m, _ = find_marks(x)
            M = norm_traces(x, m)
            M = M[~np.isnan(M[:, 0])]
        except Exception as exc:
            print(f"{fid}: skipped ({exc})")
            continue
        med0 = np.median(M[0::2], axis=0)
        med1 = np.median(M[1::2], axis=0)
        ms = med0 if med0[3090:3140].mean() < med1[3090:3140].mean() else med1
        porch = 0.5 * (med0[100:225].mean() + med1[100:225].mean())
        amp = ms[3165:3195].mean() - porch
        pic = M[:, 240:3030]
        p1, p99 = np.percentile(pic, [1, 99])
        rows.append(((p1 - porch) / amp, (p99 - porch) / amp, amp))
        print(f"{fid}: amp {amp:+.4f}  (p1-porch)/amp {(p1 - porch) / amp:+.3f}  "
              f"(p99-porch)/amp {(p99 - porch) / amp:+.3f}")
    r = np.array(rows)
    if len(r):
        print(f"white extreme {r[:, 0].min():+.3f} amp, black extreme {r[:, 1].max():+.3f} amp "
              f"(L000 binary frame: -0.745 / +0.741)")


def score_testset() -> None:
    """Decode the frozen test set with this module's landmark + constants and
    score with pipeline.quality. No timebase is passed (sync factor neutral)."""
    from . import quality
    from .testset import TEST_FRAMES

    print(f"{'frame':5} {'kind':11} {'exp':4} {'score':>6} {'shft':>5} {'drift':>6} "
          f"{'stair':>5} {'h/v':>5} {'par dB':>6}")
    comps = {}
    for fid, kind, expect in TEST_FRAMES:
        x, _ = _load(fid)
        m, _ = find_marks(x)
        img = decode_marks(x, m)
        lo, hi = np.percentile(img, [1, 99])
        imgn = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)
        fs = quality.frame_report(imgn, None, frame_id=fid)
        mm = fs.metrics
        comps[fid] = (mm["composite"], imgn)
        print(f"{fid:5} {kind:11} {expect:4} {mm['composite']:6.1f} {mm['shift_rms']:5.2f} "
              f"{mm['drift_span']:6.1f} {mm['stair_support']:5.2f} {mm['coh_ratio']:5.3f} "
              f"{mm['parity_db']:6.1f}")
    vals = np.array([v[0] for v in comps.values()])
    print(f"mean composite {vals.mean():.1f}  (baseline 15.1)")
    if "L000" in comps:
        from . import quality as q

        cm = q.circle_metrics(comps["L000"][1])
        print(f"CIRCLE L000: ok={cm['ok']} ratio {cm['axis_ratio']:.4f} rms {cm['radial_rms']:.2f} "
              f"centre ({cm['center_dx']:+.1f},{cm['center_dy']:+.1f}) cov {cm['coverage_deg']:.0f} deg")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="re-measure the frame geometry from the master")
    ap.add_argument("--all", action="store_true", help="run every measurement")
    ap.add_argument("--landmark", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--isotropy", action="store_true")
    ap.add_argument("--roundness", action="store_true")
    ap.add_argument("--levels", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args(argv)
    spread = [f"L{i:03d}" for i in range(0, 78, 6)] + [f"R{i:03d}" for i in range(0, 78, 6)]
    if args.landmark or args.all:
        print("== landmark stability ==")
        measure_landmark_stability(["L000", "L020", "L055", "L077", "R000", "R040", "R077"])
    if args.window or args.all:
        print("== active window ==")
        measure_active_window(spread)
    if args.parity or args.all:
        print("== parity independence ==")
        measure_parity_independence(["L055", "R040", "R056"])
    if args.isotropy or args.all:
        print("== isotropy (calibration ring, signal space) ==")
        measure_isotropy()
    if args.roundness or args.all:
        print("== roundness of circular features, square-pixel render ==")
        measure_roundness()
    if args.levels or args.all:
        print("== levels ==")
        measure_levels([f"L{i:03d}" for i in range(0, 78, 11)]
                       + [f"R{i:03d}" for i in range(0, 78, 11)])
    if args.score or args.all:
        print("== test-set score with geometry constants ==")
        score_testset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
