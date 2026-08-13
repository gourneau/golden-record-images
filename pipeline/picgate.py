"""Where the picture starts and stops, measured on each frame's own audio.

WHY THIS MODULE EXISTS
----------------------
`decode.py` uses ONE pair of constants for all 156 frames:

    PICTURE_START = 232/3200      PICTURE_END = 3040/3200

`pipeline/oracle.py` (tier 3 -- it is allowed to look at the answer) found that
letting the decoder pick settings per image raises the gradient NCC from 0.2181
to 0.3080, and that the gate moved on 10 of 10 images. That made the gate the
single largest identified cause of the oracle gap, and the one that ought to be
blindly recoverable: the sync burst is identical on every trace and the picture
is not, so where the picture begins and ends is a property of the signal, not of
the slide.

This module estimates the gate per frame from the audio alone -- no reference
image, no catalogue, nothing but the WAV and a recovered `Timebase`. Its module
level imports numpy and nothing else, so `decode.py` could adopt it without
ending its property of importing only numpy + dotclock + sync.

THE ANSWER, UP FRONT: THE GATE DOES NOT MOVE. This is a null.
-------------------------------------------------------------
Over all 156 frames the picture-close landmark below sits at bin 3036.75
(median) with a standard deviation of 1.38 bins, against a per-frame measurement
noise of 0.27 bins measured by splitting each frame's own traces. So the gate's
true frame-to-frame movement is ~1.36 bins = 1.36 samples at 384 kHz = 0.18 of
one output row out of 377. The picture-OPEN edge cannot be measured at all.

The oracle's gate preference therefore cannot be the picture gate: its search
grid moved the window by +-9.6 and +-19.2 bins, seven to fourteen times the
entire measured spread, and it moved it on 10 of 10 images including frames this
module measures as sitting within a quarter of a bin of the record median. What
those shifts buy is alignment against the SCANNED REFERENCE SLIDES, whose crop
and centring are properties of the scan and not of the record. That is tier 2
all the way down, there is nothing in the audio that encodes it, and it is not
recoverable blindly. A large part of the 41% oracle gap is not headroom.

WHAT IS ACTUALLY MEASURABLE, AND WHAT IS NOT
--------------------------------------------
The two edges are NOT symmetric, and that asymmetry is the whole result.

**The picture CLOSE is measurable, content-free, on every frame.** Fold the
frame onto its own 3200-bin grid and take, per bin, the mean over traces of
|d/db| -- the ACTIVITY profile. Inside blanking every trace is a smooth level,
so activity sits on a floor of ~4e-4. At the instant the video gate shuts, every
trace steps from its own last picture value to the common blanking level; the
steps have different signs and sizes but they all happen at the same bin, so the
profile carries a bump there whatever the picture is. Measured: present on
156/156 frames, on both parities, at 1.8-3.3x the floor.

**The picture OPEN is not measurable at all.** The mirror-image argument fails
in practice. On frames whose top rows are featureless the activity profile stays
on its floor from the end of the sync edge's own ringing (~bin 110) until the
first picture content, which over this record ranges from bin 191 to bin 784 and
is undetectable at all on 48 of 156 frames. Its spread is 128 bins -- ninety
times the close landmark's -- and it is manifestly the SLIDE: it is late exactly
on the frames with empty skies. There is no gate-open transient in the median
profile either: on L055 and L011 the parity-median profile crosses bin 232
without a step, sitting within 1e-3 (L055) and 2e-3 (L011) of the back-porch
level from bin 215 to bin 300 -- against a sync amplitude of 0.14-0.18, so any
gate-open pedestal is below 1.5% of it. `geometry.py` reached the same
conclusion by a different route, from the across-trace
variance ("frames with featureless top rows rise late, so the gate is the
distribution's EARLY edge"); this is an independent statistic agreeing with it.

So the SPAN cannot be estimated per frame, only a SHIFT, and `estimate()`
returns exactly that: slide the global window so its close lands on this frame's
measured close. It is a differential correction with zero mean over the record,
because the bump's centroid sits an unknown number of bins inside the true gate
(the step is smeared by the chain's rise time and by the sample-and-hold
plateau) and only the frame-to-frame VARIATION of the landmark is meaningful.

CONSEQUENCE FOR THE ARBITER NAMED IN THE BRIEF. The calibration circle's axis
ratio measures the SPAN and nothing else -- an ellipse fit absorbs a pure
translation exactly. Since the signal supports no per-frame span estimate, the
circle is structurally incapable of arbitrating this correction, and measurement
confirms it: L000 global 1.0054 / 0.833 px, per-frame 1.0053 / 0.835 px, with
the ring's centre moving +1.01 px for a -7.90-bin (-1.06-row) shift, which is
the shift and nothing else. Radial rms wanders 0.866 +- 0.017 px over a +-44-bin
span scan, so 0.835 vs 0.833 is a tenth of that metric's own noise.
What the circle DOES settle, and it is worth having, is the global span: a
21-point scan puts its preferred span at 2812.8 bins against decode.py's 2808,
agreeing to 4.8 bins, and resolving no better than ~+-5 bins. The gate's measured
movement is a quarter of that. `--circle` reruns all of it.

WHERE THE MOVEMENT THAT DOES EXIST COMES FROM
----------------------------------------------
Split the record by `Timebase.parity_offset` -- sync.py's own per-frame
classification, which knows nothing about the picture (`--parity`):

    group                n     shift mean    sd     range        rows sd
    normal             147       -0.09     0.50   -1.64..+1.03    0.067
    no alternation       6       -5.98     1.55   -7.90..-3.04    0.208
    sign-flipped         2       +4.04     0.23   +3.82..+4.27    0.030
    parity fit failed    1       +2.69       -        +2.69         -

Every frame more than 2 bins from the record median is a frame `sync.py` already
flags: the six whose sync pulse width does not alternate (L000, L004, L005,
L021, L037, L075), the two whose alternation is sign-flipped (R015, R025), and
L032, whose parity fit returns a nonsense 79.8 samples. On the 147 frames with
normal sync parity the shift NEVER exceeds 1.64 bins = 0.22 output rows and its
standard deviation is 0.067 rows.

So this is not a property of the picture gate at all. It is a residual
half-dot-scale error in the per-frame timebase ORIGIN, on exactly the frames
whose parity structure breaks the alternation `sync._zero_cross_offset` centres
its median over. The right fix belongs in `sync.py`, which owns the origin.
Measuring it from the picture is a workaround worth about 1.1 output rows on
L000 -- the calibration frame, which is one of the six -- and nothing at all on
94% of the record.

A SEPARATE, GLOBAL OBSERVATION, NOT ACTED ON
---------------------------------------------
PICTURE_END sits at 3040 and the median close landmark at 3036.75, so the window
runs ~3.2 bins past the landmark on every frame, and the encroachment test finds
a bottom-row correlation deficit of -0.154 on the 147 normal frames -- i.e. the
last dot row of every decoded frame appears to be part blanking. That is a
question about a GLOBAL constant, not about per-frame estimation, and this module
does not answer it: the bump's centroid is not the gate (the step is smeared by
the chain's rise time and by the one-dot sample-and-hold), so the landmark's
absolute calibration is unknown and a 3-bin global move cannot be justified from
it. Recorded as a lead for whoever owns geometry.py, which chose 3040 on the
different and defensible ground that "content is gone by 3040".

TESTS RUN, INCLUDING THE ONES THAT FAILED
------------------------------------------
1. Split-half, interleaved (`--survey`): per-frame noise 0.27 bins against a
   1.38-bin spread, so the per-frame estimate is real at ~5:1, not noise. A
   SEQUENTIAL split measures 3.97 bins instead -- the top and the bottom of a
   picture carry different content and the bump's amplitude rides on content --
   which is why the interleaved split is the honest one.
2. Blanking encroachment in IMAGE space (`--encroach`), 156 frames, dot-native:
   adjacent dot rows of a picture correlate across traces at 0.984; a row that
   is really blanking does not. With the global gate the bottom dot row
   correlates at 0.828 -- a deficit of -0.156 against its own two-rows-up
   control -- and that deficit correlates with how much blanking the window
   captured (r = -0.25, sign as predicted). With the per-frame gate the
   correlation with captured blanking vanishes (+0.12). Split by group, the
   result says exactly what the survey says:

       9 parity-anomalous frames   deficit -0.190 -> -0.086, improved 6/9
     147 normal frames             deficit -0.154 -> -0.152, improved 77/147

   An order-of-magnitude improvement where the shift is real (L000 -0.246 ->
   -0.036, L021 -0.343 -> -0.141) and a coin flip -- 52% -- everywhere else.
   Two of the nine get WORSE and should: R015 and R025 sit +4 bins late, so the
   global window happened to stop just short of their gate, and equalising them
   to the record median moves them into the same 3.2 bins of blanking every
   other frame already has. This is a CONSISTENCY check, not an independent
   phenomenon: the gate step is what both instruments see.
3. Second instrument, independent of the first (`--knee`): push the window's end
   past the gate and watch the bottom dot row stop being picture; extrapolate
   the falling arm back to zero deficit. Over 18 frames it tracks the activity
   bump with r = 0.60 and slope 1.11, i.e. one for one, but with 5.0 bins of
   scatter -- enough to confirm the effect's existence and sign, not enough to
   be a better estimator. Its first version was WRONG and is recorded as such:
   scanning past bin ~3050 makes BOTH bottom rows blanking, which re-correlate
   through the chain's droop memory, so the deficit curve turns back up and a
   V-fit through it returned knees as absurd as 3155. Capping the scan at 3046
   (geometry.py puts the burst's leading edge at 3048 +- 4) fixed it.
4. Brightness control: splitting traces into the darkest and brightest halves
   moves the landmark by 3.11 bins rms, which looks alarming next to 0.27 bins
   of noise -- but a brightness split is spatially contiguous on a photograph,
   so it is the same content confound as the sequential split (3.97 bins) and
   not evidence about the gate. Reported because it is the control that would
   have refuted this if it had been clean, and it is not clean. The
   content-free evidence is the fold's own floor, and the fact that the nine
   outliers are the nine parity-anomalous frames rather than nine unusual
   pictures.
5. Tripwires, on 16 frames spanning both channels and all four parity groups:
   `quality` composite +0.67 mean, improving on 9 of 16 -- no signal, which is
   what a sub-pixel window shift should produce -- and `sharpness` -0.03% mean,
   |change| < 1% on every frame. These metrics are known unreliable (they
   reward blur and improve monotonically past the correct setting) and chose
   nothing here; they are checked only to confirm nothing broke.
6. NOT RUN, and it should be: whether correcting those nine frames improves
   registration between the three separations of a colour triplet. That is the
   one genuinely independent arbiter available -- the correction comes from each
   frame's own blanking transition and the test is agreement between DIFFERENT
   frames -- but none of the nine sits in a triplet, and on the other 147 the
   effect (0.067 rows sd) is well below what cross-separation registration can
   resolve. A second agent is currently working on colour registration; if they
   want a target, this is it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BINS = 3200  # measurement grid, matching pipeline/geometry.py: 1 bin = period/3200

# Search window for the close landmark. Its floor comes from the picture side
# (bins 2880..3010, always inside the gate) and its top is clipped below the
# sync burst's leading edge, which `_burst_edge` locates per frame -- the burst
# is 100-1000x the floor and would otherwise own the centroid. 3044 is the hard
# ceiling: geometry.py measures the long-burst leading edge at 3048 +- 4.
_CLOSE_LO = 3010
_CLOSE_HI = 3044
_FLOOR_LO, _FLOOR_HI = 2880, 3010
_BURST_FACTOR = 20.0  # x floor; the burst is 100-1000x, the gate step 1.8-3.3x

# Onset search for the picture OPEN. Diagnostic only -- see the module
# docstring; this measures the slide's first content, not the gate.
_ONSET_LO, _ONSET_HI = 120, 900
_ONSET_FLOOR = (130, 200)
_ONSET_FACTOR = 2.5
_ONSET_RUN = 8  # bins that must all exceed the threshold, so one tick cannot fire

# Median close landmark over all 156 frames (`--survey`). The correction is
# DIFFERENTIAL against this: the bump's centroid sits some unknown number of
# bins inside the true gate, so its absolute value may not move the gate, only
# its frame-to-frame variation may.
RECORD_CLOSE_BIN = 3036.75


@dataclass(frozen=True)
class PicGate:
    """One frame's gate measurement, in bins of period/3200 after the landmark."""

    close: float          # measured picture-close landmark, bins
    close_snr: float      # bump peak / blanking floor; ~1.0 means not detected
    close_noise: float    # split-half estimate of this frame's own noise, bins
    onset: float          # first along-trace content (NOT the gate), bins or nan
    burst_edge: int       # sync burst leading edge on the measured parity, bins
    parity: int           # which trace parity carried the measurement
    n_traces: int

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.close)) and self.close_snr > 1.6

    @property
    def shift_bins(self) -> float:
        """How far this frame's picture sits from the record's own median."""
        return self.close - RECORD_CLOSE_BIN


def fold(x: np.ndarray, tb, n_traces: int = 512) -> np.ndarray:
    """(traces, 3200) matrix: bin b of trace k is period-fraction b/3200 of it.

    Linear interpolation on the SAME grid decode.py samples the picture on, so a
    bin here is directly a bin there.
    """
    x = np.asarray(x, dtype=np.float64)
    n = min(n_traces, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n)])
    t = starts[:, None] + (np.arange(BINS)[None, :] / BINS) * tb.period
    i = np.floor(t).astype(np.int64)
    ok = (i[:, 0] >= 0) & (i[:, -1] + 1 < len(x))
    f = t - i
    i = np.clip(i, 0, len(x) - 2)
    return ((1.0 - f) * x[i] + f * x[i + 1])[ok]


def activity(M: np.ndarray) -> np.ndarray:
    """Mean over traces of |d/db|: the blanking floor, and every gate step.

    |d/db| rather than the across-trace variance on purpose. The variance does
    not return to its blanking floor after the gate shuts -- the chain's
    AC-coupling leaves each trace's blanking level carrying a memory of that
    trace's own brightness, which is real across-trace variation with no
    along-trace structure. |d/db| is blind to it and sees only the step.
    """
    a = np.zeros(BINS)
    a[:-1] = np.mean(np.abs(np.diff(np.asarray(M, dtype=np.float64), axis=1)), axis=0)
    a[-1] = a[-2]
    return a


def _floor(a: np.ndarray) -> float:
    return float(np.median(a[_FLOOR_LO:_FLOOR_HI]))


def _burst_edge(a: np.ndarray, floor: float) -> int:
    """First bin at or after 3038 where the sync burst dwarfs the floor."""
    idx = np.where(a[3038:BINS] > _BURST_FACTOR * max(floor, 1e-12))[0]
    return int(3038 + idx[0]) if len(idx) else BINS


def close_landmark(a: np.ndarray, hi: int | None = None) -> tuple[float, float]:
    """Centroid of the gate-close activity bump, in bins, and its floor ratio.

    The centroid, not the peak or a half-rise: the bump is ~12 bins wide (one
    sample-and-hold dot) and asymmetric, and a centroid over a fixed window is
    the only summary of it that needs no threshold chosen by eye.
    """
    floor = _floor(a)
    if hi is None:
        hi = min(_CLOSE_HI, _burst_edge(a, floor) - 3)
    lo = _CLOSE_LO
    if hi - lo < 8:
        return float("nan"), 0.0
    w = np.clip(a[lo:hi] - floor, 0.0, None)
    if w.sum() <= 0:
        return float("nan"), 0.0
    b = np.arange(lo, hi) + 0.5
    snr = float(a[lo:hi].max() / max(floor, 1e-12))
    return float((b * w).sum() / w.sum()), snr


def content_onset(a: np.ndarray) -> float:
    """First bin where along-trace structure appears and stays. NOT the gate.

    Kept because the negative result needs a number attached to it: this is the
    best signal-side estimator of the picture OPEN that exists, and its spread
    across the record is ninety times the close landmark's.
    """
    floor = float(np.median(a[_ONSET_FLOOR[0]:_ONSET_FLOOR[1]]))
    thr = _ONSET_FACTOR * floor
    for b in range(_ONSET_LO, _ONSET_HI):
        if np.all(a[b:b + _ONSET_RUN] > thr):
            return float(b)
    return float("nan")


def _measure_parity(M: np.ndarray) -> tuple[int, np.ndarray, int]:
    """Pick the late-burst (short-pulse) parity and return its window ceiling.

    The sync pulse WIDTH alternates with trace parity by design, so one parity's
    burst leading edge arrives ~100 bins before the other's and can crowd the
    close window. Measure on the late one.
    """
    idx = np.arange(M.shape[0])
    a_p = [activity(M[idx[idx % 2 == p]]) for p in (0, 1)]
    edges = [_burst_edge(a, _floor(a)) for a in a_p]
    p = 0 if edges[0] >= edges[1] else 1
    return p, a_p[p], edges[p]


def measure(x: np.ndarray, tb, n_traces: int = 512) -> PicGate:
    """Measure one frame's gate landmarks from its signal and timebase."""
    M = fold(x, tb, n_traces)
    idx = np.arange(M.shape[0])
    p, a, edge = _measure_parity(M)
    sel = idx[idx % 2 == p]
    hi = min(_CLOSE_HI, edge - 3)

    close, snr = close_landmark(a, hi)

    # Split-half on the SAME parity: the frame's own measurement noise, so the
    # across-frame spread can be compared against something rather than admired.
    # INTERLEAVED halves, not the first and second half of the frame: the top
    # and the bottom of a picture carry different content and the bump's
    # amplitude rides on content, so a sequential split measures the slide as
    # well as the noise (measured: 3.97 bins sequential vs 0.27 interleaved).
    cA, _ = close_landmark(activity(M[sel[0::2]]), hi)
    cB, _ = close_landmark(activity(M[sel[1::2]]), hi)
    noise = abs(cA - cB) / 2.0 if np.isfinite(cA) and np.isfinite(cB) else float("nan")

    return PicGate(close=close, close_snr=snr, close_noise=noise,
                   onset=content_onset(activity(M)), burst_edge=edge,
                   parity=p, n_traces=int(M.shape[0]))


def content_control(x: np.ndarray, tb, n_traces: int = 512) -> tuple[float, float]:
    """Close landmark from the darkest half of traces, and from the brightest.

    CONFOUNDED, and reported anyway (module docstring, test 4): on a photograph
    a brightness split is spatially contiguous, so this measures content as well
    as any real brightness dependence and cannot separate them.
    """
    M = fold(x, tb, n_traces)
    idx = np.arange(M.shape[0])
    p, _, edge = _measure_parity(M)
    sel = idx[idx % 2 == p]
    hi = min(_CLOSE_HI, edge - 3)
    order = np.argsort(M[sel][:, 400:2900].mean(axis=1))
    dark = sel[order[: len(order) // 2]]
    light = sel[order[len(order) // 2:]]
    return (close_landmark(activity(M[dark]), hi)[0],
            close_landmark(activity(M[light]), hi)[0])


def gate_for(g: PicGate, start: float, span: float,
             record_close: float = RECORD_CLOSE_BIN) -> tuple[float, float]:
    """Per-frame (picture_start, picture_span) as fractions of the period.

    A pure SHIFT of the global window onto this frame's measured close. The span
    is not touched, because the open edge is not measurable (module docstring)
    and a span built from an unmeasurable edge would be noise dressed up as a
    correction.
    """
    if not g.ok:
        return start, span
    return start + (g.close - record_close) / BINS, span


# --------------------------------------------------------------------------
# measurement harnesses -- everything the docstring claims, re-runnable
# --------------------------------------------------------------------------


def _loader():
    """WAV + catalogue, imported lazily so the estimator stays numpy-only."""
    from pathlib import Path

    from . import catalog as catalog_mod, wav

    repo = Path(__file__).resolve().parent.parent
    info = wav.probe(repo / "data" / "master" / "384kHzStereo.wav")
    return repo, wav.memmap(info), catalog_mod.build()


def _frame_signal(mm, cat, fid: str, n_traces: int = 512):
    from . import sync as sync_mod

    fr = cat.by_id(fid)
    n = int(sync_mod.NOMINAL_PERIOD * (n_traces + 6))
    seg = np.asarray(mm[fr.seed_sample - 2000: fr.seed_sample + n, fr.channel],
                     dtype=np.float64)
    return seg, sync_mod.recover(seg, n_traces=n_traces), ("L" if fr.channel == 0 else "R")


def survey(frame_ids=None, n_traces: int = 512, control: bool = False) -> list[dict]:
    """Measure the gate on every frame."""
    _, mm, cat = _loader()
    ids = frame_ids or [f.id for f in cat.frames]
    rows = []
    for fid in ids:
        try:
            seg, tb, _ = _frame_signal(mm, cat, fid, n_traces)
            g = measure(seg, tb, n_traces)
        except Exception as exc:  # a frame we cannot lock is not a gate result
            rows.append({"id": fid, "error": str(exc)})
            continue
        row = {"id": fid, "close": g.close, "snr": g.close_snr, "noise": g.close_noise,
               "onset": g.onset, "burst": g.burst_edge, "parity": g.parity,
               "shift_bins": g.shift_bins, "period": tb.period,
               "located": float(tb.located.mean())}
        if control:
            row["dark"], row["light"] = content_control(seg, tb, n_traces)
        rows.append(row)
    return rows


def circle_test(fid: str = "L000", scan: bool = True) -> dict:
    """The arbiter named in the brief, and its resolving power.

    Axis ratio and radial rms for the global gate and the per-frame gate, plus a
    span scan that measures how much span change the circle can see at all.
    quality.circle_metrics reports major/minor, which is >= 1 by construction, so
    the scan is V-shaped and its vertex -- not its value -- is the span the ring
    says is round.
    """
    from . import decode as D, quality as Q

    _, mm, cat = _loader()
    seg, tb, ch = _frame_signal(mm, cat, fid)
    g = measure(seg, tb)
    base = D.Settings(channel=ch)
    st, sp = gate_for(g, base.picture_start, base.picture_span)

    out = {"frame": fid, "close": g.close, "shift_bins": g.shift_bins, "cases": []}
    for name, s0, s1 in (("global", base.picture_start, base.picture_span),
                         ("per-frame", st, sp)):
        m = Q.circle_metrics(D.decode(seg, base.replace(picture_start=s0,
                                                        picture_span=s1), tb).image)
        out["cases"].append({"case": name, "start_bin": s0 * BINS, "span_bin": s1 * BINS,
                             "axis_ratio": m["axis_ratio"], "radial_rms": m["radial_rms"],
                             "center_dy": m["center_dy"], "coverage_deg": m["coverage_deg"]})
    if scan:
        xs, ys, rs = [], [], []
        for d in range(-44, 45, 4):
            sp2 = (base.picture_span * BINS + d) / BINS
            m = Q.circle_metrics(D.decode(seg, base.replace(picture_span=sp2), tb).image)
            xs.append(base.picture_span * BINS + d)
            ys.append(m["axis_ratio"])
            rs.append(m["radial_rms"])
        xs, ys, rs = np.array(xs), np.array(ys), np.array(rs)
        k = int(np.argmin(ys))
        lo, hi = xs < xs[k], xs > xs[k]
        a1, a2 = np.polyfit(xs[lo], ys[lo], 1), np.polyfit(xs[hi], ys[hi], 1)
        out["scan"] = {"vertex_span_bin": float((a2[1] - a1[1]) / (a1[0] - a2[0])),
                       "slope_per_bin": float(abs(a2[0])),
                       "radial_rms_mean": float(rs.mean()),
                       "radial_rms_sd": float(rs.std()),
                       "points": [[float(a), float(b), float(c)] for a, b, c in zip(xs, ys, rs)]}
    return out


def _rowcorr(img: np.ndarray, i: int, j: int) -> float:
    a = img[i] - img[i].mean()
    b = img[j] - img[j].mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def encroach_test(frame_ids=None) -> list[dict]:
    """Blanking encroachment, measured in IMAGE space, global gate vs per-frame.

    Adjacent dot rows of a picture correlate across traces. A row that is really
    blanking does not -- it carries the chain's droop memory instead. No
    reference image and no image-quality metric is involved: this is a
    correlation the picture has by construction and blanking does not.
    """
    from . import decode as D

    _, mm, cat = _loader()
    ids = frame_ids or [f.id for f in cat.frames]
    rows = []
    for fid in ids:
        try:
            seg, tb, ch = _frame_signal(mm, cat, fid)
            g = measure(seg, tb)
        except Exception:
            continue
        base = D.Settings(channel=ch, dot_native=True)
        st, sp = gate_for(g, base.picture_start, base.picture_span)
        row = {"id": fid, "close": g.close, "encroach": 3040.0 - g.close}
        for tag, s0 in (("glob", base.picture_start), ("pf", st)):
            img = D.decode(seg, base.replace(picture_start=s0, picture_span=sp), tb).image
            n = img.shape[0]
            row[tag + "_last"] = _rowcorr(img, n - 1, n - 2)
            row[tag + "_ctrl"] = _rowcorr(img, n - 3, n - 4)
        rows.append(row)
    return rows


def parity_groups(frame_ids=None) -> list[dict]:
    """Per-frame shift against `Timebase.parity_offset`, the sync-side anomaly.

    The classification is sync.py's own, measured per frame from the landmark
    residuals, and nothing about the picture enters it.
    """
    from . import sync as sync_mod  # noqa: F401  (used via _frame_signal)

    _, mm, cat = _loader()
    ids = frame_ids or [f.id for f in cat.frames]
    rows = []
    for fid in ids:
        try:
            seg, tb, _ = _frame_signal(mm, cat, fid)
            g = measure(seg, tb)
        except Exception:
            continue
        p = float(tb.parity_offset)
        grp = ("no alternation" if abs(p) < 2.0 else
               "sign-flipped" if p < -2.0 else
               "parity fit failed" if p > 20.0 else "normal")
        rows.append({"id": fid, "parity_offset": p, "shift_bins": g.shift_bins, "group": grp})
    return rows


def knee_test(frame_ids=None, ends=None) -> list[dict]:
    """Second instrument for the close, independent of the activity profile.

    Push the window's end past the gate and watch the bottom dot row stop being
    picture; extrapolate the falling arm back to zero deficit. Capped at bin 3046
    on purpose: past ~3050 BOTH bottom rows are blanking and re-correlate through
    the droop memory, which turns the curve back up and breaks any fit through it.
    """
    from . import decode as D

    _, mm, cat = _loader()
    ids = frame_ids or ["L000", "L004", "L005", "L021", "L037", "L075", "R015", "R025",
                        "L020", "L055", "R040", "R010", "L034", "R056", "L011", "L002",
                        "L022", "L032", "L042", "R070"]
    ends = np.asarray(ends if ends is not None else np.arange(3010, 3047, 2.0), dtype=float)
    rows = []
    for fid in ids:
        try:
            seg, tb, ch = _frame_signal(mm, cat, fid)
            g = measure(seg, tb)
        except Exception:
            continue
        base = D.Settings(channel=ch, dot_native=True)
        v = []
        for e in ends:
            img = D.decode(seg, base.replace(picture_start=232.0 / BINS,
                                             picture_span=(e - 232.0) / BINS), tb).image
            n = img.shape[0]
            v.append(_rowcorr(img, n - 1, n - 2) - _rowcorr(img, n - 3, n - 4))
        v = np.array(v)
        m = (v < -0.04) & (v > -0.40)
        knee = float("nan")
        if m.sum() >= 3:
            a = np.polyfit(ends[m], v[m], 1)
            if a[0] != 0:
                knee = float(-a[1] / a[0])
        rows.append({"id": fid, "bump": g.close, "knee": knee})
    return rows


# --------------------------------------------------------------------------

def _report_survey(rows: list[dict]) -> None:
    good = [r for r in rows if "error" not in r and np.isfinite(r["close"])]
    c = np.array([r["close"] for r in good])
    nz = np.array([r["noise"] for r in good])
    on = np.array([r["onset"] for r in good], dtype=float)
    snr = np.array([r["snr"] for r in good])
    noise = float(np.sqrt(np.nanmean(nz ** 2)))
    spread = float(np.std(c))
    true = float(np.sqrt(max(spread ** 2 - noise ** 2, 0.0)))

    print(f"\nPICTURE GATE, measured per frame on {len(good)} of {len(rows)} frames")
    print(f"  close landmark      median {np.median(c):8.2f} bins   sd {spread:5.2f}"
          f"   min {c.min():.2f}   max {c.max():.2f}")
    print(f"  bump strength       {snr.min():.1f}-{snr.max():.1f} x the blanking floor")
    print(f"  per-frame noise     {noise:5.2f} bins (interleaved split-half, same parity)")
    print(f"  TRUE spread         {true:5.2f} bins = {true / 7.440:5.3f} output rows of 377")
    print(f"  decode.py constant  {3040.0:8.2f} bins  (a global choice, not per frame)")

    fin = np.isfinite(on)
    print("\n  content onset -- the OPEN edge. NOT a gate measurement:")
    print(f"    detected on {fin.sum()}/{len(good)} frames   median {np.median(on[fin]):.0f}"
          f"   sd {np.std(on[fin]):.1f}   min {on[fin].min():.0f}   max {on[fin].max():.0f}")

    if "dark" in good[0]:
        d = np.array([r["dark"] for r in good], dtype=float)
        l = np.array([r["light"] for r in good], dtype=float)
        m = np.isfinite(d) & np.isfinite(l)
        print(f"\n  brightness control (CONFOUNDED, see docstring): |dark-light| rms "
              f"{np.sqrt(np.mean((d[m]-l[m])**2)):.2f} bins on {m.sum()} frames")

    out = sorted(good, key=lambda r: abs(r["close"] - float(np.median(c))), reverse=True)
    print("\n  frames furthest from the record median:")
    for r in out[:10]:
        d = r["close"] - float(np.median(c))
        print(f"    {r['id']}  {r['close']:8.2f}  {d:+6.2f} bins  {d / 7.440:+6.2f} rows"
              f"   burst edge {r['burst']}")
    n_big = sum(1 for r in good if abs(r["close"] - float(np.median(c))) > 3.0)
    print(f"\n  frames more than 3 bins (0.40 rows) off the median: {n_big}/{len(good)}")


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="", help="comma-separated ids; default all 156")
    ap.add_argument("--control", action="store_true", help="also run the brightness control")
    ap.add_argument("--circle", action="store_true", help="the calibration-ring arbiter")
    ap.add_argument("--encroach", action="store_true", help="blanking encroachment in image space")
    ap.add_argument("--knee", action="store_true", help="the second, image-space instrument")
    ap.add_argument("--parity", action="store_true", help="shift vs sync parity anomaly")
    args = ap.parse_args()
    ids = [s for s in args.frames.split(",") if s] or None
    repo = Path(__file__).resolve().parent.parent

    if args.circle:
        r = circle_test(ids[0] if ids else "L000")
        print(f"\nCALIBRATION RING -- {r['frame']}, per-frame shift "
              f"{r['shift_bins']:+.2f} bins ({r['shift_bins']/7.44:+.2f} rows)")
        print(f"  {'case':10s} {'start_bin':>10s} {'span_bin':>10s} {'axis_ratio':>11s}"
              f" {'radial_rms':>11s} {'centre_dy':>10s} {'cov_deg':>8s}")
        for c in r["cases"]:
            print(f"  {c['case']:10s} {c['start_bin']:10.2f} {c['span_bin']:10.2f}"
                  f" {c['axis_ratio']:11.4f} {c['radial_rms']:11.3f}"
                  f" {c['center_dy']:10.2f} {c['coverage_deg']:8.0f}")
        s = r["scan"]
        print(f"\n  span scan: the ring is roundest at span {s['vertex_span_bin']:.1f} bins"
              f"  (decode.py uses 2808)")
        print(f"  sensitivity {s['slope_per_bin']:.5f} of axis ratio per bin of span;"
              f" radial_rms over the whole scan {s['radial_rms_mean']:.3f} +- {s['radial_rms_sd']:.3f} px")
        print("  => the circle measures the SPAN. A pure shift is invisible to it, and the"
              "\n     gate's whole measured movement is a quarter of what the scan can resolve.")
    elif args.encroach:
        rows = encroach_test(ids)
        arr = lambda k: np.array([r[k] for r in rows], dtype=float)
        E = arr("encroach")
        print(f"\nBLANKING ENCROACHMENT, {len(rows)} frames, dot-native decode")
        for tag, nm in (("glob", "global gate"), ("pf", "per-frame gate")):
            d = arr(tag + "_last") - arr(tag + "_ctrl")
            m = np.isfinite(d)
            print(f"  {nm:16s} bottom-row corr {np.nanmean(arr(tag+'_last')):.4f}"
                  f"   control {np.nanmean(arr(tag+'_ctrl')):.4f}"
                  f"   deficit {np.nanmean(d):+.4f}"
                  f"   corr(deficit, blanking captured) {np.corrcoef(E[m], d[m])[0,1]:+.3f}")
        dg = arr("glob_last") - arr("glob_ctrl")
        dp = arr("pf_last") - arr("pf_ctrl")
        print(f"  per-frame gate improves the deficit on {(dp > dg).sum()}/{len(dg)} frames")
        o = np.argsort(E)[::-1]
        ids_ = [r["id"] for r in rows]
        print(f"\n  {'frame':6s} {'E_bins':>8s} {'global':>9s} {'per-frame':>10s} {'change':>9s}")
        for i in o[:10]:
            print(f"  {ids_[i]:6s} {E[i]:8.2f} {dg[i]:9.4f} {dp[i]:10.4f} {dp[i]-dg[i]:+9.4f}")
        (repo / "data" / "picgate_encroach.json").write_text(json.dumps(rows, indent=1))
    elif args.parity:
        rows = parity_groups(ids)
        groups: dict[str, list[float]] = {}
        for r in rows:
            groups.setdefault(r["group"], []).append(r["shift_bins"])
        print("\nPER-FRAME SHIFT vs SYNC PARITY ANOMALY (sync.py's own classification)")
        for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            a = np.array(v)
            print(f"  {k:20s} n={len(a):3d}   shift mean {a.mean():+6.2f}  sd {a.std():5.2f}"
                  f"  min {a.min():+6.2f}  max {a.max():+6.2f} bins"
                  f"   ({a.std()/7.44:.3f} output rows sd)")
        n = np.array(groups.get("normal", []))
        if len(n):
            print(f"\n  on the {len(n)} frames with normal sync parity the shift never exceeds"
                  f" {np.abs(n).max():.2f} bins = {np.abs(n).max()/7.44:.2f} output rows.")
        (repo / "data" / "picgate_parity.json").write_text(json.dumps(rows, indent=1))
    elif args.knee:
        rows = knee_test(ids)
        b = np.array([r["bump"] for r in rows])
        k = np.array([r["knee"] for r in rows])
        m = np.isfinite(k)
        print(f"\n  {'frame':6s} {'bump':>9s} {'knee':>9s} {'knee-bump':>10s}")
        for r in rows:
            print(f"  {r['id']:6s} {r['bump']:9.2f} {r['knee']:9.2f} {r['knee']-r['bump']:10.2f}")
        print(f"\n  corr(bump, knee) {np.corrcoef(b[m], k[m])[0,1]:.3f} on {m.sum()} frames;"
              f" slope {np.polyfit(b[m], k[m], 1)[0]:.3f};"
              f" scatter {np.std(k[m]-b[m]):.2f} bins")
    else:
        rows = survey(ids, control=args.control)
        _report_survey(rows)
        p = repo / "data" / "picgate.json"
        p.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {p}")
