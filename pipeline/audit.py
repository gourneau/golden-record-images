"""Quality control over all 156 frames: what still decodes badly, and why.

Read-only with respect to the rest of the pipeline. Everything here runs the
SHIPPING decoder (pipeline/decode.py via pipeline/testset.decode_frame) on the
384 kHz master and measures the result; nothing here feeds back into pixel
recovery, and no reference image is used anywhere.

Sub-commands
------------
  scan       decode every frame twice -- current (uncouple ON, per-channel tau)
             and pre-droop (uncouple OFF) -- and record quality metrics plus
             every signal-domain diagnostic the decoder exposes. Writes JSON.
  sheets     contact sheets of the worst frames, and pre/post pairs for the
             frames the composite says regressed, for visual adjudication.
  parity     survey the per-frame sync parity offset across all 156 frames and
             characterise the outliers (L032).
  dotclock   why 66 frames fail to lock the dot clock, and what recovers them,
             with a split-half cross-validation a wrong answer cannot game.

Honest-criterion notes
----------------------
* the composite of pipeline/quality.py is used only for RANKING candidates; it
  is known to score smeared frames above resolved ones, so every conclusion
  below is confirmed on a criterion the metric cannot game (split-half
  agreement of the dot clock, the dots/trace cluster, gap-line flatness, the
  L000 circle fit, or direct inspection of the picture).

WHAT THIS PASS FOUND (2026-08, all 156 frames, decoded from the master)
----------------------------------------------------------------------
1. THE COMPOSITE'S RANKING IS INVALID ON THE CURRENT DECODER. Its spread is
   97% f_drift (corr of log composite with log f_drift 0.967); f_stair is
   0.997-1.000 on every one of the 156 frames, i.e. the diagonal staircase the
   score was built to catch is extinct. drift_span now measures the SCENE'S
   DIAGONAL (see column_slope): its five worst frames -- L057 sand dunes,
   R007/R008/R009 jungle, L011 Mars -- are clean, well resolved decodes, while
   a visibly banded frame (R055) sits 9th BEST. Use residual_drift() instead.
2. ONE frame is genuinely broken: L032 (see reanchored_timebase).
3. NO decode is a real regression from the droop fix. 59 frames score lower
   than the undrooped decode; f_parity accounts for more than half the loss on
   45 of them (median share 1.05, i.e. the other factors improved), and the
   odd/even stripe it responds to has an amplitude of 0.12-0.74% of black-white
   (median 0.18%). Every pair inspected side by side is better after the fix.
4. The dot clock is recoverable on 147/156 frames with a matched filter, not
   90/156 (see dotclock_recover), and 17 of the current 90 "locks" are false.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import catalog as catalog_mod
from . import decode as decode_mod
from . import dotclock as dot_mod
from . import quality as quality_mod
from . import sync as sync_mod
from . import wav

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "data" / "master" / "384kHzStereo.wav"
DECIM = 4
NOMINAL_96 = sync_mod.NOMINAL_PERIOD / DECIM
PRE_384 = int(round(1.5 * sync_mod.NOMINAL_PERIOD))

OUT = Path(
    "/private/tmp/claude-501/-Users-josh-code-golden-record-images/"
    "484b41e6-9824-495a-97cb-a0b8bfe1c684/scratchpad/audit"
)


# --------------------------------------------------------------------------
# frame access
# --------------------------------------------------------------------------


def frame_signal(frame: catalog_mod.Frame, master: Path = MASTER) -> np.ndarray:
    """The 96 kHz window build.py/testset.py decode, for one frame."""
    info = wav.probe(master)
    span = int(math.ceil((sync_mod.TRACES_PER_FRAME + 6) * sync_mod.NOMINAL_PERIOD))
    start = frame.seed_sample - PRE_384
    if start < 0:
        raise ValueError(f"{frame.id}: seed too close to file start")
    x = np.asarray(wav.read(info, frame.channel, start, PRE_384 + span), dtype=np.float64)
    return resample_poly(x, 1, DECIM)


def decode_both(frame: catalog_mod.Frame):
    """(timebase, current decode, pre-droop decode) for one frame."""
    x96 = frame_signal(frame)
    tb = sync_mod.recover(x96, period_guess=NOMINAL_96, n_traces=sync_mod.TRACES_PER_FRAME)
    xf = np.asarray(x96, dtype=np.float32)
    cur = decode_mod.decode(xf, decode_mod.Settings(channel=frame.id[0]), tb)
    pre = decode_mod.decode(xf, decode_mod.Settings(channel="", uncouple=False), tb)
    return x96, tb, cur, pre


# --------------------------------------------------------------------------
# extra, signal-domain defect measures (none of them gameable by sharpening)
# --------------------------------------------------------------------------


def rail_clipping(img: np.ndarray) -> dict:
    """Fraction of picture pixels pinned to the black/white rails.

    decode() clips to [0,1] after anchoring on the signal's own references, so
    a large pinned fraction means the intensity anchors are wrong for this
    frame (bad porch, bad sync-amplitude measurement) or the droop inverse ran
    away -- not that the scene is contrasty: real scenes measure 0-4%.
    """
    a = np.asarray(img)
    return {
        "clip_black": float(np.mean(a >= 0.999)),
        "clip_white": float(np.mean(a <= 0.001)),
    }


def gate_flatness(x: np.ndarray, tb, cfg=None) -> dict:
    """Flatness of the content-free back porch and of the blanking level.

    porch_pp / porch_rms: peak-to-peak and RMS of the per-trace porch median
    after the decoder's own clamp would have removed the slow trend. A frame
    whose sync detection wandered onto picture content shows a porch that is
    not content-free, i.e. a big spread.
    """
    starts = np.array([tb.trace_start(i) for i in range(tb.n_traces)])
    porch, ok = decode_mod._region(x, starts, tb.period, decode_mod.PORCH_START,
                                   decode_mod.PORCH_END)
    lvl = np.median(porch[ok], axis=1)
    if len(lvl) < 16:
        return {"porch_pp": float("nan"), "porch_rms": float("nan")}
    med = np.median(lvl)
    return {
        "porch_pp": float(np.percentile(lvl, 99) - np.percentile(lvl, 1)),
        "porch_rms": float(np.sqrt(np.mean((lvl - med) ** 2))),
    }


def frame_edge_check(x: np.ndarray, tb) -> dict:
    """Does the 512-trace window sit inside this frame, or straddle the gap?

    The inter-frame gap is content-free: its traces carry a picture band whose
    std collapses. Report the fraction of the first/last 32 traces whose
    picture-band std is below 25% of the frame median -- a frame start error
    of more than a few traces shows up here and nowhere else.
    """
    starts = np.array([tb.trace_start(i) for i in range(tb.n_traces)])
    pic, ok = decode_mod._region(x, starts, tb.period, decode_mod.PICTURE_START,
                                 decode_mod.PICTURE_START + decode_mod.PICTURE_SPAN)
    s = pic.std(axis=1)
    s[~ok] = np.nan
    med = float(np.nanmedian(s))
    if not np.isfinite(med) or med <= 0:
        return {"blank_head": float("nan"), "blank_tail": float("nan"), "blank_frac": float("nan")}
    low = s < 0.25 * med
    return {
        "blank_head": float(np.nanmean(low[:32])),
        "blank_tail": float(np.nanmean(low[-32:])),
        "blank_frac": float(np.nanmean(low)),
    }


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------


def scan(ids=None, save_png: bool = True) -> list[dict]:
    cat = catalog_mod.build()
    ids = ids or cat.ids()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cur").mkdir(exist_ok=True)
    (OUT / "pre").mkdir(exist_ok=True)
    rows = []
    for fid in ids:
        frame = cat.by_id(fid)
        try:
            x96, tb, cur, pre = decode_both(frame)
        except Exception as exc:  # noqa: BLE001
            rows.append({"id": fid, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{fid} FAILED {exc}", flush=True)
            continue
        mc = quality_mod.frame_report(cur.image, tb, frame_id=fid).metrics
        mp = quality_mod.frame_report(pre.image, tb, frame_id=fid).metrics
        row = {
            "id": fid,
            "title": frame.title,
            "n": frame.image_number,
            "color": frame.color,
            "composite": mc["composite"],
            "composite_pre": mp["composite"],
            "metrics": {k: float(v) for k, v in mc.items()},
            "metrics_pre": {k: float(v) for k, v in mp.items()},
            "sync": {
                "located_frac": float(tb.located.mean()),
                "jitter_rms": float(tb.jitter_rms),
                "meas_noise": float(tb.measurement_noise),
                "parity_offset": float(tb.parity_offset),
                "lock_quality": float(tb.lock_quality),
                "period": float(tb.period),
                "trace0": float(tb.smoothed[0]),
            },
            "decode": {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                           else v)
                       for k, v in cur.diagnostics.items() if k != "channel"},
        }
        row.update({"rails_" + k: v for k, v in rail_clipping(cur.image).items()})
        row.update({"pre_rails_" + k: v for k, v in rail_clipping(pre.image).items()})
        row.update(gate_flatness(x96, tb))
        row.update(frame_edge_check(x96, tb))
        rows.append(row)
        if save_png:
            _png(cur.image, OUT / "cur" / f"{fid}.png")
            _png(pre.image, OUT / "pre" / f"{fid}.png")
        print(f"{fid} comp {mc['composite']:6.2f} (pre {mp['composite']:6.2f}) "
              f"lock {row['decode'].get('dot_locked')} "
              f"str {row['decode'].get('dot_clock_strength')}", flush=True)
    (OUT / "scan.json").write_text(json.dumps(rows, indent=1))
    return rows


def _png(img: np.ndarray, path: Path) -> None:
    from PIL import Image
    arr = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr, "L").save(path)


# --------------------------------------------------------------------------
# parity survey
# --------------------------------------------------------------------------


def parity_survey(ids=None) -> list[dict]:
    """Per-frame sync parity offset, plus the ingredients the estimator uses.

    sync.recover() measures `alt` as mean(residual on even gated traces) minus
    mean(residual on odd gated traces) and then REMOVES half of it from every
    trace start. If the two parities are not equally represented among the
    gated traces, or if one parity's residuals carry an outlier, `alt` picks up
    the difference of two different populations -- which is what breaks L032.
    """
    cat = catalog_mod.build()
    ids = ids or cat.ids()
    rows = []
    for fid in ids:
        frame = cat.by_id(fid)
        x96 = frame_signal(frame)
        tb = sync_mod.recover(x96, period_guess=NOMINAL_96, n_traces=sync_mod.TRACES_PER_FRAME)
        loc = tb.located
        k = np.arange(len(loc))
        line = tb.phase + k * tb.period
        res = tb.positions - line
        ev = loc & (k % 2 == 0)
        od = loc & (k % 2 == 1)
        rows.append({
            "id": fid,
            "parity_offset": float(tb.parity_offset),
            "n_even": int(ev.sum()),
            "n_odd": int(od.sum()),
            "mean_even": float(res[ev].mean()) if ev.any() else float("nan"),
            "mean_odd": float(res[od].mean()) if od.any() else float("nan"),
            "med_even": float(np.median(res[ev])) if ev.any() else float("nan"),
            "med_odd": float(np.median(res[od])) if od.any() else float("nan"),
            # robust version of the same statistic
            "parity_offset_robust": (float(np.median(res[ev]) - np.median(res[od]))
                                     if ev.any() and od.any() else float("nan")),
            "res_std_even": float(res[ev].std()) if ev.any() else float("nan"),
            "res_std_odd": float(res[od].std()) if od.any() else float("nan"),
            "located_frac": float(loc.mean()),
        })
        print(f"{fid} par {rows[-1]['parity_offset']:+8.3f} "
              f"robust {rows[-1]['parity_offset_robust']:+8.3f} "
              f"nE {rows[-1]['n_even']:3d} nO {rows[-1]['n_odd']:3d}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity.json").write_text(json.dumps(rows, indent=1))
    return rows


# --------------------------------------------------------------------------
# dot-clock recovery experiments
# --------------------------------------------------------------------------


def _clock_spectrum(x, tb, lo_f, hi_f, n_traces, search=0.03):
    """Return (freqs, excess, predicted_f) over the search band -- the exact
    quantity dotclock.measure thresholds, exposed for study."""
    R = dot_mod._picture_stack(x, tb, lo_f, hi_f, n_traces)
    n = R.shape[1]
    w = np.hanning(n)
    P = np.mean(np.abs(np.fft.rfft(R * w, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n)
    bg = np.convolve(P, np.ones(41) / 41, mode="same")
    excess = P / np.maximum(bg, 1e-30)
    predicted_f = dot_mod.DOTS_PER_TRACE / tb.period
    band = (f > predicted_f * (1 - search)) & (f < predicted_f * (1 + search))
    return f[band], excess[band], predicted_f


def _peak(fb, eb):
    k = int(np.argmax(eb))
    peak = fb[k]
    if 0 < k < len(eb) - 1:
        y0, y1, y2 = eb[k - 1], eb[k], eb[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            peak = fb[k] + 0.5 * (y0 - y2) / den * (fb[1] - fb[0])
    return float(peak), float(eb[k])


def dotclock_study(ids=None) -> list[dict]:
    """For every frame: the standard measurement, plus recovery experiments.

    full      the shipping measurement over the whole gate (what decode uses)
    halves    the SAME measurement on the first and second half of the traces,
              independently. The cross-validation: a real clock line gives the
              same rate in both halves to well under the 0.043 dots/trace
              record-wide spread; a spectral accident does not.
    longwin   the picture band widened to the whole gate and the trace count
              raised to every available trace (longer averaging baseline).
    """
    cat = catalog_mod.build()
    ids = ids or cat.ids()
    lo_f = decode_mod.PICTURE_START + 0.01
    hi_f = decode_mod.PICTURE_START + decode_mod.PICTURE_SPAN - 0.01
    rows = []
    for fid in ids:
        frame = cat.by_id(fid)
        x96 = frame_signal(frame)
        tb = sync_mod.recover(x96, period_guess=NOMINAL_96, n_traces=sync_mod.TRACES_PER_FRAME)
        clock = dot_mod.measure(x96, tb, lo_f=lo_f, hi_f=hi_f)

        # split-half, same estimator, independent trace populations
        half = {}
        R = dot_mod._picture_stack(x96, tb, lo_f, hi_f, 460)
        n = R.shape[1]
        w = np.hanning(n)
        f = np.fft.rfftfreq(n)
        predicted_f = dot_mod.DOTS_PER_TRACE / tb.period
        band = (f > predicted_f * 0.97) & (f < predicted_f * 1.03)
        for name, sl in (("h1", slice(0, R.shape[0] // 2)), ("h2", slice(R.shape[0] // 2, None))):
            P = np.mean(np.abs(np.fft.rfft(R[sl] * w, axis=1)) ** 2, axis=0)
            bg = np.convolve(P, np.ones(41) / 41, mode="same")
            ex = P / np.maximum(bg, 1e-30)
            pk, st = _peak(f[band], ex[band])
            half[name] = {"dots": float(tb.period * pk), "strength": st}

        rows.append({
            "id": fid,
            "period": float(tb.period),
            "strength": float(clock.strength),
            "measured": bool(clock.measured),
            "dots": float(clock.dots_per_trace),
            "h1_dots": half["h1"]["dots"], "h1_strength": half["h1"]["strength"],
            "h2_dots": half["h2"]["dots"], "h2_strength": half["h2"]["strength"],
            "half_disagree": abs(half["h1"]["dots"] - half["h2"]["dots"]),
        })
        print(f"{fid} str {clock.strength:6.2f} dots {clock.dots_per_trace:8.3f} "
              f"h1 {half['h1']['dots']:8.3f} h2 {half['h2']['dots']:8.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dotclock.json").write_text(json.dumps(rows, indent=1))
    return rows


def _fold_coherence(d: np.ndarray, starts, period, lo_f, hi_f, spds, sl=slice(None)):
    """Matched filter for the plateau grid: |dx| folded at each candidate rate.

    Sample-and-hold means the signal can only change at a plateau edge, so
    |dx| is periodic at the dot rate whatever the picture is. The circular
    first moment's magnitude, normalised by total |dx|, is the evidence; it
    needs no spectral background model and no narrow FFT bin, so a longer
    baseline keeps helping (unlike the per-trace FFT, which is limited by the
    gate length of ~700 samples = 230 dots).
    """
    gated = np.zeros(len(d))
    for i in np.arange(len(starts))[sl]:
        a = int(starts[i] + lo_f * period)
        b = int(starts[i] + hi_f * period)
        if a < 0 or b > len(d) or b <= a:
            continue
        gated[a:b] = d[a:b]
    tot = float(gated.sum()) + 1e-12
    # One zero-padded DFT serves every candidate rate: padding to 2^22 puts the
    # bin spacing 20x inside the main lobe, so nearest-bin lookup is exact for
    # this purpose and the whole sweep costs one FFT.
    n = 1 << 22
    A = np.fft.rfft(gated, n)
    k = np.rint(n / np.asarray(spds, dtype=np.float64)).astype(np.int64)
    k = np.clip(k, 0, len(A) - 1)
    # the +0.5-sample offset used by the per-sample form is a pure phase term
    return np.abs(A[k]) / tot


def column_slope(a: np.ndarray, w: int = 9) -> np.ndarray:
    """Per-column dominant content orientation, dy/dx, from the structure tensor.

    THE POINT. quality.drift_metrics accumulates the column-to-column vertical
    shift and reports its peak-to-peak. But for a picture whose content is
    locally diagonal -- I(x, y) = f(y - m x) -- adjacent columns genuinely DO
    line up at a shift of m, so the accumulated "drift" measures the scene's
    diagonal, not the decoder. Measured on the real decodes, the mean column
    shift equals this estimate frame for frame (L057 +0.54 vs +0.38, R008
    -0.38 vs -0.30, R066 -0.26 vs -0.27, L055 +0.011 vs +0.017), so
    drift_span is currently ranking pictures by how diagonal they are.

    For I = f(y - m x): Ix = -m f', Iy = f', so m = -<Ix Iy> / <Iy^2>.
    """
    b = np.asarray(a, dtype=np.float64)
    b = b - b.mean()
    gy, gx = np.gradient(b)
    k = np.ones(w) / w
    num = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 1, -(gx * gy)).sum(axis=0)
    den = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 1, gy * gy).sum(axis=0)
    return num / np.maximum(den, 1e-12)


def residual_drift(img: np.ndarray, trim: int = 8) -> dict:
    """drift_span with the scene's own diagonal removed -- the honest version.

    Validated the way the metric module demands: it still catches an injected
    staircase (L055 3.7 px clean -> 12.5 px at 24 px/8 steps, 6.4 at 8 px/4)
    and it is NOT inflated by cosmetic sharpening (unsharp x1.5: 3.7 -> 4.2),
    while the frames drift_span called worst collapse to clean values (L011
    127 -> 7.6 px, R000 138 -> 10.0) and the frames the frozen test set labels
    as genuinely damaged rise to the top (L020 -> 28.9, L001 -> 62.8).
    """
    a = np.asarray(img, dtype=np.float64)
    d = quality_mod.column_shifts(a[:, trim:-trim])
    ok = np.isfinite(d)
    if ok.sum() < 8:
        return {"res_drift": float("nan"), "res_rms": float("nan"), "res_jump_frac": float("nan")}
    m = column_slope(a)[trim:-trim][:-1]
    e = d - m
    med = float(np.median(e[ok]))
    res = np.where(ok, e - med, 0.0)
    return {
        "res_drift": float(np.ptp(np.cumsum(res))),
        "res_rms": float(np.sqrt(np.mean((e[ok] - med) ** 2))),
        "res_jump_frac": float(np.mean(np.abs(e[ok] - med) > 1.5)),
    }


MAX_PARITY_DOTS = 1.0  # the field-ID alternation is HALF a dot; a whole dot is
# already 6x the record-wide spread (-1.62 +- 0.03 samples at 96 kHz = 0.53 dot)


def reanchored_timebase(x: np.ndarray, tb, half_window: float = 0.02):
    """Proposed guard, implemented here so it can be measured before it ships.

    Re-detects every landmark on the ZERO-CROSSING grid the timebase was
    finally anchored to, using the per-trace steepest-fall discriminator that
    sync._zero_cross_offset already trusts (geometry.py validated it at
    99.6-100% of traces, L000 included), then re-fits period / wow / parity
    from those detections. Returns (positions, smoothed, parity_offset).

    Why this and not just clamping the parity estimate: on L032 the drop-score
    detector latches a PICTURE edge on one parity, so the bogus parity offset
    is a symptom. Clamping it alone leaves the trend fitted to junk residuals.
    """
    x = np.asarray(x, dtype=np.float64)
    period = tb.period
    n = tb.n_traces
    r = max(3, int(round(half_window * period)))
    k = max(3, int(round(0.003 * period))) | 1
    g = np.convolve(x, np.ones(k) / k, mode="same")
    d = np.diff(g)

    pos = np.full(n, np.nan)
    for i in range(n):
        s = tb.smoothed[i]
        a = int(round(s)) - r
        b = int(round(s)) + r
        if a < 1 or b + 2 >= len(x):
            continue
        j = a + int(np.argmin(d[a:b]))  # steepest fall near the anchored grid
        # refine to the nearest downward zero crossing (fall back to the
        # half-amplitude level where the trough never reaches zero)
        seg = x[max(j - 6, 0): j + 8]
        base = max(j - 6, 0)
        for lvl in (0.0, 0.5 * (float(seg.max()) + float(seg.min()))):
            cr = np.where((seg[:-1] >= lvl) & (seg[1:] < lvl))[0]
            if len(cr):
                jj = cr[np.argmin(np.abs(base + cr - j))]
                pos[i] = base + jj + (seg[jj] - lvl) / (seg[jj] - seg[jj + 1])
                break
    ok = np.isfinite(pos)
    if ok.sum() < 32:
        return None
    kv = np.where(ok)[0].astype(np.float64)
    slope, icpt = sync_mod._robust_line(kv, pos[ok])
    line = icpt + np.arange(n) * slope
    res = pos - line
    par = kv.astype(int) % 2
    alt = float(np.median(res[ok][par == 0]) - np.median(res[ok][par == 1]))
    if abs(alt) > MAX_PARITY_DOTS * period / dot_mod.DOTS_PER_TRACE:
        alt = 0.0
    signed = np.where(np.arange(n) % 2 == 0, 0.5, -0.5) * alt
    res_d = (pos - signed - line)[ok]
    r_all = np.interp(np.arange(n), kv, res_d)
    s = 1.4826 * np.median(np.abs(res_d - np.median(res_d))) + 1e-9
    r_all = np.clip(r_all, np.median(res_d) - 6 * s, np.median(res_d) + 6 * s)
    smoothed = line + sync_mod._savgol(r_all, 31)
    return pos, smoothed, alt, slope, ok


def porch_survey(ids=None) -> list[dict]:
    """Is the back porch really content-free on every frame?

    decode.py clamps every trace on frac 0.020..0.072 and measures the frame's
    absolute black/white anchors there. That window is content-free only while
    the picture's left edge is dark: the geometry gate starts at 0.0725 but
    the converter's video starts earlier, so a slide that is BRIGHT at the very
    left of the raster puts picture inside the clamp window.

    Measured content-free: the profile is aligned on the frame's own smoothed
    grid (median over 360 traces, so scene structure averages out), levels are
    quoted in units of the frame's own sync amplitude, and `pic_start` is the
    first sample after the crossing where the profile leaves the early-porch
    level by more than 15% of that amplitude.
    """
    cat = catalog_mod.build()
    ids = ids or cat.ids()
    rows = []
    for fid in ids:
        frame = cat.by_id(fid)
        x = frame_signal(frame)
        tb = sync_mod.recover(x, period_guess=NOMINAL_96, n_traces=sync_mod.TRACES_PER_FRAME)
        acc = []
        for i in range(40, 400):
            s = int(round(tb.smoothed[i]))
            if s < 0 or s + int(tb.period) > len(x):
                continue
            acc.append(x[s : s + int(tb.period)])
        P = np.median(np.asarray(acc), axis=0)
        p = tb.period
        early = float(np.median(P[int(0.020 * p) : int(0.035 * p)]))
        late = float(np.median(P[int(0.055 * p) : int(0.072 * p)]))
        plateau = float(np.median(P[int(0.99 * p) :]))  # sync high, both parities
        amp = abs(plateau - early) + 1e-9
        thr = 0.15 * amp
        k = int(0.016 * p)
        j = k
        while j < int(0.10 * p) and abs(P[j] - early) < thr:
            j += 1
        rows.append({
            "id": fid,
            "porch_step_frac": (late - early) / amp,
            "pic_start_frac": j / p,
            "amp": amp,
        })
        print(f"{fid} porch step {rows[-1]['porch_step_frac']:+7.3f} amp "
              f"first departure at frac {rows[-1]['pic_start_frac']:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "porch.json").write_text(json.dumps(rows, indent=1))
    return rows


def dotclock_recover(ids=None, band_dots: float = 0.6, n_grid: int = 241) -> list[dict]:
    """Can the clock be recovered on the frames that fail MIN_STRENGTH?

    Two changes, both testable:
      * search only +-`band_dots` dots around 262.5 instead of +-3% (=+-7.9
        dots). The rate is a hardware constant of the 1977 converter; 22
        frames measured it at 262.519 +- 0.043, so a peak 7 dots away is by
        construction not the clock.
      * score the plateau grid with a MATCHED FILTER (|dx| folded at the
        candidate rate over the whole frame) instead of a per-trace FFT bin.

    Every candidate is cross-validated on trace halves, which noise cannot
    game: a real hardware clock gives the same rate on both halves.
    """
    cat = catalog_mod.build()
    ids = ids or cat.ids()
    lo_f = decode_mod.PICTURE_START + 0.01
    hi_f = decode_mod.PICTURE_START + decode_mod.PICTURE_SPAN - 0.01
    rows = []
    for fid in ids:
        frame = cat.by_id(fid)
        x96 = frame_signal(frame)
        tb = sync_mod.recover(x96, period_guess=NOMINAL_96, n_traces=sync_mod.TRACES_PER_FRAME)
        starts = np.array([tb.trace_start(i) for i in range(tb.n_traces)])
        d = np.abs(np.diff(np.asarray(x96, dtype=np.float64)))
        dots_grid = dot_mod.DOTS_PER_TRACE + np.linspace(-band_dots, band_dots, n_grid)
        spds = tb.period / dots_grid

        base = dot_mod.measure(x96, tb, lo_f=lo_f, hi_f=hi_f)
        c_all = _fold_coherence(d, starts, tb.period, lo_f, hi_f, spds)
        c_h1 = _fold_coherence(d, starts, tb.period, lo_f, hi_f, spds, slice(0, 256))
        c_h2 = _fold_coherence(d, starts, tb.period, lo_f, hi_f, spds, slice(256, 512))

        def pk(c):
            k = int(np.argmax(c))
            return float(dots_grid[k]), float(c[k])
        d_all, v_all = pk(c_all)
        d_h1, _ = pk(c_h1)
        d_h2, _ = pk(c_h2)
        # control: the same statistic half a dot away, where nothing should be
        far = np.abs(dots_grid - d_all) > 0.35
        floor = float(np.median(c_all[far])) if far.any() else float("nan")
        rows.append({
            "id": fid,
            "fft_strength": float(base.strength),
            "fft_measured": bool(base.measured),
            "fft_dots": float(base.dots_per_trace),
            "mf_dots": d_all,
            "mf_coh": v_all,
            "mf_floor": floor,
            "mf_excess": v_all / max(floor, 1e-12),
            "mf_h1": d_h1,
            "mf_h2": d_h2,
            "mf_half_disagree": abs(d_h1 - d_h2),
        })
        r = rows[-1]
        print(f"{fid} fft {r['fft_strength']:5.2f} {'LOCK' if r['fft_measured'] else '    '} "
              f"{r['fft_dots']:8.3f} | mf {r['mf_dots']:8.3f} exc {r['mf_excess']:5.2f} "
              f"halves {r['mf_half_disagree']:6.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dotclock_recover.json").write_text(json.dumps(rows, indent=1))
    return rows


# --------------------------------------------------------------------------
# contact sheets
# --------------------------------------------------------------------------


def sheet(ids, path: Path, src=("cur",), cols: int = 5, scale: float = 0.55) -> None:
    """Grid of decoded frames with labels, for visual adjudication."""
    from PIL import Image, ImageDraw
    tiles = []
    for fid in ids:
        ims = []
        for s in src:
            p = OUT / s / f"{fid}.png"
            if p.exists():
                ims.append(Image.open(p).convert("L"))
        if not ims:
            continue
        w = sum(i.width for i in ims) + 4 * (len(ims) - 1)
        h = max(i.height for i in ims)
        strip = Image.new("L", (w, h), 255)
        xo = 0
        for i in ims:
            strip.paste(i, (xo, 0))
            xo += i.width + 4
        strip = strip.resize((int(w * scale), int(h * scale)))
        tiles.append((fid, strip))
    if not tiles:
        return
    tw, th = tiles[0][1].size
    rows = (len(tiles) + cols - 1) // cols
    sheet_img = Image.new("L", (cols * (tw + 6), rows * (th + 16)), 200)
    d = ImageDraw.Draw(sheet_img)
    for k, (fid, t) in enumerate(tiles):
        r, c = divmod(k, cols)
        x, y = c * (tw + 6), r * (th + 16)
        sheet_img.paste(t, (x, y + 14))
        d.text((x + 2, y + 2), fid, fill=0)
    sheet_img.save(path)
    print(f"wrote {path} ({len(tiles)} frames)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["scan", "parity", "dotclock", "dcrecover", "porch", "sheet"])
    ap.add_argument("--frames", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--src", default="cur")
    args = ap.parse_args(argv)
    ids = [s.strip() for s in args.frames.split(",") if s.strip()] or None
    if args.cmd == "scan":
        scan(ids)
    elif args.cmd == "parity":
        parity_survey(ids)
    elif args.cmd == "dotclock":
        dotclock_study(ids)
    elif args.cmd == "porch":
        porch_survey(ids)
    elif args.cmd == "dcrecover":
        dotclock_recover(ids)
    elif args.cmd == "sheet":
        sheet(ids or [], Path(args.out or (OUT / "sheet.png")), tuple(args.src.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
