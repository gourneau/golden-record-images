"""Objective decode-quality metrics. No reference image required.

Everything downstream of sync/decode changes is judged against these numbers,
so each metric is built to detect one *specific, observed* failure mode and to
be hard to game:

  column-shift drift   The signature failure of a broken timebase is a coherent
                       vertical mis-registration between adjacent traces that
                       accumulates across the frame (the "diagonal staircase").
                       We measure, per adjacent column pair, the vertical shift
                       that maximises normalised cross-correlation. In a
                       correctly-timed frame this is 0 for essentially every
                       pair; timing faults produce sustained non-zero shifts
                       and a large accumulated drift. Sharpening cannot move
                       a correlation peak, so this cannot be inflated by
                       cosmetic filtering.
  staircase ridge      The staircase also leaves a bright ridge in the
                       column-to-column difference field that advances linearly
                       with trace index. We remove row-static structure (so the
                       fixed sync band cannot vote), then Hough-vote the
                       per-column argmax of the difference field over
                       (slope, intercept) lines. The winning line's support
                       fraction is the score; a clean frame has scattered,
                       content-driven argmaxes and low support.
  axis coherence       Timing errors damage across-trace (horizontal) structure
                       only; along-trace (vertical) structure rides on the
                       audio waveform and survives. lag-1 correlation measured
                       both ways; their ratio isolates timing damage from
                       "this image is just soft".
  parity energy        Residual odd/even trace artefacts (hum removal failure,
                       per-parity timing offset) concentrate at the Nyquist
                       frequency *across* columns. We compare spectral power in
                       the top x-frequency band against a reference band just
                       below it, in dB. White noise scores 0; content scores
                       negative; parity striping scores strongly positive.
  sync lock            Fraction of traces whose detected position agrees with
                       the smooth timebase trend to within 0.4% of a line
                       period, straight from the Timebase the decoder used.

The composite score multiplies per-failure factors, so a decoder cannot buy
back a staircase by over-sharpening, and prints in [0, 100].

The reference metric (`circle_metrics`) runs on the decode of L000, the
calibration circle, which is ground truth by design: fit an ellipse to the
ring and report axis ratio (1.000 when the geometry is right), RMS radial
deviation of the ring from the fit (timing error in pixels that sharpening
cannot fake), and centre offset. The detector asserts a minimum inlier count
and angular coverage so it can never silently "fit" four border pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# column-to-column vertical shift (the staircase, measured as a displacement)
# ---------------------------------------------------------------------------


def column_shifts(
    img: np.ndarray,
    max_shift: int = 12,
    min_std: float = 0.01,
    min_corr: float = 0.5,
) -> np.ndarray:
    """Vertical displacement of each column relative to its left neighbour.

    Returns an (n_cols - 1,) array in pixels; NaN where the pair carried too
    little signal or never correlated well enough to trust (flat columns,
    dropouts), and NaN where the correlation peak saturated the search range,
    which marks a mismatch rather than a measurement. Positive means the right
    column's content sits lower.

    Row-static structure (the fixed sync band, residual hum bands) is removed
    first: it is identical in every column, so left in place it anchors every
    correlation at zero shift and *masks* a real staircase.

    The estimate is the argmax over integer shifts of the Pearson correlation
    of the overlapping segments, refined by parabolic interpolation. It keys on
    *where content lines up*, which no monotone tone-mapping or sharpening can
    move.
    """
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape
    if w < 2 or h < 4 * max_shift:
        return np.full(max(w - 1, 0), np.nan)
    a = a - np.median(a, axis=1, keepdims=True)  # silence row-static bands

    A = a[:, :-1]
    B = a[:, 1:]
    shifts = np.arange(-max_shift, max_shift + 1)
    corr = np.full((len(shifts), w - 1), -np.inf)

    for si, s in enumerate(shifts):
        # B[y] ~= A[y - d]  =>  sum_y A[y] * B[y + s] peaks at s = d.
        if s >= 0:
            aa, bb = A[: h - s], B[s:]
        else:
            aa, bb = A[-s:], B[: h + s]
        am = aa - aa.mean(axis=0)
        bm = bb - bb.mean(axis=0)
        num = (am * bm).sum(axis=0)
        den = np.sqrt((am**2).sum(axis=0) * (bm**2).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr[si] = np.where(den > 0, num / den, -np.inf)

    k = np.argmax(corr, axis=0)
    peak = corr[k, np.arange(w - 1)]

    # Parabolic sub-pixel refinement where the peak is interior.
    d = shifts[k].astype(np.float64)
    interior = (k > 0) & (k < len(shifts) - 1)
    ki = k[interior]
    cols = np.arange(w - 1)[interior]
    y0, y1, y2 = corr[ki - 1, cols], corr[ki, cols], corr[ki + 1, cols]
    den = y0 - 2 * y1 + y2
    ok = den != 0
    d[cols[ok]] += 0.5 * (y0[ok] - y2[ok]) / den[ok]

    stds = a.std(axis=0)
    weak = (stds[:-1] < min_std) | (stds[1:] < min_std)
    saturated = np.abs(d) >= max_shift - 0.5
    d[weak | saturated | (peak < min_corr) | ~np.isfinite(peak)] = np.nan
    return d


# Columns trimmed from each side before scoring: the outermost few columns of a
# frame carry sync-adjacent junk from the neighbouring frame, and their bogus
# "shifts" would otherwise dominate drift_span (measured on L000: two saturated
# +12 px detections at columns 4-5 contributed 24 of a 27 px span).
BORDER_TRIM = 8


def drift_metrics(img: np.ndarray, max_shift: int = 12) -> dict:
    """Summarise column_shifts into frame-level numbers.

    shift_rms   RMS inter-column shift over valid pairs, px. Clean: ~0.1.
    drift_span  Peak-to-peak of the accumulated shift, px: how far the frame's
                content wanders vertically end to end. Clean: a few px.
    step_frac   Fraction of pairs displaced by >= 1.5 px -- discrete staircase
                jumps.
    valid_frac  Fraction of pairs that could be measured at all.
    """
    t = BORDER_TRIM
    d = column_shifts(np.asarray(img)[:, t:-t], max_shift=max_shift)
    valid = np.isfinite(d)
    if valid.sum() < 8:
        return {
            "shift_rms": float("nan"),
            "drift_span": float("nan"),
            "step_frac": float("nan"),
            "valid_frac": float(valid.mean()) if len(d) else 0.0,
        }
    dv = d[valid]
    filled = np.where(valid, d, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(filled)])
    return {
        "shift_rms": float(np.sqrt(np.mean(dv**2))),
        "drift_span": float(cum.max() - cum.min()),
        "step_frac": float(np.mean(np.abs(dv) >= 1.5)),
        "valid_frac": float(valid.mean()),
    }


# ---------------------------------------------------------------------------
# staircase ridge detector (the same failure, seen as a discontinuity line)
# ---------------------------------------------------------------------------


def staircase_metrics(
    img: np.ndarray,
    max_slope: float = 0.35,
    n_slopes: int = 71,
    band_px: int = 6,
    min_slope: float = 0.02,
) -> dict:
    """Detect a discontinuity ridge that advances linearly with trace index.

    Column-to-column absolute difference field, row-static structure removed
    (a fixed horizontal band differences identically in every column, so
    subtracting each row's cross-column mean silences it -- otherwise the sync
    band at the frame top would vote as a slope-0 'staircase' on every frame).
    Per column, take the argmax row where its prominence over the column median
    is real, then Hough-vote over lines y = intercept + slope * x. Support is
    the best line's share of the voting columns; `min_slope` (px per column)
    enforces 'advances across the frame' -- >= ~10 px end to end.
    """
    a = np.asarray(img, dtype=np.float64)[:, BORDER_TRIM:-BORDER_TRIM]
    h, w = a.shape
    if w < 32 or h < 32:
        return {"stair_support": float("nan"), "stair_slope": float("nan")}

    D = np.abs(np.diff(a, axis=1))
    D = D - D.mean(axis=1, keepdims=True)  # silence row-static structure
    # Light vertical smoothing so single-pixel noise cannot claim the argmax.
    k = np.ones(5) / 5.0
    D = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, D)

    ys = np.argmax(D, axis=0)
    xs = np.arange(D.shape[1])
    peak = D[ys, xs]
    med = np.median(D, axis=0)
    mad = 1.4826 * np.median(np.abs(D - med[None, :]), axis=0) + 1e-12
    strong = (peak - med) / mad > 4.0
    if strong.sum() < 16:
        return {"stair_support": 0.0, "stair_slope": 0.0}

    xs, ys = xs[strong], ys[strong].astype(np.float64)
    n = len(xs)
    best_support, best_slope = 0.0, 0.0
    for slope in np.linspace(-max_slope, max_slope, n_slopes):
        if abs(slope) < min_slope:
            continue
        icpt = ys - slope * xs
        # Vote into intercept bins of band_px; a ridge fills one bin.
        bins = np.round(icpt / band_px).astype(np.int64)
        counts = np.bincount(bins - bins.min())
        support = counts.max() / n
        if support > best_support:
            best_support, best_slope = float(support), float(slope)
    return {"stair_support": best_support, "stair_slope": best_slope}


# ---------------------------------------------------------------------------
# axis coherence, parity energy
# ---------------------------------------------------------------------------


def axis_coherence(img: np.ndarray) -> dict:
    """lag-1 correlation across traces (h) and along traces (v), plus ratio.

    Timing errors damage h only, so h/v isolates them from overall softness:
    blurring or sharpening the whole image moves h and v together.
    """
    a = np.asarray(img, dtype=np.float64)
    a = a - a.mean()

    def lag1(u: np.ndarray, v: np.ndarray) -> float:
        u = u - u.mean()
        v = v - v.mean()
        den = np.sqrt((u**2).sum() * (v**2).sum())
        return float((u * v).sum() / den) if den > 0 else float("nan")

    h = lag1(a[:, :-1].ravel(), a[:, 1:].ravel())
    v = lag1(a[:-1, :].ravel(), a[1:, :].ravel())
    return {"h_coh": h, "v_coh": v, "coh_ratio": h / v if v else float("nan")}


def parity_metrics(img: np.ndarray) -> dict:
    """Residual odd/even column energy after de-humming, in dB.

    Rowwise power spectrum across columns. Parity artefacts live at the x-axis
    Nyquist; we compare the top 4% of bins against the band just below
    (78-92% of Nyquist) as the noise-floor reference. 0 dB = white noise,
    negative = normal content roll-off, positive = parity striping.
    """
    a = np.asarray(img, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(a, axis=1)) ** 2
    nb = P.shape[1]
    top = P[:, int(0.96 * nb):].mean()
    ref = P[:, int(0.78 * nb): int(0.92 * nb)].mean()
    if ref <= 0 or top <= 0:
        return {"parity_db": float("nan")}
    return {"parity_db": float(10 * np.log10(top / ref))}


# ---------------------------------------------------------------------------
# sync lock quality (from the Timebase the decoder actually used)
# ---------------------------------------------------------------------------


def sync_metrics(timebase, rel_tol: float = 0.00375) -> dict:
    """Fraction of traces whose detection agrees with the smooth trend.

    A detection is 'accepted' when it lies within rel_tol of a line period
    (0.375% = 3 samples at 96 kHz, 12 at 384 kHz) of the smoothed trajectory;
    everything else was effectively coasted over. Also passes through the
    timebase's own jitter and measurement-noise numbers.
    """
    pos = np.asarray(timebase.positions, dtype=np.float64)
    smo = np.asarray(timebase.smoothed, dtype=np.float64)
    n = min(len(pos), len(smo))
    if n == 0:
        return {"accept_frac": 0.0, "jitter_rms": float("nan"), "meas_noise": float("nan")}
    dev = np.abs(pos[:n] - smo[:n])
    thr = rel_tol * float(timebase.period)
    return {
        "accept_frac": float(np.mean(dev <= thr)),
        "jitter_rms": float(timebase.jitter_rms),
        "meas_noise": float(timebase.measurement_noise),
    }


# ---------------------------------------------------------------------------
# sharpness -- reported ONLY to prove the score is not restating it
# ---------------------------------------------------------------------------


def sharpness(img: np.ndarray) -> float:
    """Mean gradient magnitude. A self-flattering metric on purpose: any
    sharpening inflates it. It is printed alongside the composite so drift
    between the two is visible, and it carries zero weight in the score."""
    a = np.asarray(img, dtype=np.float64)
    gy, gx = np.gradient(a)
    return float(np.mean(np.hypot(gx, gy)))


# ---------------------------------------------------------------------------
# the composite
# ---------------------------------------------------------------------------


@dataclass
class FrameScore:
    frame_id: str
    metrics: dict = field(default_factory=dict)

    @property
    def composite(self) -> float:
        return self.metrics.get("composite", float("nan"))


def composite_score(m: dict) -> float:
    """One number in [0, 100]. Multiplicative: each detected failure scales the
    whole score down, so no metric can buy back another's damage.

    Factor calibration (from the baseline run on the real master):
      clean line-art frames measure shift_rms ~0.1-0.3 px, drift_span < 8 px,
      stair_support ~< 0.12 (chance level of the Hough vote), parity_db < 0,
      accept_frac > 0.9. Staircased photographs measure shift_rms 1-4 px,
      drift_span 20-200 px, stair_support 0.3-0.9, accept_frac < 0.5.
    """
    def get(k, default=np.nan):
        v = m.get(k, default)
        return default if v is None else v

    shift_rms = get("shift_rms")
    drift = get("drift_span")
    stair = get("stair_support")
    parity = get("parity_db")
    ratio = get("coh_ratio")
    accept = get("accept_frac")

    f_shift = np.exp(-max(shift_rms - 0.15, 0.0) / 1.0) if np.isfinite(shift_rms) else 0.0
    f_drift = np.exp(-max(drift - 4.0, 0.0) / 30.0) if np.isfinite(drift) else 0.0
    f_stair = 1.0 - 0.9 * min(max(stair - 0.12, 0.0) / 0.6, 1.0) if np.isfinite(stair) else 0.1
    f_parity = 1.0 / (1.0 + max(parity, 0.0) / 6.0) if np.isfinite(parity) else 0.5
    f_coh = min(max(ratio, 0.0), 1.0) ** 2 if np.isfinite(ratio) else 0.0
    f_sync = min(max(accept, 0.0), 1.0) if np.isfinite(accept) else 0.0

    return float(100.0 * f_shift * f_drift * f_stair * f_parity * f_coh * f_sync)


def frame_report(img: np.ndarray, timebase=None, frame_id: str = "") -> FrameScore:
    """All no-reference metrics for one decoded frame image (height, traces)."""
    m: dict = {}
    m.update(drift_metrics(img))
    m.update(staircase_metrics(img))
    m.update(axis_coherence(img))
    m.update(parity_metrics(img))
    if timebase is not None:
        m.update(sync_metrics(timebase))
    else:
        m.update({"accept_frac": float("nan"), "jitter_rms": float("nan"),
                  "meas_noise": float("nan")})
    m["sharpness"] = sharpness(img)
    m["composite"] = composite_score(m)
    return FrameScore(frame_id=frame_id, metrics=m)


# ---------------------------------------------------------------------------
# reference metric: the calibration circle (left channel, frame 0)
# ---------------------------------------------------------------------------


def _fit_ellipse(x: np.ndarray, y: np.ndarray):
    """Fitzgibbon direct least-squares ellipse fit. Returns conic coefficients
    (a, b, c, d, e, f) for ax^2 + bxy + cy^2 + dx + ey + f = 0, or None."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    D1 = np.column_stack([x**2, x * y, y**2])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ T
        C = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=np.float64)
        M = np.linalg.solve(C, M)
        vals, vecs = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    cond = 4 * vecs[0] * vecs[2] - vecs[1] ** 2
    good = np.where(np.isreal(vals) & (cond > 0))[0]
    if len(good) == 0:
        return None
    a1 = np.real(vecs[:, good[0]])
    a2 = T @ a1
    return tuple(np.concatenate([a1, a2]))


def _ellipse_geometry(coef):
    """Conic -> (cx, cy, major, minor, phi). Raises ValueError if degenerate."""
    a, b, c, d, e, f = coef
    den = b * b - 4 * a * c
    if den >= 0:
        raise ValueError("not an ellipse")
    cx = (2 * c * d - b * e) / den
    cy = (2 * a * e - b * d) / den
    # Evaluate the conic at the centre to get the scale term.
    fc = a * cx * cx + b * cx * cy + c * cy * cy + d * cx + e * cy + f
    m = np.array([[a, b / 2], [b / 2, c]])
    vals, vecs = np.linalg.eigh(m)
    axes2 = -fc / vals
    if np.any(axes2 <= 0):
        raise ValueError("degenerate ellipse")
    axes = np.sqrt(axes2)
    order = np.argsort(axes)[::-1]
    major, minor = axes[order]
    vmaj = vecs[:, order[0]]
    phi = float(np.arctan2(vmaj[1], vmaj[0]))
    return float(cx), float(cy), float(major), float(minor), phi


def _radial_residuals(x, y, cx, cy, major, minor, phi):
    """Signed distance of each point from the ellipse along the centre ray."""
    dx, dy = x - cx, y - cy
    ca, sa = np.cos(phi), np.sin(phi)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    r = np.hypot(u, v)
    th = np.arctan2(v, u)
    r_ell = (major * minor) / np.sqrt(
        (minor * np.cos(th)) ** 2 + (major * np.sin(th)) ** 2
    )
    return r - r_ell


def circle_metrics(
    img: np.ndarray,
    n_angles: int = 256,
    margin: int = 10,
    min_inliers: int = 96,
    min_coverage_deg: float = 200.0,
) -> dict:
    """Fit the calibration ring of a decoded L000 and score the geometry.

    Returns:
      ok            fit accepted (enough inliers, enough angular coverage)
      axis_ratio    major/minor; 1.000 when the timebase geometry is right
      radial_rms    RMS radial deviation of ring points from the fit, px --
                    residual timing error that sharpening cannot fake
      center_dx/dy  ellipse centre minus frame centre, px
      inliers, coverage_deg, major, minor

    Robustness: row- and column-static structure is removed first, so the sync
    band and vertical striping cannot pose as the ring; candidate ring points
    are the strongest band-pass ridge per angular bin around a robust centre;
    the fit iterates with outlier trimming; and a minimum inlier count plus
    angular coverage is asserted, because an earlier attempt happily "fitted"
    four frame-border pixels.
    """
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape
    out = {
        "ok": False, "axis_ratio": float("nan"), "radial_rms": float("nan"),
        "center_dx": float("nan"), "center_dy": float("nan"),
        "inliers": 0, "coverage_deg": 0.0,
        "major": float("nan"), "minor": float("nan"), "reason": "",
    }
    if h < 64 or w < 64:
        out["reason"] = "image too small"
        return out

    # Remove static rows/columns: full-width bands and full-height stripes
    # cannot then outvote the ring anywhere.
    b = a - np.median(a, axis=1, keepdims=True)
    b = b - np.median(b, axis=0, keepdims=True)

    # Band-pass tuned to a thin ring: difference of box blurs, sigma ~1.5 vs 5.
    def blur1d(u, k, axis):
        kern = np.ones(k) / k
        return np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), axis, u)

    fine = blur1d(blur1d(b, 3, 0), 3, 1)
    coarse = blur1d(blur1d(b, 11, 0), 11, 1)
    resp = np.abs(fine - coarse)
    resp[:margin, :] = 0
    resp[-margin:, :] = 0
    resp[:, :margin] = 0
    resp[:, -margin:] = 0

    # Robust starting centre and radius from the strongest 1% of response.
    thr = np.percentile(resp, 99.0)
    if thr < 0.005:
        # No band-pass ridge anywhere above the noise: blank frame, or nothing
        # but row/column-static structure, which was removed above. Without
        # this guard a blank image "fits" the numerical dust.
        out["reason"] = f"no ridge response (p99 {thr:.2g})"
        return out
    py, px = np.where(resp >= thr)
    if len(px) < min_inliers:
        out["reason"] = f"only {len(px)} candidate ridge pixels"
        return out
    cx, cy = float(np.median(px)), float(np.median(py))
    # Initial radius from the *mode* of the radius histogram, not the median:
    # a second arc (e.g. picture content wrapped in from the neighbouring
    # window) biases a median, but the dominant ring still owns the tallest
    # histogram bin.
    rad_all = np.hypot(px - cx, py - cy)
    hist, edges = np.histogram(rad_all, bins=np.arange(0, max(h, w), 8))
    r0 = float(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]))
    if not (10 < r0 < 0.9 * max(h, w)):
        out["reason"] = f"implausible initial radius {r0:.1f}"
        return out

    for it in range(5):
        # Per angular bin, the strongest ridge pixel in a radial band.
        ang = np.arctan2(py - cy, px - cx)
        rad = np.hypot(px - cx, py - cy)
        band_lo, band_hi = (0.7, 1.3) if it == 0 else (0.6, 1.5)
        band = (rad > band_lo * r0) & (rad < band_hi * r0)
        if band.sum() < min_inliers:
            out["reason"] = f"only {int(band.sum())} pixels in radial band"
            return out
        bins = ((ang[band] + np.pi) / (2 * np.pi) * n_angles).astype(int) % n_angles
        vals = resp[py[band], px[band]]
        bx, by = px[band], py[band]
        sel_x, sel_y = [], []
        for k in range(n_angles):
            idx = np.where(bins == k)[0]
            if len(idx) == 0:
                continue
            j = idx[np.argmax(vals[idx])]
            sel_x.append(bx[j])
            sel_y.append(by[j])
        sel_x = np.asarray(sel_x, dtype=np.float64)
        sel_y = np.asarray(sel_y, dtype=np.float64)
        if len(sel_x) < min_inliers:
            out["reason"] = f"only {len(sel_x)} angular bins populated"
            return out

        coef = _fit_ellipse(sel_x, sel_y)
        if coef is None:
            out["reason"] = "ellipse fit failed"
            return out
        try:
            cx, cy, major, minor, phi = _ellipse_geometry(coef)
        except ValueError as exc:
            out["reason"] = str(exc)
            return out

        res = _radial_residuals(sel_x, sel_y, cx, cy, major, minor, phi)
        sigma = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        # Trim to whichever is stricter: 3 robust sigma, or the best 75% by
        # |residual|. Under heavy contamination sigma is inflated and the 3sigma
        # gate keeps everything; the percentile gate still peels the worst
        # quarter each round, so the fit converges onto the dominant ring.
        # Never trim below the inlier quorum, though: keep the best
        # min_inliers points and let the reported radial_rms carry whatever
        # contamination is left, rather than failing with no numbers at all.
        cut = min(max(3.0 * sigma, 2.0), float(np.percentile(np.abs(res), 75.0)))
        cut = max(cut, 1.0)
        if len(res) >= min_inliers:
            kth = float(np.partition(np.abs(res), min_inliers - 1)[min_inliers - 1])
            cut = max(cut, kth)
        keep = np.abs(res) <= cut
        if keep.sum() < min_inliers:
            out["inliers"] = int(keep.sum())
            out["reason"] = f"only {int(keep.sum())} inliers after trimming"
            return out
        sel_x, sel_y = sel_x[keep], sel_y[keep]
        coef = _fit_ellipse(sel_x, sel_y)
        if coef is None:
            out["reason"] = "ellipse refit failed"
            return out
        try:
            cx, cy, major, minor, phi = _ellipse_geometry(coef)
        except ValueError as exc:
            out["reason"] = str(exc)
            return out
        r0 = 0.5 * (major + minor)

    res = _radial_residuals(sel_x, sel_y, cx, cy, major, minor, phi)
    ang = np.arctan2(sel_y - cy, sel_x - cx)
    bins = np.unique(((ang + np.pi) / (2 * np.pi) * 72).astype(int) % 72)
    coverage = float(len(bins) / 72.0 * 360.0)

    out.update(
        ok=bool(len(sel_x) >= min_inliers and coverage >= min_coverage_deg),
        axis_ratio=float(major / minor),
        radial_rms=float(np.sqrt(np.mean(res**2))),
        center_dx=float(cx - w / 2.0),
        center_dy=float(cy - h / 2.0),
        inliers=int(len(sel_x)),
        coverage_deg=coverage,
        major=float(major),
        minor=float(minor),
        reason="" if len(sel_x) >= min_inliers and coverage >= min_coverage_deg
        else f"inliers={len(sel_x)} coverage={coverage:.0f}deg",
    )
    return out


# ---------------------------------------------------------------------------
# perturbations, used only to validate the metric itself
# ---------------------------------------------------------------------------


def inject_staircase(img: np.ndarray, total_px: float = 24.0, n_steps: int = 8) -> np.ndarray:
    """Vertically shift columns in n_steps discrete jumps totalling total_px --
    a synthetic copy of the observed failure. A trustworthy metric must crash
    on this while sharpness barely moves."""
    a = np.asarray(img, dtype=np.float64).copy()
    h, w = a.shape
    edges = np.linspace(0, w, n_steps + 1).astype(int)
    for i in range(n_steps):
        s = int(round(total_px * (i + 1) / n_steps))
        a[:, edges[i]: edges[i + 1]] = np.roll(a[:, edges[i]: edges[i + 1]], s, axis=0)
    return a


def inject_parity(img: np.ndarray, amp: float = 0.05) -> np.ndarray:
    """Add an alternating per-column offset -- synthetic residual hum."""
    a = np.asarray(img, dtype=np.float64).copy()
    a[:, 0::2] += amp
    a[:, 1::2] -= amp
    return a


def unsharp(img: np.ndarray, amount: float = 1.5, k: int = 7) -> np.ndarray:
    """Cosmetic sharpening. A trustworthy metric must NOT reward this."""
    a = np.asarray(img, dtype=np.float64)
    kern = np.ones(k) / k
    lo = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0, a)
    lo = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 1, lo)
    return np.clip(a + amount * (a - lo), 0.0, 1.0)
