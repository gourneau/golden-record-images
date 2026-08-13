"""Camera-response measurement and regularised deconvolution, in the dot domain.

WORK IN PROGRESS -- numbers filled in by measurement, not by assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

LOG_K = 2.0 * np.log(9.0)      # logistic: 10-90 width W -> slope 4.394/W
ERF_K = 2.0 * 0.9061938024     # erf: 10-90 width W -> sigma = W/2.5631


def _step_logistic(t, x0, w):
    return 1.0 / (1.0 + np.exp(-np.clip((t - x0) * (LOG_K / w), -50, 50)))


def _step_erf(t, x0, w):
    return 0.5 * (1.0 + erf((t - x0) * (ERF_K / w)))


STEPS = {"logistic": _step_logistic, "erf": _step_erf}


@dataclass
class EdgeFit:
    x0: float          # sub-sample edge position, in units of the input index
    width: float       # 10-90 rise, same units
    amp: float         # step height (signed)
    slope: float       # fitted linear background
    rms: float         # residual rms, signal units
    ok: bool


def fit_edge(y: np.ndarray, model: str = "logistic", with_slope: bool = True,
             wmax: float = 12.0) -> EdgeFit:
    """Fit a parametric step to ONE 1-D profile.

    `y` is a short window containing a single monotone transition. The fit is
    a + b*S((t-x0)/w) [+ c*(t-x0)]; the returned width is the 10-90 rise of S
    in samples of the input index. One profile = one trace (or one dot row),
    so nothing here can be widened by trace-to-trace timing error.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    t = np.arange(n, dtype=np.float64)
    step = STEPS[model]
    k = max(1, n // 5)
    a0 = float(np.mean(y[:k]))
    b0 = float(np.mean(y[-k:])) - a0
    if abs(b0) < 1e-12:
        return EdgeFit(np.nan, np.nan, 0.0, 0.0, np.inf, False)
    d = np.diff(y)
    x00 = float(np.argmax(np.abs(d))) + 0.5

    if with_slope:
        def f(t, a, b, x0, w, c):
            return a + b * step(t, x0, w) + c * (t - x0)
        p0 = [a0, b0, x00, 1.5, 0.0]
        lo = [a0 - 3 * abs(b0), -4 * abs(b0), x00 - 3.0, 0.25, -abs(b0)]
        hi = [a0 + 3 * abs(b0), 4 * abs(b0), x00 + 3.0, wmax, abs(b0)]
    else:
        def f(t, a, b, x0, w):
            return a + b * step(t, x0, w)
        p0 = [a0, b0, x00, 1.5]
        lo = [a0 - 3 * abs(b0), -4 * abs(b0), x00 - 3.0, 0.25]
        hi = [a0 + 3 * abs(b0), 4 * abs(b0), x00 + 3.0, wmax]
    if b0 < 0:
        lo[1], hi[1] = -4 * abs(b0), 4 * abs(b0)
    try:
        p, _ = curve_fit(f, t, y, p0=p0, bounds=(lo, hi), maxfev=4000)
    except Exception:
        return EdgeFit(np.nan, np.nan, 0.0, 0.0, np.inf, False)
    res = y - f(t, *p)
    rms = float(np.sqrt(np.mean(res**2)))
    c = float(p[4]) if with_slope else 0.0
    railed = p[3] > 0.995 * wmax or p[3] < 1.005 * 0.25
    return EdgeFit(float(p[2]), float(p[3]), float(p[1]), c, rms, not railed)


# ---------------------------------------------------------------------------
# candidate selection: isolated, well-formed steps only
# ---------------------------------------------------------------------------


def step_windows(p: np.ndarray, half: int = 6, guard: int = 3,
                 min_step: float = 0.25, max_wobble: float = 0.20,
                 iso: float = 0.25):
    """Indices of isolated single steps in one profile.

    Yields `i` such that the transition sits between p[i] and p[i+1] and
    p[i-half+1 : i+half+1] is a clean window: flat plateaus at both ends,
    no reversal larger than `iso` x the step, and a step of at least
    `min_step` (absolute, in the caller's units).
    """
    p = np.asarray(p, dtype=np.float64)
    d = np.diff(p)
    ad = np.abs(d)
    n = len(p)
    for i in range(half - 1, n - half - 1):
        if ad[i] < ad[max(0, i - 2):i + 3].max() - 1e-15:
            continue  # not the local peak of |gradient|
        w = p[i - half + 1:i + half + 1]
        A, B = w[:guard], w[-guard:]
        step = B.mean() - A.mean()
        if abs(step) < min_step:
            continue
        if max(np.ptp(A), np.ptp(B)) > max_wobble * abs(step):
            continue
        dd = np.diff(w) * np.sign(step)
        if -dd[dd < 0].sum() > iso * abs(step):
            continue
        yield i


def profile_edges(p: np.ndarray, half: int = 6, guard: int = 3,
                  min_step: float = 0.25, model: str = "logistic",
                  with_slope: bool = True, max_rms: float = 0.06):
    """Fit every clean step in one profile. Returns list of (i, EdgeFit)."""
    out = []
    for i in step_windows(p, half, guard, min_step):
        w = p[i - half + 1:i + half + 1]
        f = fit_edge(w, model=model, with_slope=with_slope)
        if not f.ok or not np.isfinite(f.width):
            continue
        # the edge must sit where the window put it, and the model must fit
        if abs(f.x0 - (half - 0.5)) > 1.5:
            continue
        if f.rms > max_rms * abs(f.amp):
            continue
        out.append((i, EdgeFit(f.x0 + i - half + 1, f.width, f.amp, f.slope, f.rms, True)))
    return out


def fit_edge_best(y: np.ndarray, with_slope: bool = True):
    """Fit both step shapes; return (best_model, fit_logistic, fit_erf)."""
    fl = fit_edge(y, "logistic", with_slope)
    fe = fit_edge(y, "erf", with_slope)
    best = "logistic" if fl.rms <= fe.rms else "erf"
    return best, fl, fe


# ---------------------------------------------------------------------------
# survey: every clean edge in a frame, both axes, with local slant
# ---------------------------------------------------------------------------

EDGE_DTYPE = np.dtype([
    ("line", np.int32),     # which trace (axis 0 survey) or dot row (axis 1)
    ("pos", np.float64),    # sub-sample edge position along the profile
    ("width", np.float64),  # 10-90 (logistic), in profile units (dots/traces)
    ("width_erf", np.float64),
    ("amp", np.float64),
    ("rms", np.float64),
    ("rms_erf", np.float64),
    ("slant", np.float64),  # d(pos)/d(line) of the edge, profile units per line
])


def _slants(img: np.ndarray, sigma: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed image gradients (d/d-dot, d/d-trace) for local edge slant."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(img, dtype=np.float64)
    gy = gaussian_filter(a, sigma, order=(1, 0), mode="nearest")
    gx = gaussian_filter(a, sigma, order=(0, 1), mode="nearest")
    return gy, gx


def edge_survey(img: np.ndarray, axis: int, min_step: float = 0.25,
                half: int = 6, guard: int = 3, max_rms: float = 0.06,
                slant_sigma: float = 1.2) -> np.ndarray:
    """Fit every clean single-profile edge of `img` along `axis`.

    axis=0 scans ALONG the trace (profiles are columns of the dot matrix: dot
    index within ONE trace); axis=1 scans ACROSS traces (profiles are dot
    rows). One profile is one trace, or one dot row, so a fitted width cannot
    contain trace-to-trace timing error by construction.

    `slant` is the local orientation of the edge, from the smoothed 2-D
    gradient: a slanted edge is measured through a longer chord and so must
    read wider (width^2 grows as slant^2 x the other axis' width^2). That
    dependence is a check no jitter artefact can imitate; gate on |slant| to
    isolate the axis-aligned response.
    """
    a = np.asarray(img, dtype=np.float64)
    gy, gx = _slants(a, slant_sigma)
    if axis == 1:
        a, gy, gx = a.T, gx.T, gy.T
    n_lines = a.shape[1]
    out = []
    for j in range(n_lines):
        p = a[:, j]
        for i in step_windows(p, half, guard, min_step):
            w = p[i - half + 1:i + half + 1]
            best, fl, fe = fit_edge_best(w)
            f = fl if best == "logistic" else fe
            if not (fl.ok and fe.ok):
                continue
            if abs(f.x0 - (half - 0.5)) > 1.5:
                continue
            if f.rms > max_rms * abs(f.amp):
                continue
            k = int(round(i - half + 1 + f.x0))
            k = min(max(k, 0), a.shape[0] - 1)
            gpar, gperp = gy[k, j], gx[k, j]
            slant = -gperp / gpar if abs(gpar) > 1e-12 else np.nan
            out.append((j, i - half + 1 + f.x0, fl.width, fe.width, f.amp,
                        fl.rms, fe.rms, slant))
    return np.array(out, dtype=EDGE_DTYPE)


# ---------------------------------------------------------------------------
# noise floor, measured from adjacent differences in the flattest blocks
# ---------------------------------------------------------------------------


def measure_noise(img: np.ndarray, block: int = 12, q: float = 0.05) -> dict:
    """White-noise variance per sample, measured separately along each axis.

    Adjacent differences double the noise variance and cancel any constant, so
    var(diff)/2 is the noise variance wherever the picture is locally flat.
    The blocks are ranked by their peak-to-peak content and only the flattest
    `q` are used, so scene structure cannot inflate the estimate. Along-trace
    and across-trace are reported separately: they differ, and the difference
    IS the per-trace (stripe) noise, which only the across-trace filter has to
    fight.
    """
    a = np.asarray(img, dtype=np.float64)
    out = {}
    for axis, key in ((0, "along"), (1, "across")):
        d = np.diff(a, axis=axis)
        H, W = d.shape
        v, p = [], []
        for r in range(0, H - block + 1, block):
            for c in range(0, W - block + 1, block):
                v.append(np.mean(d[r:r + block, c:c + block] ** 2) / 2.0)
                p.append(np.ptp(a[r:r + block, c:c + block]))
        v = np.asarray(v); p = np.asarray(p)
        k = max(1, int(q * len(v)))
        sel = np.argsort(p)[:k]
        out[key] = float(np.median(v[sel]))
        out[key + "_all"] = float(np.median(v))
    out["stripe"] = float(max(out["across"] - out["along"], 0.0))
    return out


# ---------------------------------------------------------------------------
# the operator: a measured Gaussian ESF per axis, and its regularised inverse
# ---------------------------------------------------------------------------

SIGMA_PER_1090 = 1.0 / 2.5631  # erf/Gaussian: 10-90 rise -> sigma


def mtf(f: np.ndarray, width: float) -> np.ndarray:
    """|H| of a Gaussian ESF with 10-90 rise `width`, at `f` cycles/unit."""
    s = width * SIGMA_PER_1090
    return np.exp(-2.0 * np.pi**2 * s**2 * np.asarray(f, dtype=np.float64) ** 2)


def _reflect_pad(a: np.ndarray, axis: int, pad: int) -> np.ndarray:
    w = [(0, 0)] * a.ndim
    w[axis] = (pad, pad)
    return np.pad(a, w, mode="reflect")


def wiener_axis(img: np.ndarray, axis: int, width: float, noise_var: float,
                lam: float = 1.0, gmax: float = 8.0,
                f_stop: float = 0.5, f_taper: float = 0.08) -> tuple[np.ndarray, dict]:
    """Regularised (Wiener) deconvolution of a Gaussian ESF along one axis.

    The signal spectrum is not assumed: it is estimated from the frame's own
    average power spectrum along the axis with the measured noise floor
    subtracted, Ps = (Pobs - Pn)/|H|^2, which gives

        G(f) = (Pobs - Pn) / ( |H| * ((Pobs - Pn) + lam*Pn) )

    -- unit gain where the signal dominates, zero gain where the frame's own
    spectrum has reached the measured noise floor, and `lam` as the single
    conservatism knob (lam = 1 is the statistically matched Wiener filter;
    larger backs off). `gmax` hard-caps noise amplification and `f_stop`
    cuts the filter off entirely above a chosen frequency.
    """
    a = np.asarray(img, dtype=np.float64)
    pad = int(max(8, np.ceil(4 * width)))
    b = _reflect_pad(a, axis, pad)
    n = b.shape[axis]
    f = np.fft.rfftfreq(n)
    X = np.fft.rfft(b, axis=axis)
    P = np.mean(np.abs(X) ** 2, axis=1 - axis) / n          # per-bin power
    H = np.maximum(mtf(f, width), 1e-6)
    Pn = float(noise_var)
    Ps = np.maximum(P - Pn, 1e-12 * max(P.max(), 1e-30))
    G = Ps / (H * (Ps + lam * Pn))
    G = np.minimum(G, gmax)
    G[0] = 1.0
    if f_stop < 0.5:
        G = 1.0 + (G - 1.0) * np.clip((f_stop - f) / max(f_taper, 1e-6) + 1.0, 0.0, 1.0)
    sl = [slice(None)] * a.ndim
    sl[1 - axis] = None
    out = np.fft.irfft(X * G[tuple(sl)], n=n, axis=axis)
    sl2 = [slice(None)] * a.ndim
    sl2[axis] = slice(pad, pad + a.shape[axis])
    info = {"gain_max": float(G.max()), "f_at_gain_max": float(f[int(np.argmax(G))]),
            "noise_gain": float(np.sqrt(np.mean(G**2)))}
    return out[tuple(sl2)], info


# ---------------------------------------------------------------------------
# what the filter does to real edges: the stacked edge profile (ringing)
# ---------------------------------------------------------------------------


def edge_stack(img: np.ndarray, edges: np.ndarray, axis: int, span: int = 10,
               step: float = 0.25) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean normalised edge profile, aligned on each edge's own fitted centre.

    Every edge is scaled so its far plateaus sit at 0 (before) and 1 (after),
    so the stack reads directly as a fraction of edge amplitude: values below 0
    or above 1 away from the transition ARE the ringing, and the far tail is
    "the level behind a bright object versus distance".
    """
    a = np.asarray(img, dtype=np.float64)
    if axis == 1:
        a = a.T
    grid = np.arange(-span, span + step / 2, step)
    acc = np.zeros(len(grid))
    cnt = np.zeros(len(grid))
    n = 0
    for rec in edges:
        j, pos = int(rec["line"]), float(rec["pos"])
        i0 = int(np.floor(pos)) - span
        i1 = i0 + 2 * span + 1
        if i0 < 0 or i1 > a.shape[0] or j >= a.shape[1]:
            continue
        w = a[i0:i1, j].astype(np.float64)
        lo = w[:3].mean()
        hi = w[-3:].mean()
        if abs(hi - lo) < 1e-9:
            continue
        w = (w - lo) / (hi - lo)
        t = np.arange(i0, i1) - pos
        k = np.round((t - grid[0]) / step).astype(int)
        ok = (k >= 0) & (k < len(grid))
        np.add.at(acc, k[ok], w[ok])
        np.add.at(cnt, k[ok], 1)
        n += 1
    m = cnt > max(8, 0.02 * n)
    return grid[m], acc[m] / np.maximum(cnt[m], 1), n


def ringing(grid: np.ndarray, prof: np.ndarray, guard: float = 1.5) -> dict:
    """Peak excursion past the plateaus, as a fraction of edge amplitude."""
    pre = (grid <= -guard)
    post = (grid >= guard)
    under = float(-np.min(prof[pre])) if pre.any() else np.nan
    over = float(np.max(prof[post]) - 1.0) if post.any() else np.nan
    return {"undershoot": under, "overshoot": over,
            "ring": float(np.nanmax([under, over]))}
