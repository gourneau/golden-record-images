"""The half-dot quincunx: is the 262.5-dot line period a free 2x vertical dither?

THE CLAIM UNDER TEST
--------------------
262.5 dots per trace is the NTSC half-line, so if the line period is P and one
dot is spd = P/262.5, then P mod spd = 0.5*spd EXACTLY: the scan converter's
sample-and-hold grid slips half a dot every trace. Even traces therefore
sample the scene at along-trace positions offset by half a dot from odd
traces, and the picture is not a uniform grid at all but a quincunx --
a built-in 2x vertical sampling dither the 1977 hardware hands us for free.
If that is real, a joint non-uniform reconstruction should recover vertical
detail that no per-column interpolator can.

WHAT IS MEASURED HERE (see the CLI subcommands; every number from the master)
----------------------------------------------------------------------------
`phase`    The alternation exists, and it is exactly half a dot. Plateau
           transition phase measured per trace on a TRACE-RELATIVE axis
           (complex moment of |dx| at the dot rate, blocks of 32 traces, no
           absolute-grid assumption anywhere): even-minus-odd = 0.4966..0.5086
           dots on 8 frames, circular sd 0.022-0.042. So the geometry is real.

`parity`   The alternation is in the SCENE, not just in the plateau edges: the
           even/odd mean-profile difference that decode.py's dehum subtracts as
           "hum" is predicted by 0.5 x d(mean profile)/dy.

`holdout`  The decisive test. Hold out every 8th trace, predict its plateau
           samples from the retained traces with ONE estimator (weighted local
           2-D polynomial), run three ways: dy ignoring the offsets, dy using
           the measured offsets, dy using SHUFFLED offsets. Same window, same
           weights, same order -- the only difference is the offset model.

`mtf`      Prediction-error spectrum vs vertical frequency on held-out data:
           where in the band, if anywhere, does the offset model actually buy
           something. This is a recovered-MTF statement a wrong answer cannot
           game, unlike the crispness of a reconstructed image.

Honest by construction: the only data input is the WAV. Sample positions come
from sync.py + dotclock.py, i.e. from the signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import catalog as catalog_mod
from . import decode as decode_mod
from . import dotclock as dot_mod
from . import sync as sync_mod
from . import wav

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "data" / "master" / "384kHzStereo.wav"
PRE_384 = int(round(1.5 * sync_mod.NOMINAL_PERIOD))


# --------------------------------------------------------------------------
# the sample set
# --------------------------------------------------------------------------


@dataclass
class DotField:
    """One frame's plateau samples with their true along-trace positions.

    `v[i, k]` is the exact integral of the plateau assigned to output row k of
    trace i. `y[i, k] = k + delta[i]` is where that plateau's CENTRE sits, in
    dots, on the common relative-time grid every trace shares (relative time
    measured from the trace's own uniform-grid start; the picture content sits
    on that grid -- see sync.py). delta alternates ~0.5 with trace parity and
    drifts across the frame; that alternation is the whole point.
    """

    v: np.ndarray  # (n_traces, n_dots) plateau integrals, porch-clamped
    delta: np.ndarray  # (n_traces,) dots, in (-0.5, +0.5]
    spd: float  # samples per dot
    period: float
    amp: float  # sync amplitude (signal units) -- the intensity yardstick
    dropouts: np.ndarray  # (n_traces,) bool
    strength: float  # dot-clock spectral excess
    coherence: float  # phase-tracker coherence

    @property
    def n_traces(self) -> int:
        return self.v.shape[0]

    @property
    def n_dots(self) -> int:
        return self.v.shape[1]


def load_frame(fid: str, cat=None, decim: int = 1) -> tuple:
    """Read one frame from the 384 kHz master and recover its timebase."""
    cat = cat or catalog_mod.build()
    fr = cat.by_id(fid)
    info = wav.probe(MASTER)
    span = int(math.ceil((sync_mod.TRACES_PER_FRAME + 6) * sync_mod.NOMINAL_PERIOD))
    x = np.asarray(
        wav.read(info, fr.channel, fr.seed_sample - PRE_384, PRE_384 + span),
        dtype=np.float64,
    )
    if decim > 1:
        from scipy.signal import resample_poly

        x = resample_poly(x, 1, decim)
    tb = sync_mod.recover(
        x,
        period_guess=sync_mod.NOMINAL_PERIOD / decim,
        n_traces=sync_mod.TRACES_PER_FRAME,
    )
    return fr, x, tb


def dot_field(
    fid: str,
    cat=None,
    *,
    decim: int = 1,
    uncouple: bool = True,
    dehum: bool = False,
    clamp: bool = True,
) -> DotField:
    """Plateau samples + their true positions, the decode.py way.

    Mirrors decode.decode()'s dot-locked branch exactly (undroop in raster
    order on the raw signal, per-parity running-median porch clamp, plateau
    integration on the tracked phase) but stops before `_rows_from_dots`,
    which is the step that throws the non-uniform positions away.

    `dehum` defaults OFF here: decode's odd/even median subtraction forces the
    two parities' mean profiles to be equal, which is exactly what the half-dot
    offset makes them NOT be (see the `parity` subcommand).
    """
    fr, x, tb = load_frame(fid, cat, decim=decim)
    period = tb.period
    n_tr = min(sync_mod.TRACES_PER_FRAME, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])

    tau = 0.0
    if uncouple:
        ch = fr.id[0]
        tau = decode_mod.UNCOUPLE_TAU_384[ch] * period / (sync_mod.NOMINAL_PERIOD / decim)
        x = decode_mod.undroop(x, tau)

    ps, span_f = decode_mod.PICTURE_START, decode_mod.PICTURE_SPAN
    lo_f, hi_f = ps + 0.01, ps + span_f - 0.01
    clock = dot_mod.measure(x, tb, lo_f=lo_f, hi_f=hi_f)
    spd = clock.samples_per_dot
    span = span_f * period
    n_dots = int(round(span / spd))
    psi, coherence = dot_mod.track_phase(x, starts, period, spd, ps, ps + span_f)

    cs = np.concatenate([[0.0], np.cumsum(x)])
    v = np.empty((n_tr, n_dots))
    delta = np.empty(n_tr)
    for i in range(n_tr):
        mid = starts[i] + (ps + span_f / 2.0) * period
        v[i] = dot_mod.sample_plateaus(x, mid, n_dots, spd, psi[i], cs=cs)
        c0 = mid - (n_dots - 1) / 2.0 * spd
        k0 = np.floor((c0 - psi[i]) / spd)
        delta[i] = (psi[i] + (k0 + 0.5) * spd - c0) / spd

    porch, porch_ok = decode_mod._region(
        x, starts, period, decode_mod.PORCH_START, decode_mod.PORCH_END
    )
    porch_level = np.median(porch, axis=1)
    med = np.median(porch_level)
    mad = 1.4826 * np.median(np.abs(porch_level - med)) + 1e-9
    dropouts = (np.abs(porch_level - med) > 6 * mad) | (~porch_ok)

    if clamp:
        lvl = porch_level.copy()
        for p in (0, 1):
            s = porch_level[p::2]
            pad = np.pad(s, 3, mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(pad, 7)
            lvl[p::2] = np.median(win, axis=1)
        v = v - lvl[:, None]

    if dehum:
        profile, _ = decode_mod.measure_hum(v)
        v[1::2] -= profile
        v[0::2] += profile

    ref = decode_mod.measure_levels(x, starts, period)
    amp = float(ref[1]) if ref is not None else float(v.std())

    return DotField(
        v=v,
        delta=delta,
        spd=float(spd),
        period=float(period),
        amp=amp,
        dropouts=dropouts,
        strength=float(clock.strength),
        coherence=float(coherence),
    )


# --------------------------------------------------------------------------
# step 1: does the dot phase really alternate half a dot?
# --------------------------------------------------------------------------


def cross_spectrum_shift(V, delta, ok, lag=1, trim=8, fmin=0.02, fmax=0.45):
    """Along-trace shift between column pairs, measured on the PICTURE.

    Returns (groups, table) where each group is a set of pairs sharing the
    sign of the dot clock's predicted offset. The estimator is the phase slope
    of the pair-averaged cross-spectrum: no resampling kernel touches the data,
    so there is no half-integer interpolation-loss bias (a Lanczos-shift
    residual search has one, and it pulls minima toward 0 -- exactly the
    direction that would hide this effect).

    Convention: `shift` is the measured Delta with v[j+lag](k) = v[j](k+Delta),
    which the quincunx model predicts equals delta[j+lag] - delta[j].
    """
    Vt = V[:, trim:-trim]
    n = Vt.shape[1]
    w = np.hanning(n)
    F = np.fft.rfft((Vt - Vt.mean(axis=1, keepdims=True)) * w, axis=1)
    f = np.fft.rfftfreq(n)
    A = np.arange(V.shape[0] - lag)
    good = ok[A] & ok[A + lag]
    pred = delta[A + lag] - delta[A]  # NOT wrapped: delta already in (-.5,.5]
    C = F[A + lag] * np.conj(F[A])
    P = np.abs(F[A]) ** 2 + np.abs(F[A + lag]) ** 2
    band = (f > fmin) & (f < fmax)
    out = []
    for name, sel in (("pred>0", good & (pred > 0.15)), ("pred<0", good & (pred < -0.15))):
        if sel.sum() < 20:
            continue
        Cm = C[sel].mean(axis=0)
        Pm = P[sel].mean(axis=0)
        coh = np.abs(Cm) / (0.5 * Pm + 1e-30)
        ph = np.unwrap(np.angle(Cm[band]))
        wt = np.abs(Cm[band])
        fb = f[band]
        s = float((wt * ph * fb).sum() / max((wt * fb * fb).sum(), 1e-30)) / (2 * np.pi)
        out.append(
            dict(group=name, n=int(sel.sum()), pred=float(pred[sel].mean()), shift=s,
                 coh=float(np.average(coh[band], weights=wt)), f=f, coh_f=coh, Cm=Cm)
        )
    return out


def phase_by_parity(x, tb, spd, lo_f, hi_f, block=32, n_tr=512):
    """Plateau-transition phase of even and odd traces, TRACE-RELATIVE.

    Sample-and-hold means |dx| concentrates on plateau boundaries. Fold it at
    the dot rate on an axis measured from each trace's own start, accumulating
    even and odd traces into separate complex moments, in blocks of `block`
    traces so the frame's phase drift cannot smear the comparison. Returns a
    list of (block centre, z_even, z_odd, |dx|_even, |dx|_odd).

    Nothing here assumes an absolute dot grid: that assumption is what would
    make the answer 0.5 by construction.
    """
    d = np.abs(np.diff(np.asarray(x, dtype=np.float64)))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])
    P = tb.period
    out = []
    for b0 in range(0, n_tr - block + 1, block):
        zs = [0j, 0j]
        ts = [0.0, 0.0]
        for i in range(b0, b0 + block):
            s = starts[i]
            a, b = int(s + lo_f * P), int(s + hi_f * P)
            if a < 0 or b > len(d):
                continue
            t = np.arange(a, b) + 0.5 - s
            seg = d[a:b]
            p = i & 1
            zs[p] += np.sum(seg * np.exp(2j * np.pi * t / spd))
            ts[p] += float(seg.sum())
        out.append((b0 + block / 2.0, zs[0], zs[1], ts[0], ts[1]))
    return out


# --------------------------------------------------------------------------
# the hold-out: does using the measured offsets beat ignoring them?
# --------------------------------------------------------------------------


def _mls_weights(dx, dy, deg=3, sigx=1.6, sigy=1.6, ridge=1e-8):
    """Linear predictor at (0,0) from samples at offsets (dx, dy).

    Weighted local polynomial (moving least squares): Gaussian weights, all
    monomials dx^a dy^b with a+b <= deg. The returned row w satisfies
    prediction = w @ values, so the SAME estimator can be run with different
    dy -- which is the whole experiment: only the offset model changes.
    """
    terms = [(a, b) for d in range(deg + 1) for a in range(d + 1) for b in [d - a]]
    B = np.stack([dx ** a * dy ** b for a, b in terms], axis=1)
    wt = np.exp(-0.5 * ((dx / sigx) ** 2 + (dy / sigy) ** 2))
    BtW = B.T * wt
    M = BtW @ B + ridge * np.eye(B.shape[1])
    e0 = np.zeros(B.shape[1])
    e0[terms.index((0, 0))] = 1.0
    return np.linalg.solve(M, e0) @ BtW


def _lanczos_shift_rows(V, s, a=4):
    """Resample every row of V at k + s[row] (out[j,k] = V[j, k+s[j]])."""
    V = np.atleast_2d(V)
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = V.shape[1]
    u = np.arange(n)[None, :] + s[:, None]  # (rows, n)
    base = np.floor(u).astype(np.int64)
    idx = base[:, :, None] + np.arange(-a + 1, a + 1)[None, None, :]
    t = u[:, :, None] - idx
    w = np.sinc(t) * np.sinc(t / a)
    w[np.abs(t) >= a] = 0.0
    w /= np.maximum(w.sum(axis=2, keepdims=True), 1e-12)
    np.clip(idx, 0, n - 1, out=idx)
    rows = np.arange(V.shape[0])[:, None, None]
    return (V[rows, idx] * w).sum(axis=2)


METHODS = ("naive", "true", "shuffled", "separable")


def holdout(
    df: DotField,
    *,
    stride: int = 8,
    offset: int = 0,
    Wx: int = 3,
    Wy: int = 3,
    deg: int = 3,
    sigx: float = 1.6,
    sigy: float = 1.6,
    seed: int = 0,
    methods=METHODS,
) -> dict:
    """Hold out every `stride`-th trace; predict its plateau samples.

    All four arms see exactly the same retained samples and the same target.
    `naive` and `true` are the SAME estimator with the same window, weights
    and polynomial order -- they differ only in whether dy carries the
    measured half-dot offsets. `shuffled` uses offsets drawn from the same
    frame but assigned to the wrong traces: the control that a merely
    "more flexible" model cannot pass. `separable` is what decode.py does
    today (per-column Lanczos onto a common grid, then interpolate).

    Returns per-method rms error (in sync-amplitude units) plus the error
    spectrum against vertical frequency.
    """
    V, delta, ok = df.v, df.delta, ~df.dropouts
    n_tr, n_dots = V.shape
    rng = np.random.default_rng(seed)
    shuf = delta[rng.permutation(n_tr)]

    held = [i for i in range(Wx, n_tr - Wx) if i % stride == offset and ok[i]]
    ks = np.arange(Wy, n_dots - Wy)
    dks = np.arange(-Wy, Wy + 1)

    # `separable`: every column resampled onto the common grid once, the way
    # decode._rows_from_dots does it (per-column Lanczos at its own delta).
    U = _lanczos_shift_rows(V, -delta)
    Ublk = np.stack([U[:, ks + dk] for dk in dks], axis=2)  # (n_tr, nk, ndk)

    err = {m: [] for m in methods}
    truth = []
    for i in held:
        js = np.array([j for j in range(i - Wx, i + Wx + 1) if j != i and ok[j]])
        if len(js) < 4:
            continue
        DX = np.repeat(js - i, len(dks)).astype(float)
        DK = np.tile(dks, len(js)).astype(float)
        block = np.stack(
            [V[j, ks[:, None] + dks[None, :]] for j in js], axis=1
        ).reshape(len(ks), -1)  # (nk, njs*ndk)
        tgt = V[i, ks]
        truth.append(tgt)
        for m in methods:
            if m == "separable":
                # same estimator, same window -- but the offsets are handled by
                # per-column band-limited resampling instead of jointly, so the
                # 2-D fit runs on the common grid (dy = dk) and the answer is
                # shifted back to where this trace actually sampled.
                w = _mls_weights(DX, DK, deg=deg, sigx=sigx, sigy=sigy)
                est = Ublk[js][:, :, :].transpose(1, 0, 2).reshape(len(ks), -1) @ w
                pred = _lanczos_shift_rows(est[None, :], np.array([delta[i]]))[0]
                err[m].append(pred - tgt)
                continue
            if m == "naive":
                DY = DK.copy()
            elif m == "true":
                DY = DK + np.repeat(delta[js] - delta[i], len(dks))
            else:
                DY = DK + np.repeat(shuf[js] - shuf[i], len(dks))
            w = _mls_weights(DX, DY, deg=deg, sigx=sigx, sigy=sigy)
            err[m].append(block @ w - tgt)

    truth = np.concatenate(truth)
    out = {
        "n_held": len(err[methods[0]]),
        "n_samples": int(truth.size),
        "truth_rms": float(np.sqrt(np.mean(truth ** 2)) / df.amp),
    }
    for m in methods:
        e = np.concatenate(err[m])
        out[m] = float(np.sqrt(np.mean(e ** 2)) / df.amp)
        out[m + "_err"] = np.stack(err[m])
    return out
