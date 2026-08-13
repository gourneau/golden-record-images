"""One linear operator for the whole chain, and a regularised inverse of it.

WHY
---
pipeline/decode.py corrects the recording in SEQUENCE: invert the one-pole
high-pass on the raw signal, clamp each trace on its porch, subtract the
parity-locked hum, integrate each sample-and-hold plateau, Lanczos the dot rows
onto square pixels.  Every stage is linear, and each one inherits whatever the
previous stage got wrong (most visibly: the exact pole inverse integrates the
ADC's noise into a random walk, which the porch clamp then has to fight).

The forward model is now known well enough to write down in one piece:

    scene x  ->  camera PSF K  ->  sample-and-hold S  ->  recording-path blur R
             ->  one-pole high-pass H  ->  y (the samples in the WAV)

all linear, so y = A x + n with A = H R S K, and the principled estimate is

    xhat = argmin || W (A z - y) ||^2 + lambda R(z)

with W a data weight (used for hold-out), z = (scene, blanking, hum) and R
either Tikhonov (squared gradient) or total variation.  This module builds A
matrix-free with an exact adjoint, checks the adjoint by dot product, solves
with LSMR, and -- the only part that matters -- compares the result against the
sequential decoder on criteria a nicer-looking wrong answer cannot game.

WHAT IS IN A, and where each number came from (all quoted, none re-derived)
--------------------------------------------------------------------------
  timebase        sync.recover(): per-trace starts, fitted period, parity term
                  removed (the picture sits on a uniform grid).
  dot clock       dotclock.measure()/track_phase(): samples per plateau and the
                  per-trace absolute plateau phase psi.  262.5 plateaus/trace.
  picture gate    decode.PICTURE_START .. +PICTURE_SPAN (0.0725..0.9500).
  camera PSF      Gaussian, 10-90 rise CAM_ROWS dots along the trace and
                  CAM_TRACES traces across.  The along-trace default comes from
                  the project's measured full-chain edge (15.3 samples at
                  384 kHz ~ 1.26 plateaus); the across-trace one is NOT
                  measured on this master, so it is a free knob and is chosen
                  by HOLD-OUT prediction error, never by how the picture looks.
  recording blur  Gaussian, 10-90 rise 2.0 samples at 384 kHz (the measured
                  single-trace edge rise of the recording path, 1.75-2.37).
  high-pass       H(z) = (1 - z^-1)/(1 - a z^-1), a = exp(-1/tau),
                  tau = decode.UNCOUPLE_TAU_384[channel] scaled by the measured
                  period.  Applied in RASTER order over the whole window,
                  straight across trace boundaries, exactly as droop.py
                  measured it.
  blanking        every sample outside the picture gates is a FREE UNKNOWN, one
                  per sample.  It is not scene, we do not model sync pulses,
                  and it must be in the state because the high-pass carries the
                  sync burst into the next trace's picture.  Being exactly
                  determined by its own samples it costs nothing statistically;
                  it is what lets the porch reference emerge from the fit
                  instead of being clamped on afterwards.
  hum             optional per-row profile added with alternating trace sign,
                  i.e. the same fixed pattern decode.measure_hum() removes,
                  estimated jointly instead of afterwards.

Absolute DC is unknowable (the chain blocked it) so A has a near-null constant
direction; it is damped, and the output is re-anchored to the porch and to the
sync amplitude with decode.measure_levels(), the same anchors the sequential
decoder uses, so the two images are on the same intensity scale.

NO REFERENCE IMAGE ENTERS ANY OF THIS.  The only inputs are the WAV, the
timebase and the constants above.

WHAT IT ACTUALLY MEASURED -- A NULL RESULT (2026-08)
----------------------------------------------------
Everything below was run on the real master at 384 kHz, 512 traces, ~230
plateaus per trace: 322,500 unknowns against 1,637,112 equations.

1. THE ADJOINT IS EXACT.  <Ax,r> vs <x,A^T r> agrees to 2e-13 relative on all
   four operator variants (psf on/off x hum on/off), at 96 and at 384 kHz.

2. THE SEQUENTIAL CHAIN IS ALREADY THE ML SOLUTION.  With lambda = 0 the MAP
   solve reproduces decode.py's plateau values to 2.4e-4 rms against a signal
   std of 3.9e-2 -- 0.6% -- and the entire remaining difference is a per-trace
   DC drift (removing each trace's mean takes 1.26e-2 rms down to 2.4e-4),
   i.e. exactly the direction the high-pass made unobservable and the porch
   clamp pins anyway.  Sequential correction is NOT costing accuracy here:
   undroop-then-average-each-plateau IS the maximum-likelihood estimator of
   this forward model.  This also means the camera deconvolution is NOT a
   joint-estimation problem: with lambda = 0 the fitted plateau values are
   bit-for-bit identical whether the camera PSF is in A or not (residual
   1.9426e-3 either way), because the PSF is invertible and adds no fit power.
   Deblurring is decided entirely by the prior, so it belongs where
   pipeline/deconv.py already puts it.

3. HOLD-OUT (whole traces, the only split with no shortcut).  Drop every 8th
   trace entirely and predict its plateau values back:

                                            L055       R040
     neighbour mean, UNREGISTERED         0.00487    0.00892
     neighbour mean, REGISTERED           0.00386    0.00707   <- best
     MAP tikhonov, best at full data fit  0.00546    0.00745
     MAP TV, best at full data fit        0.00535    0.00773
     MAP, sub-plateau offsets discarded   0.00609    0.00920

   The MAP loses, on both frames, to averaging the two neighbouring traces
   after a Lanczos shift onto the held trace's measured sub-plateau offsets.
   It can be made to win -- TV at lambda 0.1, anisotropy 3 reaches 0.00358 on
   L055 -- but only by degrading the fit to the RETAINED samples from 1.82e-3
   to 3.23e-3, i.e. by blurring across traces, which is what this particular
   metric rewards.  Read hold-out and training residual together or this test
   lies to you.

4. A GENUINE POSITIVE, and it is about the record, not about this module:
   registering the sub-plateau offsets beats ignoring them by 21% rms on BOTH
   frames (0.00386 vs 0.00487; 0.00707 vs 0.00892), with a mean offset of
   0.496 / 0.501 plateaus to the neighbouring trace.  That is an independent,
   ungameable confirmation of decode.py's `delta` correction, from prediction
   of data the estimate never saw.

5. IMAGE-DOMAIN CRITERIA: no regression, no gain.  Composite (quality.py of
   2026-08-12 19:38), sequential vs MAP-tikhonov vs MAP-TV:
   L000 89.2 / 88.9 / 90.3, L055 90.4 / 91.0 / 91.7, L020 38.1 / 37.5 / 41.7,
   R040 78.6 / 79.2 / 80.3.  L000's circle: axis ratio 1.0065 -> 1.0066 /
   1.0059, radial rms 0.89 -> 0.87 / 0.86 px, no regression.  L000's flat
   field, ring interior, residual after the source's own linear shading:
   0.0017 -> 0.0018 / 0.0017, unchanged.  TV's advantage is 1-4 composite
   points and it comes with more flat area (L000 6.05% vs 5.99% of pixels
   below a 1e-3 gradient), which is the cartoon effect starting, not detail.

6. COST.  8-11 s per frame per solve on this machine (Tikhonov 250 LSMR
   iterations; TV 8 ADMM x 32 LSMR), against 0.1 s for decode.decode -- 80-100x.
   Not worth offering as a "best quality" mode, because it is not better.

TRAPS THIS MODULE FELL INTO, recorded so nobody repeats them
------------------------------------------------------------
  * UNPRECONDITIONED LSMR.  A scene column whose plateau is held out is held
    up only by the prior (norm ~lambda) while a column with data has norm
    ~sqrt(samples per plateau); 10^4 in the column norms is 10^8 in the normal
    equations.  The first hold-out run looked like a substantive MAP defeat
    and was mostly non-convergence -- doubling the iteration count moved the
    number 15% and NOT monotonically.  column_scaling() fixes it; every number
    above passed a 2x-iteration guard.
  * READING A HOLD-OUT WITHOUT ITS TRAINING RESIDUAL (point 3).
  * A CONDITION NUMBER MEASURED ON THE BOUNDARY.  The per-trace row-resample
    operator looks like cond 5e3, which suggested registration was hopeless;
    cropping 10 rows off each end gives 1.05-35.  It was the zero-padded edge.
  * THE HUM BLOCK IS DEGENERATE: see scene_to_image.

RESULTS ABOVE ARE REPRODUCED BY
    python -m pipeline.mapsolve --frame L055 --adjoint
    python -m pipeline.mapsolve --frame L055 --holdout --solve tik --lam 0.03
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import lfilter, resample_poly
from scipy.sparse.linalg import LinearOperator, lsmr

from . import catalog as catalog_mod
from . import decode as decode_mod
from . import dotclock as dot_mod
from . import sync as sync_mod
from . import wav

MASTER = catalog_mod.DATA_DIR / "master" / "384kHzStereo.wav"

# 10-90 rise of the camera's along-trace (plateau-to-plateau) edge response, in
# plateaus.  15.3 samples at 384 kHz / 12.18 samples per plateau.
CAM_ROWS = 1.26
# Across traces the camera response has never been measured jitter-free on this
# master.  Default = no blur; set_psf() takes it and the hold-out judges it.
CAM_TRACES = 0.0
# Recording-path 10-90 rise, samples at 384 kHz (decode.measure_psf's range).
REC_RISE_384 = 2.0
# 10-90 rise -> Gaussian sigma.
SIGMA_PER_1090 = 1.0 / 2.5631
# One plateau of trace-time is this many trace widths (geometry.py: 7.4406 bins
# per trace, 3200/262.5 = 12.19 bins per plateau).  Isotropic priors need it.
ROW_PER_TRACE = (3200.0 / dot_mod.DOTS_PER_TRACE) / 7.4406


def _gauss(sigma: float) -> np.ndarray:
    if sigma <= 1e-6:
        return np.array([1.0])
    r = int(max(1, math.ceil(4.0 * sigma)))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


_L3_A = 3


def _row_kernel(sigma: float, a: int = _L3_A, half: float = 16.0,
                dt: float = 0.002):
    """Continuous row kernel = Lanczos-3 interpolation (x) Gaussian blur.

    The scene lives on a unit grid but the plateau centres land anywhere
    between grid rows, so the kernel has to do two jobs: interpolate (which is
    what the sinc-like Lanczos window is for -- with sigma = 0 the operator is
    then pure resampling and adds no blur of its own) and apply the camera's
    vertical aperture.  Convolving the two numerically keeps both honest at
    every sigma instead of pretending a narrow Gaussian can interpolate.
    """
    t = np.arange(-half, half + dt / 2, dt)
    lz = np.sinc(t) * np.sinc(t / a)
    lz[np.abs(t) >= a] = 0.0
    if sigma > 1e-6:
        g = np.exp(-0.5 * (t / sigma) ** 2)
        g /= g.sum() * dt
        k = np.convolve(lz, g, mode="same") * dt
    else:
        k = lz
    return t, k


# ---------------------------------------------------------------------------
# loading one frame, exactly the way testset.py does
# ---------------------------------------------------------------------------

PRE_384 = int(round(1.5 * sync_mod.NOMINAL_PERIOD))


def load_frame(fid: str, decim: int = 1, n_traces: int = sync_mod.TRACES_PER_FRAME):
    """Return (frame, signal, timebase) for one frame id, at 384/decim kHz."""
    cat = catalog_mod.build()
    frame = cat.by_id(fid)
    info = wav.probe(MASTER)
    head = PRE_384
    span = int(math.ceil((n_traces + 6) * sync_mod.NOMINAL_PERIOD))
    start = frame.seed_sample - head
    if start < 0:
        raise ValueError(f"{fid}: seed too close to file start")
    x = np.asarray(wav.read(info, frame.channel, start, head + span), dtype=np.float64)
    if decim > 1:
        x = resample_poly(x, 1, decim)
    tb = sync_mod.recover(
        x, period_guess=sync_mod.NOMINAL_PERIOD / decim, n_traces=n_traces
    )
    return frame, x, tb


# ---------------------------------------------------------------------------
# geometry: which sample belongs to which plateau of which trace
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    n_samples: int
    n_tr: int
    n_rows: int
    period: float
    spd: float
    starts: np.ndarray
    psi: np.ndarray
    a: float                       # high-pass pole, exp(-1/tau)
    tau: float
    # plateau list
    pl_trace: np.ndarray           # (n_pl,) int32
    pl_centre: np.ndarray          # (n_pl,) gate-relative plateau centre, rows
    pl_edges: np.ndarray           # (n_pl, 2) absolute sample coords of the plateau
    # sample -> plateau
    pic_n: np.ndarray              # (n_pic,) sample index
    pic_pl: np.ndarray             # (n_pic,) plateau index
    u_n: np.ndarray                # (n_u,) samples that are NOT picture
    # scene -> plateau weights (row direction), built for a given camera sigma
    w_idx: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int64))
    w_val: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    row_sigma: float = 0.0
    trace_kernel: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    rec_kernel: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    dot_locked: bool = False
    clock_strength: float = 0.0

    @property
    def n_pl(self) -> int:
        return len(self.pl_trace)


def build_geometry(
    x: np.ndarray,
    tb: sync_mod.Timebase,
    channel: str,
    n_traces: int | None = None,
    cam_rows: float = CAM_ROWS,
    cam_traces: float = CAM_TRACES,
    rec_rise: float = REC_RISE_384,
) -> Geometry:
    """Everything about A that does not depend on the unknowns."""
    period = tb.period
    n_tr = min(n_traces or sync_mod.TRACES_PER_FRAME, len(tb.smoothed))
    starts = np.array([tb.trace_start(i) for i in range(n_tr)])

    clock = dot_mod.measure(
        x, tb,
        lo_f=decode_mod.PICTURE_START + 0.01,
        hi_f=decode_mod.PICTURE_START + decode_mod.PICTURE_SPAN - 0.01,
    )
    spd = clock.samples_per_dot if clock.measured else period / dot_mod.DOTS_PER_TRACE
    psi, _ = dot_mod.track_phase(
        x, starts, period, spd,
        decode_mod.PICTURE_START,
        decode_mod.PICTURE_START + decode_mod.PICTURE_SPAN,
    )

    span = decode_mod.PICTURE_SPAN * period
    n_rows = int(round(span / spd))

    # tau is quoted at 384 kHz and scales with the measured period
    tau = decode_mod.UNCOUPLE_TAU_384[channel[:1].upper()] * period / sync_mod.NOMINAL_PERIOD
    a = float(np.exp(-1.0 / tau))

    n_end = int(np.floor(starts[n_tr - 1] + (decode_mod.PICTURE_START + decode_mod.PICTURE_SPAN + 0.06) * period))
    n_samples = min(len(x), max(n_end, 1))

    pl_trace, pl_centre, pl_lo, pl_hi = [], [], [], []
    pic_n, pic_pl = [], []
    is_pic = np.zeros(n_samples, dtype=bool)
    for i in range(n_tr):
        a_i = starts[i] + decode_mod.PICTURE_START * period
        b_i = a_i + span
        n0 = int(np.ceil(a_i))
        n1 = int(np.floor(b_i))
        if n0 < 0 or n1 >= n_samples:
            continue
        ns = np.arange(n0, n1 + 1)
        k = np.floor((ns - psi[i]) / spd).astype(np.int64)
        g = (psi[i] + (k + 0.5) * spd - a_i) / spd      # plateau centre, rows
        keep = (g >= 0.0) & (g <= n_rows)
        ns, k, g = ns[keep], k[keep], g[keep]
        if len(ns) == 0:
            continue
        uk, inv = np.unique(k, return_inverse=True)
        base = len(pl_trace)
        pl_trace.extend([i] * len(uk))
        ucentre = (psi[i] + (uk + 0.5) * spd - a_i) / spd
        pl_centre.extend(ucentre.tolist())
        pl_lo.extend((psi[i] + uk * spd).tolist())
        pl_hi.extend((psi[i] + (uk + 1) * spd).tolist())
        pic_n.append(ns)
        pic_pl.append(base + inv)
        is_pic[ns] = True

    geo = Geometry(
        n_samples=n_samples,
        n_tr=n_tr,
        n_rows=n_rows,
        period=period,
        spd=spd,
        starts=starts,
        psi=psi,
        a=a,
        tau=tau,
        pl_trace=np.asarray(pl_trace, dtype=np.int64),
        pl_centre=np.asarray(pl_centre, dtype=np.float64),
        pl_edges=np.stack([np.asarray(pl_lo), np.asarray(pl_hi)], axis=1),
        pic_n=np.concatenate(pic_n),
        pic_pl=np.concatenate(pic_pl),
        u_n=np.where(~is_pic)[0],
        dot_locked=bool(clock.measured),
        clock_strength=float(clock.strength),
    )
    set_psf(geo, cam_rows, cam_traces, rec_rise)
    return geo


def set_psf(geo: Geometry, cam_rows: float, cam_traces: float, rec_rise: float,
            registered: bool = True, interp_a: int = _L3_A) -> None:
    """(Re)build the camera / recording-path kernels of A.

    `registered` False forces every trace's plateau centres onto the integer
    row grid, i.e. it throws away the measured sub-plateau offsets.  It exists
    only so the hold-out can test whether those offsets carry information.
    """
    sigma = max(cam_rows, 0.0) * SIGMA_PER_1090
    geo.row_sigma = sigma
    rad = int(max(3, math.ceil(interp_a + 3.5 * sigma)))
    c = geo.pl_centre - 0.5                       # in row-index coordinates
    if not registered:
        c = np.round(c)
    tt, kk = _row_kernel(sigma, interp_a)
    base = np.round(c).astype(np.int64) - rad
    idx = base[:, None] + np.arange(2 * rad + 1)[None, :]
    t = idx - c[:, None]
    w = np.interp(t, tt, kk, left=0.0, right=0.0)
    ok = (idx >= 0) & (idx < geo.n_rows)
    w = w * ok                                     # zero padding outside the gate
    s = w.sum(axis=1, keepdims=True)
    w = np.where(np.abs(s) > 1e-9, w / np.where(np.abs(s) > 1e-9, s, 1.0), w)
    geo.w_idx = np.clip(idx, 0, geo.n_rows - 1)
    geo.w_val = w
    geo.trace_kernel = _gauss(cam_traces * SIGMA_PER_1090) if cam_traces > 0 else np.array([1.0])
    # rec_rise is quoted at 384 kHz; scale by the sample rate via the period
    r = rec_rise * geo.period / sync_mod.NOMINAL_PERIOD
    geo.rec_kernel = _gauss(r * SIGMA_PER_1090) if r > 0.3 else np.array([1.0])


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------


class Chain:
    """A, matrix-free, with an exact adjoint.

    Unknown blocks (any subset may be held fixed):
      "x"   scene, (n_tr, n_rows), or plateau values when psf=False
      "u"   blanking / out-of-gate raster samples, (n_u,)
      "hum" parity-locked row profile, (n_rows,)
    """

    def __init__(self, geo: Geometry, psf: bool = True, hum: bool = True,
                 blocks: tuple[str, ...] = ("x", "u", "hum"),
                 weights: np.ndarray | None = None):
        self.geo = geo
        self.psf = psf
        self.use_hum = hum
        self.blocks = tuple(b for b in blocks if b != "hum" or hum)
        self.w = weights
        g = geo
        self.sizes = {
            "x": g.n_tr * g.n_rows if psf else g.n_pl,
            "u": len(g.u_n),
            "hum": g.n_rows,
        }
        self.offsets, o = {}, 0
        for b in self.blocks:
            self.offsets[b] = o
            o += self.sizes[b]
        self.n = o
        self.m = g.n_samples
        self.sign = np.where(np.arange(g.n_tr) % 2 == 1, 1.0, -1.0)
        self._fixed = {b: np.zeros(self.sizes[b]) for b in ("x", "u", "hum")}
        self.matvecs = 0

    # -- packing -----------------------------------------------------------
    def unpack(self, z: np.ndarray) -> dict:
        out = dict(self._fixed)
        for b in self.blocks:
            out[b] = z[self.offsets[b]: self.offsets[b] + self.sizes[b]]
        return out

    def pack(self, parts: dict) -> np.ndarray:
        return np.concatenate([np.asarray(parts[b]).ravel() for b in self.blocks])

    def set_fixed(self, **parts) -> None:
        for k, v in parts.items():
            self._fixed[k] = np.asarray(v, dtype=np.float64).ravel().copy()

    # -- pieces ------------------------------------------------------------
    def plateaus(self, parts: dict) -> np.ndarray:
        g = self.geo
        if self.psf:
            xs = parts["x"].reshape(g.n_tr, g.n_rows)
            if len(g.trace_kernel) > 1:
                xs = convolve1d(xs, g.trace_kernel, axis=0, mode="constant")
            p = np.einsum("pt,pt->p", xs[g.pl_trace[:, None], g.w_idx], g.w_val)
        else:
            p = parts["x"].copy()
        if self.use_hum:
            p = p + self.sign[g.pl_trace] * parts["hum"][
                np.clip(np.round(g.pl_centre - 0.5).astype(np.int64), 0, g.n_rows - 1)
            ]
        return p

    def plateaus_adj(self, pa: np.ndarray, out: dict) -> None:
        g = self.geo
        if self.use_hum:
            rows = np.clip(np.round(g.pl_centre - 0.5).astype(np.int64), 0, g.n_rows - 1)
            out["hum"] = np.bincount(rows, weights=self.sign[g.pl_trace] * pa,
                                     minlength=g.n_rows)
        if self.psf:
            flat = (g.pl_trace[:, None] * g.n_rows + g.w_idx).ravel()
            wsum = np.bincount(flat, weights=(g.w_val * pa[:, None]).ravel(),
                               minlength=g.n_tr * g.n_rows).reshape(g.n_tr, g.n_rows)
            if len(g.trace_kernel) > 1:
                wsum = convolve1d(wsum, g.trace_kernel[::-1], axis=0, mode="constant")
            out["x"] = wsum.ravel()
        else:
            out["x"] = pa

    def raster(self, parts: dict) -> np.ndarray:
        g = self.geo
        p = self.plateaus(parts)
        v = np.zeros(g.n_samples)
        v[g.pic_n] = p[g.pic_pl]
        v[g.u_n] = parts["u"]
        return v

    def forward_raster(self, v: np.ndarray) -> np.ndarray:
        g = self.geo
        if len(g.rec_kernel) > 1:
            v = np.convolve(v, g.rec_kernel, mode="same")
        return lfilter([1.0, -1.0], [1.0, -g.a], v)

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.matvecs += 1
        y = self.forward_raster(self.raster(self.unpack(z)))
        return y if self.w is None else y * self.w

    def adjoint(self, r: np.ndarray) -> np.ndarray:
        g = self.geo
        r = r if self.w is None else r * self.w
        # H^T = D^T (I - a Z)^-T
        zz = lfilter([1.0], [1.0, -g.a], r[::-1])[::-1]
        v = zz - np.concatenate([zz[1:], [0.0]])
        if len(g.rec_kernel) > 1:
            v = np.convolve(v, g.rec_kernel[::-1], mode="same")
        parts = {}
        pa = np.bincount(g.pic_pl, weights=v[g.pic_n], minlength=g.n_pl)
        self.plateaus_adj(pa, parts)
        parts["u"] = v[g.u_n]
        return np.concatenate([parts[b].ravel() for b in self.blocks])

    def as_linop(self) -> LinearOperator:
        return LinearOperator((self.m, self.n), matvec=self.forward,
                              rmatvec=self.adjoint, dtype=np.float64)

    def offset(self) -> np.ndarray:
        """A applied to the FIXED blocks alone (subtract from y)."""
        parts = dict(self._fixed)
        y = self.forward_raster(self.raster(parts))
        return y if self.w is None else y * self.w


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------


class Grad:
    """Gradient of the scene block only, with the dot/trace aspect folded in.

    `anis` multiplies the ACROSS-TRACE difference relative to the along-trace
    one.  anis = 1 is the geometrically isotropic prior (one row of trace-time
    is ROW_PER_TRACE trace widths, so the row difference is divided by that);
    it is not the right prior for this record, because the across-trace
    direction carries per-trace noise the along-trace direction does not.  The
    value is chosen on a hold-out split that is NOT the one used to score.
    """

    def __init__(self, chain: Chain, aspect: float = ROW_PER_TRACE, anis: float = 1.0):
        self.chain = chain
        g = chain.geo
        self.shape2 = (g.n_tr, g.n_rows)
        self.n = chain.n
        self.off = chain.offsets["x"]
        self.nx = chain.sizes["x"]
        # one plateau of trace-time is `aspect` trace widths, so a difference of
        # one row spans `aspect` times the distance of a difference of one trace
        self.sr = 1.0 / aspect
        self.st = float(anis)
        self.m = 2 * self.nx

    def forward(self, z: np.ndarray) -> np.ndarray:
        a = z[self.off: self.off + self.nx].reshape(self.shape2)
        dt = np.zeros_like(a); dt[:-1] = (a[1:] - a[:-1]) * self.st
        dr = np.zeros_like(a); dr[:, :-1] = (a[:, 1:] - a[:, :-1]) * self.sr
        return np.concatenate([dt.ravel(), dr.ravel()])

    def adjoint(self, q: np.ndarray) -> np.ndarray:
        dt = q[: self.nx].reshape(self.shape2) * self.st
        dr = q[self.nx:].reshape(self.shape2)
        a = np.zeros(self.shape2)
        a[1:] += dt[:-1]; a[:-1] -= dt[:-1]
        a[:, 1:] += dr[:, :-1] * self.sr; a[:, :-1] -= dr[:, :-1] * self.sr
        out = np.zeros(self.n)
        out[self.off: self.off + self.nx] = a.ravel()
        return out


class Stacked(LinearOperator):
    """[ A ; s1*G1 ; s2*G2 ; ... ] as one LinearOperator."""

    def __init__(self, chain: Chain, extras: list[tuple[object, float]]):
        self.chain = chain
        self.extras = extras
        m = chain.m + sum(int(e.m) for e, _ in extras)
        super().__init__(np.float64, (m, chain.n))

    def _matvec(self, z):
        out = [self.chain.forward(z)]
        for e, s in self.extras:
            out.append(s * e.forward(z))
        return np.concatenate(out)

    def _rmatvec(self, r):
        m0 = self.chain.m
        out = self.chain.adjoint(r[:m0])
        o = m0
        for e, s in self.extras:
            out = out + s * e.adjoint(r[o: o + int(e.m)])
            o += int(e.m)
        return out


class ColScaled(LinearOperator):
    """op with its columns rescaled to unit norm: z = d * z'.

    Without it LSMR crawls.  The scene columns of A that still have data
    attached have norm ~sqrt(samples per plateau) ~ 3.5, while a column whose
    plateau has been HELD OUT is held up only by the prior and has norm ~lambda
    ~ 0.01 -- a spread of 10^4 in the column norms, i.e. 10^8 in the normal
    equations, which no Krylov method will resolve in a few hundred iterations.
    The first hold-out run of this module measured exactly that and it looked
    like a substantive result until the iteration count was varied.
    """

    def __init__(self, op: LinearOperator, d: np.ndarray):
        self.op = op
        self.d = d
        super().__init__(np.float64, op.shape)

    def _matvec(self, z):
        return self.op.matvec(self.d * z)

    def _rmatvec(self, r):
        return self.d * self.op.rmatvec(r)


def column_scaling(op: LinearOperator, probes: int = 24, seed: int = 3,
                   floor: float = 1e-6) -> np.ndarray:
    """1/||A_j||, estimated by Rademacher probes: E[(A^T g)_j^2] = ||A_j||^2."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(op.shape[1])
    for _ in range(probes):
        g = rng.integers(0, 2, size=op.shape[0]).astype(np.float64) * 2.0 - 1.0
        acc += op.rmatvec(g) ** 2
    nrm = np.sqrt(acc / probes)
    nrm = np.maximum(nrm, floor * max(nrm.max(), 1e-30))
    return 1.0 / nrm


class Ridge:
    """Tiny identity on every unknown: pins the unobservable DC direction."""

    def __init__(self, chain: Chain):
        self.n = chain.n
        self.m = chain.n

    def forward(self, z):
        return z

    def adjoint(self, q):
        return q


# ---------------------------------------------------------------------------
# solves
# ---------------------------------------------------------------------------


@dataclass
class Solution:
    parts: dict
    chain: Chain
    geo: Geometry
    prior: str
    lam: float
    iters: int
    seconds: float
    resid_rms: float
    history: list = field(default_factory=list)

    @property
    def scene(self) -> np.ndarray:
        g = self.geo
        if self.chain.psf:
            return self.parts["x"].reshape(g.n_tr, g.n_rows)
        return plateaus_to_grid(g, self.parts["x"])


def plateaus_to_grid(geo: Geometry, p: np.ndarray) -> np.ndarray:
    """Put plateau values on the uniform (trace, row) grid by nearest row."""
    out = np.zeros((geo.n_tr, geo.n_rows))
    cnt = np.zeros((geo.n_tr, geo.n_rows))
    r = np.clip(np.round(geo.pl_centre - 0.5).astype(np.int64), 0, geo.n_rows - 1)
    flat = geo.pl_trace * geo.n_rows + r
    np.add.at(out.reshape(-1), flat, p)
    np.add.at(cnt.reshape(-1), flat, 1.0)
    m = cnt > 0
    out[m] /= cnt[m]
    # rows a trace never reached: fill from the nearest filled row
    for i in range(geo.n_tr):
        idx = np.where(m[i])[0]
        if len(idx) and len(idx) < geo.n_rows:
            out[i] = np.interp(np.arange(geo.n_rows), idx, out[i, idx])
    return out


def solve_tikhonov(chain: Chain, y: np.ndarray, lam: float, ridge: float = 1e-4,
                   maxiter: int = 200, x0: np.ndarray | None = None,
                   atol: float = 1e-7, precond: bool = True,
                   anis: float = 1.0) -> Solution:
    t0 = time.time()
    extras: list[tuple[object, float]] = [(Grad(chain, anis=anis), lam)] if lam > 0 else []
    extras.append((Ridge(chain), ridge))
    op = Stacked(chain, extras)
    b = np.zeros(op.shape[0])
    yy = y if chain.w is None else y * chain.w
    b[: chain.m] = yy - chain.offset()
    if precond:
        d = column_scaling(op)
        sop = ColScaled(op, d)
        res = lsmr(sop, b, atol=atol, btol=atol, maxiter=maxiter,
                   x0=None if x0 is None else x0 / d)
        z = res[0] * d
    else:
        res = lsmr(op, b, atol=atol, btol=atol, maxiter=maxiter, x0=x0)
        z = res[0]
    r = chain.forward(z) + chain.offset() - yy
    sec = time.time() - t0
    return Solution(chain.unpack(z), chain, chain.geo, "tikhonov", lam,
                    int(res[2]), sec, float(np.sqrt(np.mean(r**2))))


def solve_tv(chain: Chain, y: np.ndarray, lam: float, rho: float | None = None,
             outer: int = 8, inner: int = 60, ridge: float = 1e-4,
             isotropic: bool = True, x0: np.ndarray | None = None,
             verbose: bool = False, precond: bool = True,
             anis: float = 1.0) -> Solution:
    """Split-Bregman / ADMM total variation on the scene block."""
    t0 = time.time()
    G = Grad(chain, anis=anis)
    rho = rho if rho is not None else max(lam, 1e-6)
    extras = [(G, math.sqrt(rho)), (Ridge(chain), ridge)]
    op = Stacked(chain, extras)
    d = column_scaling(op) if precond else np.ones(chain.n)
    sop = ColScaled(op, d) if precond else op
    yy = y if chain.w is None else y * chain.w
    data = yy - chain.offset()

    z = x0 if x0 is not None else np.zeros(chain.n)
    dsplit = np.zeros(G.m)
    bdual = np.zeros(G.m)
    hist = []
    nx = G.nx
    for it in range(outer):
        b = np.zeros(op.shape[0])
        b[: chain.m] = data
        b[chain.m: chain.m + G.m] = math.sqrt(rho) * (dsplit - bdual)
        res = lsmr(sop, b, atol=1e-9, btol=1e-9, maxiter=inner, x0=z / d)
        z = res[0] * d
        gz = G.forward(z)
        t = gz + bdual
        if isotropic:
            mag = np.sqrt(t[:nx] ** 2 + t[nx:] ** 2)
            shrink = np.maximum(mag - lam / rho, 0.0) / np.maximum(mag, 1e-12)
            dsplit = np.concatenate([t[:nx] * shrink, t[nx:] * shrink])
        else:
            dsplit = np.sign(t) * np.maximum(np.abs(t) - lam / rho, 0.0)
        bdual = bdual + gz - dsplit
        r = chain.forward(z) + chain.offset() - yy
        tv = float(np.sum(np.sqrt(gz[:nx] ** 2 + gz[nx:] ** 2)))
        hist.append({"iter": it, "resid_rms": float(np.sqrt(np.mean(r**2))), "tv": tv,
                     "primal": float(np.sqrt(np.mean((gz - dsplit) ** 2)))})
        if verbose:
            print(f"    tv iter {it}: resid {hist[-1]['resid_rms']:.6f} "
                  f"tv {tv:.1f} primal {hist[-1]['primal']:.5f}")
    r = chain.forward(z) + chain.offset() - yy
    sec = time.time() - t0
    return Solution(chain.unpack(z), chain, chain.geo, "tv", lam,
                    outer * inner, sec, float(np.sqrt(np.mean(r**2))), hist)


# ---------------------------------------------------------------------------
# the sequential decoder's own estimate, on the same plateau list
# ---------------------------------------------------------------------------


def sequential_plateaus(x: np.ndarray, geo: Geometry,
                        mask: np.ndarray | None = None) -> np.ndarray:
    """What decode.py's chain estimates for every plateau, in signal units.

    undroop the raw signal (exact pole inverse, raster order), then average it
    over each plateau -- which is exactly dotclock.sample_plateaus.  `mask`
    (per sample, True = usable) supports the hold-out comparison: held-out
    samples are dropped from the average and plateaus that lose all their
    samples are interpolated along the row axis, the best a chain with no
    missing-data model can do.
    """
    v = decode_mod.undroop(x[: geo.n_samples], geo.tau)
    n = geo.pic_n
    p = geo.pic_pl
    if mask is not None:
        keep = mask[n]
        n, p = n[keep], p[keep]
    s = np.bincount(p, weights=v[n], minlength=geo.n_pl)
    c = np.bincount(p, minlength=geo.n_pl)
    out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    if np.isnan(out).any():
        for i in range(geo.n_tr):
            sel = np.where(geo.pl_trace == i)[0]
            if len(sel) == 0:
                continue
            vals = out[sel]
            good = np.isfinite(vals)
            if good.sum() >= 2:
                vals[~good] = np.interp(np.where(~good)[0], np.where(good)[0], vals[good])
            elif good.sum() == 1:
                vals[~good] = vals[good][0]
            else:
                vals[:] = 0.0
            out[sel] = vals
    return out


def sequential_u(x: np.ndarray, geo: Geometry) -> np.ndarray:
    return decode_mod.undroop(x[: geo.n_samples], geo.tau)[geo.u_n]


# ---------------------------------------------------------------------------
# hold-out: fit on part of the data, predict the part held back
# ---------------------------------------------------------------------------


def holdout_samples(geo: Geometry, frac: float, seed: int = 0):
    """Drop a random `frac` of PICTURE samples (blanking is always kept)."""
    rng = np.random.default_rng(seed)
    sel = rng.random(len(geo.pic_n)) < frac
    w = np.ones(geo.n_samples)
    w[geo.pic_n[sel]] = 0.0
    return w, geo.pic_n[sel]


def holdout_plateaus(geo: Geometry, frac: float, seed: int = 0):
    """Drop EVERY sample of a random `frac` of plateaus.

    This is the hard test: a held-out plateau has no data of its own left, so
    predicting it needs the model and the prior.  A method that invents detail
    fails here; a method that recovers real detail passes.
    """
    rng = np.random.default_rng(seed)
    held = rng.random(geo.n_pl) < frac
    w = np.ones(geo.n_samples)
    w[geo.pic_n[held[geo.pic_pl]]] = 0.0
    return w, np.where(held)[0]





# ---------------------------------------------------------------------------
# from a scene to an image on the same intensity scale as decode.py
# ---------------------------------------------------------------------------


def scene_to_image(geo: Geometry, scene: np.ndarray, u: np.ndarray,
                   height: int = decode_mod.SQUARE_ROWS, dehum: bool = True,
                   invert: bool = True) -> tuple[np.ndarray, tuple]:
    """(height, traces) in [0,1], anchored the way decode.py anchors.

    `dehum` is ON and must stay on.  MEASURED (2026-08): the MAP solve leaves a
    per-trace DC that ALTERNATES with trace parity at amplitude 0.0138 signal
    units on L055 and 0.0136 on L000, against 0.00053 / 0.00203 for the
    sequential decoder -- a 7-26x regression that shows up as parity_db 13.0 /
    10.4 dB against 1.2 / 3.2.  Adding the hum block to the solve does NOT fix
    it: a profile h[r] applied with alternating trace sign can be absorbed
    exactly into the scene (scene[i,r] += s_i h[r]) at zero data cost, so the
    two are degenerate and the quadratic prior splits them arbitrarily
    (hum-on and hum-off measure the same to 3 decimals).  The sequential
    chain's odd/even MEDIAN is what breaks the tie -- a robust estimator that
    assumes picture content is uncorrelated with parity, which no quadratic
    term encodes.  Applying it after the solve restores parity_db to 0.72 /
    3.15, i.e. at or below the sequential decoder.  The physical origin is the
    one decode.py already documents: the sync pulse width alternates, so the
    porch the model ties the picture DC to alternates with it.
    """
    v = np.zeros(geo.n_samples)
    rows = np.clip(np.round(geo.pl_centre - 0.5).astype(np.int64), 0, geo.n_rows - 1)
    v[geo.pic_n] = scene[geo.pl_trace, rows][geo.pic_pl]
    v[geo.u_n] = u
    ref = decode_mod.measure_levels(v, geo.starts[: geo.n_tr], geo.period)

    porch, ok = decode_mod._region(v, geo.starts[: geo.n_tr], geo.period,
                                   decode_mod.PORCH_START, decode_mod.PORCH_END)
    porch_level = np.median(porch, axis=1)
    clamp = porch_level.copy()
    if len(porch_level) >= 16:
        for par in (0, 1):
            s = porch_level[par::2]
            pad = np.pad(s, 3, mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(pad, 7)
            clamp[par::2] = np.median(win, axis=1)
    pic = scene - clamp[:, None]
    if dehum:
        prof, _ = decode_mod.measure_hum(pic)
        pic[1::2] -= prof
        pic[0::2] += prof
    pic = decode_mod._rows_from_dots(pic, height, 3, None)
    img = pic.T
    if invert:
        img = -img
    sgn = -1.0 if invert else 1.0
    if ref is not None:
        porch_med, amp, _ = ref
        lo = sgn * decode_mod.UNCOUPLE_BLACK_REF * amp
        hi = sgn * decode_mod.UNCOUPLE_WHITE_REF * amp
        levels = (porch_med + decode_mod.UNCOUPLE_BLACK_REF * amp,
                  porch_med + decode_mod.UNCOUPLE_WHITE_REF * amp)
    else:
        lo = float(np.percentile(img, 1.0))
        hi = float(np.percentile(img, 99.0))
        levels = (lo, hi)
    if abs(hi - lo) < 1e-9:
        hi = lo + 1e-9
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32), levels


# ---------------------------------------------------------------------------
# the adjoint test.  Skipping this is how these projects go wrong silently.
# ---------------------------------------------------------------------------


def adjoint_test(chain: Chain, trials: int = 3, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        z = rng.standard_normal(chain.n)
        r = rng.standard_normal(chain.m)
        lhs = float(chain.forward(z) @ r)
        rhs = float(z @ chain.adjoint(r))
        scale = max(abs(lhs), abs(rhs), 1e-300)
        out.append(abs(lhs - rhs) / scale)
    return out


# ---------------------------------------------------------------------------
# the trace hold-out: the only test here that a wrong answer cannot game
# ---------------------------------------------------------------------------


def _lanczos_shift(v: np.ndarray, s: float, a: int = 3) -> np.ndarray:
    n = len(v)
    u = np.arange(n) + s
    base = np.floor(u).astype(np.int64) - a + 1
    idx = base[:, None] + np.arange(2 * a)[None, :]
    t = u[:, None] - idx
    ww = np.sinc(t) * np.sinc(t / a)
    ww[np.abs(t) >= a] = 0.0
    ww /= ww.sum(axis=1, keepdims=True)
    return (v[np.clip(idx, 0, n - 1)] * ww).sum(axis=1)


def trace_holdout(x: np.ndarray, geo: Geometry, every: int = 8):
    """Drop every `every`-th trace whole, and return the scoring machinery.

    A whole trace is the only hold-out on this signal with no shortcut: the
    nearest surviving sample is a full line period away, so nothing can be
    recovered by interpolating the waveform.  The truth is the held trace's own
    plateau values, measured from its own samples with the full data.
    """
    truth = sequential_plateaus(x, geo).reshape(geo.n_tr, geo.n_rows)
    centres = (geo.pl_centre - 0.5).reshape(geo.n_tr, geo.n_rows)
    held = np.arange(3, geo.n_tr - 3, every)
    mask = np.zeros(geo.n_tr, dtype=bool)
    mask[held] = True
    w = np.ones(geo.n_samples)
    w[geo.pic_n[mask[geo.pl_trace][geo.pic_pl]]] = 0.0

    def score(P: np.ndarray) -> float:
        d = P[held] - truth[held]
        d = d - d.mean(axis=1, keepdims=True)      # per-trace DC is unobservable
        return float(np.sqrt(np.mean(d[:, 8:-8] ** 2)))

    unreg = truth.copy()
    reg = truth.copy()
    for i in held:
        unreg[i] = 0.5 * (truth[i - 1] + truth[i + 1])
        reg[i] = 0.5 * (_lanczos_shift(truth[i - 1], float(np.mean(centres[i] - centres[i - 1])))
                        + _lanczos_shift(truth[i + 1], float(np.mean(centres[i] - centres[i + 1]))))
    return w, held, truth, score, {"unregistered": unreg, "registered": reg}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", default="L055")
    ap.add_argument("--decim", type=int, default=1)
    ap.add_argument("--traces", type=int, default=sync_mod.TRACES_PER_FRAME)
    ap.add_argument("--adjoint", action="store_true", help="dot-product adjoint test")
    ap.add_argument("--holdout", action="store_true", help="whole-trace hold-out")
    ap.add_argument("--solve", default="", help="tik | tv")
    ap.add_argument("--lam", type=float, default=0.03)
    ap.add_argument("--anis", type=float, default=1.0)
    ap.add_argument("--cam", type=float, default=0.0, help="camera 10-90 rise, dots")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--png", default="")
    args = ap.parse_args(argv)

    frame, x, tb = load_frame(args.frame, args.decim, args.traces)
    geo = build_geometry(x, tb, frame.id[0], args.traces, cam_rows=args.cam)
    y = x[: geo.n_samples]
    print(f"{frame.id}: {geo.n_samples} samples, {geo.n_tr} traces, "
          f"{geo.n_rows} rows, {geo.n_pl} plateaus, spd {geo.spd:.4f}, "
          f"tau {geo.tau:.1f}, dot_locked {geo.dot_locked} "
          f"(strength {geo.clock_strength:.2f})")

    if args.adjoint:
        print("adjoint test  <Ax,r> vs <x,A^T r>")
        for psf in (False, True):
            for hum in (False, True):
                ch = Chain(geo, psf=psf, hum=hum)
                print(f"  psf={psf!s:5} hum={hum!s:5} n={ch.n} m={ch.m} "
                      f"rel err {max(adjoint_test(ch)):.3e}")

    if args.holdout:
        w, held, truth, score, base = trace_holdout(x, geo)
        print(f"\nwhole-trace hold-out: {len(held)} of {geo.n_tr} traces")
        for k, v in base.items():
            print(f"  neighbour mean, {k:14} {score(v):.5f}")
        ch = Chain(geo, psf=True, hum=False, weights=w)
        sol = (solve_tv(ch, y, args.lam, outer=8, inner=args.iters // 8, ridge=1e-6,
                        anis=args.anis)
               if args.solve == "tv" else
               solve_tikhonov(ch, y, args.lam, ridge=1e-6, maxiter=args.iters,
                              atol=1e-11, anis=args.anis))
        ch2 = Chain(geo, psf=True, hum=False)
        P = ch2.plateaus(sol.parts).reshape(geo.n_tr, geo.n_rows)
        print(f"  MAP {args.solve or 'tik'} lam={args.lam} anis={args.anis} "
              f"cam={args.cam}: {score(P):.5f}  (train resid {sol.resid_rms:.5f}, "
              f"{sol.seconds:.1f}s)")
        print("  NOTE: read the hold-out and the training residual TOGETHER -- this "
              "metric\n  rewards across-trace smoothing, which a big lambda buys by "
              "wrecking the fit.")

    if args.solve:
        ch = Chain(geo, psf=True, hum=True)
        sol = (solve_tv(ch, y, args.lam, outer=8, inner=args.iters // 8, ridge=1e-6,
                        anis=args.anis, verbose=True)
               if args.solve == "tv" else
               solve_tikhonov(ch, y, args.lam, ridge=1e-6, maxiter=args.iters,
                              atol=1e-11, anis=args.anis))
        print(f"\n{args.solve}: {sol.seconds:.1f}s, residual rms {sol.resid_rms:.5f}")
        if args.png:
            from PIL import Image
            img, lv = scene_to_image(
                geo, sol.parts["x"].reshape(geo.n_tr, geo.n_rows), sol.parts["u"])
            Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8),
                            "L").save(args.png)
            print(f"  wrote {args.png}  (black {lv[0]:.4f} white {lv[1]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
