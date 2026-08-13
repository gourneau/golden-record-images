"""The recording chain as a differentiable operator, and its adjoint.

Everything the ML plan wants -- fitting a continuous scene through the measured
physics, deep image priors, self-calibration -- needs one thing first: a
forward operator `A` that maps a latent slide to what the record actually
contains, differentiable end to end, with a VERIFIED adjoint.

The adjoint test is not a formality. `<Ax, y> == <x, A^T y>` to machine
precision is the only cheap way to know that the gradient your optimiser
follows is the gradient of the thing you wrote down. Get it wrong and nothing
raises: the optimiser still runs, still converges, still produces a plausible
picture, and the picture is a solution to a problem you did not pose. This
project's retraction history is entirely made of results that looked fine and
were not, so the test runs before anything is built on top.

The operator, in the order the 1977 chain applied it:

  1. CAMERA APERTURE. The television camera's spot has finite width. Modelled
     as a separable Gaussian; its width is the measured ceiling, and it is the
     term that makes super-resolution futile -- we already sample 1.4-1.9x
     finer than the camera resolved in both axes.

  2. SAMPLE AND HOLD. The scan converter sampled once per NTSC scan line and
     held: 262.5 plateaus per trace. A boxcar of one dot, which in frequency is
     sinc(nu). This is the term the aperture work inverts.

  3. AC-COUPLING DROOP. A one-pole high-pass in RASTER order -- along the trace
     and straight across every trace boundary, because that is how time ran in
     the recording. H(z) = (1 - z^-1)/(1 - a z^-1). Not a per-trace filter;
     applying it per trace is a bug that has shipped here before.

  4. NEGATIVE. More signal means darker.

Provenance: every parameter here is measured from the audio (provenance tier 0)
or is closed-form. No reference image is involved, and this module imports only
torch and numpy so it cannot acquire one by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Measured constants, mirrored from the modules that measured them. Duplicated
# deliberately rather than imported: importing decode/dotclock here would drag
# their whole graph into anything that uses the operator, and these four numbers
# are the entire physical content.
DOTS_PER_TRACE = 262.5      # dotclock.py: NTSC half-line, measured 262.519+-0.043
TRANSITION_1090 = 0.49      # aperture.py: camera 10-90 rise, in dots
TAU_384 = {"L": 530.0, "R": 295.0}   # decode.py: droop time constant, samples
GAUSS_PER_1090 = 1.0 / 2.563   # sigma of a Gaussian with unit 10-90 transition


@dataclass(frozen=True)
class ChainParams:
    """One channel's measured chain. All lengths in DOTS unless stated."""
    psf_sigma: float = TRANSITION_1090 * GAUSS_PER_1090
    hold: bool = True
    droop_a: float = 0.0        # exp(-1/tau) in dots; 0.0 disables
    negative: bool = True

    @staticmethod
    def for_channel(ch: str, period_samples: float = 3197.4) -> "ChainParams":
        """Convert the measured per-channel tau from samples into dots."""
        tau_samples = TAU_384.get(ch, 530.0)
        samples_per_dot = period_samples / DOTS_PER_TRACE
        tau_dots = tau_samples / samples_per_dot
        return ChainParams(droop_a=float(np.exp(-1.0 / tau_dots)))


def _gauss_kernel(sigma: float, device, dtype) -> torch.Tensor:
    """Normalised 1-D Gaussian, truncated at 4 sigma and odd-length."""
    r = max(1, int(np.ceil(4.0 * sigma)))
    n = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (n / sigma) ** 2)
    return k / k.sum()


def _conv_sep(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Separable convolution with reflect padding, shape preserved.

    Reflect rather than zero: a zero pad tells the operator the slide is black
    outside its own edge, which it is not, and the resulting edge response would
    be a feature the optimiser cheerfully fits.
    """
    r = (len(k) - 1) // 2
    x = x[None, None]
    xp = torch.nn.functional.pad(x, (r, r, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(xp, k.view(1, 1, 1, -1))
    xp = torch.nn.functional.pad(x, (0, 0, r, r), mode="reflect")
    x = torch.nn.functional.conv2d(xp, k.view(1, 1, -1, 1))
    return x[0, 0]


def _hold(x: torch.Tensor) -> torch.Tensor:
    """Sample-and-hold over one dot: a 2-tap boxcar along the trace axis.

    The scene is represented on the dot lattice, so one dot is one sample here
    and the hold is the unit boxcar (1/2, 1/2) -- sinc(nu) in frequency, which
    is exactly the A(f) the aperture correction inverts.
    """
    k = torch.tensor([0.5, 0.5], device=x.device, dtype=x.dtype)
    xp = torch.nn.functional.pad(x[None, None], (1, 0, 0, 0), mode="reflect")
    return torch.nn.functional.conv2d(xp, k.view(1, 1, 1, -1))[0, 0]


def _droop(x: torch.Tensor, a: float) -> torch.Tensor:
    """One-pole high-pass in RASTER order: y[n] = x[n] - x[n-1] + a*y[n-1].

    Flattened across trace boundaries on purpose -- the chain did not reset
    between traces, and a per-trace version of this is a bug that has shipped
    here before. Implemented as a scan so it stays differentiable.
    """
    if a <= 0.0:
        return x
    h, w = x.shape
    flat = x.reshape(-1)
    d = flat - torch.nn.functional.pad(flat[:-1], (1, 0))
    # y[n] = d[n] + a y[n-1] is a first-order IIR. Written as a cumulative sum
    # in the exponentially-weighted domain so it is one vectorised pass rather
    # than a Python loop over 130k samples.
    # Scale factors underflow for long sequences, so work relative to a moving
    # reference: split into blocks short enough that a**k stays representable.
    #
    # THE BLOCK LENGTH MUST COME FROM THE DTYPE, and getting this wrong is how
    # the first version of this file shipped broken. It hard-coded the float64
    # limit (a**k down to e^-700). Run in float32, where the smallest normal is
    # about e^-87, a**k underflowed to zero a quarter of the way into each
    # block; `cumsum(seg / w_) * w_` then divided by zero and produced 189166
    # NaNs out of 193024. Nothing raised. The adjoint test passed the whole time
    # because it runs in float64 -- testing a different precision from the one
    # you compute in tests a different program. Hence `adjoint_test_f32`.
    lim = 80.0 if x.dtype == torch.float32 else 700.0
    out = torch.empty_like(d)
    blk = max(1, int(lim / -np.log(a))) if a < 1.0 else len(d)
    blk = min(blk, len(d))
    carry = torch.zeros((), device=x.device, dtype=x.dtype)
    for s in range(0, len(d), blk):
        e = min(s + blk, len(d))
        m = e - s
        p = torch.arange(m, device=x.device, dtype=x.dtype)
        w_ = a ** p
        seg = d[s:e]
        acc = torch.cumsum(seg / w_, dim=0) * w_
        out[s:e] = acc + carry * (w_ * a)
        carry = out[e - 1]
    return out.reshape(h, w)


def forward(scene: torch.Tensor, p: ChainParams) -> torch.Tensor:
    """Latent slide -> what the record carries, on the dot lattice."""
    y = _conv_sep(scene, _gauss_kernel(p.psf_sigma, scene.device, scene.dtype))
    if p.hold:
        y = _hold(y)
    if p.droop_a > 0.0:
        y = _droop(y, p.droop_a)
    if p.negative:
        y = -y
    return y


def adjoint_by_autograd(y: torch.Tensor, p: ChainParams, shape) -> torch.Tensor:
    """A^T y, obtained as the vector-Jacobian product of `forward`.

    For a LINEAR operator the VJP *is* the adjoint, exactly. Using autograd
    rather than a hand-written transpose is the point: the thing the optimiser
    will actually follow is this VJP, so this is the object that must be tested,
    not a separate implementation that might agree with the maths while
    disagreeing with the code.
    """
    x = torch.zeros(shape, device=y.device, dtype=y.dtype, requires_grad=True)
    out = forward(x, p)
    (g,) = torch.autograd.grad(out, x, grad_outputs=y)
    return g


def adjoint_test(shape=(64, 96), p: ChainParams | None = None,
                 seed: int = 0, device: str = "cpu") -> dict:
    """<Ax, y> vs <x, A^T y>. They must agree to floating-point precision."""
    p = p or ChainParams(droop_a=float(np.exp(-1.0 / 40.0)))
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float64).to(device)
    y = torch.randn(shape, generator=g, dtype=torch.float64).to(device)
    lhs = torch.sum(forward(x, p) * y).item()
    rhs = torch.sum(x * adjoint_by_autograd(y, p, shape)).item()
    scale = max(abs(lhs), abs(rhs), 1e-30)
    return {"lhs": lhs, "rhs": rhs,
            "abs_err": abs(lhs - rhs), "rel_err": abs(lhs - rhs) / scale}


def finite_test(shape=(512, 377), p: ChainParams | None = None,
                dtype=torch.float32, seed: int = 0) -> dict:
    """Does the operator produce finite numbers at REAL frame size and dtype?

    Separate from the adjoint test on purpose. The adjoint test runs small and
    in float64 because that is what makes the inner-product identity sharp --
    and that is exactly why it cannot see an underflow that only happens in
    float32 over 193024 samples. This test exists because that bug shipped.
    """
    p = p or ChainParams.for_channel("L")
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g).to(dtype)
    y = forward(x, p)
    n_bad = int((~torch.isfinite(y)).sum().item())
    return {"n": int(y.numel()), "n_nonfinite": n_bad,
            "max_abs": float(y[torch.isfinite(y)].abs().max().item()) if n_bad < y.numel() else float("nan")}


def adjoint_test_f32(shape=(256, 192), p: ChainParams | None = None, seed: int = 0) -> dict:
    """The adjoint identity in the precision the optimiser actually runs in.

    Tolerance is necessarily loose -- float32 inner products over 49k terms
    accumulate real rounding -- so this is not a substitute for the float64
    test. It is a check that the float32 path computes the same OPERATOR, which
    the underflow bug proved is a separate question.
    """
    p = p or ChainParams.for_channel("L")
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g)
    y = torch.randn(shape, generator=g)
    lhs = torch.sum(forward(x, p) * y).item()
    rhs = torch.sum(x * adjoint_by_autograd(y, p, shape)).item()
    scale = max(abs(lhs), abs(rhs), 1e-30)
    return {"lhs": lhs, "rhs": rhs, "rel_err": abs(lhs - rhs) / scale}


def linearity_test(shape=(32, 48), p: ChainParams | None = None, seed: int = 0) -> dict:
    """A(ax + bz) == aA(x) + bA(z). The adjoint test assumes linearity; check it.

    Worth its own test because the adjoint identity would still hold for the
    LINEARISATION of a nonlinear operator, so a passing adjoint test alone does
    not prove the operator is what it claims to be.
    """
    p = p or ChainParams(droop_a=float(np.exp(-1.0 / 40.0)))
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float64)
    z = torch.randn(shape, generator=g, dtype=torch.float64)
    a, b = 0.37, -1.9
    lhs = forward(a * x + b * z, p)
    rhs = a * forward(x, p) + b * forward(z, p)
    err = (lhs - rhs).abs().max().item()
    return {"max_abs_err": err, "scale": rhs.abs().max().item()}


def droop_roundtrip_test(n: int = 4096, tau_dots: float = 40.0, seed: int = 0) -> dict:
    """The droop scan must match the recursion it claims to implement.

    The vectorised block form is an optimisation, and an optimisation of a
    recursion is exactly the kind of thing that is subtly wrong at the block
    boundary and right everywhere else. Compared against the naive loop.
    """
    a = float(np.exp(-1.0 / tau_dots))
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((1, n), generator=g, dtype=torch.float64)
    fast = _droop(x, a).reshape(-1).numpy()
    flat = x.reshape(-1).numpy()
    slow = np.empty(n)
    prev_y = 0.0
    prev_x = 0.0
    for i in range(n):
        d = flat[i] - prev_x
        prev_y = d + a * prev_y
        slow[i] = prev_y
        prev_x = flat[i]
    err = float(np.max(np.abs(fast - slow)))
    return {"max_abs_err": err, "scale": float(np.max(np.abs(slow)))}


if __name__ == "__main__":  # pragma: no cover
    print("FORWARD OPERATOR -- verification before anything is built on it\n")

    r = droop_roundtrip_test()
    ok_d = r["max_abs_err"] / max(r["scale"], 1e-30) < 1e-10
    print(f"  droop scan vs naive recursion   max abs err {r['max_abs_err']:.3e} "
          f"(scale {r['scale']:.3e})   {'PASS' if ok_d else 'FAIL'}")

    r = linearity_test()
    ok_l = r["max_abs_err"] / max(r["scale"], 1e-30) < 1e-12
    print(f"  linearity A(ax+bz)=aAx+bAz      max abs err {r['max_abs_err']:.3e} "
          f"(scale {r['scale']:.3e})   {'PASS' if ok_l else 'FAIL'}")

    print()
    ok_a = True
    for name, p in (
        ("psf only        ", ChainParams(hold=False, droop_a=0.0, negative=False)),
        ("psf + hold      ", ChainParams(hold=True, droop_a=0.0, negative=False)),
        ("psf + hold + neg", ChainParams(hold=True, droop_a=0.0, negative=True)),
        ("full chain      ", ChainParams(droop_a=float(np.exp(-1.0 / 40.0)))),
        ("full, L channel ", ChainParams.for_channel("L")),
        ("full, R channel ", ChainParams.for_channel("R")),
    ):
        t = adjoint_test(p=p)
        good = t["rel_err"] < 1e-12
        ok_a &= good
        print(f"  adjoint {name}  <Ax,y> {t['lhs']:14.6f}  <x,A^T y> {t['rhs']:14.6f}  "
              f"rel err {t['rel_err']:.2e}  {'PASS' if good else 'FAIL'}")

    print()
    ok_f = True
    for ch in ("L", "R"):
        for dt, nm in ((torch.float32, "float32"), (torch.float64, "float64")):
            t = finite_test(p=ChainParams.for_channel(ch), dtype=dt)
            good = t["n_nonfinite"] == 0
            ok_f &= good
            print(f"  finite  {ch} channel, {nm}, 512x377   non-finite {t['n_nonfinite']:6d} "
                  f"of {t['n']}   {'PASS' if good else 'FAIL'}")
    t = adjoint_test_f32()
    ok_f32 = t["rel_err"] < 1e-4
    print(f"  adjoint in float32 (the precision the optimiser runs in)   "
          f"rel err {t['rel_err']:.2e}   {'PASS' if ok_f32 else 'FAIL'}")

    print()
    if ok_d and ok_l and ok_a and ok_f and ok_f32:
        print("  ALL PASS -- the gradient an optimiser follows is the gradient of this operator.")
    else:
        print("  FAILURES ABOVE. Do not build on this operator until they are fixed.")
