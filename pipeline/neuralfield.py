"""Fit a continuous scene through the measured forward operator.

The classical decoder recovers a value per dot. This represents the latent slide
as a CONTINUOUS function f(row, col) -> intensity, and fits it by pushing it
through `forward.py` and comparing against the decoded dot matrix.

Why a continuous representation is the right shape for THIS signal and not just
fashionable: our samples do not sit on a rectangular lattice. 262.5 dots per
trace is the NTSC half-line, so the dot phase advances exactly half a dot every
trace -- even columns sample the scene at positions offset half a dot from odd
columns. `dotclock.sample_plateaus` currently throws that away by snapping row
centres to plateaus. A continuous field consumes it natively.

No training data. The network's weights are fitted to this one frame and to
nothing else, so there is no corpus of other photographs for it to import detail
from. That matters for provenance: this is tier 1 (universal priors), not tier
2, because an alien could run it too.

WHAT THIS CANNOT DO, stated up front because the temptation is real: it cannot
recover detail the 1977 camera did not resolve. We already sample 1.4-1.9x finer
than the camera's own limit in both axes. Any structure a fit produces beyond
that is invented, and the hold-out test below is what distinguishes the two.

THE TEST THAT DECIDES. Reference metrics are not the arbiter -- every prior-based
method produces something crisper, and crisper is not truer. The arbiter is a
hold-out on the MEASUREMENTS: fit using a subset of the observed dots, predict
the dots that were withheld, and compare against the classical interpolator on
those same withheld dots. A method that recovers real structure predicts unseen
measurements better. A method that invents structure does not, however good it
looks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import forward as fwd


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FourierFeatures(torch.nn.Module):
    """Random Fourier features: the fix for coordinate MLPs' spectral bias.

    A plain MLP on raw (row, col) fits smooth functions and essentially refuses
    to fit high frequencies. Lifting the coordinates through random sinusoids
    removes that bias. `scale` sets the bandwidth and is the single most
    important knob here -- too high and the field has enough freedom to fit the
    noise, which is exactly the failure the hold-out is there to catch.
    """

    def __init__(self, n_features: int = 128, scale: float = 6.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn((2, n_features), generator=g) * scale
        self.register_buffer("B", B)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * xy @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SceneField(torch.nn.Module):
    """f(row, col) -> intensity, on coordinates normalised to [-1, 1]."""

    def __init__(self, n_features: int = 128, scale: float = 6.0,
                 width: int = 128, depth: int = 3, seed: int = 0):
        super().__init__()
        self.ff = FourierFeatures(n_features, scale, seed)
        layers: list[torch.nn.Module] = [torch.nn.Linear(2 * n_features, width), torch.nn.GELU()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.GELU()]
        layers += [torch.nn.Linear(width, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(self.ff(xy)).squeeze(-1)

    def render(self, h: int, w: int, device) -> torch.Tensor:
        r = torch.linspace(-1, 1, h, device=device)
        c = torch.linspace(-1, 1, w, device=device)
        gr, gc = torch.meshgrid(r, c, indexing="ij")
        xy = torch.stack([gr.reshape(-1), gc.reshape(-1)], dim=-1)
        return self(xy).reshape(h, w)


@dataclass
class FitResult:
    scene: np.ndarray
    loss_history: list[float]
    heldout_rmse: float
    baseline_rmse: float
    iters: int

    @property
    def beats_baseline(self) -> bool:
        return self.heldout_rmse < self.baseline_rmse


def _checker_mask(h: int, w: int, keep: float, seed: int) -> np.ndarray:
    """Which measured dots the fit is allowed to see.

    Random per-dot rather than whole rows or columns: withholding a whole column
    would let a smooth prior win by interpolating along the OTHER axis, which
    tests smoothness rather than fidelity.
    """
    rng = np.random.default_rng(seed)
    return rng.random((h, w)) < keep


def _baseline_predict(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The honest thing to beat: fill withheld dots from their seen neighbours.

    A 4-neighbour average over seen dots only, iterated until every hole is
    covered. This is what the classical chain effectively does at a dropout, and
    a learned method that cannot beat it has demonstrated nothing.
    """
    out = np.where(mask, y, np.nan)
    for _ in range(64):
        holes = np.isnan(out)
        if not holes.any():
            break
        pad = np.pad(out, 1, mode="edge")
        stack = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(stack, axis=0)
        out = np.where(holes, avg, out)
    return np.nan_to_num(out, nan=float(np.nanmean(y[mask])))


def fit(dots: np.ndarray, params: fwd.ChainParams | None = None, *,
        iters: int = 1500, lr: float = 3e-3, keep: float = 0.85,
        ff_scale: float = 6.0, seed: int = 0, verbose: bool = False) -> FitResult:
    """Fit a continuous scene to one frame's dot matrix, with a hold-out.

    `dots` is the decoded (traces x dots) matrix -- the measurement. The field is
    fitted so that forward(field) matches the SEEN dots; the withheld dots are
    never in the loss and are used only to score.
    """
    params = params or fwd.ChainParams()
    dev = _device()
    h, w = dots.shape
    y = np.asarray(dots, dtype=np.float64)
    y = (y - y.mean()) / (y.std() + 1e-12)

    mask = _checker_mask(h, w, keep, seed)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    mt = torch.tensor(mask, dtype=torch.bool, device=dev)

    field = SceneField(scale=ff_scale, seed=seed).to(dev)
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    hist: list[float] = []
    for i in range(iters):
        opt.zero_grad(set_to_none=True)
        scene = field.render(h, w, dev)
        pred = fwd.forward(scene, params)
        loss = torch.mean((pred[mt] - yt[mt]) ** 2)
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
        if verbose and (i % 250 == 0 or i == iters - 1):
            print(f"    iter {i:5d}  train mse {hist[-1]:.5f}")

    with torch.no_grad():
        scene = field.render(h, w, dev)
        pred = fwd.forward(scene, params).cpu().numpy().astype(np.float64)
        scene_np = scene.cpu().numpy().astype(np.float64)

    held = ~mask
    hr = float(np.sqrt(np.mean((pred[held] - y[held]) ** 2))) if held.any() else float("nan")
    br = float(np.sqrt(np.mean((_baseline_predict(y, mask)[held] - y[held]) ** 2))) if held.any() else float("nan")
    return FitResult(scene=scene_np, loss_history=hist, heldout_rmse=hr,
                     baseline_rmse=br, iters=iters)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    from pathlib import Path
    from . import sync as sync_mod, decode as decode_mod, wav, catalog as catalog_mod

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="L000,L020,L055")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--scales", default="4,6,9")
    args = ap.parse_args()

    REPO = Path(__file__).resolve().parent.parent
    MASTER = REPO / "data" / "master" / "384kHzStereo.wav"
    info = wav.probe(MASTER); mm = wav.memmap(info)
    cat = catalog_mod.build()

    print(f"device: {_device()}\n")
    print("HOLD-OUT ON THE MEASUREMENTS -- fit on 85% of dots, predict the other 15%.")
    print("A method that recovers real structure beats neighbour-fill. One that")
    print("invents structure does not, however good the picture looks.\n")

    rows = []
    for fid in args.frames.split(","):
        fr = cat.by_id(fid)
        seg = np.asarray(mm[fr.seed_sample: fr.seed_sample + int(sync_mod.NOMINAL_PERIOD * 520),
                            fr.channel], dtype=np.float64)
        tb = sync_mod.recover(seg)
        dec = decode_mod.decode(seg, decode_mod.Settings(traces=512, rotate=0), tb=tb)
        dots = dec.image
        p = fwd.ChainParams.for_channel(fid[0], period_samples=float(tb.period))
        print(f"  {fid}  {dots.shape[0]}x{dots.shape[1]}")
        for sc in (float(s) for s in args.scales.split(",")):
            r = fit(dots, p, iters=args.iters, ff_scale=sc)
            gain = (r.baseline_rmse - r.heldout_rmse) / r.baseline_rmse * 100
            rows.append((fid, sc, r))
            print(f"     ff_scale {sc:4.1f}   held-out rmse {r.heldout_rmse:.4f}   "
                  f"neighbour-fill {r.baseline_rmse:.4f}   "
                  f"{gain:+6.1f}%   {'BEATS' if r.beats_baseline else 'loses to'} baseline")
        print()

    wins = sum(1 for _, _, r in rows if r.beats_baseline)
    print(f"{wins}/{len(rows)} configurations beat neighbour-fill on withheld measurements.")
    if wins == 0:
        print("VERDICT: no support for the neural field on this signal. Report it as a null.")
