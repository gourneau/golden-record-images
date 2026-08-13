"""Deep image prior: a reconstruction whose only prior is the shape of a network.

The neural field in `neuralfield.py` asks a coordinate MLP to be the scene. This
asks a convolutional decoder instead, driven by a fixed random input, and fits
its weights so that the image it produces -- pushed through the measured
physics -- matches the measurements. The prior is entirely ARCHITECTURAL: a
convolutional generator finds natural-looking structure easily and noise only
with difficulty, so early in fitting it explains the scene and late in fitting
it starts explaining the noise.

Two reasons it belongs in this project specifically.

It uses NO TRAINING DATA. Nothing is learned from other photographs, so nothing
from other photographs can appear in the output. That is what keeps it at
provenance tier 1 rather than tier 2 -- an alien could run it, and it cannot
smuggle Earth in. A diffusion prior trained on a photo corpus could not make
that claim, which is why this project does not use one however much better the
pictures would look.

And its known failure mode is exactly the one this artifact punishes. DIP
eventually fits the noise, so WHEN YOU STOP is the whole method. Choosing that
by eye, or by an image metric, means choosing the point where the picture looks
nicest -- and this project has already measured that the composite metric
rewards blur and that every method's metrics keep improving past the correct
setting. So the stopping point is chosen by a hold-out on the measurements, and
the same hold-out decides whether the method was worth running at all.

MEASURED RESULT (2026-08, data/master/384kHzStereo.wav; 2000 iterations, 15% of
dots withheld, the stopping iterate chosen by those withheld dots alone):

  frame   stopped at   train mse   held-out   neighbour-fill   vs baseline
  L000       1800       0.0007      0.0351        0.0550          +36.2%
  L055       1975       0.0010      0.0477        0.0538          +11.3%
  L020       1925       0.0010      0.0620        0.1092          +43.2%

Three of three beat the baseline on measurements the fit never saw. That is a
better result than the neural field in `neuralfield.py` managed (one of six),
and the difference is the expected one: a convolutional generator's inductive
bias suits photographs, a coordinate MLP's does not.

TWO THINGS THAT QUALIFY IT, both visible in the numbers above:

  * THE BUDGET WAS BINDING ON TWO OF THREE FRAMES. L000 and L055 stopped at
    1800 and 1975 of 2000 iterations, i.e. the held-out error was still at or
    near its best when the run ended. Those gains are a floor, not a ceiling,
    and equally the stopping point was not freely chosen -- it was cut off.
  * L020 IS THE ONE THAT BEHAVED AS THE LITERATURE PREDICTS, and it is the most
    informative frame here: held-out error fell to 0.0620 at iteration 1925 and
    then rose to 0.4136 by 2000. That is the network starting to fit the noise,
    caught by data it never saw. It is also the evidence that the early stopping
    is doing real work rather than decorating a monotonic curve.

WHAT THIS DOES NOT SHOW. The hold-out asks the model to predict dots that were
withheld, which is interpolation. Doing that better than neighbour-fill means
the model represents the image better, and that supports denoising -- but it is
not the same measurement as denoising, and no claim of a denoising gain is made
here. `n2n.py` measures that directly on the colour triplets, and it has a blur
control, which this does not.

WHAT IT CANNOT DO. Not resolution. The 1977 camera resolved 138-172 elements
along a trace and we already sample 230; across traces it resolved 260-324 and
we sample 512. There is no hidden detail to uncover, and any that appears is
invented. What is genuinely available is DENOISING -- and denoising is testable,
because a real denoiser predicts measurements it was not shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import forward as fwd
from .neuralfield import _baseline_predict, _checker_mask, _device, residual_params


class Decoder(torch.nn.Module):
    """A small U-net-ish decoder. Depth and width ARE the prior, so they matter.

    Deliberately modest: a larger network fits noise sooner, which moves the
    early-stopping point earlier and buys nothing. Skip connections are omitted
    for the same reason -- they make it far easier for the network to reproduce
    its input exactly, which is precisely the behaviour we do not want.
    """

    def __init__(self, in_ch: int = 32, width: int = 64, depth: int = 4, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        layers: list[torch.nn.Module] = []
        c = in_ch
        for _ in range(depth):
            layers += [
                torch.nn.Conv2d(c, width, 3, padding=1, padding_mode="reflect"),
                torch.nn.BatchNorm2d(width),
                torch.nn.LeakyReLU(0.1, inplace=True),
                torch.nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
                torch.nn.BatchNorm2d(width),
                torch.nn.LeakyReLU(0.1, inplace=True),
            ]
            c = width
        layers += [torch.nn.Conv2d(width, 1, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)[0, 0]


@dataclass
class DipResult:
    scene: np.ndarray
    best_iter: int
    heldout_rmse: float
    baseline_rmse: float
    train_mse: float
    curve: list[tuple[int, float, float]]   # (iter, train mse, held-out rmse)

    @property
    def beats_baseline(self) -> bool:
        return self.heldout_rmse < self.baseline_rmse

    @property
    def operator_fits(self) -> bool:
        return self.train_mse < 0.25


def fit(dots: np.ndarray, params: fwd.ChainParams | None = None, *,
        iters: int = 3000, lr: float = 1e-2, keep: float = 0.85,
        width: int = 64, depth: int = 4, seed: int = 0,
        check_every: int = 25, noise_sd: float = 0.03) -> DipResult:
    """Fit a decoder to one frame, stopping where the WITHHELD dots say to stop.

    `keep` of the dots are visible to the loss. The rest are scored every
    `check_every` iterations and the best-scoring iterate is the one returned --
    which is early stopping chosen by data the fit never saw, rather than by
    looking at the picture.
    """
    params = params or residual_params(rows=dots.shape[0])
    dev = _device()
    h, w = dots.shape
    y = np.asarray(dots, dtype=np.float64)
    y = (y - y.mean()) / (y.std() + 1e-12)

    mask = _checker_mask(h, w, keep, seed)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    mt = torch.tensor(mask, dtype=torch.bool, device=dev)

    g = torch.Generator(device="cpu").manual_seed(seed)
    z0 = torch.randn((1, 32, h, w), generator=g).to(dev)
    net = Decoder(width=width, depth=depth, seed=seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    best = (float("inf"), 0, None, float("nan"))
    curve: list[tuple[int, float, float]] = []
    held = ~mask
    for i in range(iters):
        opt.zero_grad(set_to_none=True)
        # Jittering the input each step is the standard DIP regulariser: it
        # slows the network's descent into fitting individual noise samples
        # without changing what it is being asked to explain.
        z = z0 + noise_sd * torch.randn_like(z0)
        scene = net(z)
        pred = fwd.forward(scene, params)
        loss = torch.mean((pred[mt] - yt[mt]) ** 2)
        loss.backward()
        opt.step()

        if i % check_every == 0 or i == iters - 1:
            with torch.no_grad():
                p = fwd.forward(net(z0), params).cpu().numpy().astype(np.float64)
            hr = float(np.sqrt(np.mean((p[held] - y[held]) ** 2)))
            tm = float(np.mean((p[mask] - y[mask]) ** 2))
            curve.append((i, tm, hr))
            if hr < best[0]:
                with torch.no_grad():
                    sc = net(z0).cpu().numpy().astype(np.float64)
                best = (hr, i, sc, tm)

    br = float(np.sqrt(np.mean((_baseline_predict(y, mask)[held] - y[held]) ** 2)))
    return DipResult(scene=best[2] if best[2] is not None else np.zeros_like(y),
                     best_iter=best[1], heldout_rmse=best[0], baseline_rmse=br,
                     train_mse=best[3], curve=curve)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    from pathlib import Path
    from . import sync as sync_mod, decode as decode_mod, wav, catalog as catalog_mod

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="L000,L020,L055")
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()

    REPO = Path(__file__).resolve().parent.parent
    info = wav.probe(REPO / "data" / "master" / "384kHzStereo.wav")
    mm = wav.memmap(info)
    cat = catalog_mod.build()

    print(f"device: {_device()}")
    print("Early stopping is chosen by WITHHELD dots, not by eye and not by an")
    print("image metric -- both of which reward the point where it looks nicest.\n")

    rows = []
    for fid in args.frames.split(","):
        fr = cat.by_id(fid)
        seg = np.asarray(mm[fr.seed_sample: fr.seed_sample + int(sync_mod.NOMINAL_PERIOD * 520),
                            fr.channel], dtype=np.float64)
        tb = sync_mod.recover(seg)
        dec = decode_mod.decode(seg, decode_mod.Settings(traces=512, rotate=0), tb=tb)
        r = fit(dec.image, iters=args.iters)
        rows.append((fid, r))
        gain = (r.baseline_rmse - r.heldout_rmse) / r.baseline_rmse * 100
        verdict = ("BEATS" if r.beats_baseline else "loses to") if r.operator_fits \
                  else "OPERATOR MISMATCH -- no conclusion from"
        print(f"  {fid}  stopped at iter {r.best_iter:5d}   train mse {r.train_mse:.4f}   "
              f"held-out {r.heldout_rmse:.4f}   neighbour-fill {r.baseline_rmse:.4f}   "
              f"{gain:+6.1f}%   {verdict} baseline")
        # The shape of the curve is the finding, whichever way it goes: a real
        # method's held-out error falls then rises (it starts fitting noise); a
        # method with nothing to offer never falls below the baseline at all.
        lo = min(c[2] for c in r.curve)
        print(f"        held-out over the run: start {r.curve[0][2]:.4f}  "
              f"best {lo:.4f} at iter {r.best_iter}  end {r.curve[-1][2]:.4f}"
              f"{'   (rises again -- fitting noise, as documented)' if r.curve[-1][2] > lo * 1.02 else ''}")

    usable = [r for _, r in rows if r.operator_fits]
    wins = sum(1 for r in usable if r.beats_baseline)
    print(f"\n{len(usable)}/{len(rows)} usable fits; {wins} beat neighbour-fill on withheld dots.")
    if usable and wins == 0:
        print("VERDICT: no support for DIP on this signal. A real null -- report it.")
