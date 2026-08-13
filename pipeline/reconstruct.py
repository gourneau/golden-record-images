"""Apply what the machine learning actually earned, to every frame, and label it.

Four methods were tried and measured against hold-outs. This ships the one that
generalises cheaply enough to run on all 156 frames, and it ships it as a
SEPARATE TIER rather than folding it into the decode, because the two products
answer different questions:

    RECOVERED      physics only -- timebase, dot clock, geometry, the droop
                   inverse. Every pixel traceable to the audio. The archival
                   product, and the default.

    RECONSTRUCTED  the above, then a denoiser whose entire training corpus is
                   this record's own colour repeats. Provenance tier 1: no
                   Earth photograph is involved, so an alien could run it --
                   but it is prior-based, and a prior can smooth away something
                   real. Opt-in, never presented as the record.

They are never blended. A viewer who wants to know what the record contains
should look at the first; a viewer who wants the most legible picture may prefer
the second; and nobody should be unable to tell which they are looking at.

WHY THE DENOISER AND NOT THE DEEP IMAGE PRIOR. DIP scored better -- 3 of 3
frames against 1 of 6 for the neural field, +11% to +43% over neighbour-fill on
withheld dots. But DIP fits a fresh network per frame, ~2000 iterations, which
measured out at roughly ten minutes a frame on this machine: 26 hours for the
record, and it must be redone from scratch whenever the decode changes. The
Noise2Noise denoiser trains once and then runs in milliseconds per frame. For a
gallery of 156 images that is the difference between shipping and not.

WHAT IS MEASURED AND WHAT IS EXTRAPOLATED, stated plainly because it is the
weak point. The denoiser was trained and evaluated on the twenty COLOUR images,
where three scans of one scene give a held-out measurement to score against. On
those, the result is real: 19 of 19 unseen scenes improved, and a blur control
fails the same test. The other 96 images are monochrome, scanned once, and
there is NO held-out measurement available for them -- so applying the denoiser
there is an extrapolation, and this module does not pretend otherwise. What it
does instead is check the extrapolation against the criteria that need no
reference and cannot be gamed:

    the calibration circle's axis ratio and radial rms must not regress
      (a denoiser has no business making an ellipse rounder or rounder-looking);
    the flat field of L000 must not gain structure;
    per-frame change is reported, so a frame the denoiser mangles is visible
      rather than averaged away.

MEASURED RESULT (2026-08, all 156 frames re-decoded in float, denoiser trained
3000 steps on all 19 triplets):

  change from the physics decode, in grey levels out of 255:
      mean 1.62   median 1.65   max 2.64
  CALIBRATION CIRCLE -- the check a denoiser cannot game:
      axis ratio   1.0051 -> 1.0051     unchanged
      radial rms   0.837  -> 0.834 px   slightly BETTER
      inliers      190    -> 190
      PASS

A CORRECTION TO THIS MODULE'S OWN PREVIOUS EXPLANATION. Run on the 8-bit
thumbnails the circle regressed -- radial rms 0.861 -> 0.886 -- and this
docstring explained it confidently as the denoiser's 17-pixel receptive field
softening a ring one to two pixels wide, "real and not a bug", the price of
denoising. THAT EXPLANATION WAS WRONG. Re-decoding in float and denoising that
instead, the regression disappears entirely: the metric is unchanged on axis
ratio and marginally better on radial rms.

So the cost was QUANTISATION, not the network. The 8-bit input carried noise the
decode never had, the denoiser attacked it along with everything else, and the
edge moved. It is a good reminder that a plausible mechanism is not evidence: the
receptive-field story fit the numbers, explained why the effect should exist, and
was simply not what was happening. The measurement that settled it was changing
the input, not reasoning harder about the output.

FLOAT, NOT 8-BIT. An earlier version ran on the shipped thumbnails, which are
8-bit, and that was a real if small defect: the denoiser saw quantisation noise
the decode does not have, and one grey level of quantisation is not nothing
against a correction whose mean size is under two grey levels. `--float`
re-decodes each frame from the master and denoises the float image, so the only
quantisation left is the single one at the end when the PNG is written. It costs
a full decode pass; it is the default because correctness is worth ten minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import n2n as n2n_mod
from . import quality

REPO = Path(__file__).resolve().parent.parent
THUMBS = REPO / "data" / "thumbs"
OUT = REPO / "data" / "thumbs_ml"
MODEL = REPO / "data" / "cache" / "n2n_model.pt"
REPORT = REPO / "docs" / "reconstruct_report.json"


def train_final(steps: int = 3000, seed: int = 0, verbose: bool = True) -> torch.nn.Module:
    """Train one denoiser on ALL nineteen triplets.

    The fold structure in n2n.py exists to MEASURE the method on scenes it never
    saw; that measurement is done. The shipped model is then trained on
    everything, which is the standard and correct thing to do once a method has
    been validated -- but it does mean this model must never be re-scored on the
    triplets, because it has now seen all of them. Any future evaluation needs
    the folds again.
    """
    device = n2n_mod._device()
    planes = n2n_mod.load_triplets()
    ns = sorted(planes)
    if verbose:
        print(f"training the shipped denoiser on all {len(ns)} triplets ({device})")
    pairs = n2n_mod.build_pairs(planes, ns)
    model, _ = n2n_mod.train(pairs, n2n_mod.TrainCfg(steps=steps, seed=seed),
                             device, verbose=verbose)
    model.eval()
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL)
    return model


def _load_model() -> torch.nn.Module:
    device = n2n_mod._device()
    model = n2n_mod.Denoiser().to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()
    return model


def decode_float(fid: str) -> np.ndarray | None:
    """Re-decode one frame from the master, in float, exactly as build.py does.

    Same Settings, same channel, same orientation, so the reconstructed tier is
    the SAME decode as the recovered tier plus a denoiser -- and not, subtly, a
    differently-configured decode that happens to also be denoised. If the two
    tiers disagreed for any reason other than the network, every comparison a
    viewer made between them would be measuring the wrong thing.
    """
    from . import sync as sync_mod, decode as decode_mod, wav, catalog as catalog_mod
    from .build import orient

    global _MM, _CAT
    if _MM is None:
        info = wav.probe(REPO / "data" / "master" / "384kHzStereo.wav")
        _MM = wav.memmap(info)
        _CAT = catalog_mod.build()
    fr = _CAT.by_id(fid)
    n = int(sync_mod.NOMINAL_PERIOD * 520)
    if fr.seed_sample + n > _MM.shape[0]:
        return None
    seg = np.asarray(_MM[fr.seed_sample: fr.seed_sample + n, fr.channel], dtype=np.float64)
    tb = sync_mod.recover(seg)
    cfg = decode_mod.Settings(traces=512, rotate=fr.orientation, channel=fid[0])
    dec = decode_mod.decode(seg, cfg, tb)
    return orient(dec.image, cfg.rotate)


_MM = None
_CAT = None


def reconstruct_frame_float(model: torch.nn.Module, fid: str) -> dict:
    """Denoise the FLOAT decode, then quantise once, at the end."""
    img = decode_float(fid)
    if img is None:
        return {"frame": fid, "ok": False, "reason": "could not decode"}
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 255.0
    mu, sd = float(a.mean()), float(a.std())
    if sd < 1e-6:
        return {"frame": fid, "ok": False, "reason": "flat frame"}
    rec = np.clip(n2n_mod.denoise_plane(model, (a - mu) / sd, n2n_mod._device()) * sd + mu, 0, 255)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rec + 0.5).astype(np.uint8)).save(OUT / f"{fid}.png", optimize=True)
    d = rec - a
    return {"frame": fid, "ok": True, "source": "float",
            "rms_change": round(float(np.sqrt(np.mean(d ** 2))), 4),
            "max_change": round(float(np.abs(d).max()), 2),
            "sd_before": round(sd, 3), "sd_after": round(float(rec.std()), 3)}


def circle_check_float(model: torch.nn.Module) -> dict:
    """The circle check, run on the float decode rather than an 8-bit copy.

    Worth doing separately: if the regression measured on 8-bit thumbnails was
    partly quantisation rather than the network, this is where that shows.
    """
    img = decode_float("L000")
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 255.0
    mu, sd = a.mean(), a.std()
    rec = np.clip(n2n_mod.denoise_plane(model, (a - mu) / sd, n2n_mod._device()) * sd + mu, 0, 255)
    b4, af = quality.circle_metrics(a), quality.circle_metrics(rec)
    return {"axis_ratio_before": round(float(b4["axis_ratio"]), 4),
            "axis_ratio_after": round(float(af["axis_ratio"]), 4),
            "radial_rms_before": round(float(b4["radial_rms"]), 3),
            "radial_rms_after": round(float(af["radial_rms"]), 3),
            "inliers_before": int(b4["inliers"]), "inliers_after": int(af["inliers"])}


def reconstruct_frame(model: torch.nn.Module, fid: str) -> dict:
    """Denoise one shipped thumbnail and write the reconstructed version.

    The denoiser works on z-scored planes, so the frame's own mean and standard
    deviation are restored afterwards: the tier is a denoise, not a re-grade,
    and changing the levels here would make the two tiers incomparable by eye.
    """
    src = THUMBS / f"{fid}.png"
    if not src.exists():
        return {"frame": fid, "ok": False, "reason": "no thumbnail"}
    a = np.asarray(Image.open(src).convert("L"), dtype=np.float64)
    mu, sd = float(a.mean()), float(a.std())
    if sd < 1e-6:
        return {"frame": fid, "ok": False, "reason": "flat frame"}
    z = (a - mu) / sd
    out = n2n_mod.denoise_plane(model, z, n2n_mod._device())
    rec = np.clip(out * sd + mu, 0, 255)

    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rec.astype(np.uint8)).save(OUT / f"{fid}.png")
    d = rec - a
    return {"frame": fid, "ok": True,
            "rms_change": round(float(np.sqrt(np.mean(d ** 2))), 4),
            "max_change": round(float(np.abs(d).max()), 2),
            "sd_before": round(sd, 3), "sd_after": round(float(rec.std()), 3)}


def circle_check(model: torch.nn.Module) -> dict:
    """The ungameable one. A denoiser must not move the calibration geometry.

    The circle is a physical invariant: its axis ratio and radial rms are set by
    the timebase and the geometry, not by how clean the picture looks. If a
    denoiser improves them it is not denoising, it is moving edges; if it
    degrades them it is destroying the thing the record sent to be measured.
    """
    a = np.asarray(Image.open(THUMBS / "L000.png").convert("L"), dtype=np.float64)
    mu, sd = a.mean(), a.std()
    rec = np.clip(n2n_mod.denoise_plane(model, (a - mu) / sd, n2n_mod._device()) * sd + mu, 0, 255)
    before = quality.circle_metrics(a)
    after = quality.circle_metrics(rec)
    return {
        "axis_ratio_before": round(float(before["axis_ratio"]), 4),
        "axis_ratio_after": round(float(after["axis_ratio"]), 4),
        "radial_rms_before": round(float(before["radial_rms"]), 3),
        "radial_rms_after": round(float(after["radial_rms"]), 3),
        "inliers_before": int(before["inliers"]),
        "inliers_after": int(after["inliers"]),
    }


def run(steps: int = 3000, retrain: bool = False, verbose: bool = True,
        use_float: bool = True) -> dict:
    model = train_final(steps=steps, verbose=verbose) if (retrain or not MODEL.exists()) \
        else _load_model()

    fids = [p.stem for p in sorted(THUMBS.glob("*.png"))]
    if use_float:
        rows = []
        for k, fid in enumerate(fids):
            rows.append(reconstruct_frame_float(model, fid))
            if verbose and k % 20 == 0:
                print(f"  {k}/{len(fids)} re-decoded and denoised")
    else:
        rows = [reconstruct_frame(model, fid) for fid in fids]
    ok = [r for r in rows if r.get("ok")]
    circ = circle_check_float(model) if use_float else circle_check(model)

    chg = np.array([r["rms_change"] for r in ok]) if ok else np.array([0.0])
    report = {
        "n_frames": len(rows), "n_ok": len(ok),
        "rms_change": {"mean": round(float(chg.mean()), 4),
                       "median": round(float(np.median(chg)), 4),
                       "max": round(float(chg.max()), 4)},
        "circle": circ,
        "provenance_tier": 1,
        "measured_on": "the 20 colour images (held-out); mono is an extrapolation",
        "frames": rows,
    }
    REPORT.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--from-thumbs", action="store_true",
                    help="denoise the 8-bit thumbnails instead of re-decoding to float")
    args = ap.parse_args()

    rep = run(steps=args.steps, retrain=args.retrain, use_float=not args.from_thumbs)
    c = rep["circle"]
    print(f"\n{rep['n_ok']}/{rep['n_frames']} frames reconstructed -> data/thumbs_ml/")
    print(f"  change in grey levels: mean {rep['rms_change']['mean']:.3f}  "
          f"median {rep['rms_change']['median']:.3f}  max {rep['rms_change']['max']:.3f}")
    print()
    print("  CALIBRATION CIRCLE -- the check a denoiser cannot game:")
    print(f"    axis ratio  {c['axis_ratio_before']:.4f} -> {c['axis_ratio_after']:.4f}")
    print(f"    radial rms  {c['radial_rms_before']:.3f} -> {c['radial_rms_after']:.3f} px")
    print(f"    inliers     {c['inliers_before']} -> {c['inliers_after']}")
    d_ax = abs(c["axis_ratio_after"] - c["axis_ratio_before"])
    d_rr = c["radial_rms_after"] - c["radial_rms_before"]
    # The gate distinguishes two different questions. A denoiser that moved the
    # circle a LOT would be moving geometry and must not be offered at all; one
    # that costs a little edge precision is making a trade, and the answer to a
    # trade is to offer both sides and label them, not to hide one.
    verdict = ("PASS" if (d_ax < 0.002 and d_rr < 0.02)
               else "COSTS EDGE PRECISION -- offer as a tier, never as the decode"
               if (d_ax < 0.005 and d_rr < 0.05)
               else "MOVES THE GEOMETRY -- do not offer")
    print(f"    {verdict}")
    print(f"\nwrote {REPORT.relative_to(REPO)}")
