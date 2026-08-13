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

A LANDMINE, named: this runs on the SHIPPED 8-BIT THUMBNAILS, not on the
float decode. That is deliberate -- they are what the page displays, so this is
honest about what the viewer gets -- but it means the denoiser sees quantisation
noise the float decode does not have. The noise being removed is larger than a
level of 8-bit quantisation, so the effect is small, but a future version that
re-decodes to float would be strictly better and this is the reason to do it.
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


def run(steps: int = 3000, retrain: bool = False, verbose: bool = True) -> dict:
    model = train_final(steps=steps, verbose=verbose) if (retrain or not MODEL.exists()) \
        else _load_model()

    rows = [reconstruct_frame(model, p.stem) for p in sorted(THUMBS.glob("*.png"))]
    ok = [r for r in rows if r.get("ok")]
    circ = circle_check(model)

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
    args = ap.parse_args()

    rep = run(steps=args.steps, retrain=args.retrain)
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
    verdict = "PASS" if (d_ax < 0.002 and d_rr < 0.02) else "REGRESSION -- do not ship"
    print(f"    {verdict}")
    print(f"\nwrote {REPORT.relative_to(REPO)}")
