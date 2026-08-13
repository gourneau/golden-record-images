"""Register the three separations before compositing a colour image.

A defect hiding in plain sight: the twenty colour images on the gallery are
built by stacking three separations straight into R, G and B, and NOBODY
ALIGNED THEM. The registration code has existed in `fuse.py` for some time, and
`n2n.py` uses it before training -- but the shipped composites never did.

The misalignment is measured, not suspected (fuse.py section 2): across-trace
offsets are random with sd 0.55 px and reach 2.26 px on image 60, and the
along-trace part is a systematic creep of +0.33 px per frame -- the three scans
of one slide were taken minutes apart while the tape transport drifted, so the
third separation is reliably about two thirds of a pixel further along than the
first. At 2.26 px on a 512-wide frame that is a visible colour fringe, and on
every colour image it is at least a softening, because stacking three
mutually-shifted planes into RGB blurs the luminance as well as fringing the
colour.

WHAT THIS IS NOT. It is not a correction to the decode: each separation is
already decoded correctly. It is a correction to how three CORRECT decodes are
combined, which is why it belongs here and not in decode.py -- and why it stays
provenance tier 0. Nothing about which plane is red is used (that would be tier
2, and it is not needed): the planes are aligned onto the MIDDLE one in record
order, and alignment is symmetric in colour.

The estimator is fuse.register: band-limited phase correlation below 0.25
cyc/px, where the 1977 camera's response actually lives, with upsampled-DFT
refinement. Band-limiting matters -- above the camera's response the planes
share no signal, only noise, and a full-band phase correlation would be trying
to align two different noise fields.

THE CHECK THAT MATTERS is not that the offsets look plausible. It is that
registration REDUCES the disagreement between planes where they should agree.
Colour separations of one scene share their luminance and differ in chroma, so
the residual after aligning must fall in the band where the camera has response
and must not be manufactured out of the noise band. Reported per image, so a
triplet the estimator gets wrong is visible rather than averaged away.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import fuse as fuse_mod

REPO = Path(__file__).resolve().parent.parent
THUMBS = REPO / "data" / "thumbs"
OUT = REPO / "data" / "thumbs_reg"
REPORT = REPO / "docs" / "colour_registration.json"


def _planes(frames: list[str], src: Path) -> list[np.ndarray] | None:
    out = []
    for fid in frames:
        p = src / f"{fid}.png"
        if not p.exists():
            return None
        out.append(np.asarray(Image.open(p).convert("L"), dtype=np.float64))
    shapes = {a.shape for a in out}
    return out if len(shapes) == 1 else None


def _disagreement(planes: list[np.ndarray]) -> float:
    """How much the planes disagree, in the band where they should agree.

    Low-pass first: the separations genuinely differ in chroma at low
    frequencies and share nothing but noise at high ones, so the honest place to
    measure alignment is the middle -- structure the camera resolved, common to
    all three. Measured as the rms of each outer plane minus the middle one
    after removing each plane's own mean and scale, so a pure tone difference
    between channels does not read as misalignment.
    """
    from scipy.ndimage import gaussian_filter
    z = []
    for a in planes:
        b = gaussian_filter(a, 1.0) - gaussian_filter(a, 6.0)   # band-pass
        z.append(b / (b.std() + 1e-9))
    return float(np.sqrt(np.mean((z[0] - z[1]) ** 2 + (z[2] - z[1]) ** 2) / 2))


# The misalignment is a measured quantity, not an unknown: fuse.py finds sd
# 0.55 px across-trace and a maximum of 2.26 px over all twenty triplets, and
# the along-trace creep is +0.33 px per frame. So a shift of more than a few
# pixels is not a large misalignment, it is a FAILED ESTIMATE, and applying it
# would wreck an image that was merely slightly soft.
#
# This is not a hypothetical guard. Image 8, the solar spectrum, returns
# -104 px: phase correlation assumes the planes share a scene, and on a spectrum
# the content MOVES WITH WAVELENGTH, so the three "separations" genuinely are
# three different pictures. n2n.py excludes image 8 for exactly this reason. The
# cap catches it on the physics rather than by name, which is what makes it a
# rule instead of a special case.
MAX_PLAUSIBLE_SHIFT = 8.0   # px; measured worst case is 2.26


def register_image(n: int, frames: list[str], src: Path = THUMBS,
                   dst: Path = OUT) -> dict:
    planes = _planes(frames, src)
    if planes is None:
        return {"n": n, "ok": False, "reason": "missing or mismatched planes"}

    before = _disagreement(planes)
    reg, offsets = fuse_mod.register(planes)
    big = max((max(abs(float(o.get("dy", 0.0))), abs(float(o.get("dx", 0.0))))
               for o in offsets), default=0.0)
    rejected = big > MAX_PLAUSIBLE_SHIFT
    if rejected:
        reg = planes                      # ship it unregistered rather than wrong
    after = _disagreement(reg)

    dst.mkdir(parents=True, exist_ok=True)
    for fid, a in zip(frames, reg):
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(dst / f"{fid}.png",
                                                                  optimize=True)
    shifts = [(round(float(o.get("dy", 0.0)), 3), round(float(o.get("dx", 0.0)), 3))
              for o in offsets]
    mag = max((abs(dy) + abs(dx)) for dy, dx in shifts) if shifts else 0.0
    return {"n": n, "ok": True, "frames": frames, "shifts": shifts,
            "max_shift_px": round(float(mag), 3),
            "rejected": bool(rejected),
            "reason": ("estimate implausible (> %.0f px); shipped unregistered"
                       % MAX_PLAUSIBLE_SHIFT) if rejected else None,
            "disagreement_before": round(before, 5),
            "disagreement_after": round(after, 5),
            "improved": bool(after < before)}


def run(src: Path = THUMBS, dst: Path = OUT) -> dict:
    cat = json.loads((REPO / "web" / "public" / "data" / "catalog.json").read_text())
    colour = [im for im in cat["images"] if im.get("color") and len(im["frames"]) == 3]
    rows = [register_image(im["n"], im["frames"], src, dst) for im in colour]
    ok = [r for r in rows if r.get("ok")]
    improved = [r for r in ok if r["improved"]]
    rejected = [r for r in ok if r.get("rejected")]
    rep = {
        "n_colour_images": len(colour), "n_registered": len(ok),
        "n_improved": len(improved),
        "n_rejected": len(rejected),
        "rejected_images": [r["n"] for r in rejected],
        "max_shift_px": round(max((r["max_shift_px"] for r in ok), default=0.0), 3),
        "mean_disagreement_before": round(float(np.mean([r["disagreement_before"] for r in ok])), 5),
        "mean_disagreement_after": round(float(np.mean([r["disagreement_after"] for r in ok])), 5),
        "images": rows,
    }
    REPORT.write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(THUMBS))
    ap.add_argument("--dst", default=str(OUT))
    args = ap.parse_args()

    rep = run(Path(args.src), Path(args.dst))
    print(f"{rep['n_registered']}/{rep['n_colour_images']} colour images registered "
          f"-> {Path(args.dst).relative_to(REPO)}")
    print(f"  largest correction applied: {rep['max_shift_px']:.2f} px")
    print(f"  plane disagreement (band-passed, lower is better):")
    print(f"    before {rep['mean_disagreement_before']:.5f}   "
          f"after {rep['mean_disagreement_after']:.5f}   "
          f"({(rep['mean_disagreement_after']/rep['mean_disagreement_before']-1)*100:+.1f}%)")
    print(f"  improved on {rep['n_improved']}/{rep['n_registered']} images")
    print()
    worst = sorted((r for r in rep["images"] if r.get("ok")),
                   key=lambda r: -r["max_shift_px"])[:6]
    print("  largest misalignments found:")
    for r in worst:
        print(f"    image {r['n']:3d}  {r['max_shift_px']:.2f} px  "
              f"shifts {r['shifts']}  "
              f"{'improved' if r['improved'] else 'NO IMPROVEMENT'}")
    if rep["n_rejected"]:
        print(f"\n  ESTIMATE REJECTED on images {rep['rejected_images']} and shipped "
              f"unregistered: the shift exceeded {MAX_PLAUSIBLE_SHIFT:.0f} px, which is not a "
              f"large misalignment but a failed estimate (measured worst case is 2.26 px).")
