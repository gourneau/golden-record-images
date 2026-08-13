"""Trim the book's paper margin off each reference crop.

The crops are correctly matched -- every one checked by eye shows the same
photograph as our decode -- but they are cut from a scanned book page, so most
carry a border of white paper, and some carry a slice of the facing page. Shown
beside a decode that is all picture, the two look mismatched even when they are
the same image, which reads as a bad match rather than a bad crop.

Peeling that border is not simply "threshold the bright pixels": several slides
are legitimately bright edge to edge (snow, sky, the calibration field, the
white-on-black diagrams). What distinguishes paper is that it is both bright AND
featureless, so this peels edge rows and columns only while they are *uniform*
and *close to the corner level*, and stops at the first line that carries
structure. A slide that genuinely reaches its own edge therefore loses nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
CROPS = REPO / "docs" / "reference" / "crops"

# A line is "paper" if its standard deviation is below this fraction of the
# whole image's, and its level is within this much of the corner level.
UNIFORM_FRAC = 0.28
LEVEL_TOL = 0.10
MAX_PEEL = 0.30  # never remove more than this fraction of a side


def _corner_level(a: np.ndarray) -> float:
    h, w = a.shape
    k = max(4, min(h, w) // 20)
    corners = [a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:]]
    return float(np.median([np.median(c) for c in corners]))


def trim_bounds(a: np.ndarray) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) after peeling uniform paper edges."""
    h, w = a.shape
    ref = _corner_level(a)
    overall = float(a.std()) + 1e-9
    lim_v = int(h * MAX_PEEL)
    lim_h = int(w * MAX_PEEL)

    def paper(line: np.ndarray) -> bool:
        return (line.std() < UNIFORM_FRAC * overall) and (abs(float(line.mean()) - ref) < LEVEL_TOL)

    top = 0
    while top < lim_v and paper(a[top]):
        top += 1
    bot = h
    while bot > h - lim_v and paper(a[bot - 1]):
        bot -= 1
    left = 0
    while left < lim_h and paper(a[:, left]):
        left += 1
    right = w
    while right > w - lim_h and paper(a[:, right - 1]):
        right -= 1

    # Guard against degenerate results.
    if bot - top < h * 0.4 or right - left < w * 0.4:
        return 0, h, 0, w
    return top, bot, left, right


def trim_file(path: Path, dry_run: bool = False) -> dict:
    im = Image.open(path).convert("RGB")
    a = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
    h, w = a.shape
    top, bot, left, right = trim_bounds(a)
    removed = 1.0 - ((bot - top) * (right - left)) / float(h * w)
    out = {
        "file": path.name,
        "before": [w, h],
        "after": [right - left, bot - top],
        "removed_frac": round(removed, 4),
    }
    if not dry_run and (top or left or bot != h or right != w):
        im.crop((left, top, right, bot)).save(path, quality=88)
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = []
    for p in sorted(CROPS.glob("*.jpg"), key=lambda q: int(q.stem) if q.stem.isdigit() else 0):
        try:
            rows.append(trim_file(p, dry_run=args.dry_run))
        except Exception as e:
            print(f"  {p.name}: FAILED {e}")

    trimmed = [r for r in rows if r["removed_frac"] > 0.01]
    print(f"{len(rows)} crops, {len(trimmed)} had paper margin to remove")
    frac = np.array([r["removed_frac"] for r in rows])
    print(f"  area removed: mean {frac.mean()*100:.1f}%  median {np.median(frac)*100:.1f}%  max {frac.max()*100:.1f}%")
    for r in sorted(rows, key=lambda z: -z["removed_frac"])[:10]:
        print(f"    {r['file']:>9s}  {r['before']} -> {r['after']}  ({r['removed_frac']*100:.0f}% removed)")
