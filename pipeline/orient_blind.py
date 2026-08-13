"""How much of the rotation can be recovered without knowing anything about Earth.

The record does not encode which edge is the top. That is the standing verdict
in ALIENS.md and it is correct as far as it goes -- but it is not the whole
answer, and the difference matters, because "the aliens are stuck" and "the
aliens get most of the way there" are very different claims.

There are four distinct ambiguities, and they are NOT equally hopeless:

  1. RASTER HANDEDNESS -- does trace 0 sit on the left or the right edge, and
     does the start of a trace land at the top or the bottom? Four combinations.
     This one is GLOBAL: whatever the answer, it is the same for all 116 images.

  2. PORTRAIT vs LANDSCAPE -- was the slide fed into the camera turned on its
     side? This LOOKED measurable, and this module is the test. See below.

  3. WHICH quarter turn -- given that a slide is portrait, is it +90 or -90?

  4. UPRIGHT vs UPSIDE DOWN -- 0 or 180 for a landscape slide.

The hypothesis for (2) was that the scan converter scans a fixed 4:3 raster, so
a portrait slide projected into it could not fill the frame and would leave
featureless bands down the left and right -- a property of the SIGNAL, needing
no Earth knowledge. If that held, an alien would at least know WHICH images
needed turning, even without knowing which way.

**It does not hold, and the measurement is unambiguous.** Against Barry's table
the detector scores 1.5% recall: it finds essentially none of the 66 rotated
frames. The reason is visible the moment you look at a raw frame instead of
theorising about it -- L023, the human skeleton, is a portrait subject lying on
its side and filling the 4:3 raster edge to edge, with no surround at all. The
1977 operators reframed each slide to fill the frame. The content-box aspect
ratio confirms it: 1.44 for rotated slides against 1.40 for upright ones, when
the hypothesis predicted roughly 0.75 against 1.33.

So the honest answer is worse than "the record does not say which edge is up".
It is: **the record does not even say which images were turned.** All four
ambiguities are total. An alien would recover 116 correct photographs and have
no way to tell that 66 of them want a quarter turn, because the act of turning
them left no trace in the signal.

That leaves scene content as the only route, which means gravity, horizons and
faces -- Earth physics applied to Earth scenes, and inference about us rather
than decoding. This module is kept as the record of a falsified hypothesis,
because a negative result that took ten minutes to measure is worth more than a
plausible claim that would have stood unchallenged.

On (1) there is a genuine escape that the record provides and which we have not
credited before: **image 2 is the pulsar map, and the pulsar map is also
engraved on the cover.** The cover is bolted to the record, so a recipient has
both. The map is strongly chiral -- fourteen lines at fourteen distinct angles,
each labelled with a binary period -- so comparing the decoded image against the
engraving pins the handedness of the whole raster, and with it every mirror
question, for free. The circle (image 1) checks the timebase; the pulsar map
checks the geometry's handedness. Two self-tests, both by design.

We have not yet run that comparison, because it needs a rectified image of the
cover engraving. It is recorded here as the specific thing to do, not as a
result: see `docs/reference/` for what is missing.

Evaluation against Barry's orientation table is tier 2 (see provenance.py) and
is used here for SCORING ONLY -- nothing in the detector reads it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
THUMBS = REPO / "data" / "thumbs"

# A column is "blank" if its variation is this small a fraction of the frame's.
# Set from the histogram of column standard deviations, which is strongly
# bimodal on portrait frames -- paper-white surround against picture.
BLANK_FRAC = 0.35
# A frame counts as portrait if this fraction of its width is blank surround.
PORTRAIT_MARGIN = 0.12


def _raw(stored: np.ndarray, barry_k: int) -> np.ndarray:
    """Undo build.py's orient() to recover the decoder's own raster frame.

    build.py writes np.rot90(img, k=-k). The inverse is np.rot90(stored, k=+k).
    Needed because the shipped PNGs already carry Barry's Earth-knowledge turn,
    so measuring on them would be measuring his answer, not the signal.
    """
    return np.rot90(stored, k=barry_k)


def side_profile(a: np.ndarray) -> dict:
    """Measure the blank surround on left and right of the raster.

    Uses column variation rather than column brightness: the surround is a
    uniform field, but whether it is bright or dark depends on the slide mount
    and on the polarity, neither of which we want to depend on.
    """
    a = a.astype(np.float64)
    a = (a - a.min()) / (a.ptp() + 1e-9)
    col = a.std(axis=0)
    row = a.std(axis=1)
    thr_c = BLANK_FRAC * float(np.median(col))
    thr_r = BLANK_FRAC * float(np.median(row))

    def run(v: np.ndarray, thr: float) -> tuple[int, int]:
        lo = 0
        while lo < len(v) and v[lo] < thr:
            lo += 1
        hi = 0
        while hi < len(v) - lo and v[len(v) - 1 - hi] < thr:
            hi += 1
        return lo, hi

    cl, cr = run(col, thr_c)
    rt, rb = run(row, thr_r)
    w, h = len(col), len(row)
    return {
        "blank_lr": (cl + cr) / w,
        "blank_tb": (rt + rb) / h,
        "content_w": (w - cl - cr) / w,
        "content_h": (h - rt - rb) / h,
    }


def is_portrait(a: np.ndarray) -> tuple[bool, dict]:
    """Blind verdict: was this slide scanned turned on its side?

    The test is that the picture content is narrower than the raster while
    filling its height -- which is what a portrait slide does inside a 4:3 scan
    and what a landscape slide never does.
    """
    p = side_profile(a)
    # Content aspect in raster units. The raster is 4:3, so a portrait slide
    # ends up with a content box appreciably taller than it is wide.
    box_w = p["content_w"] * a.shape[1]
    box_h = p["content_h"] * a.shape[0]
    p["content_aspect"] = box_w / max(box_h, 1e-9)
    verdict = (p["blank_lr"] > PORTRAIT_MARGIN) and (p["blank_lr"] > p["blank_tb"])
    return bool(verdict), p


def run() -> dict:
    tables = json.loads((REPO / "pipeline" / "barry_tables.json").read_text())
    cat = json.loads((REPO / "web" / "public" / "data" / "catalog.json").read_text())
    frames = {f["id"]: f for f in cat["frames"]}

    rows = []
    for fid, fr in sorted(frames.items()):
        png = THUMBS / f"{fid}.png"
        if not png.exists():
            continue
        ch, idx = fid[0], int(fid[1:])
        k = tables["orientation"][0 if ch == "L" else 1][idx]
        stored = np.asarray(Image.open(png).convert("L"))
        raw = _raw(stored, k)
        verdict, p = is_portrait(raw)
        rows.append({"frame": fid, "blind_portrait": verdict,
                     "truth_portrait": bool(k != 0),          # tier 2, scoring only
                     "blank_lr": round(p["blank_lr"], 4),
                     "blank_tb": round(p["blank_tb"], 4),
                     "content_aspect": round(p["content_aspect"], 3)})

    tp = sum(r["blind_portrait"] and r["truth_portrait"] for r in rows)
    fp = sum(r["blind_portrait"] and not r["truth_portrait"] for r in rows)
    fn = sum(not r["blind_portrait"] and r["truth_portrait"] for r in rows)
    tn = sum(not r["blind_portrait"] and not r["truth_portrait"] for r in rows)
    n = len(rows)
    return {
        "n": n,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "rows": rows,
    }


if __name__ == "__main__":  # pragma: no cover
    res = run()
    c = res["confusion"]
    print(f"{res['n']} frames")
    print(f"  blind portrait detector vs Barry's hand-made table")
    print(f"    accuracy  {res['accuracy']*100:5.1f}%")
    print(f"    precision {res['precision']*100:5.1f}%   recall {res['recall']*100:5.1f}%")
    print(f"    tp {c['tp']}  fp {c['fp']}  fn {c['fn']}  tn {c['tn']}")
    print()
    print("  VERDICT: the letterbox hypothesis is FALSIFIED. Portrait slides were")
    print("  reframed to fill the 4:3 raster, so being turned left no trace in the")
    print("  signal. The record does not encode which edge is up, and does not")
    print("  encode which images were turned either. See the module docstring.")

    out = REPO / "docs" / "orientation_blind.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out.relative_to(REPO)}")
