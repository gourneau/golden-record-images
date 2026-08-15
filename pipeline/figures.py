"""Regenerate the README figures from the current pipeline.

They were made by hand once and then rotted, which is a failure this project has
now had three times in different forms: `build.py` silently dropped the tier-2
presentation metadata, `n2n.py`'s docstring described a model that no longer
existed, and the README's hero image showed a decode from before the chain
correction while captioned as this decoder's output.

The common cause is that the RECORD OF THE WORK is produced by a different
process from the work, so nothing fails when they drift apart. A committed
figure is a claim about current behaviour, and a claim nothing checks is a claim
that will eventually be false. This module makes the figures reproducible, and
`--check` makes the drift detectable.

Run `python -m pipeline.figures` after anything that changes the decode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs" / "img"
CAL = "L000"

# Barry's 2017 settings, for the comparison figure. His picture window was 2680
# samples against the 2808 measured here, which is the 7% that made his circle
# an ellipse he had to nudge by hand -- so this is the figure's whole point and
# it must be decoded, not asserted.
BARRY_PICTURE_SAMPLES = 2680.0


def _decode(fid: str, **over):
    from . import sync as sync_mod, decode as decode_mod, wav, catalog as catalog_mod
    info = wav.probe(REPO / "data" / "master" / "384kHzStereo.wav")
    mm = wav.memmap(info)
    fr = catalog_mod.build().by_id(fid)
    n = int(sync_mod.NOMINAL_PERIOD * 520)
    seg = np.asarray(mm[fr.seed_sample: fr.seed_sample + n, fr.channel], dtype=np.float64)
    cfg = decode_mod.Settings(traces=512, rotate=0, channel=fid[0], **over)
    return decode_mod.decode(seg, cfg, sync_mod.recover(seg)).image


def _png(a) -> Image.Image:
    return Image.fromarray((np.clip(np.asarray(a, float), 0, 1) * 255 + 0.5).astype(np.uint8))


def _from_thumb(name: str, fid: str = CAL) -> np.ndarray:
    return np.asarray(Image.open(REPO / "data" / name / f"{fid}.png").convert("L"), float)


def level_fall(a: np.ndarray) -> float:
    """How far the field drops from the top of a trace to the bottom, in grey."""
    c = np.asarray(a, float)[15:345, 25:485].mean(axis=1)
    return float(c[0] - c[-1])


def build(verbose: bool = True) -> dict:
    IMG.mkdir(parents=True, exist_ok=True)
    out = {}

    # 1. Barry's method: his picture window, no dot clock, no chain correction.
    barry = _decode(CAL, picture_span=BARRY_PICTURE_SAMPLES / 3200.0,
                    dot_lock=False, uncumulate=0.0)
    _png(barry).save(IMG / "calibration-barry.png", optimize=True)

    # 2. This decoder, as it currently ships: the denoised tier the gallery
    #    defaults to. Taken from the shipped file rather than re-decoded, so the
    #    figure is literally what a visitor sees.
    cur = _from_thumb("thumbs_ml")
    Image.fromarray(cur.astype(np.uint8)).save(IMG / "calibration-restored.png", optimize=True)

    # 3. The chain correction, before and after, on the field it is measured on.
    nc = _from_thumb("thumbs_nc")
    h, w = nc.shape
    pair = Image.new("L", (w * 2 + 10, h), 20)
    pair.paste(Image.fromarray(nc.astype(np.uint8)), (0, 0))
    pair.paste(Image.fromarray(cur.astype(np.uint8)), (w + 10, 0))
    pair.save(IMG / "chain-correction.png", optimize=True)

    # 4. The link preview card, from the same shipped frames. Part of this run
    #    rather than a separate command, so it cannot be the one figure that
    #    was forgotten -- which is the failure this whole module exists for.
    card = build_card(verbose=verbose)

    from . import quality
    mb, mc = quality.circle_metrics(barry), quality.circle_metrics(cur / 255.0)
    out = {"social_card": card["images"],
        "barry_axis_ratio": round(float(mb["axis_ratio"]), 4),
        "current_axis_ratio": round(float(mc["axis_ratio"]), 4),
        "level_fall_before": round(level_fall(nc), 1),
        "level_fall_after": round(level_fall(cur), 1),
    }
    write_manifest()
    if verbose:
        print(f"  calibration-barry.png     axis ratio {out['barry_axis_ratio']:.4f}  "
              f"(his 2680-sample window)")
        print(f"  calibration-restored.png  axis ratio {out['current_axis_ratio']:.4f}  "
              f"(the shipped denoised tier)")
        print(f"  chain-correction.png      field falls {out['level_fall_before']:.0f} -> "
              f"{out['level_fall_after']:.0f} grey levels")
    return out


# --------------------------------------------------------------------------
# the link preview card
# --------------------------------------------------------------------------
# What a person sees before they see the page: pasted into a chat window or a
# timeline, a link without og:image is a grey rectangle and a URL. This is a
# gallery, so the card is made of the gallery -- real decoded frames, composited
# here the same way the browser composites them, so the card cannot show
# something the page does not.
#
# Built from the shipped thumbnails rather than drawn by hand, for the reason
# every other figure here is: a hand-made card is a claim about the decode that
# nothing checks, and this project has watched three of those rot.

CARD = "social-card.png"
CARD_W, CARD_H = 1200, 630
BAND = 196                       # the text band across the foot
BG = (10, 11, 13)
GOLD = (201, 162, 39)

# Ten frames that read at thumbnail size and between them say what the record
# is: the calibration circle it is decoded against, a planet, a diagram, a
# building, a dancer, a bird, and Earth from space. Four are colour, composited
# here exactly as the browser composites them.
#
# Chosen for a preview that appears with no context and no warning -- expanded
# automatically in somebody's timeline. The record contains documentary
# photographs of unclothed people, which belong in the gallery and do not belong
# on a card that opens itself.
CARD_PICKS = [1, 12, 11, 112, 90, 64, 104, 78, 114, 39]


def _oriented(n: int, pres: dict, cat: dict):
    """One image as the gallery shows it: colour composited, then turned."""
    import json
    entry = cat["images"][n - 1]
    assert entry["n"] == n, f"catalog is not in image order at {n}"
    planes = [Image.open(REPO / "data" / "thumbs_ml" / f"{f}.png").convert("L")
              for f in entry["frames"]]
    img = (Image.merge("RGB", planes[:3]) if entry.get("color") and len(planes) >= 3
           else planes[0].convert("RGB"))
    turn = int((pres["images"].get(str(n)) or {}).get("displayRotate", 0))
    if turn:
        img = img.rotate(-turn, expand=True)      # PIL turns anticlockwise
    return img


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """Fill w x h, cropping the overflow -- a contact sheet, not letterboxes."""
    s = max(w / img.width, h / img.height)
    r = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return r.crop(((r.width - w) // 2, (r.height - h) // 2,
                   (r.width - w) // 2 + w, (r.height - h) // 2 + h))


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    for p in ("/System/Library/Fonts/SFNS.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        try:
            f = ImageFont.truetype(p, size)
            try:
                f.set_variation_by_name("Bold" if bold else "Regular")
            except Exception:
                pass
            return f
        except Exception:
            continue
    return ImageFont.load_default()


def build_card(verbose: bool = True) -> dict:
    import json
    from PIL import ImageDraw
    cat = json.loads((REPO / "web" / "public" / "data" / "catalog.json").read_text())
    pres = json.loads((REPO / "docs" / "presentation.json").read_text())

    card = Image.new("RGB", (CARD_W, CARD_H), BG)
    cols, rows = 5, 2
    tw, th = CARD_W // cols, (CARD_H - BAND) // rows
    for i, n in enumerate(CARD_PICKS[: cols * rows]):
        tile = _cover(_oriented(n, pres, cat), tw, th)
        card.paste(tile, ((i % cols) * tw, (i // cols) * th))

    # Sink the mosaic towards the band so the text sits on darkness rather than
    # on somebody's face.
    top = card.crop((0, 0, CARD_W, CARD_H - BAND)).convert("RGB")
    px = top.load()
    fade_from = (CARD_H - BAND) - 90
    for y in range(fade_from, CARD_H - BAND):
        k = 1.0 - 0.85 * (y - fade_from) / 90.0
        for x in range(CARD_W):
            r, g, b = px[x, y]
            px[x, y] = (int(r * k), int(g * k), int(b * k))
    card.paste(top, (0, 0))

    d = ImageDraw.Draw(card)
    d.rectangle([0, CARD_H - BAND, CARD_W, CARD_H], fill=BG)
    d.line([(0, CARD_H - BAND), (CARD_W, CARD_H - BAND)], fill=(46, 40, 20), width=2)
    y = CARD_H - BAND + 34
    d.text((56, y), "The 116 pictures on the", font=_font(44, True), fill=(232, 234, 237))
    d.text((56, y + 54), "Voyager Golden Record", font=_font(44, True), fill=GOLD)
    d.text((56, y + 120),
           "Decoded from the sound they are stored as, on a record launched in 1977",
           font=_font(23), fill=(154, 163, 173))

    out = IMG / CARD
    card.save(out, optimize=True)
    info = {"file": str(out.relative_to(REPO)), "size": [CARD_W, CARD_H],
            "bytes": out.stat().st_size, "images": CARD_PICKS[: cols * rows]}
    if verbose:
        print(f"  {CARD}          {CARD_W}x{CARD_H}, {info['bytes'] / 1024:.0f} KiB, "
              f"images {info['images']}")
    return info


# The figures depend on these and on nothing else that matters. Change any of
# them and a committed figure may no longer show what the code does.
SOURCES = ("pipeline/decode.py", "pipeline/sync.py", "pipeline/dotclock.py",
           "pipeline/figures.py")
MANIFEST = IMG / "FIGURES.json"


def fingerprint() -> str:
    """A hash of the code that determines what the figures look like.

    NOT modification times. The first version of this check compared the
    figures' mtimes against the decode's, which works in my own tree and is
    meaningless anywhere else: git does not preserve mtimes, so in a fresh clone
    -- or in CI -- every file has the same checkout timestamp and the check
    silently passes whatever the truth is. It would have reported "all figures
    current" on a repository where they were a year out of date.
    """
    import hashlib
    h = hashlib.sha256()
    for rel in SOURCES:
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    return h.hexdigest()[:16]


def write_manifest() -> dict:
    import hashlib, json
    figs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(IMG.glob("*.png"))}
    m = {"source_fingerprint": fingerprint(), "figures": figs,
         "note": ("Regenerate with `python -m pipeline.figures`. The fingerprint is a hash of "
                  "the code that determines what these images look like, so a stale figure is "
                  "detectable in any clone -- which mtimes are not.")}
    MANIFEST.write_text(json.dumps(m, indent=1) + "\n")
    return m


def check() -> list[str]:
    """Do the committed figures match the code that is committed beside them?

    Works in a fresh clone and in CI, where the master WAV is absent and nothing
    can be re-decoded -- which is exactly where a stale figure would otherwise
    go unnoticed.
    """
    import hashlib, json
    problems = []
    if not MANIFEST.exists():
        return [f"{MANIFEST.relative_to(REPO)} missing -- run `python -m pipeline.figures`"]
    m = json.loads(MANIFEST.read_text())
    if m.get("source_fingerprint") != fingerprint():
        problems.append("the decode has changed since the figures were built "
                        "-- run `python -m pipeline.figures`")
    for name, want in (m.get("figures") or {}).items():
        p = IMG / name
        if not p.exists():
            problems.append(f"docs/img/{name} is missing")
        elif hashlib.sha256(p.read_bytes()).hexdigest()[:16] != want:
            problems.append(f"docs/img/{name} does not match the manifest")
    return problems


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report stale figures instead of rebuilding them")
    args = ap.parse_args()

    if args.check:
        bad = check()
        print("\n".join(f"  STALE: {b}" for b in bad) if bad
              else "  figures match the code committed beside them")
        raise SystemExit(1 if bad else 0)

    print("regenerating the README figures from the current pipeline")
    build()
    bad = check()
    print("\n  " + ("figures and manifest written" if not bad
                            else "STILL STALE: " + "; ".join(bad)))
