"""The oracle tier: what would cheating actually buy?

THE QUESTION
------------
Our decoder is blind -- it never sees a reference image. That is the right
design, and it is what makes the result an archival recovery rather than a
guess. But it leaves an obvious question unanswered: how much are we losing by
being honest?

This module answers it by building a deliberately dishonest decoder. The ORACLE
is allowed to look at the true slide and pick, per image, whichever decoder
settings maximise agreement with it. It is not a decoder anyone could ship --
it needs the answer in order to produce the answer -- but the gap between it and
the blind decoder is a real and useful measurement:

    oracle - blind  =  what per-image parameter knowledge is worth

If that gap is large, our blind decoder is leaving recoverable quality on the
table and we should work out how to infer those parameters from the signal. If
it is small, then the remaining distance to the true slide is NOT about
parameter choice -- it is the camera's own resolution limit and the noise floor,
and no amount of cleverness in parameter selection will close it.

Either answer is worth having, and only one of them is flattering.

WHAT THE ORACLE MAY TUNE
------------------------
Only settings that are genuinely ambiguous per image -- gate placement, level
mapping, correction strengths. It may not, for instance, be handed the reference
as an initialiser or blended with it. The oracle picks settings; the pixels
still come from the audio.

The search runs in two stages (structure, then rendering) because the full grid
is ~200 decodes per image and the stages are close to separable.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NOMINAL = 3197.4


# --------------------------------------------------------------------------
# search space
# --------------------------------------------------------------------------

# Stage 1: structural choices that change what part of the signal becomes the
# picture, and which corrections run at all.
STRUCTURAL = [
    {"uncouple": u, "dot_lock": d, "picture_shift": g}
    for u in (True, False)
    for d in (True, False)
    for g in (-0.006, -0.003, 0.0, 0.003, 0.006)
]

# Stage 2: rendering choices, searched on top of the best structure.
RENDERING = [
    {"deconv": dc, "gamma": gm, "levels": lv}
    for dc in (0.0, 0.3, 0.6)
    for gm in (0.8, 1.0, 1.25)
    for lv in ("reference", "percentile")
]


def _settings(base, over: dict):
    from . import decode as D

    cfg = replace(base)
    shift = over.get("picture_shift", 0.0)
    kw = {k: v for k, v in over.items() if k != "picture_shift"}
    cfg = replace(cfg, **kw)
    if shift:
        cfg = replace(
            cfg,
            picture_start=cfg.picture_start + shift,
            picture_span=cfg.picture_span,
        )
    return cfg


def _score(img: np.ndarray, ref_path: Path) -> float:
    from . import evalset as E

    return float(E.compare(img, ref_path)["gradient_ncc"])


def optimise_one(args) -> dict:
    """Blind score and oracle score for one image. Runs in a worker process."""
    n, title, frame_id, channel, ref_path, start_sample = args
    import numpy as np

    from . import decode as D, sync as S, wav

    info = wav.probe(REPO / "data" / "master" / "384kHzStereo.wav")
    mm = wav.memmap(info)
    ch = 0 if channel == "L" else 1
    x = wav.read(info, ch, start_sample - 2000, int(540 * NOMINAL) + 14000, mm=mm).astype(np.float64)
    try:
        tb = S.recover(x)
    except Exception as e:  # a frame we cannot lock is not the oracle's fault
        return {"n": n, "title": title, "error": f"sync: {e}"}

    base = D.Settings(channel=channel)
    ref = Path(ref_path)

    try:
        blind_img = D.decode(x, base, tb).image
        blind = _score(blind_img, ref)
    except Exception as e:
        return {"n": n, "title": title, "error": f"blind decode: {e}"}

    best = {"score": blind, "over": {}}
    for over in STRUCTURAL:
        try:
            img = D.decode(x, _settings(base, over), tb).image
            s = _score(img, ref)
        except Exception:
            continue
        if s > best["score"]:
            best = {"score": s, "over": dict(over)}

    stage2 = dict(best["over"])
    for over in RENDERING:
        trial = {**stage2, **over}
        try:
            img = D.decode(x, _settings(base, trial), tb).image
            s = _score(img, ref)
        except Exception:
            continue
        if s > best["score"]:
            best = {"score": s, "over": dict(trial)}

    return {
        "n": n,
        "title": title,
        "frame": frame_id,
        "blind": blind,
        "oracle": best["score"],
        "gain": best["score"] - blind,
        "settings": best["over"],
    }


def run(limit: int | None = None, jobs: int = 8, heldout: bool = False) -> list[dict]:
    from multiprocessing import Pool

    from . import evalset as E

    cat = json.loads((REPO / "web" / "public" / "data" / "catalog.json").read_text())
    byid = {f["id"]: f for f in cat["frames"]}
    refs = E.load_references(include_heldout=True)
    refs = [r for r in refs if (r.split == "heldout") == heldout]
    if limit:
        refs = refs[:limit]

    tasks = []
    for r in refs:
        if not r.frames:
            continue
        fid = r.frames[0]
        fr = byid.get(fid)
        if not fr:
            continue
        tasks.append((r.n, r.title, fid, "L" if fid[0] == "L" else "R",
                      str(r.ref_path), int(fr["startSample"])))

    with Pool(jobs) as pool:
        out = pool.map(optimise_one, tasks)
    return [o for o in out if "error" not in o]


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--heldout", action="store_true")
    args = ap.parse_args()

    rows = run(limit=args.limit, jobs=args.jobs, heldout=args.heldout)
    if not rows:
        raise SystemExit("no images scored")

    blind = np.array([r["blind"] for r in rows])
    orac = np.array([r["oracle"] for r in rows])
    gain = orac - blind

    split = "HELD-OUT" if args.heldout else "DEV"
    print(f"\nTIER COMPARISON — {split} split, {len(rows)} images")
    print(f"  {'blind (shipped decoder)':32s} gradient NCC  mean {blind.mean():+.4f}  median {np.median(blind):+.4f}")
    print(f"  {'oracle (allowed to cheat)':32s} gradient NCC  mean {orac.mean():+.4f}  median {np.median(orac):+.4f}")
    print(f"  {'what cheating buys':32s}               mean {gain.mean():+.4f}  median {np.median(gain):+.4f}")
    print(f"  images where cheating helps at all: {(gain > 0.005).sum()}/{len(rows)}")

    rows.sort(key=lambda r: -r["gain"])
    print("\n  biggest gains from cheating:")
    for r in rows[:8]:
        s = {k: v for k, v in r["settings"].items()}
        print(f"    #{r['n']:3d} {r['title'][:26]:28s} {r['blind']:+.3f} -> {r['oracle']:+.3f} "
              f"({r['gain']:+.3f})  {s}")
    print("\n  where blind is already optimal:")
    for r in rows[-5:]:
        print(f"    #{r['n']:3d} {r['title'][:26]:28s} {r['blind']:+.3f} -> {r['oracle']:+.3f} ({r['gain']:+.3f})")

    from collections import Counter
    c = Counter()
    for r in rows:
        for k, v in r["settings"].items():
            c[f"{k}={v}"] += 1
    print("\n  settings the oracle most often preferred (a hint at what to infer blindly):")
    for k, v in c.most_common(10):
        print(f"    {k:28s} {v:3d}/{len(rows)}")

    out = REPO / "data" / ("eval_tiers_heldout.json" if args.heldout else "eval_tiers_dev.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out.relative_to(REPO)}")
