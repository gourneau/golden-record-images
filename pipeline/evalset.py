"""Reference-based evaluation of the decoder, with a held-out split.

WHY THIS EXISTS, AND WHY IT COMES FIRST
---------------------------------------
Until now "the decode got better" has been an aesthetic judgement backed by
physical sanity checks (is the calibration field flat, is the circle round).
Those are good but they are not a measure of how close a decoded picture is to
the photograph that was actually on the slide. We now have 47 of the original
slides, so that measure exists.

THE RULE THAT MAKES THIS LEGITIMATE
-----------------------------------
Reference images are an EVALUATION SET. They never enter pixel recovery.
pipeline/decode.py imports only numpy, dotclock and sync, and `assert_decode_is_blind`
below checks that automatically so the guarantee cannot rot.

And a subtler rule: choosing decoder settings by looking at all 47 references is
overfitting the test set -- the same error as cheating, only slower and easier to
miss. So the references are split into a DEVELOPMENT half, used freely for
choosing settings, and a HELD-OUT half that is not to be looked at until a single
final evaluation. The split is deterministic (hash of the image number), so it is
stable across runs and cannot be quietly reshuffled until it flatters a result.

WHAT IS COMPARED, AND WHAT THAT IS WORTH
----------------------------------------
The decode and the reference differ legitimately in ways that are not decoder
error: overall brightness and contrast (the record stores a negative through an
unknown transfer), sub-pixel alignment and scale (the reference is a rescanned
print of unknown crop), and tone (the slide was photographed, printed, scanned).
So the metrics here normalise brightness/contrast and register before comparing,
and the headline number is deliberately a STRUCTURAL one.

Even then these numbers are weaker evidence than the physical criteria. A crop
mismatch of a few per cent will dominate any real decoder improvement. Treat a
change of a few points as noise; treat the hold-out prediction test in
pipeline/uncertainty.py as the real arbiter of whether detail is genuine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REF_MAP = REPO / "docs" / "reference" / "map.json"
NASA_INDEX = REPO / "docs" / "reference" / "nasa" / "index.json"
NASA_MATCH = REPO / "docs" / "reference" / "nasa" / "match_draft.json"
CATALOG = REPO / "web" / "public" / "data" / "catalog.json"
THUMBS = REPO / "data" / "thumbs"

# Fraction of references reserved for the final evaluation.
HELDOUT_FRACTION = 0.5


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------


def split_of(n: int) -> str:
    """Deterministic dev/held-out assignment for canonical image number `n`.

    Hashed rather than random so it is identical on every machine and every run,
    and so it cannot be resampled until the answer looks good.
    """
    h = hashlib.sha256(f"golden-record-eval-{n}".encode()).digest()
    return "heldout" if (h[0] / 255.0) < HELDOUT_FRACTION else "dev"


@dataclass(frozen=True)
class RefItem:
    n: int
    title: str
    frames: list[str]
    ref_path: Path
    split: str


def load_references(include_heldout: bool = False) -> list[RefItem]:
    """Reference images with their canonical image numbers.

    By default the held-out half is NOT returned. Asking for it is an explicit,
    visible act -- which is the point.
    """
    if not CATALOG.exists():
        raise FileNotFoundError(f"{CATALOG} missing; run the build first")
    cat = json.loads(CATALOG.read_text())
    titles = {i["n"]: (i.get("title") or "") for i in cat["images"]}
    frames = {i["n"]: i["frames"] for i in cat["images"]}

    found: dict[int, Path] = {}

    # Prefer the NASA slides: individual photographs, not book pages with
    # several pictures on them.
    if NASA_MATCH.exists() and NASA_INDEX.exists():
        idx = json.loads(NASA_INDEX.read_text())
        for row in json.loads(NASA_MATCH.read_text()):
            if row.get("score", 0) < 0.55:
                continue  # low-confidence slug match; vision has not confirmed it
            slug = row["slug"]
            if slug in idx:
                p = REPO / idx[slug]["file"]
                if p.exists():
                    found.setdefault(int(row["n"]), p)

    # Fall back to whatever the reference map offers (Wikimedia scans, crops).
    if REF_MAP.exists():
        for k, v in json.loads(REF_MAP.read_text()).items():
            if not k.isdigit():
                continue  # the map carries metadata keys such as "_about"
            n = int(k)
            if n in found:
                continue
            for key in ("crop", "file"):
                if v.get(key):
                    p = REPO / v[key]
                    if p.exists():
                        found[n] = p
                        break

    out = [
        RefItem(n, titles.get(n, ""), frames.get(n, []), p, split_of(n))
        for n, p in sorted(found.items())
        if n in frames
    ]
    if not include_heldout:
        out = [r for r in out if r.split == "dev"]
    return out


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def _to_gray(path: Path, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("L").resize((shape[1], shape[0]), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def _normalise(a: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance. Removes the brightness/contrast difference that
    is not decoder error."""
    a = a - a.mean()
    s = a.std()
    return a / s if s > 1e-9 else a


def _register(a: np.ndarray, b: np.ndarray, max_shift: int = 24) -> tuple[np.ndarray, tuple[int, int]]:
    """Align `b` to `a` by integer shift, via FFT phase correlation.

    Without this the metrics measure crop mismatch, not decode quality -- and
    the crop of a rescanned print is unknown, so it must be estimated.
    """
    A = np.fft.rfft2(_normalise(a))
    B = np.fft.rfft2(_normalise(b))
    cps = A * np.conj(B)
    mag = np.abs(cps)
    cps = np.where(mag > 1e-12, cps / np.maximum(mag, 1e-12), 0)
    corr = np.fft.irfft2(cps, s=a.shape)
    corr = np.fft.fftshift(corr)
    c = np.array(a.shape) // 2
    win = corr[
        max(0, c[0] - max_shift) : c[0] + max_shift + 1,
        max(0, c[1] - max_shift) : c[1] + max_shift + 1,
    ]
    k = np.unravel_index(int(np.argmax(win)), win.shape)
    dy = k[0] - min(max_shift, c[0])
    dx = k[1] - min(max_shift, c[1])
    return np.roll(np.roll(b, dy, axis=0), dx, axis=1), (int(dy), int(dx))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float((_normalise(a) * _normalise(b)).mean())


def _gradient_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of gradient magnitude: structure, not tone.

    This is the headline number. The decode's brightness transfer differs from a
    scanned print's in ways that are not decoder error, so comparing structure is
    fairer than comparing levels.
    """
    def grad(z):
        gy, gx = np.gradient(z)
        return np.hypot(gy, gx)

    return _ncc(grad(a), grad(b))


def _scaled(path: Path, shape: tuple[int, int], scale: float, pol: int) -> np.ndarray:
    """Reference resampled at `scale` about its centre, cropped/padded to `shape`."""
    from PIL import Image

    im = Image.open(path).convert("L")
    w = max(8, int(round(shape[1] * scale)))
    h = max(8, int(round(shape[0] * scale)))
    a = np.asarray(im.resize((w, h), Image.LANCZOS), dtype=np.float64) / 255.0
    if pol < 0:
        a = 1.0 - a
    out = np.full(shape, float(a.mean()))
    y0 = (shape[0] - h) // 2
    x0 = (shape[1] - w) // 2
    ys, xs = max(0, y0), max(0, x0)
    ay, ax = max(0, -y0), max(0, -x0)
    hh = min(shape[0] - ys, h - ay)
    ww = min(shape[1] - xs, w - ax)
    out[ys : ys + hh, xs : xs + ww] = a[ay : ay + hh, ax : ax + ww]
    return out


def compare(decoded: np.ndarray, ref_path: Path, scales=None) -> dict:
    """Score one decoded image against its reference slide.

    Searches polarity, scale and shift, because none of those three is decoder
    error: the record stores a negative, the reference is a rescanned print of
    unknown crop, and the two need not share a bounding box. Without the scale
    search this metric measured crop mismatch -- shifts of 19-23 px appeared on
    images whose decode is visibly fine, which is registration failure wearing
    the costume of a decode failure.
    """
    if scales is None:
        scales = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.18, 1.3)
    best = None
    for pol in (1, -1):
        for sc in scales:
            r = _scaled(ref_path, decoded.shape, sc, pol)
            aligned, shift = _register(decoded, r)
            s = {
                "polarity": pol,
                "scale": sc,
                "shift": shift,
                "ncc": _ncc(decoded, aligned),
                "gradient_ncc": _gradient_ncc(decoded, aligned),
            }
            if best is None or s["gradient_ncc"] > best["gradient_ncc"]:
                best = s
    return best


# --------------------------------------------------------------------------
# the honesty guard
# --------------------------------------------------------------------------


def assert_decode_is_blind() -> list[str]:
    """Fail if the decode path can see reference imagery.

    Automated because a guarantee maintained by memory is not a guarantee. Walks
    the decoder's own imports and greps for anything that would read a reference.
    """
    banned = ("docs/reference", "reference/", "images.json", "commons", "ozma", "nasa")
    seen: list[str] = []
    problems: list[str] = []
    stack = ["decode.py", "sync.py", "dotclock.py"]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.append(name)
        p = REPO / "pipeline" / name
        if not p.exists():
            problems.append(f"{name}: missing")
            continue
        src = p.read_text()
        for b in banned:
            for i, line in enumerate(src.splitlines(), 1):
                if b in line and not line.lstrip().startswith("#"):
                    problems.append(f"{name}:{i} references {b!r}: {line.strip()[:70]}")
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("from . import ") or s.startswith("from .") and " import " in s:
                for mod in s.split("import", 1)[1].split(","):
                    mod = mod.strip().split(" as ")[0].strip()
                    if mod and not mod.startswith("."):
                        stack.append(f"{mod}.py")
    return problems


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout", action="store_true",
                    help="include the held-out half (final evaluation only)")
    args = ap.parse_args()

    problems = assert_decode_is_blind()
    print("decode-path blindness:", "OK" if not problems else "VIOLATIONS")
    for p in problems:
        print("   ", p)

    every = load_references(include_heldout=True)
    n_dev = sum(1 for r in every if r.split == "dev")
    n_held = len(every) - n_dev
    refs = every if args.heldout else [r for r in every if r.split == "dev"]
    print(f"\nreferences available: {len(every)}  (dev {n_dev}, held-out {n_held})")
    if not args.heldout:
        print("held-out half withheld by default -- pass --heldout for the final evaluation")
    for r in refs[:12]:
        print(f"   #{r.n:3d} [{r.split:7s}] {r.title[:42]:44s} {r.ref_path.name[:40]}")
