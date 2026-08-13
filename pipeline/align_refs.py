"""Warp each reference plate into the decode's own frame.

The crops are correctly matched but they are a DIFFERENT CROP of the same
photograph: the book printed a plate, we decoded the record's scan, and the two
have different fields of view and different aspect ratios. Shown side by side in
one box they look mismatched even though the match is right -- which reads as a
bad reference rather than a bad presentation.

This estimates a similarity transform (scale, rotation, translation) from the
reference to the decode using SIFT correspondences with RANSAC, and renders the
reference into the decode's exact geometry. After it, the two images overlay:
the same feature is at the same pixel, so a viewer can see what the decode got
right and what it lost, rather than being distracted by framing.

SIFT rather than intensity correlation because the two differ in tone, contrast,
polarity and halftone screening -- all the things a correlation is sensitive to
and a keypoint descriptor is not. Where SIFT cannot find a transform (plain
fields like the calibration circle give it nothing to grip) the reference is
passed through unwarped and flagged, rather than warped by a bad fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CROPS = REPO / "docs" / "reference" / "crops"
ALIGNED = REPO / "docs" / "reference" / "aligned"
THUMBS = REPO / "data" / "thumbs"

MIN_INLIERS = 12


def _prep(a: np.ndarray) -> np.ndarray:
    """Normalise for matching: 8-bit, contrast-equalised, modest size."""
    a = a.astype(np.float32)
    a -= a.min()
    if a.max() > 0:
        a /= a.max()
    u = (a * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u)


def estimate(decode: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray | None, int, bool]:
    """Similarity transform mapping ref -> decode. Tries both polarities."""
    sift = cv2.SIFT_create(nfeatures=4000)
    d8 = _prep(decode)
    kd, dd = sift.detectAndCompute(d8, None)
    if dd is None or len(kd) < 8:
        return None, 0, False, False

    # Search reflections too. estimateAffinePartial2D fits rotation, uniform
    # scale and translation -- 4 degrees of freedom, which CANNOT represent a
    # mirror. Several published reproductions of these slides are laterally
    # inverted relative to the record (the elephant faces the other way), so
    # without trying a flip those pairs can never align and simply score badly.
    # The record is the authority on orientation; the reference is what gets
    # flipped to match it.
    best = (None, 0, False, False)
    for mirror in (False, True):
      base = ref[:, ::-1] if mirror else ref
      for inv in (False, True):
        r = 255 - base if inv else base
        r8 = _prep(r.astype(np.float32) / 255.0)
        kr, dr = sift.detectAndCompute(r8, None)
        if dr is None or len(kr) < 8:
            continue
        matches = cv2.BFMatcher().knnMatch(dr, dd, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            continue
        src = np.float32([kr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kd[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=5000
        )
        n_in = int(mask.sum()) if mask is not None else 0
        if M is not None and n_in > best[1]:
            best = (M, n_in, inv, mirror)
    return best


def align_one(n: int, frame_id: str) -> dict:
    src = CROPS / f"{n}.jpg"
    thumb = THUMBS / f"{frame_id}.png"
    if not src.exists() or not thumb.exists():
        return {"n": n, "ok": False, "reason": "missing input"}

    dec = cv2.imread(str(thumb), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if dec is None or ref is None:
        return {"n": n, "ok": False, "reason": "unreadable"}

    M, n_in, inv, mirror = estimate(dec, ref)
    ALIGNED.mkdir(parents=True, exist_ok=True)
    out = ALIGNED / f"{n}.jpg"

    # Inlier count alone does not mean the fit is sane. Image 8, the solar
    # spectrum, is a soft band on black: RANSAC finds 40 "inliers" and returns a
    # transform with scale 0.0, which collapses the whole plate to a point. A
    # degenerate warp that passes the inlier gate is worse than no warp, because
    # it looks like a considered answer. The reference is a crop of the same
    # photograph at broadly the same size, so anything outside this range is a
    # failed fit whatever its inlier count says.
    if M is not None:
        sc = float(np.hypot(M[0, 0], M[0, 1]))
        if not (0.2 <= sc <= 5.0):
            M, n_in = None, 0

    if M is None or n_in < MIN_INLIERS:
        # No trustworthy transform: letterbox the reference into the decode's
        # frame so the pair at least shares a box, and flag it. Warping by a bad
        # fit would be worse than not warping.
        h, w = dec.shape
        sc = min(w / ref.shape[1], h / ref.shape[0])
        rw, rh = max(1, int(ref.shape[1] * sc)), max(1, int(ref.shape[0] * sc))
        canvas = np.full((h, w), int(np.median(ref)), np.uint8)
        r = cv2.resize(ref, (rw, rh), interpolation=cv2.INTER_AREA)
        canvas[(h - rh) // 2 : (h - rh) // 2 + rh, (w - rw) // 2 : (w - rw) // 2 + rw] = r
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return {"n": n, "ok": False, "reason": f"only {n_in} inliers", "inliers": n_in,
                "file": f"docs/reference/aligned/{n}.jpg", "warped": False}

    base = ref[:, ::-1] if mirror else ref
    src_img = 255 - base if inv else base
    warped = cv2.warpAffine(
        src_img, M, (dec.shape[1], dec.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(np.median(src_img)),
    )
    cv2.imwrite(str(out), warped, [cv2.IMWRITE_JPEG_QUALITY, 88])
    scale = float(np.hypot(M[0, 0], M[0, 1]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return {"n": n, "ok": True, "inliers": n_in, "scale": round(scale, 4),
            "rotation_deg": round(rot, 2), "polarity_inverted": bool(inv),
            "mirrored": bool(mirror),
            "file": f"docs/reference/aligned/{n}.jpg", "warped": True}


if __name__ == "__main__":  # pragma: no cover
    cat = json.loads((REPO / "web" / "public" / "data" / "catalog.json").read_text())
    imgs = {i["n"]: i for i in cat["images"]}
    rows = []
    for n in sorted(imgs):
        fr = imgs[n]["frames"][0]
        rows.append(align_one(n, fr))
        r = rows[-1]
        flag = "" if r.get("ok") else f"   <- {r.get('reason')}"
        print(f"  #{n:3d} inliers {r.get('inliers', 0):4d} "
              f"scale {r.get('scale', float('nan')):>6} rot {r.get('rotation_deg', float('nan')):>7}{flag}")

    ok = [r for r in rows if r.get("ok")]
    print(f"\n{len(ok)}/{len(rows)} aligned by transform; {len(rows)-len(ok)} letterboxed and flagged")
    (REPO / "docs" / "reference" / "aligned" / "index.json").write_text(json.dumps(rows, indent=1))
