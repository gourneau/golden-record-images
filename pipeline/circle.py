"""Everything the calibration frame can tell us.

Image 1 on the record is a black ring on a uniform white field. The cover says
so -- a picture-within-a-picture showing a circle -- which means using it as a
calibration target is sanctioned by the record's own designers. Unlike the
photographs, this is NOT outside knowledge: a recipient who read the cover knows
image 1 is a circle and can do everything below.

That makes it the single most valuable frame on the record, and it can calibrate
almost the whole chain at once:

  the field is known UNIFORM     -> flatness measures residual droop
                                 -> its variation measures the noise floor
                                 -> its level is absolute WHITE
  the ring core                  -> absolute BLACK
  the ring is a known thin line  -> its profile IS the line spread function,
                                    and because the ring is curved it presents
                                    an edge at every angle and every sub-pixel
                                    phase, so the LSF can be super-resolved
  the ring is a known CIRCLE     -> roundness measures geometry
                                 -> radial scatter measures timing error, in a
                                    way sharpening cannot fake

And the decisive test, which nothing else on the record enables: synthesise the
ideal slide, push it through our measured forward model, and compare the result
to the ACTUAL RECORDED AUDIO. If the model is right the residual is noise. Its
structure is a map of everything the model still gets wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# ring geometry
# --------------------------------------------------------------------------


@dataclass
class Ring:
    cy: float
    cx: float
    ry: float
    rx: float
    axis_ratio: float
    radial_rms: float  # px, scatter of the edge about the fitted ellipse
    n_points: int
    coverage_deg: float
    width: float  # measured ring thickness, px


def fit_ring(img: np.ndarray) -> Ring | None:
    """Fit the ring, delegating to pipeline.quality.circle_metrics.

    An earlier version here grew its own detector and fitter and could not fit
    the ring at all: a darkness-maximum scan locks onto the frame's dark edge
    regions (the bottom-left corner reads 0.18 against a field of 0.81), and a
    least-squares conic seeded from that point set goes nowhere. quality.py had
    already solved this -- row/column-static removal so the sync band cannot
    pose as the ring, a radius-histogram MODE for the initial estimate, the
    strongest band-pass ridge per angular bin, iterative outlier trimming, and
    asserted inlier and angular-coverage floors. Reuse beats reinvention.
    """
    from . import quality as Q

    m = Q.circle_metrics(img)
    if not m.get("ok"):
        return None
    major = float(m["major"])
    minor = float(m["minor"])
    h, w = img.shape
    cx = w / 2.0 + float(m["center_dx"])
    cy = h / 2.0 + float(m["center_dy"])
    # quality.py reports major/minor; map them onto the image axes. The decode
    # is wider than tall, and the ring is measured in that frame.
    rx, ry = (major, minor) if w >= h else (minor, major)
    ring = Ring(cy, cx, ry, rx, float(rx / ry), float(m["radial_rms"]),
                int(m.get("inliers", 0)), float(m.get("coverage_deg", 0.0)), float("nan"))
    prof, edges = _radial_profile(img, cy, cx, ry, rx)
    fin = np.isfinite(prof)
    if fin.sum() > 8:
        half = np.nanmin(prof) + 0.5 * (np.nanmax(prof) - np.nanmin(prof))
        below = edges[fin][prof[fin] < half]
        if len(below) > 1:
            ring.width = float(below.max() - below.min())
    return ring


def _radial_profile(img, cy, cx, ry, rx, span=8.0, bins=161):
    """Darkness against normalised radius, averaged over all angles.

    This is the super-resolved edge profile: every angle contributes at a
    different sub-pixel phase, so the profile is sampled far more finely than
    the pixel grid. That is the ISO-12233 slanted-edge trick, and a circle
    supplies every angle for free.
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - cx) / rx, (yy - cy) / ry)
    scale = 0.5 * (rx + ry)
    d = (rr - 1.0) * scale  # signed distance from the ring, in px
    m = np.abs(d) <= span
    edges = np.linspace(-span, span, bins)
    idx = np.clip(np.digitize(d[m], edges) - 1, 0, bins - 1)
    prof = np.zeros(bins)
    cnt = np.zeros(bins)
    np.add.at(prof, idx, img[m])
    np.add.at(cnt, idx, 1.0)
    good = cnt > 0
    prof[good] /= cnt[good]
    prof[~good] = np.nan
    return prof, edges


def mtf_from_ring(img: np.ndarray, ring: Ring) -> dict:
    """MTF from the ring's own profile.

    The ring is a LINE, so its darkness profile is the line spread function
    directly -- no differentiation, and therefore none of the noise amplification
    that wrecks a differentiated edge. Fourier transform gives the MTF.
    """
    prof, edges = _radial_profile(img, ring.cy, ring.cx, ring.ry, ring.rx)
    ok = np.isfinite(prof)
    lsf = prof[ok].max() - prof[ok]
    lsf = lsf - np.median(np.concatenate([lsf[:8], lsf[-8:]]))
    lsf = np.clip(lsf, 0, None)
    if lsf.sum() <= 0:
        return {}
    step = float(edges[1] - edges[0])
    n = 1024
    F = np.abs(np.fft.rfft(lsf * np.hanning(len(lsf)), n))
    F /= F[0]
    freq = np.fft.rfftfreq(n, step)  # cycles per pixel

    def crossing(target):
        below = np.where(F < target)[0]
        if not len(below):
            return float("nan")
        k = below[0]
        if k == 0:
            return 0.0
        f0, f1 = freq[k - 1], freq[k]
        a, b = F[k - 1], F[k]
        return float(f0 + (a - target) * (f1 - f0) / (a - b))

    mtf50 = crossing(0.5)
    return {
        "mtf50_cycles_per_px": mtf50,
        "mtf10_cycles_per_px": crossing(0.1),
        "lsf_fwhm_px": float(np.sum(lsf > 0.5 * lsf.max()) * step),
        "resolvable_px_per_trace": float(2 * mtf50 * img.shape[0]) if mtf50 == mtf50 else float("nan"),
    }


def field_stats(img: np.ndarray, ring: Ring) -> dict:
    """Flatness, noise and absolute levels from the known-uniform field."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - ring.cx) / ring.rx, (yy - ring.cy) / ring.ry)
    inside = rr < 0.80
    outside = (rr > 1.20) & (yy > 6) & (yy < h - 6) & (xx > 6) & (xx < w - 6)
    core = np.abs(rr - 1.0) < (0.35 * max(ring.width, 1.0) / (0.5 * (ring.rx + ring.ry)))

    def tilt(mask):
        v = img[mask]
        ys = yy[mask].astype(float)
        if len(v) < 50:
            return float("nan")
        A = np.stack([ys, np.ones_like(ys)], axis=1)
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        return float(c[0] * h)  # level change across the full height

    # Noise from neighbouring-pixel differences inside the flat field: real
    # variation of a uniform slide is zero, so what is left is noise.
    d = np.diff(img, axis=1)
    dm = inside[:, :-1] & inside[:, 1:]
    noise = float(np.std(d[dm]) / np.sqrt(2)) if dm.sum() > 100 else float("nan")

    white = float(np.median(img[inside])) if inside.sum() else float("nan")
    black = float(np.percentile(img[core], 10)) if core.sum() > 20 else float("nan")
    return {
        "white_level": white,
        "black_level": black,
        "contrast": white - black,
        "field_tilt_inside": tilt(inside),
        "field_tilt_outside": tilt(outside),
        "field_std_inside": float(np.std(img[inside])) if inside.sum() else float("nan"),
        "noise_rms": noise,
        "snr_db": float(20 * np.log10(abs(white - black) / noise)) if noise and noise == noise else float("nan"),
    }


def ideal_slide(shape: tuple[int, int], ring: Ring, width: float | None = None) -> np.ndarray:
    """Synthesise the ideal calibration slide: a black ring on a white field.

    This is the ground truth the record was designed to give us, and it is what
    the forward model should reproduce from the recorded audio.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - ring.cx) / ring.rx, (yy - ring.cy) / ring.ry)
    scale = 0.5 * (ring.rx + ring.ry)
    d = np.abs(rr - 1.0) * scale
    half = 0.5 * (width if width else ring.width)
    # Soft edge of half a pixel so the target is band-limited rather than a
    # step, which would put energy above Nyquist that no decoder could match.
    return np.clip((d - half) / 0.5, 0.0, 1.0)


def analyse(img: np.ndarray) -> dict:
    ring = fit_ring(img)
    if ring is None:
        return {"ok": False, "reason": "ring fit failed"}
    out = {
        "ok": True,
        "ring": ring.__dict__,
        "mtf": mtf_from_ring(img, ring),
        "field": field_stats(img, ring),
    }
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(REPO))
    from PIL import Image

    from pipeline import decode as D, sync as S, wav

    T = json.loads((REPO / "pipeline" / "barry_tables.json").read_text())
    info = wav.probe(REPO / "data" / "master" / "384kHzStereo.wav")
    mm = wav.memmap(info)
    st = T["start_samples"][0][0]
    x = wav.read(info, 0, st - 2000, int(540 * 3197.4) + 14000, mm=mm).astype(np.float64)
    tb = S.recover(x)

    for label, cfg in (
        ("shipped", D.Settings(channel="L")),
        ("no undroop", D.Settings(channel="L", uncouple=False)),
        ("no dot lock", D.Settings(channel="L", dot_lock=False)),
    ):
        r = D.decode(x, cfg, tb)
        a = analyse(r.image)
        if not a["ok"]:
            print(f"{label:12s} {a['reason']}")
            continue
        g, m, f = a["ring"], a["mtf"], a["field"]
        print(f"\n=== {label} ===  image {r.image.shape}")
        print(f"  ring   axis_ratio {g['axis_ratio']:.4f}  radial_rms {g['radial_rms']:.3f} px"
              f"  width {g['width']:.2f} px  n={g['n_points']} cov={g['coverage_deg']:.0f} deg")
        print(f"  MTF    mtf50 {m.get('mtf50_cycles_per_px', float('nan')):.4f} c/px"
              f"  lsf_fwhm {m.get('lsf_fwhm_px', float('nan')):.2f} px"
              f"  -> {m.get('resolvable_px_per_trace', float('nan')):.0f} resolvable px/trace")
        print(f"  field  white {f['white_level']:.3f}  black {f['black_level']:.3f}"
              f"  contrast {f['contrast']:.3f}  noise {f['noise_rms']:.5f}"
              f"  SNR {f['snr_db']:.1f} dB")
        print(f"  tilt   inside {f['field_tilt_inside']:+.4f}  outside {f['field_tilt_outside']:+.4f}"
              f"   (0 = perfectly flat; the slide IS uniform)")
