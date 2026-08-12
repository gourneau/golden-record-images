"""Post-decode enhancement, from measurements on the 384 kHz master.

Every constant and every technique below was MEASURED on the real master
(2026-08-12 session; drivers and raw outputs in the session scratchpad under
enh/). Techniques that did not survive measurement are documented as negative
results at the bottom of this docstring rather than shipped as code.

MEASURED SYSTEM RESOLUTION (the honest numbers)
-----------------------------------------------
Target: the L000 calibration circle's ring, 182 crossings fitted one trace at
a time with a two-logistic bump model (estimator validated on synthetic edges
at matched SNR: bias < 0.5% on widths, +0.3% on MTF50), each aligned on its
own fitted centre so timing jitter cannot masquerade as blur, slant-corrected
by cos(theta) from a RANSAC circle fit.

  along-trace, full chain (artwork->vidicon->scan converter->tape->disc->
  digitisation), at 384 kHz:
      leading (white->dark) edge 10-90:  17.7 samples  (MAD 4.9)
      trailing (dark->white) edge 10-90: 12.6 samples  (MAD 6.4)
      super-resolved ESF -> MTF50 = 0.0388 c/sample, MTF10 = 0.0881 c/sample
      => per 2780-sample active picture: ~216 px at MTF50, ~490 px limiting
      => at a 384-row decode (7.24 samples/px): MTF50 at 0.281 c/px,
         MTF10 at 0.638 c/px -- detail exists slightly past 384-row Nyquist,
         so ~576 rows captures everything; Barry's 364 px sits near MTF25.
  electronics-only (sync falling edge, per parity):
      10-90 = 3.8 samples (even) / 2.0 samples (odd)
      => the audio chain is ~4x sharper than the picture; the blur is
         dominantly the vidicon/scan-converter spot, not the recording.
  across-trace (ring near left/right extremes, sector method, per-column
  droop detrended):
      outer edge 10-90 = 3.25 traces
      => ~170 effective columns of 512 at MTF50-equivalent; the 512-column
         raster oversamples the beam by ~2x. Across-trace deconvolution was
         NOT shipped: stripe + parity artefacts sit exactly in the band a
         sharpener would boost.

ACTIVE WINDOW (resolves the wrap contradiction)
-----------------------------------------------
Z = half-amplitude downward crossing of the sync falling edge. Variance census
across traces (parity pattern removed, droop high-passed away) on 7 frames
spread over both channels:
      quiet gap      [Z+40,   Z+180 ]   content-free on every frame tested
      ACTIVE PICTURE [Z+180,  Z+2960]   ~2780 samples, NO WRAP
      back porch     [Z+2960, Z+3045]
      next burst lead ~Z+3045, next Z at Z+period (3197.4 early -> 3193.1 late)
The earlier "2873-sample wrapping window" was an artefact of the notch
landmark. Barry's 2680-sample window was ~100 samples short at both ends.

NOISE
-----
Quiet-gap (content-free) noise PSD: N/S < 0.001 through the whole picture
band. The recording's additive noise is NOT the resolution limiter.
Do NOT measure noise from adjacent-trace differences while sync jitter is
10-20 samples: content leaks into the difference and poisons the estimate
(measured: 0.0072 "noise" vs 0.0029 true on the same data).

NEGATIVE RESULTS (measured, do not re-ship)
-------------------------------------------
* AC-coupling droop, one-pole model: WRONG. The burst plateau shows no sag
  and the -0.10 -> +0.017 "recovery" across the picture is scene/format
  structure (it cannot be a linear response: it exceeds the stimulus).
  decode._fit_tau rails because it fits scene. What IS measurable
  (porch-level regression on 500 traces x 7 frames): a short positive
  persistence (porch follows the last ~200 samples of picture, slope
  0.4-0.7, r up to 0.9) plus a long negative whole-trace component
  (r ~ -0.4 at w=2770). One invertible pole does not exist; recommend
  uncouple=False (A/B on the 16-frame test set: composite 19.6 -> 19.4,
  i.e. the correction was doing nothing measurable anyway).
* Triplet super-resolution: NO resolution gain possible. Registration between
  colour separations measured at |dt| <= 0.4 traces, |ds| <= 0.04 samples --
  the scan raster is locked between separations, so there is no sampling
  phase diversity to exploit. Spectral coherence between separations dies at
  the single-frame MTF anyway. What stacking DOES buy is SNR on luminance:
  high-band noise 0.145 -> 0.084 (= exactly 1/sqrt(3)) on the L007 triplet,
  partial (shared stripe noise) elsewhere. stack_triplet() ships that.
* Despike at 6 MAD: eats real detail on photos (2.6% of R040 flagged).
  Composite +0.4 at best. Use >= 10 MAD or nothing.
* 60 Hz mains-hum subtraction beyond the existing parity-median removal:
  nothing left to remove above the stripe floor.

POSITIVE, SHIPPED
-----------------
* wiener_sharpen(): capped-inverse deconvolution of the measured full-chain
  LSF, along-trace, band-limited to f < 0.11 c/sample. Measured on L000 ring
  (the filter's own source): lead/trail 10-90 17.7/12.6 -> 12.4/4.2 samples
  at cap=3. Independent target (R040, 500+ strong edges the filter never
  saw): 10.3 -> 3.5 samples. Flat-field noise cost +2%. Circle-fit
  radial_rms 0.95 -> 0.93 px (geometry unharmed). Overshoot: bump-fit
  residual 0.0053 -> 0.0090 (~10% of edge amplitude) -- visible ringing
  starts above cap~4, so the default is 3.
* corrected_porch(): the porch clamp leaks bottom-of-picture content into
  every trace's DC (measured slope 0.4-0.9). Regressing it out cuts stripe
  RMS by 26% (R040) and 48% (L055); neutral (+-5%) where the leak is weak.
  Self-tuning: the slope is fitted per frame, and gated on correlation.
* stack_triplet(): sub-pixel registration + luminance stack, SNR gain
  1.3-1.7x measured.
* Cross-channel timing (for sync.py, not implemented here): late-record wow
  (6.7-7.8 samples rms smoothed) correlates r=+0.83 between L and R at equal
  record time; where residuals are small the correlation vanishes (detector
  noise). A joint L+R wow track would nearly halve late-record timing noise.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# measured constants
# ---------------------------------------------------------------------------

# Full-chain along-trace line spread function, 37 taps at 384 kHz sample
# spacing, from the super-resolved L000 ring ESF (leading edges, slant
# corrected, windowed to +-18 samples, unit sum). Centre tap index 18.
LSF_384 = np.array([
    -0.000000, -0.000031, -0.000076, 0.000263, 0.002237, 0.003400,
    0.002479, 0.004639, 0.011008, 0.010972, 0.009769, 0.017809,
    0.015186, 0.019068, -0.002083, 0.003036, 0.007678, 0.027800,
    0.057635, 0.101578, 0.123490, 0.138845, 0.123004, 0.100116,
    0.072866, 0.063355, 0.045354, 0.026570, 0.013571, 0.005288,
    -0.000841, -0.002004, -0.000947, -0.000567, -0.000411, -0.000055,
    -0.000000,
])

MTF50_C_PER_SAMPLE = 0.0388   # measured, L000 ring, 384 kHz
MTF10_C_PER_SAMPLE = 0.0881
EDGE_1090_ALONG = 17.7        # samples, leading edge, full chain
EDGE_1090_ACROSS = 3.25       # traces, outer ring edge
F_STOP_DEFAULT = 0.11         # c/sample; ring MTF ~5 % here, only hiss beyond

# Active window relative to the sync falling edge crossing Z (384 kHz samples)
PIC_ON, PIC_OFF = 180, 2960
PORCH_ON, PORCH_OFF = 2965, 3040


def system_mtf(freqs: np.ndarray) -> np.ndarray:
    """|H| of the measured full-chain LSF at `freqs` (cycles/384k-sample)."""
    n = 2048
    kern = np.zeros(n)
    idx = (np.arange(len(LSF_384)) - 18) % n
    kern[idx] = LSF_384
    H = np.abs(np.fft.rfft(kern))
    f = np.fft.rfftfreq(n)
    return np.interp(freqs, f, H)


# ---------------------------------------------------------------------------
# deconvolution
# ---------------------------------------------------------------------------


def wiener_sharpen(
    pic: np.ndarray,
    samples_per_px: float = 1.0,
    cap: float = 3.0,
    f_stop: float = F_STOP_DEFAULT,
) -> np.ndarray:
    """Capped-inverse deconvolution of the measured LSF along axis 1.

    `pic` is (traces, n) with axis 1 along the trace. If axis 1 is already
    binned to pixels, pass samples_per_px (=active samples/rows, ~7.24 for a
    384-row decode) so the gain lands on the right frequencies; note that at
    384 rows most of the recoverable band lies above pixel Nyquist, so apply
    this BEFORE binning (samples_per_px=1) whenever possible.

    Measured at cap=3 on the master: ring edge 10-90 17.7 -> 12.4 (lead) and
    12.6 -> 4.2 (trail) samples; independent R040 edges 10.3 -> 3.5; noise
    +2%; overshoot ~10% of edge amplitude. cap>4 buys nothing further and
    rings more.
    """
    n = pic.shape[1]
    nfft = 1 << int(np.ceil(np.log2(max(n, 64))))
    f_px = np.fft.rfftfreq(nfft)                    # cycles per axis-1 unit
    f_samp = f_px / samples_per_px                  # cycles per 384k sample
    H = np.maximum(system_mtf(np.minimum(f_samp, 0.5)), 1e-3)
    G = np.minimum(1.0 / H, cap)
    G *= np.clip((f_stop - f_samp) / 0.02 + 1.0, 0.0, 1.0)
    G[0] = 1.0
    mean = pic.mean(axis=1, keepdims=True)
    X = np.fft.rfft(pic - mean, nfft, axis=1)
    out = np.fft.irfft(X * G[None, :], nfft, axis=1)[:, :n]
    return out + mean


# ---------------------------------------------------------------------------
# stripe reduction: leakage-corrected porch clamp
# ---------------------------------------------------------------------------


def corrected_porch(
    porch: np.ndarray,
    trail: np.ndarray,
    detrend: int = 31,
    min_abs_r: float = 0.3,
) -> tuple[np.ndarray, float, float]:
    """Remove the measured content->porch leak from per-trace porch levels.

    `porch`: per-trace mean of the back-porch region ([Z+2965, Z+3040]).
    `trail`: per-trace mean of the last ~200 picture samples ([Z+2750,
    Z+2950]). The chain lets ~0.4-0.9 of recent content persist into the
    porch (measured by regression across 500 traces on 7 frames), so a naive
    porch clamp stripes the image with the scene's bottom row. Returns
    (corrected porch, fitted slope, r). The correction is applied only when
    |r| >= min_abs_r; measured gains: stripe RMS -26% (R040), -48% (L055),
    neutral where the leak is weak.
    """
    porch = np.asarray(porch, dtype=np.float64)
    trail = np.asarray(trail, dtype=np.float64)
    if len(porch) < 3 * detrend or len(porch) != len(trail):
        return porch, 0.0, 0.0

    def _detrend(v):
        k = detrend
        pad = np.pad(v, k, mode="edge")
        med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(v))])
        return v - med

    pd = _detrend(porch)
    td = _detrend(trail)
    a = td - td.mean()
    b = pd - pd.mean()
    den = float(a @ a)
    if den <= 0:
        return porch, 0.0, 0.0
    slope = float(a @ b) / den
    r = float(a @ b) / (np.sqrt(den * float(b @ b)) + 1e-30)
    if abs(r) < min_abs_r:
        return porch, slope, r
    return porch - slope * td, slope, r


# ---------------------------------------------------------------------------
# triplet stacking
# ---------------------------------------------------------------------------


def register_offset(
    a: np.ndarray, b: np.ndarray, max_shift: int = 40
) -> tuple[float, float, float]:
    """(dy, dx, peak) sub-pixel offset of b relative to a by bounded phase
    correlation on high-passed images. Measured on real triplets: separations
    are raster-locked (|ds| <= 0.04 samples along-trace), so a global shift is
    the whole story."""
    from scipy.ndimage import uniform_filter

    ah = a - uniform_filter(a, size=(9, 65))
    bh = b - uniform_filter(b, size=(9, 65))
    A = np.fft.fft2(ah)
    B = np.fft.fft2(bh)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    r = np.real(np.fft.ifft2(R))
    m = max_shift
    mask = np.zeros_like(r, dtype=bool)
    mask[: m + 1, : m + 1] = True
    mask[: m + 1, -m:] = True
    mask[-m:, : m + 1] = True
    mask[-m:, -m:] = True
    rm = np.where(mask, r, -np.inf)
    k = np.unravel_index(np.argmax(rm), r.shape)

    def sub(v0, v1, v2):
        d = v0 - 2 * v1 + v2
        return 0.5 * (v0 - v2) / d if d != 0 else 0.0

    di = sub(r[(k[0] - 1) % r.shape[0], k[1]], r[k], r[(k[0] + 1) % r.shape[0], k[1]])
    dj = sub(r[k[0], (k[1] - 1) % r.shape[1]], r[k], r[k[0], (k[1] + 1) % r.shape[1]])
    off = [float(k[0]), float(k[1])]
    if off[0] > r.shape[0] // 2:
        off[0] -= r.shape[0]
    if off[1] > r.shape[1] // 2:
        off[1] -= r.shape[1]
    return off[0] + di, off[1] + dj, float(r[k])


def _shift2(a: np.ndarray, dy: float, dx: float) -> np.ndarray:
    A = np.fft.fft2(a)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    return np.real(np.fft.ifft2(A * np.exp(2j * np.pi * (fy * dy + fx * dx))))


def stack_triplet(frames: list[np.ndarray]) -> dict:
    """Register a colour triplet's separations and stack their luminance.

    Returns dict with 'stack' (registered mean of robustly normalised
    separations), 'offsets', 'peaks', and 'noise_gain' (high-band noise of a
    single separation over the stack; measured 1.3-1.7x on real triplets,
    sqrt(3)=1.73 max). Resolution does NOT improve -- the separations are
    raster-locked -- this is an SNR tool for the luminance channel only.
    """
    from scipy.ndimage import uniform_filter1d

    base = frames[0]
    regs = [base]
    offsets = [(0.0, 0.0)]
    peaks = [1.0]
    for f in frames[1:]:
        dy, dx, pk = register_offset(base, f)
        regs.append(_shift2(f, dy, dx))
        offsets.append((dy, dx))
        peaks.append(pk)
    norm = []
    for m in regs:
        med = np.median(m)
        mad = np.median(np.abs(m - med)) + 1e-12
        norm.append((m - med) / mad)
    stack = np.mean(norm, axis=0)

    def hiband_noise(m):
        hp = m - uniform_filter1d(m, 33, axis=1 if m.shape[1] >= 33 else 0)
        d = (hp[:-1] - hp[1:]) / np.sqrt(2)
        return float(np.sqrt(np.mean(d ** 2)))

    n_single = float(np.mean([hiband_noise(m) for m in norm]))
    n_stack = hiband_noise(stack)
    return {
        "stack": stack,
        "offsets": offsets,
        "peaks": peaks,
        "noise_gain": n_single / n_stack if n_stack > 0 else 1.0,
    }


# ---------------------------------------------------------------------------
# conservative despike (only >= 10 MAD is safe on photos; see docstring)
# ---------------------------------------------------------------------------


def despike(pic: np.ndarray, n_mad: float = 10.0, halfwin: int = 6) -> np.ndarray:
    """Hampel filter along axis 1 in the oversampled domain. At 10 MAD this
    touches 0.05-0.9% of samples; at 6 MAD it starts eating real photo detail
    (2.6% of R040 flagged) -- do not lower the threshold."""
    from scipy.ndimage import median_filter

    med = median_filter(pic, size=(1, 2 * halfwin + 1), mode="nearest")
    r = pic - med
    mad = 1.4826 * np.median(np.abs(r)) + 1e-12
    out = pic.copy()
    bad = np.abs(r) > n_mad * mad
    out[bad] = med[bad]
    return out
