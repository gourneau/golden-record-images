/**
 * Minimal radix-2 FFT, plus the two things the decoder actually needs it for:
 * autocorrelation and a power spectrum.
 *
 * Both matter for correctness, not just speed. The line period is recovered by
 * autocorrelation over a ~50-trace baseline, and a direct O(n^2) correlation
 * over a quarter-million samples is ~65 billion operations -- in the Python
 * reference that did not run slowly, it hung. The dot clock is found as a
 * narrow line in the averaged picture-band spectrum, which is an FFT by
 * construction.
 */

/** In-place complex FFT. `re`/`im` must be the same power-of-two length. */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;
  if ((n & (n - 1)) !== 0) throw new Error(`fft: length ${n} is not a power of two`);

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const ar = re[i + k];
        const ai = im[i + k];
        const br = re[i + k + half] * cr - im[i + k + half] * ci;
        const bi = re[i + k + half] * ci + im[i + k + half] * cr;
        re[i + k] = ar + br;
        im[i + k] = ai + bi;
        re[i + k + half] = ar - br;
        im[i + k + half] = ai - bi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Autocorrelation of `x` (mean removed), normalised so lag 0 is 1.
 * Returned for lags 0..maxLag inclusive.
 */
export function autocorr(x: Float64Array | Float32Array, maxLag: number): Float64Array {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n || 1;

  const nfft = nextPow2(2 * n);
  const re = new Float64Array(nfft);
  const im = new Float64Array(nfft);
  for (let i = 0; i < n; i++) re[i] = x[i] - mean;

  fft(re, im);
  // Power spectrum in place: F * conj(F) is real.
  for (let i = 0; i < nfft; i++) {
    const p = re[i] * re[i] + im[i] * im[i];
    re[i] = p;
    im[i] = 0;
  }
  // Inverse FFT via conjugation.
  for (let i = 0; i < nfft; i++) im[i] = -im[i];
  fft(re, im);

  const out = new Float64Array(Math.min(maxLag + 1, n));
  const norm = re[0] || 1;
  for (let k = 0; k < out.length; k++) out[k] = re[k] / norm;
  return out;
}

/** Power spectrum of a real signal, windowed with a Hann. Length n/2+1. */
export function powerSpectrum(x: Float64Array): Float64Array {
  const n = nextPow2(x.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const len = x.length;
  for (let i = 0; i < len; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (len - 1 || 1));
    re[i] = x[i] * w;
  }
  fft(re, im);
  const half = (n >> 1) + 1;
  const out = new Float64Array(half);
  for (let i = 0; i < half; i++) out[i] = re[i] * re[i] + im[i] * im[i];
  return out;
}

/**
 * Sub-sample refinement of a peak at index `k` by fitting a parabola through
 * its two neighbours. Returns a fractional index.
 */
export function parabolic(y: ArrayLike<number>, k: number): number {
  if (k <= 0 || k >= y.length - 1) return k;
  const y0 = y[k - 1];
  const y1 = y[k];
  const y2 = y[k + 1];
  const den = y0 - 2 * y1 + y2;
  if (den === 0) return k;
  return k + (0.5 * (y0 - y2)) / den;
}
