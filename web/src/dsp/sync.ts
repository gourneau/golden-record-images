/**
 * Timebase recovery, ported from pipeline/sync.py and held to the same
 * behaviour. This is the part of the decoder that matters most: a landmark
 * error of a hundred samples draws a diagonal staircase across a photograph.
 *
 * The chain is:
 *   1. estimate the line period by autocorrelation over a ~50-trace baseline;
 *   2. score every sample with a bipolar edge detector (sustained high level
 *      followed by a sustained low one) -- picture content is a negative and
 *      never swings the way the blanking burst does, so this cannot be
 *      triggered by scene detail;
 *   3. find the sync phase by folding that score at the period and letting the
 *      traces vote, which is threshold-free and survives dropouts;
 *   4. locate each trace at the sub-sample downward half-amplitude crossing of
 *      the falling edge -- NOT the pulse peak, because the pulse WIDTH
 *      alternates with trace parity (the Colorado Video field-ID code) so the
 *      peak wanders by ~100 samples while the edge does not;
 *   5. fit a robust line, measure the parity alternation of the landmark and
 *      DISCARD it (the picture does not alternate -- adjacent columns agree to
 *      ~0.12 samples on a purely linear clock -- so it is an artefact of what
 *      we measure, not of the recording), and smooth the remainder, which is
 *      the master tape's wow and flutter.
 */

import { autocorr, parabolic } from './fft';

export const NOMINAL_PERIOD_384K = 3197.4;
export const TRACES_PER_FRAME = 512;

const GAP_FRAC = 0.00375; // half-gap of the bipolar drop windows
const AVG_FRAC = 0.010; // length of each drop window
const SMOOTH_N = 3; // boxcar on the signal for the sub-sample crossing
const GATE = 0.45; // validity gate: drop score vs frame median
const SG_WINDOW = 31; // Savitzky-Golay window, in traces

export interface Timebase {
  /** Sub-sample position of each trace's landmark, wow/flutter corrected. */
  positions: Float64Array;
  /** Fitted samples per trace. */
  period: number;
  /** Fraction of traces that locked rather than coasted. */
  lock: number;
  /** RMS of our own measurement noise, in samples. */
  noiseRms: number;
  /** Measured parity alternation of the landmark, in samples (discarded). */
  parityAlt: number;
  located: Uint8Array;
  nTraces: number;
}

/** Measure the line period by autocorrelation. Must happen before any
 *  template work: the search window is predicted from the period, and an
 *  error of 0.12 samples/trace accumulates to 60 samples over 512 traces --
 *  enough to walk the search clean off the sync and onto picture content. */
export function estimatePeriod(x: Float64Array, guess: number, tolerance = 0.02): number {
  const lag = Math.round(guess * 50);
  const span = Math.min(x.length, Math.max(Math.round(guess * 80), lag * 2));
  if (span < lag + 16) return guess;
  const ac = autocorr(x.subarray(0, span), Math.min(span - 2, Math.round(lag * (1 + tolerance)) + 2));
  const lo = Math.max(1, Math.round(lag * (1 - tolerance)));
  const hi = Math.min(ac.length - 2, Math.round(lag * (1 + tolerance)));
  if (hi <= lo) return guess;
  let k = lo;
  for (let i = lo; i <= hi; i++) if (ac[i] > ac[k]) k = i;
  return parabolic(ac, k) / 50;
}

function cumsum(x: Float64Array): Float64Array {
  const c = new Float64Array(x.length + 1);
  for (let i = 0; i < x.length; i++) c[i + 1] = c[i] + x[i];
  return c;
}

/** Bipolar edge score: mean-before minus mean-after, at every sample. */
function dropScore(x: Float64Array, period: number) {
  const gap = Math.max(2, Math.round(GAP_FRAC * period));
  const k = Math.max(4, Math.round(AVG_FRAC * period));
  const n = x.length;
  const c = cumsum(x);
  const drop = new Float64Array(n);
  const left = new Float64Array(n);
  const right = new Float64Array(n);
  const clamp = (v: number) => (v < 0 ? 0 : v > n ? n : v);
  for (let i = 0; i < n; i++) {
    const lo = clamp(i - gap - k);
    const hi = clamp(i - gap);
    const lo2 = clamp(i + 1 + gap);
    const hi2 = clamp(i + 1 + gap + k);
    const l = (c[hi] - c[lo]) / Math.max(hi - lo, 1);
    const r = (c[hi2] - c[lo2]) / Math.max(hi2 - lo2, 1);
    left[i] = l;
    right[i] = r;
    drop[i] = l - r;
  }
  return { drop, left, right };
}

/** Fold the drop score at the period; the sync phase wins the vote. */
function phaseVote(drop: Float64Array, period: number, nTraces: number): number {
  const W = Math.floor(period);
  const v = new Float64Array(W);
  for (let k = 0; k < nTraces; k++) {
    const o = Math.round(k * period);
    if (o + W > drop.length) break;
    for (let i = 0; i < W; i++) v[i] += drop[o + i];
  }
  let best = 0;
  for (let i = 1; i < W; i++) if (v[i] > v[best]) best = i;
  return best;
}

/** Sub-sample position where xs crosses `mid` downward, nearest to j. */
function halfCross(xs: Float64Array, j: number, mid: number, radius: number): number | null {
  for (let d = 0; d <= radius; d++) {
    const tries = d === 0 ? [j] : [j + d, j - d];
    for (const jj of tries) {
      if (jj < 1 || jj + 1 >= xs.length) continue;
      const a = xs[jj];
      const b = xs[jj + 1];
      if (a >= mid && mid > b) return jj + (a - mid) / (a - b);
    }
  }
  return null;
}

function median(a: number[]): number {
  if (!a.length) return 0;
  const s = [...a].sort((p, q) => p - q);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/** IRLS straight-line fit with a Tukey biweight. Returns [slope, intercept]. */
function robustLine(k: number[], p: number[], iters = 5): [number, number] {
  let w = new Array(k.length).fill(1);
  let slope = 0;
  let intercept = 0;
  for (let it = 0; it < iters; it++) {
    let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < k.length; i++) {
      sw += w[i]; sx += w[i] * k[i]; sy += w[i] * p[i];
      sxx += w[i] * k[i] * k[i]; sxy += w[i] * k[i] * p[i];
    }
    const den = sw * sxx - sx * sx;
    if (Math.abs(den) < 1e-12) break;
    slope = (sw * sxy - sx * sy) / den;
    intercept = (sy - slope * sx) / sw;
    const r = k.map((kk, i) => p[i] - (intercept + slope * kk));
    const mr = median(r);
    const s = 1.4826 * median(r.map((v) => Math.abs(v - mr))) + 1e-9;
    w = r.map((v) => {
      const u = Math.max(-1, Math.min(1, v / (4.0 * s)));
      return (1 - u * u) ** 2;
    });
  }
  return [slope, intercept];
}

/** Savitzky-Golay smoothing (quadratic), edges handled by window clamping. */
function savgol(y: Float64Array, window: number): Float64Array {
  const n = y.length;
  let w = Math.min(window | 1, n % 2 ? n : n - 1);
  if (w < 5) return Float64Array.from(y);
  const half = w >> 1;
  const out = new Float64Array(n);
  // Quadratic least squares over the window, evaluated at the centre offset.
  for (let i = 0; i < n; i++) {
    const lo = Math.min(Math.max(0, i - half), n - w);
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, b0 = 0, b1 = 0, b2 = 0;
    for (let j = 0; j < w; j++) {
      const t = j - half;
      const v = y[lo + j];
      const t2 = t * t;
      s0 += 1; s1 += t; s2 += t2; s3 += t2 * t; s4 += t2 * t2;
      b0 += v; b1 += v * t; b2 += v * t2;
    }
    // Solve the 3x3 normal equations by Cramer's rule.
    const d =
      s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2);
    if (Math.abs(d) < 1e-12) { out[i] = y[i]; continue; }
    const a0 =
      b0 * (s2 * s4 - s3 * s3) - s1 * (b1 * s4 - s3 * b2) + s2 * (b1 * s3 - s2 * b2);
    const a1 =
      s0 * (b1 * s4 - s3 * b2) - b0 * (s1 * s4 - s3 * s2) + s2 * (s1 * b2 - b1 * s2);
    const a2 =
      s0 * (s2 * b2 - b1 * s3) - s1 * (s1 * b2 - b1 * s2) + b0 * (s1 * s3 - s2 * s2);
    const t = i - lo - half;
    out[i] = (a0 + a1 * t + a2 * t * t) / d;
  }
  return out;
}

export function recover(
  input: Float32Array | Float64Array,
  opts: { periodGuess?: number; nTraces?: number; passes?: number; smoothWindow?: number } = {},
): Timebase {
  const x = input instanceof Float64Array ? input : Float64Array.from(input);
  const nTraces = opts.nTraces ?? TRACES_PER_FRAME;
  const passes = opts.passes ?? 3;
  const smoothWindow = opts.smoothWindow ?? SG_WINDOW;

  const period0 = estimatePeriod(x, opts.periodGuess ?? NOMINAL_PERIOD_384K / 4);
  const { drop, left, right } = dropScore(x, period0);

  // 3-sample boxcar: kills single-sample noise on the crossing without moving
  // a ~2-sample edge.
  const xs = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    let s = 0;
    for (let d = -(SMOOTH_N >> 1); d <= SMOOTH_N >> 1; d++) {
      s += x[Math.min(x.length - 1, Math.max(0, i + d))];
    }
    xs[i] = s / SMOOTH_N;
  }

  const phi = phaseVote(drop, period0, nTraces);
  let pred = new Float64Array(nTraces);
  for (let k = 0; k < nTraces; k++) pred[k] = phi + k * period0;

  let W = Math.max(6, Math.round(0.04 * period0));
  const crossRad = Math.max(3, Math.round(0.006 * period0));
  const edgeMargin = Math.max(16, Math.round(0.02 * period0));

  const pos = new Float64Array(nTraces);
  const score = new Float64Array(nTraces);
  let located = new Uint8Array(nTraces);
  let smoothed = Float64Array.from(pred);
  let signed = new Float64Array(nTraces);
  let alt = 0;
  let slope = period0;

  for (let p = 0; p < passes; p++) {
    pos.fill(NaN);
    score.fill(0);
    for (let k = 0; k < nTraces; k++) {
      const c = Math.round(pred[k]);
      const a = c - W;
      const b = c + W + 1;
      if (a < edgeMargin || b + edgeMargin > x.length) continue;
      let j = a;
      for (let i = a; i < b; i++) if (drop[i] > drop[j]) j = i;
      const cross = halfCross(xs, j, 0.5 * (left[j] + right[j]), crossRad);
      if (cross === null) continue;
      pos[k] = cross;
      score[k] = drop[j];
    }

    const validIdx: number[] = [];
    for (let k = 0; k < nTraces; k++) if (Number.isFinite(pos[k])) validIdx.push(k);
    if (validIdx.length < 32) throw new Error(`only ${validIdx.length} traces located`);
    const med = median(validIdx.map((k) => score[k]));
    const gateIdx = validIdx.filter((k) => score[k] > GATE * med);
    if (gateIdx.length < 32) throw new Error(`only ${gateIdx.length} traces passed the sync gate`);

    located = new Uint8Array(nTraces);
    for (const k of gateIdx) located[k] = 1;

    [slope] = robustLine(gateIdx.map(Number), gateIdx.map((k) => pos[k]));
    const [sl, ic] = robustLine(gateIdx.map(Number), gateIdx.map((k) => pos[k]));
    slope = sl;
    const line = new Float64Array(nTraces);
    for (let k = 0; k < nTraces; k++) line[k] = ic + k * sl;

    // Parity alternation of the landmark, measured per frame and then removed.
    const evenRes: number[] = [];
    const oddRes: number[] = [];
    for (const k of gateIdx) (k % 2 === 0 ? evenRes : oddRes).push(pos[k] - line[k]);
    alt = evenRes.length && oddRes.length
      ? evenRes.reduce((a, b) => a + b, 0) / evenRes.length -
        oddRes.reduce((a, b) => a + b, 0) / oddRes.length
      : 0;
    signed = new Float64Array(nTraces);
    for (let k = 0; k < nTraces; k++) signed[k] = (k % 2 === 0 ? 0.5 : -0.5) * alt;

    // What remains after removing the line and the parity term is wow and
    // flutter: real, smooth, and worth correcting per trace.
    const resD = gateIdx.map((k) => pos[k] - signed[k] - line[k]);
    const mrd = median(resD);
    const s = 1.4826 * median(resD.map((v) => Math.abs(v - mrd))) + 1e-9;
    const rAll = new Float64Array(nTraces);
    for (let k = 0; k < nTraces; k++) {
      // linear interpolation over the gated traces
      let i = 0;
      while (i < gateIdx.length - 1 && gateIdx[i + 1] < k) i++;
      const k0 = gateIdx[i];
      const k1 = gateIdx[Math.min(i + 1, gateIdx.length - 1)];
      const t = k1 === k0 ? 0 : (k - k0) / (k1 - k0);
      const v = resD[i] + t * (resD[Math.min(i + 1, resD.length - 1)] - resD[i]);
      rAll[k] = Math.max(mrd - 6 * s, Math.min(mrd + 6 * s, v));
    }
    const trend = savgol(rAll, smoothWindow);
    smoothed = new Float64Array(nTraces);
    for (let k = 0; k < nTraces; k++) smoothed[k] = line[k] + trend[k];
    pred = new Float64Array(nTraces);
    for (let k = 0; k < nTraces; k++) pred[k] = smoothed[k] + signed[k];
    W = Math.max(4, Math.round(0.0075 * period0));
  }

  // Coasted traces carry the prediction.
  const out = new Float64Array(nTraces);
  for (let k = 0; k < nTraces; k++) out[k] = located[k] ? smoothed[k] : pred[k] - signed[k];

  let acc = 0;
  let cnt = 0;
  for (let k = 0; k < nTraces; k++) {
    if (!located[k]) continue;
    const d = pos[k] - signed[k] - smoothed[k];
    acc += d * d;
    cnt++;
  }
  const noiseRms = cnt ? Math.sqrt(acc / cnt) : Infinity;
  const lockFrac = located.reduce((a, b) => a + b, 0) / nTraces;

  return {
    positions: out,
    period: slope,
    lock: lockFrac / (1 + noiseRms / (0.002 * slope)),
    noiseRms,
    parityAlt: alt,
    located,
    nTraces,
  };
}
