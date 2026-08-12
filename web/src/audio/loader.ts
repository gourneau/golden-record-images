/** Frame audio acquisition: fetch a frame's FLAC and hand back 96 kHz mono float.
 *
 * The one thing this module exists to prevent: `AudioContext.decodeAudioData`
 * resamples whatever it decodes to the *context's* rate, and a page's normal
 * AudioContext runs at whatever the output device wants (44.1 or 48 kHz on most
 * machines). That would quietly throw away three quarters of our samples. The
 * decoder's whole timebase -- sub-sample trace positions, a line period of
 * 799.35 rather than 800, porch windows expressed as fractions of that period --
 * is built on the 96 kHz oversampling, so a silent resample is not a quality
 * regression, it is a wrong picture.
 *
 * So: decode inside an OfflineAudioContext pinned to 96 kHz, and then *check*
 * that both the context and the returned buffer really are at 96 kHz instead of
 * assuming the browser honoured the request. If anything about that path is
 * unavailable we fall back to a .wav sibling asset parsed by hand, which cannot
 * resample because no audio engine is involved.
 */

/** Every asset in web/public/data/frames is mono at this rate. */
export const TARGET_SAMPLE_RATE = 96000;

/** The subset of a catalog.json frame entry this module needs. */
export interface FrameRef {
  id: string;
  /** Path relative to the catalog, e.g. "frames/L000.flac". */
  file: string;
  /** Samples of audio present before the first trace start. */
  leadIn?: number;
  /** Expected decoded length; only used to size the decoder context. */
  durationSamples?: number;
  /** Pre-normalisation peak of the master slice, for absolute level restore. */
  peak?: number;
}

export type DecodePath = 'offline-audio-context' | 'wav-parser';

export interface FrameAudio {
  id: string;
  /** The URL the samples actually came from (may be the .wav fallback). */
  url: string;
  /** How it was decoded -- worth surfacing in diagnostics. */
  path: DecodePath;
  /** Mono, `sampleRate` Hz, in the master's original amplitude if `peak` was given. */
  samples: Float32Array;
  /** Verified, not assumed. */
  sampleRate: number;
  leadIn: number;
  /** Peak of `samples` as returned. */
  peak: number;
  /** Factor applied to the decoded asset to restore absolute level. */
  levelScale: number;
  /** Heap cost of `samples`, which is what the cache budgets. */
  bytes: number;
}

export type LoaderErrorCode =
  | 'aborted'
  | 'http'
  | 'no-offline-context'
  | 'context-rate'
  | 'buffer-rate'
  | 'decode'
  | 'bad-wav';

export class LoaderError extends Error {
  readonly code: LoaderErrorCode;
  readonly url?: string;
  /** The underlying failure, if any. Not called `cause`, which lib.es2022 owns. */
  readonly detail?: unknown;

  constructor(code: LoaderErrorCode, message: string, url?: string, detail?: unknown) {
    super(message);
    this.name = 'LoaderError';
    this.code = code;
    this.url = url;
    this.detail = detail;
  }
}

function abortError(): DOMException {
  return new DOMException('Frame load aborted', 'AbortError');
}

function isAbort(err: unknown): boolean {
  return (
    (err instanceof DOMException && err.name === 'AbortError') ||
    (err instanceof Error && err.name === 'AbortError')
  );
}

// --------------------------------------------------------------------------
// decoding
// --------------------------------------------------------------------------

type OfflineCtor = new (channels: number, length: number, sampleRate: number) => OfflineAudioContext;

function offlineCtor(): OfflineCtor | null {
  const g = globalThis as unknown as Record<string, unknown>;
  const c = (g.OfflineAudioContext ?? g.webkitOfflineAudioContext) as OfflineCtor | undefined;
  return c ?? null;
}

/** One decoder context per rate. decodeAudioData is re-entrant, so sharing is safe
 *  and saves the (surprisingly slow) context construction on every frame. */
const decoderContexts = new Map<number, OfflineAudioContext>();

function decoderContext(sampleRate: number, length: number): OfflineAudioContext {
  const cached = decoderContexts.get(sampleRate);
  if (cached) return cached;
  const Ctor = offlineCtor();
  if (!Ctor) {
    throw new LoaderError('no-offline-context', 'OfflineAudioContext is not available');
  }
  let ctx: OfflineAudioContext;
  try {
    ctx = new Ctor(1, Math.max(128, Math.min(length, 1 << 22)), sampleRate);
  } catch (err) {
    throw new LoaderError(
      'context-rate',
      `OfflineAudioContext refused ${sampleRate} Hz`,
      undefined,
      err,
    );
  }
  // The rate we asked for is not necessarily the rate we got: some engines clamp
  // to a supported range. If it were clamped, decodeAudioData would resample.
  if (ctx.sampleRate !== sampleRate) {
    throw new LoaderError(
      'context-rate',
      `OfflineAudioContext clamped ${sampleRate} Hz to ${ctx.sampleRate} Hz`,
    );
  }
  decoderContexts.set(sampleRate, ctx);
  return ctx;
}

function decodeAudioDataCompat(ctx: BaseAudioContext, data: ArrayBuffer): Promise<AudioBuffer> {
  return new Promise<AudioBuffer>((resolve, reject) => {
    let ret: unknown;
    try {
      // Older WebKit only has the callback form and returns undefined.
      ret = (ctx.decodeAudioData as unknown as (
        d: ArrayBuffer,
        ok: (b: AudioBuffer) => void,
        no: (e: unknown) => void,
      ) => unknown)(data, resolve, reject);
    } catch (err) {
      reject(err);
      return;
    }
    if (ret && typeof (ret as Promise<AudioBuffer>).then === 'function') {
      (ret as Promise<AudioBuffer>).then(resolve, reject);
    }
  });
}

/** Decode compressed or PCM audio through the Web Audio engine at a pinned rate. */
async function decodeViaOfflineContext(
  bytes: ArrayBuffer,
  url: string,
  sampleRate: number,
  channel: number,
  expectedLength: number,
): Promise<{ samples: Float32Array; sampleRate: number }> {
  const ctx = decoderContext(sampleRate, expectedLength || 128);
  let buffer: AudioBuffer;
  try {
    // decodeAudioData detaches the buffer it is given, and the caller may still
    // need the bytes for the hand-rolled fallback, so it gets a copy.
    buffer = await decodeAudioDataCompat(ctx, bytes.slice(0));
  } catch (err) {
    throw new LoaderError('decode', `decodeAudioData failed for ${url}`, url, err);
  }
  if (buffer.sampleRate !== sampleRate) {
    throw new LoaderError(
      'buffer-rate',
      `decoded buffer is ${buffer.sampleRate} Hz, expected ${sampleRate} Hz`,
      url,
    );
  }
  const ch = Math.min(channel, buffer.numberOfChannels - 1);
  // slice() so a stray stereo asset does not pin its other channel in the cache.
  const samples = buffer.getChannelData(ch).slice();
  return { samples, sampleRate: buffer.sampleRate };
}

function fourcc(view: DataView, offset: number): string {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

function looksLikeRiff(bytes: ArrayBuffer): boolean {
  if (bytes.byteLength < 12) return false;
  const v = new DataView(bytes);
  return fourcc(v, 0) === 'RIFF' && fourcc(v, 8) === 'WAVE';
}

/** Parse a PCM/float WAV by hand.
 *
 * This is the escape hatch: no AudioContext means nothing can resample us, so
 * whatever rate the file declares is the rate we return. Handles the
 * WAVE_FORMAT_EXTENSIBLE header ffmpeg writes and skips LIST/fact chunks.
 */
export function decodeWav(
  bytes: ArrayBuffer,
  channel = 0,
): { samples: Float32Array; sampleRate: number } {
  if (!looksLikeRiff(bytes)) throw new LoaderError('bad-wav', 'not a RIFF/WAVE file');
  const view = new DataView(bytes);
  let offset = 12;
  let fmt: { tag: number; channels: number; rate: number; bits: number } | null = null;
  let dataStart = -1;
  let dataLength = 0;

  while (offset + 8 <= bytes.byteLength) {
    const id = fourcc(view, offset);
    let size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (size === 0xffffffff || body + size > bytes.byteLength) size = bytes.byteLength - body;
    if (id === 'fmt ') {
      let tag = view.getUint16(body, true);
      const channels = view.getUint16(body + 2, true);
      const rate = view.getUint32(body + 4, true);
      const bits = view.getUint16(body + 14, true);
      // EXTENSIBLE: the real format tag is the first two bytes of the subformat GUID.
      if (tag === 0xfffe && size >= 40) tag = view.getUint16(body + 24, true);
      fmt = { tag, channels, rate, bits };
    } else if (id === 'data') {
      dataStart = body;
      dataLength = size;
    }
    offset = body + size + (size & 1); // chunks are word aligned
  }

  if (!fmt || dataStart < 0) throw new LoaderError('bad-wav', 'WAV is missing fmt or data');
  const { tag, channels, rate, bits } = fmt;
  if (channels < 1) throw new LoaderError('bad-wav', 'WAV declares no channels');
  const ch = Math.min(channel, channels - 1);
  const bytesPerSample = bits >> 3;
  const frames = Math.floor(dataLength / (bytesPerSample * channels));
  const out = new Float32Array(frames);
  const stride = bytesPerSample * channels;
  let p = dataStart + ch * bytesPerSample;

  if (tag === 1 && bits === 16) {
    for (let i = 0; i < frames; i++, p += stride) out[i] = view.getInt16(p, true) / 32768;
  } else if (tag === 1 && bits === 24) {
    for (let i = 0; i < frames; i++, p += stride) {
      const lo = view.getUint8(p) | (view.getUint8(p + 1) << 8) | (view.getInt8(p + 2) << 16);
      out[i] = lo / 8388608;
    }
  } else if (tag === 1 && bits === 32) {
    for (let i = 0; i < frames; i++, p += stride) out[i] = view.getInt32(p, true) / 2147483648;
  } else if (tag === 1 && bits === 8) {
    // 8-bit WAV is unsigned.
    for (let i = 0; i < frames; i++, p += stride) out[i] = (view.getUint8(p) - 128) / 128;
  } else if (tag === 3 && bits === 32) {
    for (let i = 0; i < frames; i++, p += stride) out[i] = view.getFloat32(p, true);
  } else if (tag === 3 && bits === 64) {
    for (let i = 0; i < frames; i++, p += stride) out[i] = view.getFloat64(p, true);
  } else {
    throw new LoaderError('bad-wav', `unsupported WAV format tag ${tag} at ${bits} bits`);
  }
  return { samples: out, sampleRate: rate };
}

function peakOf(x: Float32Array): number {
  let m = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i] < 0 ? -x[i] : x[i];
    if (v > m) m = v;
  }
  return m;
}

// --------------------------------------------------------------------------
// loader
// --------------------------------------------------------------------------

export interface LoaderOptions {
  /** Where catalog-relative frame paths resolve against. */
  baseUrl?: string;
  /** Cache budget in bytes of decoded float samples. */
  maxCacheBytes?: number;
  /** Multiply decoded samples by the frame's `peak` to undo asset normalisation. */
  restoreLevel?: boolean;
  /** Required decoded rate. Anything else is an error, never a silent accept. */
  sampleRate?: number;
  /** Channel to take if an asset is unexpectedly multi-channel. */
  channel?: number;
  /** Where to look when FLAC decoding is unavailable. */
  wavFallback?: (flacUrl: string) => string | null;
  fetchImpl?: typeof fetch;
}

export interface LoaderStats {
  entries: number;
  bytes: number;
  maxBytes: number;
  hits: number;
  misses: number;
  coalesced: number;
  evictions: number;
  fallbacks: number;
}

interface Inflight {
  promise: Promise<FrameAudio>;
  controller: AbortController;
  waiters: number;
}

function defaultWavFallback(url: string): string | null {
  return url.endsWith('.flac') ? url.slice(0, -5) + '.wav' : null;
}

export class FrameLoader {
  #baseUrl: string;
  #maxBytes: number;
  #restoreLevel: boolean;
  #sampleRate: number;
  #channel: number;
  #wavFallback: (url: string) => string | null;
  #fetch: typeof fetch;

  /** Insertion order is LRU order: `get` re-inserts to move an entry to the end. */
  #cache = new Map<string, FrameAudio>();
  #bytes = 0;
  #inflight = new Map<string, Inflight>();
  #stats = { hits: 0, misses: 0, coalesced: 0, evictions: 0, fallbacks: 0 };

  constructor(opts: LoaderOptions = {}) {
    this.#baseUrl = opts.baseUrl ?? 'data/';
    this.#maxBytes = opts.maxCacheBytes ?? 40 * 1024 * 1024;
    this.#restoreLevel = opts.restoreLevel ?? true;
    this.#sampleRate = opts.sampleRate ?? TARGET_SAMPLE_RATE;
    this.#channel = opts.channel ?? 0;
    this.#wavFallback = opts.wavFallback ?? defaultWavFallback;
    // Wrapped, because an unbound `fetch` throws "Illegal invocation" in Chrome.
    this.#fetch =
      opts.fetchImpl ?? ((...args: Parameters<typeof fetch>) => globalThis.fetch(...args));
  }

  get sampleRate(): number {
    return this.#sampleRate;
  }

  url(frame: FrameRef): string {
    const base = new URL(this.#baseUrl, globalThis.location?.href ?? 'http://localhost/');
    return new URL(frame.file, base).href;
  }

  /** Cached, coalesced, abortable frame fetch.
   *
   * Two callers asking for the same frame share one network request; a caller
   * that aborts detaches from it, and the shared request is only cancelled once
   * every caller has gone away.
   */
  get(frame: FrameRef, opts: { signal?: AbortSignal } = {}): Promise<FrameAudio> {
    const signal = opts.signal;
    if (signal?.aborted) return Promise.reject(abortError());

    const hit = this.#cache.get(frame.id);
    if (hit) {
      this.#stats.hits++;
      this.#cache.delete(frame.id);
      this.#cache.set(frame.id, hit);
      return Promise.resolve(hit);
    }

    let entry = this.#inflight.get(frame.id);
    if (entry) {
      this.#stats.coalesced++;
    } else {
      this.#stats.misses++;
      const controller = new AbortController();
      const created: Inflight = {
        controller,
        waiters: 0,
        promise: this.#load(frame, controller.signal).then(
          (audio) => {
            this.#inflight.delete(frame.id);
            this.#put(audio);
            return audio;
          },
          (err) => {
            this.#inflight.delete(frame.id);
            throw err;
          },
        ),
      };
      this.#inflight.set(frame.id, created);
      entry = created;
    }

    const shared = entry;
    shared.waiters++;
    let detached = false;
    const detach = () => {
      if (detached) return;
      detached = true;
      signal?.removeEventListener('abort', onAbort);
      shared.waiters--;
      if (shared.waiters <= 0 && this.#inflight.get(frame.id) === shared) {
        this.#inflight.delete(frame.id);
        shared.controller.abort();
      }
    };

    let rejectCaller: (e: unknown) => void = () => {};
    const onAbort = () => {
      detach();
      rejectCaller(abortError());
    };

    return new Promise<FrameAudio>((resolve, reject) => {
      rejectCaller = reject;
      signal?.addEventListener('abort', onAbort, { once: true });
      shared.promise.then(
        (audio) => {
          detach();
          resolve(audio);
        },
        (err) => {
          detach();
          reject(err);
        },
      );
    });
  }

  /** Warm the cache without caring about the outcome. */
  prefetch(frame: FrameRef): void {
    if (this.#cache.has(frame.id) || this.#inflight.has(frame.id)) return;
    void this.get(frame).catch(() => undefined);
  }

  peek(id: string): FrameAudio | undefined {
    return this.#cache.get(id);
  }

  has(id: string): boolean {
    return this.#cache.has(id);
  }

  evict(id: string): boolean {
    const e = this.#cache.get(id);
    if (!e) return false;
    this.#cache.delete(id);
    this.#bytes -= e.bytes;
    return true;
  }

  clear(): void {
    this.#cache.clear();
    this.#bytes = 0;
  }

  stats(): LoaderStats {
    return {
      entries: this.#cache.size,
      bytes: this.#bytes,
      maxBytes: this.#maxBytes,
      ...this.#stats,
    };
  }

  #put(audio: FrameAudio): void {
    const existing = this.#cache.get(audio.id);
    if (existing) this.#bytes -= existing.bytes;
    this.#cache.set(audio.id, audio);
    this.#bytes += audio.bytes;
    // Evict oldest first, but never the entry we just stored.
    for (const key of this.#cache.keys()) {
      if (this.#bytes <= this.#maxBytes) break;
      if (key === audio.id) continue;
      const e = this.#cache.get(key)!;
      this.#cache.delete(key);
      this.#bytes -= e.bytes;
      this.#stats.evictions++;
    }
  }

  async #fetchBytes(url: string, signal: AbortSignal): Promise<ArrayBuffer> {
    const res = await this.#fetch(url, { signal, credentials: 'same-origin' });
    if (!res.ok) throw new LoaderError('http', `HTTP ${res.status} for ${url}`, url);
    return await res.arrayBuffer();
  }

  async #load(frame: FrameRef, signal: AbortSignal): Promise<FrameAudio> {
    const primary = this.url(frame);
    const expected = frame.durationSamples ?? 0;
    let decoded: { samples: Float32Array; sampleRate: number } | null = null;
    let usedUrl = primary;
    let path: DecodePath = 'offline-audio-context';
    let firstError: unknown = null;

    try {
      const bytes = await this.#fetchBytes(primary, signal);
      if (looksLikeRiff(bytes)) {
        // A .wav asset: the hand parser is exact and cannot be resampled, so it
        // goes first; the engine is only there in case of an exotic encoding.
        try {
          decoded = decodeWav(bytes, this.#channel);
          path = 'wav-parser';
        } catch (err) {
          firstError = err;
        }
      }
      if (!decoded) {
        decoded = await decodeViaOfflineContext(
          bytes,
          primary,
          this.#sampleRate,
          this.#channel,
          expected,
        );
        path = 'offline-audio-context';
      }
    } catch (err) {
      if (isAbort(err) || signal.aborted) throw abortError();
      firstError = err;
    }

    if (!decoded) {
      // FLAC decoding (or the 96 kHz context) is unavailable here. Try the
      // uncompressed sibling, which we can always decode ourselves.
      const alt = this.#wavFallback(primary);
      if (!alt || alt === primary) throw firstError;
      try {
        const bytes = await this.#fetchBytes(alt, signal);
        decoded = decodeWav(bytes, this.#channel);
        usedUrl = alt;
        path = 'wav-parser';
        this.#stats.fallbacks++;
      } catch (err) {
        if (isAbort(err) || signal.aborted) throw abortError();
        throw new LoaderError(
          'decode',
          `${(firstError as Error)?.message ?? firstError}; WAV fallback also failed: ` +
            `${(err as Error)?.message ?? err}`,
          primary,
          firstError,
        );
      }
    }

    if (decoded.sampleRate !== this.#sampleRate) {
      throw new LoaderError(
        'buffer-rate',
        `${usedUrl} decoded at ${decoded.sampleRate} Hz, expected ${this.#sampleRate} Hz`,
        usedUrl,
      );
    }

    let samples = decoded.samples;
    const levelScale = this.#restoreLevel && frame.peak && frame.peak > 0 ? frame.peak : 1;
    if (levelScale !== 1) {
      // Assets are peak-normalised so 16-bit quantisation buys the most SNR;
      // multiplying back puts absolute levels on the master's scale.
      samples = new Float32Array(samples.length);
      for (let i = 0; i < samples.length; i++) samples[i] = decoded.samples[i] * levelScale;
    }

    return {
      id: frame.id,
      url: usedUrl,
      path,
      samples,
      sampleRate: decoded.sampleRate,
      leadIn: frame.leadIn ?? 0,
      peak: peakOf(samples),
      levelScale,
      bytes: samples.byteLength,
    };
  }
}

/** Shared loader for the app; tests and the probe page make their own. */
export const frames = new FrameLoader();

export function loadFrame(frame: FrameRef, opts?: { signal?: AbortSignal }): Promise<FrameAudio> {
  return frames.get(frame, opts);
}
