/** Playback of frame audio, with a clock the decoder can trust.
 *
 * The decoder paints a column per trace, and it needs to know how far the
 * *audible* signal has got so it never paints ahead of what you can hear. The
 * only clock that stays honest across a backgrounded tab, a throttled timer, a
 * dropped animation frame or a change of playback rate is the audio hardware's
 * own: `AudioContext.currentTime`. Counting requestAnimationFrame callbacks
 * drifts within seconds and stops dead when the tab is hidden, while the audio
 * keeps playing -- so rAF here is only ever used to decide *when to ask*, never
 * as the thing being asked.
 *
 * Positions are reported in samples of the frame's own timebase: sample 0 is the
 * start of trace 0, so during the asset's lead-in the position is negative. That
 * is the same coordinate the timebase recovery in pipeline/sync.py works in, so
 * `Math.floor(pos / period)` is directly the trace index being heard.
 */

export interface PlayerSource {
  id: string;
  /** Mono samples at `sampleRate`. */
  samples: Float32Array;
  sampleRate: number;
  /** Samples of audio before trace 0; the origin of the reported timebase. */
  leadIn: number;
}

export interface PlayerOptions {
  /** Supply your own if the app already has one. */
  context?: AudioContext;
  /** Report the position of what is *audible* now rather than what has been
   *  scheduled, by subtracting the output pipeline's latency. */
  latencyCompensation?: boolean;
  /** Polling period while the tab is hidden and rAF is unavailable. */
  hiddenPollMs?: number;
  /** Seconds by which playback starts and rate changes are scheduled into the
   *  future, so the baseline lands on an instant the engine has not rendered
   *  yet. See #startAt for why this is what makes the clock exact. */
  scheduleAhead?: number;
}

type Listener = (positionSamples: number, player: FramePlayer) => void;

/** A baseline pins one instant of context time to one buffer position. Position
 *  is affine in context time between baselines, so every event that changes the
 *  slope -- play, pause, seek, rate -- just lays down a new baseline. */
interface Baseline {
  ctxTime: number;
  posSamples: number;
  rate: number;
}

function makeContext(): AudioContext {
  const g = globalThis as unknown as Record<string, unknown>;
  const Ctor = (g.AudioContext ?? g.webkitAudioContext) as { new (): AudioContext } | undefined;
  if (!Ctor) throw new Error('Web Audio is not available');
  return new Ctor();
}

export class FramePlayer {
  #ctx: AudioContext;
  #gain: GainNode;
  #latencyCompensation: boolean;
  #hiddenPollMs: number;
  #scheduleAhead: number;

  #source: PlayerSource | null = null;
  #buffer: AudioBuffer | null = null;
  #node: AudioBufferSourceNode | null = null;
  #nodeToken = 0;

  #base: Baseline = { ctxTime: 0, posSamples: 0, rate: 1 };
  #playing = false;
  #rate = 1;
  #loop = false;

  #listeners = new Set<Listener>();
  #rafId = 0;
  #timerId: ReturnType<typeof setTimeout> | 0 = 0;
  #onVisibility: (() => void) | null = null;

  /** Fired when playback reaches the end of the frame (not on stop or seek). */
  onended: ((player: FramePlayer) => void) | null = null;

  constructor(opts: PlayerOptions = {}) {
    this.#ctx = opts.context ?? makeContext();
    this.#latencyCompensation = opts.latencyCompensation ?? true;
    this.#hiddenPollMs = opts.hiddenPollMs ?? 40;
    this.#scheduleAhead = opts.scheduleAhead ?? 0.025;
    this.#gain = this.#ctx.createGain();
    this.#gain.connect(this.#ctx.destination);
  }

  get context(): AudioContext {
    return this.#ctx;
  }

  get source(): PlayerSource | null {
    return this.#source;
  }

  get playing(): boolean {
    return this.#playing;
  }

  get rate(): number {
    return this.#rate;
  }

  get loop(): boolean {
    return this.#loop;
  }

  set loop(v: boolean) {
    this.#loop = v;
    if (this.#node) this.#node.loop = v;
  }

  /** Length of the loaded frame in its own timebase (trace 0 == sample 0). */
  get durationSamples(): number {
    if (!this.#source) return 0;
    return this.#source.samples.length - this.#source.leadIn;
  }

  get sampleRate(): number {
    return this.#source?.sampleRate ?? this.#ctx.sampleRate;
  }

  /** Output latency in seconds: how far behind the audible signal is. */
  get latency(): number {
    if (!this.#latencyCompensation) return 0;
    const ctx = this.#ctx as AudioContext & { outputLatency?: number; baseLatency?: number };
    const out = typeof ctx.outputLatency === 'number' ? ctx.outputLatency : 0;
    const base = typeof ctx.baseLatency === 'number' ? ctx.baseLatency : 0;
    // Firefox reports outputLatency including baseLatency; Chrome's outputLatency
    // is the device buffer alone. Taking the max is right in both, and 0 in
    // neither case is worse than pretending latency does not exist.
    return Math.max(out, base);
  }

  /** Install a frame. Playback stops and the position resets to the lead-in. */
  load(source: PlayerSource): void {
    this.#stopNode();
    this.#source = source;
    // The buffer keeps the frame's own 96 kHz rate even though the output device
    // almost certainly runs at 44.1 or 48 kHz. The engine resamples on the way
    // out, which is fine -- what matters is that one second of playback advances
    // exactly `sampleRate` samples of *our* timebase, which is what the clock
    // arithmetic below assumes.
    const buf = this.#ctx.createBuffer(1, source.samples.length, source.sampleRate);
    buf.copyToChannel(source.samples, 0);
    this.#buffer = buf;
    this.#playing = false;
    this.#base = { ctxTime: this.#ctx.currentTime, posSamples: 0, rate: this.#rate };
  }

  /** Start (or resume) playback. Must be reachable from a user gesture the first
   *  time, or the autoplay policy leaves the context suspended. */
  async play(): Promise<void> {
    if (!this.#buffer || !this.#source) throw new Error('FramePlayer.play() before load()');
    if (this.#ctx.state !== 'running') {
      await this.#ctx.resume();
    }
    if (this.#playing) return;
    const offset = this.#clampBufferPos(this.#base.posSamples);
    this.#startAt(offset);
  }

  pause(): void {
    if (!this.#playing) return;
    const pos = this.#bufferPosition(this.#ctx.currentTime);
    this.#stopNode();
    this.#playing = false;
    this.#base = { ctxTime: this.#ctx.currentTime, posSamples: pos, rate: this.#rate };
    this.#emit();
  }

  /** Suspend the whole context. `currentTime` stops advancing while suspended,
   *  so the position freezes without any extra bookkeeping. */
  async suspend(): Promise<void> {
    if (this.#ctx.state === 'running') await this.#ctx.suspend();
  }

  /** Seek in the frame's timebase; negative values land in the lead-in. */
  seekSamples(positionSamples: number): void {
    if (!this.#source) return;
    const target = this.#clampBufferPos(positionSamples + this.#source.leadIn);
    if (this.#playing) {
      this.#startAt(target); // a source node cannot be re-positioned; make a new one
    } else {
      this.#base = { ctxTime: this.#ctx.currentTime, posSamples: target, rate: this.#rate };
      this.#emit();
    }
  }

  seekSeconds(t: number): void {
    this.seekSamples(t * this.sampleRate);
  }

  /** Fraction of the frame, 0..1, measured over the picture (lead-in excluded). */
  seekFraction(f: number): void {
    this.seekSamples(f * this.durationSamples);
  }

  setRate(rate: number): void {
    const r = Math.max(0.0625, Math.min(16, rate));
    if (r === this.#rate) return;
    if (!this.#playing) {
      this.#rate = r;
      this.#base = { ...this.#base, rate: r };
      return;
    }
    // Change the slope at a named future instant for the same reason play()
    // does: audio already rendered will come out at the old rate, and the
    // baseline has to agree with it or the clock steps.
    const when = this.#ctx.currentTime + this.#scheduleAhead;
    const pos = this.#positionAt(when);
    this.#rate = r;
    this.#base = { ctxTime: when, posSamples: pos, rate: r };
    if (this.#node) this.#node.playbackRate.setValueAtTime(r, when);
  }

  setVolume(v: number): void {
    this.#gain.gain.setTargetAtTime(Math.max(0, v), this.#ctx.currentTime, 0.01);
  }

  /** Position in the frame's timebase, in samples. Negative inside the lead-in.
   *  Safe to call at any rate from any callback -- it is pure arithmetic. */
  positionSamples(): number {
    if (!this.#source) return 0;
    return this.#bufferPosition(this.#ctx.currentTime) - this.#source.leadIn;
  }

  positionSeconds(): number {
    return this.positionSamples() / this.sampleRate;
  }

  /** Trace index currently audible, given the frame's measured line period. */
  traceIndex(period: number): number {
    return Math.floor(this.positionSamples() / period);
  }

  /** Poll the clock. Uses rAF when the tab is visible for smoothness, and a
   *  timer when it is hidden -- where rAF stops but the audio does not. */
  subscribe(fn: Listener): () => void {
    this.#listeners.add(fn);
    if (this.#listeners.size === 1) this.#startTicking();
    return () => {
      this.#listeners.delete(fn);
      if (this.#listeners.size === 0) this.#stopTicking();
    };
  }

  dispose(): void {
    this.#stopNode();
    this.#stopTicking();
    this.#listeners.clear();
    this.#gain.disconnect();
    this.#buffer = null;
    this.#source = null;
  }

  // ----------------------------------------------------------------------

  #clampBufferPos(p: number): number {
    const n = this.#buffer?.length ?? 0;
    if (!(p > 0)) return 0;
    return Math.min(p, n);
  }

  /** Scheduled buffer position at context time `t`, i.e. the sample the engine
   *  is being asked to produce then -- not yet the one you can hear. */
  #positionAt(t: number): number {
    const b = this.#base;
    if (!this.#playing) return this.#clampBufferPos(b.posSamples);
    const elapsed = Math.max(0, t - b.ctxTime); // before `b.ctxTime` we have not started
    let pos = b.posSamples + elapsed * b.rate * this.sampleRate;
    const n = this.#buffer?.length ?? 0;
    if (this.#loop && n > 0 && pos > n) pos = pos % n;
    return this.#clampBufferPos(pos);
  }

  /** Audible buffer position now: the sample leaving the speaker was handed to
   *  the engine `latency` seconds ago. */
  #bufferPosition(now: number): number {
    return this.#positionAt(now - this.latency);
  }

  #startAt(offsetSamples: number): void {
    if (!this.#buffer) return;
    this.#stopNode();
    const node = this.#ctx.createBufferSource();
    node.buffer = this.#buffer;
    node.loop = this.#loop;
    node.playbackRate.value = this.#rate;
    node.connect(this.#gain);
    const token = ++this.#nodeToken;
    node.onended = () => {
      // stop() also fires onended; the token tells a real end-of-frame apart
      // from a node we replaced during a seek.
      if (token !== this.#nodeToken || !this.#playing) return;
      this.#playing = false;
      this.#base = {
        ctxTime: this.#ctx.currentTime,
        posSamples: this.#buffer?.length ?? 0,
        rate: this.#rate,
      };
      this.#emit();
      this.onended?.(this);
    };
    // start(currentTime) does *not* begin at currentTime: the engine has already
    // rendered some quanta ahead, so playback really starts whenever the graph
    // next catches up -- measured at ~30 ms on Chrome/macOS, which showed up as
    // a 3% error in the clock's slope over a short window. Scheduling a little
    // into the future puts the start on an instant we can name exactly, and the
    // baseline then describes the affine truth rather than an estimate.
    const when = this.#ctx.currentTime + this.#scheduleAhead;
    node.start(when, offsetSamples / this.sampleRate);
    this.#node = node;
    this.#playing = true;
    this.#base = { ctxTime: when, posSamples: offsetSamples, rate: this.#rate };
    this.#emit();
  }

  #stopNode(): void {
    const node = this.#node;
    this.#node = null;
    this.#nodeToken++;
    if (!node) return;
    node.onended = null;
    try {
      node.stop();
    } catch {
      // already stopped
    }
    node.disconnect();
  }

  #emit(): void {
    if (this.#listeners.size === 0) return;
    const pos = this.positionSamples();
    for (const fn of this.#listeners) fn(pos, this);
  }

  #startTicking(): void {
    const hidden = () => typeof document !== 'undefined' && document.visibilityState === 'hidden';
    const tick = () => {
      this.#emit();
      schedule();
    };
    const schedule = () => {
      if (this.#listeners.size === 0) return;
      if (!hidden() && typeof requestAnimationFrame === 'function') {
        this.#rafId = requestAnimationFrame(tick);
      } else {
        // Background timers are clamped to ~1 Hz in some browsers; that is fine,
        // because the position we read is still exact whenever we do read it.
        this.#timerId = setTimeout(tick, this.#hiddenPollMs);
      }
    };
    this.#onVisibility = () => {
      this.#cancelTick();
      schedule();
    };
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this.#onVisibility);
    }
    schedule();
  }

  #cancelTick(): void {
    if (this.#rafId) {
      cancelAnimationFrame(this.#rafId);
      this.#rafId = 0;
    }
    if (this.#timerId) {
      clearTimeout(this.#timerId);
      this.#timerId = 0;
    }
  }

  #stopTicking(): void {
    this.#cancelTick();
    if (this.#onVisibility && typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.#onVisibility);
    }
    this.#onVisibility = null;
  }
}
