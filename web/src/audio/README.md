# web/src/audio

Audio acquisition and playback for the browser decoder. Two modules, no dependencies.

- `loader.ts` — fetch a frame's FLAC, return 96 kHz mono `Float32Array`.
- `player.ts` — play a frame and expose a playhead the decoder can trust.

## The pitfall this exists to avoid

`decodeAudioData` resamples whatever it decodes to the sample rate of the context you call it
on. A page's ordinary `AudioContext` runs at the output device's rate — 48000 Hz on the machine
this was developed on, 44100 on plenty of others. Decoding a 96 kHz asset there silently returns
half the samples, and nothing in the API tells you it happened.

That is not a quality question here. The whole decoder is built on 96 kHz oversampling: the line
period is 799.35 samples rather than 800, trace starts are located to a fraction of a sample by
cross-correlation, and the picture/porch windows are fractions of that period. Resampled to
48 kHz the sync template blurs, sub-sample interpolation loses its footing, and the picture comes
out slanted and soft — a plausible-looking wrong answer, which is the worst kind.

So the loader decodes inside `new OfflineAudioContext(1, length, 96000)` and then checks two
things instead of trusting either: that the context really came back at 96000 Hz (some engines
clamp the requested rate), and that `buffer.sampleRate` really is 96000. Anything else is an
error, never a silent accept.

## loader.ts

```ts
import { FrameLoader } from './audio/loader';

const loader = new FrameLoader({ baseUrl: 'data/' });   // 40 MB cache by default
const audio = await loader.get(frame, { signal });      // frame is a catalog.json entry
audio.samples;      // Float32Array, mono, 96 kHz
audio.sampleRate;   // 96000, verified
audio.leadIn;       // samples before trace 0
audio.path;         // 'offline-audio-context' | 'wav-parser'
```

- **LRU cache**, budgeted in bytes of decoded float (a 512-trace frame is ~1.7 MB decoded, so
  the 40 MB default holds about two dozen). Eviction is oldest-first and never evicts the entry
  just stored. `stats()` reports hits, misses, coalesced requests, evictions and fallbacks.
- **Request coalescing**: two callers asking for the same frame id share one fetch and get the
  same object back.
- **Abort**: pass an `AbortSignal`. A caller that aborts detaches from the shared request; the
  underlying fetch is only cancelled once every caller has gone, so one component unmounting
  cannot cancel another component's load.
- **Level restore**: assets are peak-normalised so 16-bit quantisation buys the most SNR.
  If the catalog entry carries `peak`, samples are multiplied back onto the master's scale.
  Pass `restoreLevel: false` to skip it.
- **WAV fallback**: if the FLAC path fails for any reason — no FLAC decoder, no
  `OfflineAudioContext`, a context clamped away from 96 kHz — the loader fetches the `.wav`
  sibling and parses the RIFF itself. No audio engine is involved on that path, so nothing can
  resample it; whatever rate the file declares is the rate you get. The parser handles
  `WAVE_FORMAT_EXTENSIBLE` and skips `LIST`/`fact` chunks. Override the URL mapping with the
  `wavFallback` option, or turn it off by returning `null`.

`decodeWav()` is exported on its own; it is the one piece testable outside a browser.

### One measured quirk

Chrome's FLAC decode of a 16-bit asset does not reproduce the file's integers exactly as
`i / 32768`: positive samples come back as `i / 32767`, negatives as `i / 32768`. Verified against
the same slice decoded in Python (peak 0.250823975 in the file, 0.250831634 out of Chrome). It is
a sub-LSB asymmetry, 3e-5 of full scale, well under the 16-bit quantisation of the asset itself,
and the decoder re-derives its own levels per frame from the back porch and percentile clamps, so
it does not reach the picture. Worth knowing when comparing browser output against the Python
reference sample-for-sample: use the WAV path for that, which is bit-exact.

## player.ts

```ts
const player = new FramePlayer();
player.load({ id, samples, sampleRate, leadIn });
await player.play();          // needs a user gesture the first time
player.positionSamples();     // frame timebase: sample 0 is trace 0, lead-in is negative
player.traceIndex(period);    // which column is audible right now
```

The clock is `AudioContext.currentTime` arithmetic, never a frame counter. A baseline pins one
instant of context time to one buffer position, and position is affine in context time between
baselines; play, pause, seek and rate change each lay down a new baseline. Consequences:

- **Backgrounded tabs stay correct.** `requestAnimationFrame` stops when the tab is hidden and
  timers get clamped, but the audio keeps playing and `currentTime` keeps advancing. rAF is used
  only to decide *when to ask*; `subscribe()` switches to a timer when the tab is hidden, and a
  slow poll costs nothing in accuracy because each reading is computed, not accumulated.
- **Suspension is free.** `currentTime` does not advance while a context is suspended, so the
  position freezes and resumes without special handling.
- **Starts are scheduled, not immediate.** `start(currentTime)` does not begin at `currentTime`:
  the engine has already rendered ahead, so playback actually begins whenever the graph catches
  up — measured at roughly 30 ms on Chrome/macOS, which showed up as a 3% error in the clock's
  slope over a one-second window. Playback and rate changes are therefore scheduled
  `scheduleAhead` seconds (25 ms) into the future, so the baseline names an instant that has not
  been rendered yet and the arithmetic describes the truth rather than an estimate.
- **Positions are audible, not scheduled.** `outputLatency`/`baseLatency` is subtracted, so the
  decoder paints what you can hear rather than what has been queued. Set
  `latencyCompensation: false` to get the scheduled position instead.

The AudioBuffer keeps the frame's own 96 kHz rate even though the output device does not; the
engine resamples on the way out. That is fine — it only affects what the speaker does, while the
clock counts samples of *our* timebase, which is what `positionSamples()` returns.

## Verifying

`web/scratch/flac-probe.html` loads one frame through the loader and prints the decoded rate,
length, peak and first samples, plus coalescing, cache, abort and level-restore checks. Under
Vite just open `/scratch/flac-probe.html`; it imports `../src/audio/loader.ts` directly. Both the
asset and the module URL can be overridden: `?url=data/frames/L000.flac&mod=./loader.js`.

Measured on Chrome 151 / macOS, against a 4.375 s 96 kHz mono slice of the master:
`OfflineAudioContext(1, 128, 96000).sampleRate = 96000`, decoded buffer 96000 Hz and 420000
samples long while the device context ran at 48000 Hz; one network request for two concurrent
callers; aborted loads reject with `AbortError`; the `.wav` fallback path reproduces the Python
reference bit-exactly (peak 0.250823975, rms 0.050594125). The player's clock measured 96000
samples per second of context time with zero error over 1.5 s, and 192000 at 2x rate.
