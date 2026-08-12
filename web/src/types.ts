/**
 * The wiring contract for the whole application.
 *
 * Three agents share this build:
 *
 *   web/src/dsp/*      sync detection, timebase recovery, resampling, rendering
 *   web/src/audio/*    fetching and decoding the FLAC frame slices
 *   web/src/ui/*       this shell: navigation, dials, transport, scope   <-- here
 *   web/src/panels/cover.ts   the cover panel
 *
 * The seam is deliberately one-directional. The UI never imports the DSP.
 * Instead the DSP's entry point constructs a `DecoderHost`, then calls
 * `init(root, deps)` from `ui/panels.ts` and drives the shell through the
 * handle it gets back. That keeps the UI buildable and testable on its own,
 * and means neither side can reach into the other's internals.
 *
 * The expected `web/src/main.ts` (owned by the DSP agent) is roughly:
 *
 *     import { init } from './ui/panels';
 *     import { createDecoder } from './dsp/decoder';
 *     import { mountCover } from './panels/cover';
 *
 *     const catalog = await fetch('data/catalog.json').then((r) => r.json());
 *     const decoder = createDecoder(catalog);
 *     const ui = init(document.getElementById('app')!, { catalog, decoder, mountCover });
 *     ui.open(1);
 *
 * Signal facts quoted below are measured from the 384 kHz master and must not
 * be re-derived here. See pipeline/decode.py and pipeline/sync.py.
 */

// ---------------------------------------------------------------------------
// Trace geometry
// ---------------------------------------------------------------------------

/** Measured line period at 384 kHz. Every offset below is a fraction of it. */
export const NOMINAL_PERIOD_384K = 3197.4;

/** What the record cover specifies: 8.34 ms, i.e. 3200 samples at 384 kHz. */
export const COVER_PERIOD_384K = 3200;

export const TRACES_PER_FRAME = 512;

/**
 * Where things sit inside one trace, as fractions of the period, so they hold
 * at any sample rate. Mirrors the header comment of pipeline/decode.py.
 *
 * The DC reference is the back porch of the *previous* trace, which is why it
 * is expressed as a negative offset from this trace's start.
 */
export const TRACE_REGIONS = {
  pictureStart: 20 / NOMINAL_PERIOD_384K,
  pictureEnd: 2560 / NOMINAL_PERIOD_384K,
  plateauStart: 2560 / NOMINAL_PERIOD_384K,
  plateauEnd: 2900 / NOMINAL_PERIOD_384K,
  edgeStart: 2900 / NOMINAL_PERIOD_384K,
  edgeEnd: 2910 / NOMINAL_PERIOD_384K,
  porchStart: 2910 / NOMINAL_PERIOD_384K,
  porchEnd: 3160 / NOMINAL_PERIOD_384K,
  /** DC reference window, as a negative offset from the trace start. */
  refStart: -282 / NOMINAL_PERIOD_384K,
  refEnd: -37 / NOMINAL_PERIOD_384K,
} as const;

// ---------------------------------------------------------------------------
// Decoder settings — mirrors pipeline/decode.py Settings field for field
// ---------------------------------------------------------------------------

export type ResampleMode = 'peak' | 'box' | 'area' | 'lanczos3';
export type TransferMode = 'linear' | 'barry';

export interface DecoderSettings {
  /** Pixels along each trace, i.e. the height of the decoded image. */
  height: number;
  /** Traces used from the frame. 512 is the whole picture. */
  traces: number;

  /** Picture window inside a trace, as fractions of the line period. */
  pictureStart: number;
  pictureEnd: number;

  resample: ResampleMode;
  /** Lanczos half-width. Present in decode.py; only meaningful for lanczos3. */
  lanczosA: number;

  /** Clamp every trace on its back porch. */
  dcRestore: boolean;
  /** Invert the recording chain's high-pass. */
  uncouple: boolean;
  /** High-pass time constant in samples. 0 means "fit it from the picture". */
  tau: number;

  /** Wiener deconvolution strength against the measured PSF. 0 disables. */
  deconv: number;
  deconvNoise: number;

  /** The record stores a negative, so this is normally on. */
  invert: boolean;
  blackPct: number;
  whitePct: number;
  gamma: number;
  transfer: TransferMode;

  /** Hampel despike threshold in MADs. 0 disables. */
  despike: number;
  destripe: boolean;

  /** Quarter turns, matching Barry's orientation table. */
  rotate: number;
  flipScan: boolean;

  /** Horizontal scale applied at render time. */
  aspect: number;
}

export const DEFAULT_SETTINGS: DecoderSettings = {
  height: 384,
  traces: 512,
  pictureStart: TRACE_REGIONS.pictureStart,
  pictureEnd: TRACE_REGIONS.pictureEnd,
  resample: 'lanczos3',
  lanczosA: 3,
  dcRestore: true,
  uncouple: true,
  tau: 0,
  deconv: 0,
  deconvNoise: 0.02,
  invert: true,
  blackPct: 1,
  whitePct: 99,
  gamma: 1,
  transfer: 'linear',
  despike: 0,
  destripe: false,
  rotate: 0,
  flipScan: false,
  aspect: 1,
};

/**
 * Timebase override. This is deliberately *not* part of DecoderSettings: the
 * line period is measured per frame by sync.recover, never configured. The
 * dial exists so you can detune the measurement and watch the picture shear —
 * which is the whole argument of the piece.
 *
 * `locked` means "follow whatever the timebase recovery measured". Dragging
 * the dial clears it; the snap-to-lock button restores it.
 */
export interface TimebaseSettings {
  /** Samples per trace actually in use, at the asset sample rate. */
  period: number;
  /** True while `period` tracks the measured value. */
  locked: boolean;
  /** Sub-sample shift applied to trace 0, in samples. */
  phase: number;
}

export const DEFAULT_TIMEBASE: TimebaseSettings = {
  period: NOMINAL_PERIOD_384K / 4, // 799.35 at 96 kHz
  locked: true,
  phase: 0,
};

/**
 * Colour compositing for RGB triplets. Not part of decode.py Settings — the
 * pipeline decodes three separations independently and the browser combines
 * them, so these knobs live only here.
 *
 * TODO(signal): the true inter-separation registration offsets are unmeasured.
 * They default to zero rather than to a guessed number.
 */
export interface ColorSettings {
  rGain: number;
  gGain: number;
  bGain: number;
  /** Registration nudge per separation, in traces. [r, g, b]. */
  registration: [number, number, number];
  /** Which separation to show. 'rgb' composites all three. */
  separation: 'rgb' | 'r' | 'g' | 'b';
}

export const DEFAULT_COLOR: ColorSettings = {
  rGain: 1,
  gGain: 1,
  bGain: 1,
  registration: [0, 0, 0],
  separation: 'rgb',
};

/** One complete decoder configuration. The A/B wipe holds two of these. */
export interface LabState {
  settings: DecoderSettings;
  timebase: TimebaseSettings;
  color: ColorSettings;
}

export type SettingsSlot = 'a' | 'b';

// ---------------------------------------------------------------------------
// Catalog — matches web/public/data/catalog.json exactly
// ---------------------------------------------------------------------------

export type ColorRole = 'mono' | 'r' | 'g' | 'b';

export interface CatalogFrame {
  /** Channel letter L/R plus zero-padded frame index, e.g. "L000". */
  id: string;
  /** 0 = left, 1 = right. */
  channel: 0 | 1;
  index: number;
  /** Path relative to data/, e.g. "frames/L000.flac". */
  file: string;
  /** Samples of audio present before the first trace start. */
  leadIn: number;
  /** Position in the 384 kHz master, for reference and timecode. */
  startSample: number;
  periodGuess: number;
  color: ColorRole;
  /** e.g. "T03" when color is not "mono". */
  tripletId: string | null;
  /** Quarter turns, from Barry's table. */
  orientation: number;
  durationSamples: number;
  /** Pre-normalisation peak, so absolute level can be restored. */
  peak: number;
  imageNumber: number;
  title: string;
}

export interface CatalogImage {
  n: number;
  title: string;
  /** One frame id for mono, three for a colour triplet. */
  frames: string[];
  color: boolean;
}

export interface Catalog {
  version: number;
  sampleRate: number;
  nominalPeriod: number;
  tracesPerFrame: number;
  frames: CatalogFrame[];
  images: CatalogImage[];
}

// ---------------------------------------------------------------------------
// What the DSP reports back
// ---------------------------------------------------------------------------

export interface Diagnostics {
  /** Fitted line period, samples per trace, at the asset sample rate. */
  period?: number;
  /** RMS departure from a constant line rate, in samples — wow and flutter. */
  jitterRms?: number;
  /** RMS of what timing smoothing removed: our uncertainty, not the tape's. */
  measurementNoise?: number;
  /** Peak-to-peak back porch level across the frame. */
  porchDrift?: number;
  /** Fitted or configured high-pass time constant, in samples. */
  tauSamples?: number;
  /** Black and white points chosen by the percentile stretch. */
  levels?: [number, number];
  /** How many traces the correlator actually located. */
  tracesLocated?: number;
  /** Sub-sample position of trace 0 from the straight-line fit. */
  phase?: number;
  /** Per-trace timing residuals, in samples. Plotted as wow and flutter. */
  residuals?: Float32Array | number[];
  /** The folded matched filter used to locate traces. */
  template?: Float32Array | number[];
  /** Index within `template` treated as the trace start. */
  templateOrigin?: number;
  /** Wall-clock cost of the last full decode, in milliseconds. */
  decodeMs?: number;
}

export type DecoderStatus =
  | 'idle'
  | 'loading'
  | 'decoding'
  | 'ready'
  | 'playing'
  | 'paused'
  | 'error';

export type DecoderEvent =
  | { type: 'status'; status: DecoderStatus; message?: string }
  /** Playback position. `trace` is the column currently being painted. */
  | { type: 'time'; seconds: number; duration: number; trace: number; traces: number }
  /** A new image finished loading and is now the subject of the Lab. */
  | { type: 'image'; imageNumber: number }
  | { type: 'diagnostics'; data: Diagnostics };

/**
 * One trace of audio, pulled by the oscilloscope each animation frame.
 *
 * `samples` must cover at least the DC reference window of the previous trace
 * (TRACE_REGIONS.refStart) through the end of this trace's back porch, so the
 * scope can draw every labelled region without extrapolating.
 */
export interface TraceSnapshot {
  samples: Float32Array;
  /** Index within `samples` of the trace start, i.e. of t = 0. */
  origin: number;
  /** Samples per trace in use for this trace. */
  period: number;
  /** Which trace this is, 0-based. */
  index: number;
  /** Pixel bins across the picture span, i.e. the current `height`. */
  bins: number;
  pictureStart: number;
  pictureEnd: number;
  /** Back porch median subtracted by dcRestore, in signal units. */
  dcReference?: number;
}

// ---------------------------------------------------------------------------
// The dependency the UI is driven by
// ---------------------------------------------------------------------------

/**
 * Implemented by the DSP agent (web/src/dsp/*), consumed by the shell.
 *
 * Contract notes:
 *  - `attach` is called exactly once, before any other method, with the canvas
 *    the UI created and owns the CSS box of. The DSP sizes the backing store
 *    to whatever the decode produces (traces x height, after rotation); the
 *    UI's CSS letterboxes it. The oscilloscope is drawn entirely by the UI
 *    from `traceSnapshot`, so the DSP never sees that canvas.
 *  - Every setter is cheap to call at pointer-move rate. Coalesce internally.
 */
export interface DecoderHost {
  attach(surfaces: { image: HTMLCanvasElement }): void;

  /** Load and begin decoding an image by catalog number (1-116). */
  open(imageNumber: number): Promise<void>;

  setSettings(settings: DecoderSettings, slot: SettingsSlot): void;
  setTimebase(timebase: TimebaseSettings, slot: SettingsSlot): void;
  setColor(color: ColorSettings, slot: SettingsSlot): void;

  /** 1 = slot A fills the canvas, 0 = slot B fills it. */
  setWipe(t: number): void;

  play(): void;
  pause(): void;
  /** Seek within the current image, in seconds. */
  seek(seconds: number): void;
  setSpeed(rate: number): void;

  /** Returns an unsubscribe function. */
  subscribe(listener: (event: DecoderEvent) => void): () => void;

  /** The trace under the playhead, for the oscilloscope. */
  traceSnapshot?(): TraceSnapshot | null;

  /** Paint a gallery thumbnail. Called lazily as cards scroll into view. */
  renderThumbnail?(imageNumber: number, canvas: HTMLCanvasElement): Promise<void>;

  dispose?(): void;
}

/** Signature the cover panel (web/src/panels/cover.ts) must export. */
export type MountCover = (host: HTMLElement) => void | (() => void);

export interface UiDeps {
  catalog: Catalog;
  decoder: DecoderHost;
  /**
   * Mounts the cover panel. Pass `mountCover` imported from
   * `./panels/cover`. If omitted the shell renders a plate explaining that the
   * cover module has not been wired up, rather than failing to boot.
   */
  mountCover?: MountCover;
}

export type PanelId = 'cover' | 'lab' | 'gallery' | 'diagnostics' | 'gap';

/** What `init` hands back to the entry point. */
export interface UiHandle {
  /** Switch panels. Also updates the location hash. */
  show(panel: PanelId): void;
  /** Open an image in the Lab and switch to it. */
  open(imageNumber: number): void;
  /** Current configuration of a slot, for tests and for the entry point. */
  state(slot: SettingsSlot): LabState;
  /** Force a settings patch into a slot, as a preset button would. */
  patch(patch: Partial<DecoderSettings>, slot?: SettingsSlot): void;
  destroy(): void;
}
