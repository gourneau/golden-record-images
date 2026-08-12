/**
 * The instrument: navigation, the Lab, the gallery, diagnostics, and the essay.
 *
 * `init(root, deps)` builds the whole shell and returns a handle. The UI never
 * imports the DSP; it only ever talks to the `DecoderHost` it is handed, and
 * only ever hears back through `subscribe`. See web/src/types.ts for the full
 * contract and for the entry point this expects to be called from.
 */

import {
  COVER_PERIOD_384K,
  DEFAULT_COLOR,
  DEFAULT_SETTINGS,
  DEFAULT_TIMEBASE,
  NOMINAL_PERIOD_384K,
  TRACE_REGIONS,
  type CatalogFrame,
  type CatalogImage,
  type DecoderEvent,
  type DecoderSettings,
  type DecoderStatus,
  type Diagnostics,
  type LabState,
  type PanelId,
  type SettingsSlot,
  type TraceSnapshot,
  type UiDeps,
  type UiHandle,
} from '../types';
import {
  autoFormat,
  createDial,
  createKnob,
  createSelect,
  createToggle,
  type DialHandle,
  type Explainer,
  type KnobHandle,
  type SelectHandle,
  type ToggleHandle,
} from './dials';
import { PRESETS, createPresetBar, presetById, presetName } from './presets';

// ---------------------------------------------------------------------------
// small DOM helpers
// ---------------------------------------------------------------------------

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  html?: string,
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (html !== undefined) el.innerHTML = html;
  return el;
}

function clockText(seconds: number): string {
  const s = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')}`;
}

function toArray(x: Float32Array | number[] | undefined): ArrayLike<number> | null {
  return x && x.length ? x : null;
}

const PANELS: { id: PanelId; label: string }[] = [
  { id: 'cover', label: 'The Cover' },
  { id: 'lab', label: 'The Lab' },
  { id: 'gallery', label: 'Gallery' },
  { id: 'diagnostics', label: 'Diagnostics' },
  { id: 'gap', label: 'The Gap' },
];

const STATUS_TEXT: Record<DecoderStatus, string> = {
  idle: 'standby',
  loading: 'loading',
  decoding: 'decoding',
  ready: 'ready',
  playing: 'scanning',
  paused: 'paused',
  error: 'fault',
};

const LEGEND_DEFAULT = {
  name: 'Legend',
  text: 'Point at any control to read what it does. <em>Space</em> plays, <em>←</em> <em>→</em> seek, <em>L</em> snaps the timebase back into lock.',
};

// ---------------------------------------------------------------------------

export function init(root: HTMLElement, deps: UiDeps): UiHandle {
  const { catalog, decoder } = deps;
  const sampleRate = catalog.sampleRate || 96000;
  const nominal = catalog.nominalPeriod || NOMINAL_PERIOD_384K / 4;
  /** Master sample rate, from which the web assets were decimated. */
  const masterRate = sampleRate * 4;

  const frameById = new Map<string, CatalogFrame>(catalog.frames.map((f) => [f.id, f]));

  // --- state ------------------------------------------------------------

  const makeState = (patch: Partial<DecoderSettings>): LabState => ({
    settings: { ...DEFAULT_SETTINGS, ...patch },
    timebase: { ...DEFAULT_TIMEBASE, period: nominal },
    color: { ...DEFAULT_COLOR, registration: [0, 0, 0] },
  });

  const slots: Record<SettingsSlot, LabState> = {
    a: makeState(presetById('restored')!.settings),
    b: makeState(presetById('raw')!.settings),
  };
  let editing: SettingsSlot = 'a';
  let wipe = 1;

  let currentImage: CatalogImage | null = null;
  let currentPanel: PanelId = 'lab';
  let status: DecoderStatus = 'idle';
  let diag: Diagnostics = {};
  let measured: number | null = null;
  let playing = false;
  let duration = 0;
  let position = 0;
  let traceIndex = 0;
  let scrubbing = false;
  let firstOpen = true;
  const disposers: (() => void)[] = [];

  // ======================================================================
  // Shell
  // ======================================================================

  root.textContent = '';
  const app = h('div', 'app');

  // --- top rail ---------------------------------------------------------
  const rail = h('header', 'rail');
  rail.append(
    h(
      'div',
      'brand',
      `<span class="brand__mark">Voyager · Golden Record</span>
       <span class="brand__sub">image decoder · ${sampleRate / 1000} kHz</span>`,
    ),
  );

  const nav = h('nav', 'nav');
  nav.setAttribute('role', 'tablist');
  nav.setAttribute('aria-label', 'Sections');
  const tabs = new Map<PanelId, HTMLButtonElement>();
  for (const p of PANELS) {
    const b = h('button', 'nav__tab');
    b.type = 'button';
    b.id = `tab-${p.id}`;
    b.textContent = p.label;
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-controls', `panel-${p.id}`);
    b.setAttribute('aria-selected', 'false');
    b.addEventListener('click', () => show(p.id));
    b.addEventListener('keydown', (e) => {
      const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!dir) return;
      e.preventDefault();
      const i = PANELS.findIndex((x) => x.id === p.id);
      const next = PANELS[(i + dir + PANELS.length) % PANELS.length];
      show(next.id);
      tabs.get(next.id)?.focus();
    });
    tabs.set(p.id, b);
    nav.append(b);
  }
  rail.append(nav);

  const lamp = h('div', 'lamp', '<span class="lamp__led"></span><span class="lamp__text"></span>');
  lamp.setAttribute('role', 'status');
  const lampText = lamp.querySelector('.lamp__text') as HTMLElement;
  rail.append(lamp);

  // --- stage ------------------------------------------------------------
  const stage = h('main', 'stage');
  const panelEls = new Map<PanelId, HTMLElement>();
  for (const p of PANELS) {
    const el = h('section', 'panel');
    el.id = `panel-${p.id}`;
    el.setAttribute('role', 'tabpanel');
    el.setAttribute('aria-labelledby', `tab-${p.id}`);
    el.tabIndex = -1;
    el.hidden = true;
    panelEls.set(p.id, el);
    stage.append(el);
  }

  app.append(rail, stage);
  root.append(app);

  // ======================================================================
  // Legend window (the shared explanation strip)
  // ======================================================================

  const legendEl = h(
    'div',
    'legend',
    '<span class="legend__name"></span><span class="legend__text"></span>',
  );
  legendEl.setAttribute('aria-live', 'polite');
  const legendName = legendEl.querySelector('.legend__name') as HTMLElement;
  const legendText = legendEl.querySelector('.legend__text') as HTMLElement;

  const explainer: Explainer = {
    show(name, text) {
      legendName.textContent = name;
      legendText.textContent = text;
    },
    clear() {
      legendName.textContent = LEGEND_DEFAULT.name;
      legendText.innerHTML = LEGEND_DEFAULT.text;
    },
  };
  explainer.clear();

  // ======================================================================
  // THE LAB
  // ======================================================================

  const lab = h('div', 'lab');
  panelEls.get('lab')!.append(lab);

  // --- viewport ---------------------------------------------------------
  const view = h('div', 'view');
  const mount = h('div', 'view__mount');
  const imageCanvas = h('canvas', 'view__canvas');
  imageCanvas.width = 512;
  imageCanvas.height = 384;
  imageCanvas.setAttribute('role', 'img');
  imageCanvas.setAttribute('aria-label', 'Decoded frame');

  const scanline = h('div', 'scanline');
  const wipeEl = h('div', 'wipe', '<span class="wipe__grip"></span>');
  wipeEl.setAttribute('role', 'slider');
  wipeEl.tabIndex = 0;
  wipeEl.setAttribute('aria-label', 'A/B wipe position');
  wipeEl.setAttribute('aria-valuemin', '0');
  wipeEl.setAttribute('aria-valuemax', '100');
  const tagA = h('span', 'wipe__tag wipe__tag--a');
  const tagB = h('span', 'wipe__tag wipe__tag--b');
  wipeEl.append(tagA, tagB);
  mount.append(imageCanvas, scanline, wipeEl);

  const slate = h(
    'div',
    'view__slate',
    '<span class="view__num"></span><span class="view__title"></span><span class="view__meta"></span>',
  );
  const slateNum = slate.querySelector('.view__num') as HTMLElement;
  const slateTitle = slate.querySelector('.view__title') as HTMLElement;
  const slateMeta = slate.querySelector('.view__meta') as HTMLElement;
  const viewStage = h('div', 'view__stage');
  viewStage.append(mount);
  view.append(viewStage, slate);

  // --- oscilloscope -----------------------------------------------------
  const scope = h('div', 'scope');
  const scopeHead = h(
    'div',
    'scope__head',
    `<span class="eng">Trace</span><span class="num scope__trace">—</span>
     <span class="eng">Period</span><span class="num scope__period">—</span>
     <span class="eng">Bins</span><span class="num scope__bins">—</span>`,
  );
  const scopeTrace = scopeHead.querySelector('.scope__trace') as HTMLElement;
  const scopePeriod = scopeHead.querySelector('.scope__period') as HTMLElement;
  const scopeBins = scopeHead.querySelector('.scope__bins') as HTMLElement;
  const scopeScreen = h('div', 'scope__screen');
  const scopeCanvas = h('canvas', 'scope__canvas');
  scopeCanvas.setAttribute('role', 'img');
  scopeCanvas.setAttribute(
    'aria-label',
    'Oscilloscope of the current trace, showing the picture span, sync plateau, falling edge and back porch',
  );
  scopeScreen.append(scopeCanvas);
  scope.append(scopeHead, scopeScreen);

  // --- transport --------------------------------------------------------
  const transport = h('div', 'transport');
  const playBtn = h('button', 'tbtn');
  playBtn.type = 'button';
  playBtn.setAttribute('aria-label', 'Play');
  const ICON_PLAY = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1l9 5-9 5z"/></svg>';
  const ICON_PAUSE =
    '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>';
  playBtn.innerHTML = ICON_PLAY;
  playBtn.addEventListener('click', () => (playing ? decoder.pause() : decoder.play()));

  const scrub = document.createElement('input');
  scrub.type = 'range';
  scrub.className = 'scrub';
  scrub.min = '0';
  scrub.max = '1';
  scrub.step = '0.001';
  scrub.value = '0';
  scrub.setAttribute('aria-label', 'Playback position');
  scrub.addEventListener('pointerdown', () => (scrubbing = true));
  scrub.addEventListener('pointerup', () => (scrubbing = false));
  scrub.addEventListener('input', () => {
    position = Number(scrub.value);
    decoder.seek(position);
    paintClock();
  });
  scrub.addEventListener('keydown', () => (scrubbing = true));
  scrub.addEventListener('keyup', () => (scrubbing = false));

  const clock = h('div', 'clock', '<span class="clock__now">0:00.0</span><em> / </em><span class="clock__dur">0:00.0</span>');
  const clockNow = clock.querySelector('.clock__now') as HTMLElement;
  const clockDur = clock.querySelector('.clock__dur') as HTMLElement;

  const speed = createSelect({
    label: 'Speed',
    options: [
      { value: '0.25', label: '¼×' },
      { value: '0.5', label: '½×' },
      { value: '1', label: '1×' },
      { value: '2', label: '2×' },
      { value: '4', label: '4×' },
    ],
    value: '1',
    describe:
      'How fast the record is played back. The scan paints in step with the audio, so slowing it down slows the painting.',
    onInput: (v) => decoder.setSpeed(Number(v)),
    explainer,
  });
  speed.el.classList.add('seg--inline');

  transport.append(playBtn, scrub, clock, speed.el);

  // --- rack -------------------------------------------------------------
  const rack = h('div', 'rack');
  const rackScroll = h('div', 'rack__scroll');
  rack.append(rackScroll, legendEl);

  lab.append(view, scope, transport, rack);

  // ======================================================================
  // Presets bar + A/B
  // ======================================================================

  const presetBar = createPresetBar({
    explainer,
    onPreset: (id) => {
      const p = presetById(id);
      if (!p) return;
      Object.assign(slots[editing].settings, p.settings);
      syncControls();
      pushSettings();
    },
    onSlot: (s) => {
      editing = s;
      presetBar.setSlot(s);
      syncControls();
    },
    onWipe: setWipe,
    onSwap: () => {
      const a = slots.a;
      slots.a = slots.b;
      slots.b = a;
      syncControls();
      pushSettings('a');
      pushSettings('b');
      updateWipeTags();
    },
  });
  rackScroll.append(presetBar.el);

  function setWipe(t: number): void {
    wipe = Math.min(1, Math.max(0, t));
    decoder.setWipe(wipe);
    presetBar.setWipe(wipe);
    wipeEl.style.left = `${wipe * 100}%`;
    wipeEl.setAttribute('aria-valuenow', String(Math.round(wipe * 100)));
    wipeEl.setAttribute(
      'aria-valuetext',
      `${Math.round(wipe * 100)}% ${presetName(matchPreset('a'))}, ${Math.round(
        (1 - wipe) * 100,
      )}% ${presetName(matchPreset('b'))}`,
    );
    // Park the divider out of the way when one slot fills the frame.
    wipeEl.style.opacity = wipe > 0.995 || wipe < 0.005 ? '0.4' : '1';
    updateWipeTags();
  }

  function updateWipeTags(): void {
    tagA.textContent = presetName(matchPreset('a'));
    tagB.textContent = presetName(matchPreset('b'));
    // Hide the tag on whichever side the divider has run out of room on.
    tagA.hidden = wipe < 0.12;
    tagB.hidden = wipe > 0.88;
  }

  // drag the divider on the picture itself
  {
    let pid = -1;
    const move = (e: PointerEvent): void => {
      const r = mount.getBoundingClientRect();
      if (r.width > 0) setWipe((e.clientX - r.left) / r.width);
    };
    wipeEl.addEventListener('pointerdown', (e) => {
      pid = e.pointerId;
      wipeEl.setPointerCapture(pid);
      wipeEl.focus();
      e.preventDefault();
    });
    wipeEl.addEventListener('pointermove', (e) => {
      if (e.pointerId === pid) move(e);
    });
    const stop = (e: PointerEvent): void => {
      if (e.pointerId === pid) pid = -1;
    };
    wipeEl.addEventListener('pointerup', stop);
    wipeEl.addEventListener('pointercancel', stop);
    wipeEl.addEventListener('keydown', (e) => {
      const step = e.shiftKey ? 0.01 : 0.05;
      if (e.key === 'ArrowLeft') setWipe(wipe - step);
      else if (e.key === 'ArrowRight') setWipe(wipe + step);
      else if (e.key === 'Home') setWipe(0);
      else if (e.key === 'End') setWipe(1);
      else return;
      e.preventDefault();
    });
    wipeEl.addEventListener('pointerenter', () =>
      explainer.show(
        'A/B wipe',
        'Drag to split the picture between the two settings slots. Everything left of the line is A, everything right is B.',
      ),
    );
    wipeEl.addEventListener('pointerleave', () => explainer.clear());
  }

  // ======================================================================
  // The hero: line period
  // ======================================================================

  // +/- 3 samples per trace. Half a sample already shears the frame by more
  // than half its height, so a wider range would only buy coarser resolution
  // across the part of the travel anyone actually uses.
  const knobMin = nominal - 3;
  const knobMax = nominal + 3;

  const knob: KnobHandle = createKnob({
    min: knobMin,
    max: knobMax,
    step: 0.002,
    fine: 0.0002,
    value: nominal,
    measured: null,
    describe:
      'Samples per trace. The timebase measures this per frame; drag to detune it and the picture shears, because every trace after the first starts in the wrong place.',
    onInput: (v) => {
      slots[editing].timebase.period = v;
      slots[editing].timebase.locked = false;
      decoder.setTimebase(slots[editing].timebase, editing);
      updateShearContext();
    },
    onSnap: () => snapToLock(),
    explainer,
  });
  rackScroll.append(knob.el);

  // The rest of the timebase group lives under the knob, in the same block.
  const timebaseRest = h('div', 'hero__rest');
  knob.el.append(timebaseRest);

  function snapToLock(): void {
    const target = measured ?? nominal;
    const tb = slots[editing].timebase;
    const from = tb.period;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

    const finish = (): void => {
      tb.period = target;
      tb.locked = true;
      knob.set(target, true);
      decoder.setTimebase(tb, editing);
      updateShearContext();
    };

    if (reduced || Math.abs(target - from) < 0.002) {
      finish();
      return;
    }
    // Animating the snap is the point: you watch the picture pull itself
    // straight rather than jumping there.
    const t0 = performance.now();
    const DUR = 420;
    const step = (now: number): void => {
      const p = Math.min(1, (now - t0) / DUR);
      const e = 1 - Math.pow(1 - p, 3);
      tb.period = from + (target - from) * e;
      tb.locked = false;
      knob.set(tb.period, false);
      decoder.setTimebase(tb, editing);
      if (p < 1) requestAnimationFrame(step);
      else finish();
    };
    requestAnimationFrame(step);
  }

  function updateShearContext(): void {
    const s = slots[editing].settings;
    const p = slots[editing].timebase.period;
    knob.setContext({
      traces: s.traces,
      spanSamples: Math.max(1, (s.pictureEnd - s.pictureStart) * p),
      height: s.height,
    });
  }

  // ======================================================================
  // Dial groups
  // ======================================================================

  type NumKey =
    | 'height'
    | 'traces'
    | 'pictureStart'
    | 'pictureEnd'
    | 'lanczosA'
    | 'tau'
    | 'deconv'
    | 'deconvNoise'
    | 'blackPct'
    | 'whitePct'
    | 'gamma'
    | 'despike'
    | 'rotate'
    | 'aspect';
  type BoolKey = 'dcRestore' | 'uncouple' | 'invert' | 'destripe' | 'flipScan';

  const numDials = new Map<NumKey, DialHandle>();
  const boolToggles = new Map<BoolKey, ToggleHandle>();
  const resampleSel: { h?: SelectHandle<DecoderSettings['resample']> } = {};
  const transferSel: { h?: SelectHandle<DecoderSettings['transfer']> } = {};

  function group(name: string, open: boolean): { el: HTMLElement; body: HTMLElement; note: HTMLElement } {
    const el = h('section', 'group');
    el.dataset.open = open ? '1' : '0';
    const head = h(
      'button',
      'group__head',
      `<span class="group__name"></span><span class="group__note"></span>
       <span class="group__rule"></span><span class="group__chev"></span>`,
    );
    head.type = 'button';
    (head.querySelector('.group__name') as HTMLElement).textContent = name;
    head.setAttribute('aria-expanded', open ? 'true' : 'false');
    const body = h('div', 'group__body');
    head.addEventListener('click', () => {
      const next = el.dataset.open !== '1';
      el.dataset.open = next ? '1' : '0';
      head.setAttribute('aria-expanded', next ? 'true' : 'false');
    });
    el.append(head, body);
    rackScroll.append(el);
    return { el, body, note: head.querySelector('.group__note') as HTMLElement };
  }

  function numDial(
    key: NumKey,
    opts: {
      label: string;
      unit?: string;
      min: number;
      max: number;
      step: number;
      fine?: number;
      describe: string;
      curve?: 'lin' | 'log';
      format?: (v: number) => string;
      into: HTMLElement;
    },
  ): void {
    const d = createDial({
      label: opts.label,
      unit: opts.unit,
      min: opts.min,
      max: opts.max,
      step: opts.step,
      fine: opts.fine,
      value: slots[editing].settings[key],
      home: DEFAULT_SETTINGS[key],
      describe: opts.describe,
      curve: opts.curve,
      format: opts.format ?? autoFormat(opts.step),
      onInput: (v) => {
        slots[editing].settings[key] = v;
        if (key === 'height' || key === 'traces' || key === 'pictureStart' || key === 'pictureEnd') {
          updateShearContext();
        }
        pushSettings();
      },
      explainer,
    });
    numDials.set(key, d);
    opts.into.append(d.el);
  }

  function boolToggle(
    key: BoolKey,
    opts: { label: string; describe: string; into: HTMLElement },
  ): void {
    const t = createToggle({
      label: opts.label,
      value: slots[editing].settings[key],
      describe: opts.describe,
      onInput: (v) => {
        slots[editing].settings[key] = v;
        pushSettings();
      },
      explainer,
    });
    boolToggles.set(key, t);
    opts.into.append(t.el);
  }

  /** Picture-window dials read out in samples at 384 kHz, as decode.py does. */
  const asMasterSamples = (v: number): string => (v * NOMINAL_PERIOD_384K).toFixed(0);

  // Assigned in the Timebase block immediately below.
  let phaseDial!: DialHandle;

  // --- Timebase (under the knob) ---------------------------------------
  numDial('traces', {
    label: 'Traces',
    unit: 'lines',
    min: 16,
    max: 512,
    step: 1,
    describe: 'How many of the frame’s 512 lines to decode. Fewer means a shorter, faster scan.',
    into: timebaseRest,
  });
  numDial('pictureStart', {
    label: 'Picture start',
    unit: 'smp @384k',
    min: 0,
    max: 0.2,
    step: 0.0005,
    fine: 0.00005,
    format: asMasterSamples,
    describe: 'Where the picture begins inside a trace, measured from the sync notch.',
    into: timebaseRest,
  });
  numDial('pictureEnd', {
    label: 'Picture end',
    unit: 'smp @384k',
    min: 0.5,
    max: 0.86,
    step: 0.0005,
    fine: 0.00005,
    format: asMasterSamples,
    describe: 'Where the picture ends and the sync plateau begins. Push past it and sync bleeds in.',
    into: timebaseRest,
  });
  {
    const phase = createDial({
      label: 'Phase',
      unit: 'smp',
      min: -400,
      max: 400,
      step: 0.25,
      fine: 0.025,
      value: 0,
      home: 0,
      describe: 'Sub-sample shift of trace 0. Slides the whole picture sideways within the line.',
      onInput: (v) => {
        slots[editing].timebase.phase = v;
        decoder.setTimebase(slots[editing].timebase, editing);
      },
      explainer,
    });
    timebaseRest.append(phase.el);
    phaseDial = phase;
  }


  // --- Levels ------------------------------------------------------------
  const gLevels = group('Levels', true);
  boolToggle('dcRestore', {
    label: 'DC restore',
    describe:
      'Clamp every trace on its back porch — the flat stretch before the picture — so slow level drift cannot become brightness.',
    into: gLevels.body,
  });
  boolToggle('uncouple', {
    label: 'Uncouple',
    describe: 'Invert the recording chain’s high-pass, which otherwise leaves large flat areas tilted.',
    into: gLevels.body,
  });
  numDial('tau', {
    label: 'Tau',
    unit: 'smp',
    min: 0,
    max: 4_000_000,
    step: 100,
    fine: 10,
    curve: 'log',
    format: (v) => (v <= 0 ? 'fit' : v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0)),
    describe: 'High-pass time constant used by Uncouple. At zero the decoder fits it from the picture.',
    into: gLevels.body,
  });
  boolToggle('invert', {
    label: 'Invert',
    describe: 'The record stores a negative — more signal means darker — so this is normally on.',
    into: gLevels.body,
  });
  numDial('blackPct', {
    label: 'Black point',
    unit: '%ile',
    min: 0,
    max: 20,
    step: 0.1,
    describe: 'Percentile of the frame mapped to black. Robust to the odd click or scratch.',
    into: gLevels.body,
  });
  numDial('whitePct', {
    label: 'White point',
    unit: '%ile',
    min: 80,
    max: 100,
    step: 0.1,
    describe: 'Percentile of the frame mapped to white.',
    into: gLevels.body,
  });
  numDial('gamma', {
    label: 'Gamma',
    min: 0.3,
    max: 3,
    step: 0.01,
    describe: 'Midtone bend applied after the level stretch. Above 1 opens the shadows.',
    into: gLevels.body,
  });
  transferSel.h = createSelect<DecoderSettings['transfer']>({
    label: 'Transfer',
    options: [
      { value: 'linear', label: 'Linear' },
      { value: 'barry', label: 'Barry' },
    ],
    value: slots[editing].settings.transfer,
    describe: 'Tone curve. “Barry” is the cosine ramp from the 2017 decoder, kept so it can be compared.',
    onInput: (v) => {
      slots[editing].settings.transfer = v;
      pushSettings();
    },
    explainer,
  });
  gLevels.body.append(transferSel.h.el);

  // --- Detail ------------------------------------------------------------
  const gDetail = group('Detail', true);
  numDial('height', {
    label: 'Height',
    unit: 'px',
    min: 64,
    max: 640,
    step: 8,
    fine: 1,
    describe:
      'Pixels per trace. The picture span holds about 635 samples at 96 kHz, so past roughly 512 you are magnifying, not resolving.',
    into: gDetail.body,
  });
  resampleSel.h = createSelect<DecoderSettings['resample']>({
    label: 'Resample',
    options: [
      { value: 'peak', label: 'Peak' },
      { value: 'box', label: 'Box' },
      { value: 'area', label: 'Area' },
      { value: 'lanczos3', label: 'Lanczos' },
    ],
    value: slots[editing].settings.resample,
    describe:
      'How samples become pixels. Peak takes the loudest sample in each bin, area averages them, Lanczos filters properly and keeps sub-sample timing.',
    onInput: (v) => {
      slots[editing].settings.resample = v;
      pushSettings();
    },
    explainer,
  });
  gDetail.body.append(resampleSel.h.el);
  numDial('deconv', {
    label: 'Deconvolve',
    min: 0,
    max: 1.5,
    step: 0.01,
    describe:
      'Wiener strength against the impulse response measured from the sync edge. Sharpens, then rings.',
    into: gDetail.body,
  });
  numDial('deconvNoise', {
    label: 'Deconv noise',
    min: 0.001,
    max: 0.5,
    step: 0.001,
    curve: 'log',
    format: (v) => v.toFixed(3),
    describe: 'Noise floor assumed by the deconvolution. Lower is sharper and grainier.',
    into: gDetail.body,
  });

  // --- Geometry ----------------------------------------------------------
  const gGeom = group('Geometry', false);
  numDial('rotate', {
    label: 'Rotate',
    unit: '× 90°',
    min: 0,
    max: 3,
    step: 1,
    describe: 'Quarter turns, following the orientation column of Barry’s frame table.',
    into: gGeom.body,
  });
  boolToggle('flipScan', {
    label: 'Flip scan',
    describe: 'Reverse the order of the traces, mirroring the picture along the scan direction.',
    into: gGeom.body,
  });
  numDial('aspect', {
    label: 'Aspect',
    unit: '×',
    min: 0.5,
    max: 2,
    step: 0.01,
    describe: 'Horizontal scale at render time. The encoder’s pixel was not square.',
    into: gGeom.body,
  });

  // --- Colour ------------------------------------------------------------
  const gColour = group('Colour', false);
  const colourSel = createSelect<'rgb' | 'r' | 'g' | 'b'>({
    label: 'Separation',
    options: [
      { value: 'rgb', label: 'RGB' },
      { value: 'r', label: 'R' },
      { value: 'g', label: 'G' },
      { value: 'b', label: 'B' },
    ],
    value: 'rgb',
    describe:
      'Colour photographs were sent as three consecutive frames — a red, a green and a blue separation. Show the composite or one separation on its own.',
    onInput: (v) => {
      slots[editing].color.separation = v;
      decoder.setColor(slots[editing].color, editing);
    },
    explainer,
  });
  gColour.body.append(colourSel.el);

  const gainDials = {} as Record<'r' | 'g' | 'b', DialHandle>;
  const regDials = {} as Record<'r' | 'g' | 'b', DialHandle>;

  const setGain = (ch: 'r' | 'g' | 'b', v: number): void => {
    const col = slots[editing].color;
    if (ch === 'r') col.rGain = v;
    else if (ch === 'g') col.gGain = v;
    else col.bGain = v;
  };

  (['r', 'g', 'b'] as const).forEach((ch, i) => {
    const name = { r: 'Red', g: 'Green', b: 'Blue' }[ch];
    gainDials[ch] = createDial({
      label: `${name} gain`,
      unit: '×',
      min: 0,
      max: 2.5,
      step: 0.01,
      value: 1,
      home: 1,
      describe: `Weight of the ${name.toLowerCase()} separation in the composite. The three passes were not exposed alike.`,
      onInput: (v) => {
        setGain(ch, v);
        decoder.setColor(slots[editing].color, editing);
      },
      explainer,
    });
    regDials[ch] = createDial({
      label: `${name} register`,
      unit: 'traces',
      min: -12,
      max: 12,
      step: 0.25,
      fine: 0.05,
      value: 0,
      home: 0,
      describe: `Slide the ${name.toLowerCase()} separation along the scan to line it up with the others.`,
      onInput: (v) => {
        slots[editing].color.registration[i] = v;
        decoder.setColor(slots[editing].color, editing);
      },
      explainer,
    });
    gColour.body.append(gainDials[ch].el, regDials[ch].el);
  });

  // --- Noise -------------------------------------------------------------
  const gNoise = group('Noise', false);
  numDial('despike', {
    label: 'Despike',
    unit: 'MAD',
    min: 0,
    max: 12,
    step: 0.1,
    format: (v) => (v <= 0 ? 'off' : v.toFixed(1)),
    describe:
      'Hampel threshold for pressing out clicks before the samples are binned. A click is a few samples wide here, and a whole pixel afterwards.',
    into: gNoise.body,
  });
  boolToggle('destripe', {
    label: 'Destripe',
    describe:
      'Remove the constant brightness difference between neighbouring traces. Each line went through the chain at a slightly different moment.',
    into: gNoise.body,
  });

  // ======================================================================
  // Settings plumbing
  // ======================================================================

  function matchPreset(slot: SettingsSlot): string | null {
    const s = slots[slot].settings;
    for (const p of PRESETS) {
      let ok = true;
      for (const k of Object.keys(p.settings) as (keyof DecoderSettings)[]) {
        if (s[k] !== p.settings[k]) {
          ok = false;
          break;
        }
      }
      if (ok) return p.id;
    }
    return null;
  }

  function pushSettings(slot: SettingsSlot = editing): void {
    decoder.setSettings(slots[slot].settings, slot);
    presetBar.setActive(slot, matchPreset(slot));
    updateWipeTags();
    updateShearContext();
  }

  /** Repaint every control from whichever slot is being edited. */
  function syncControls(): void {
    const st = slots[editing];
    numDials.forEach((d, k) => d.set(st.settings[k]));
    boolToggles.forEach((t, k) => t.set(st.settings[k]));
    resampleSel.h?.set(st.settings.resample);
    transferSel.h?.set(st.settings.transfer);
    colourSel.set(st.color.separation);
    gainDials.r.set(st.color.rGain);
    gainDials.g.set(st.color.gGain);
    gainDials.b.set(st.color.bGain);
    regDials.r.set(st.color.registration[0]);
    regDials.g.set(st.color.registration[1]);
    regDials.b.set(st.color.registration[2]);
    phaseDial.set(st.timebase.phase);
    knob.set(st.timebase.period, st.timebase.locked);
    presetBar.setActive(editing, matchPreset(editing));
    updateShearContext();
  }

  // ======================================================================
  // GALLERY
  // ======================================================================

  const gallery = h('div', 'sheet');
  gallery.innerHTML = `
    <div class="sheet__head">
      <h1 class="sheet__title">Gallery</h1>
      <p class="sheet__note">${catalog.images.length} images · ${catalog.frames.length} frames<br />
      thumbnails are decoded on demand</p>
    </div>`;
  const grid = h('div', 'grid');
  gallery.append(grid);
  panelEls.get('gallery')!.append(gallery);

  const cardByImage = new Map<number, HTMLElement>();
  const thumbDone = new WeakSet<HTMLCanvasElement>();

  const thumbObserver =
    typeof IntersectionObserver !== 'undefined'
      ? new IntersectionObserver(
          (entries) => {
            const render = decoder.renderThumbnail;
            if (!render) return;
            for (const en of entries) {
              if (!en.isIntersecting) continue;
              const card = en.target as HTMLElement;
              const canvas = card.querySelector('canvas') as HTMLCanvasElement;
              const n = Number(card.dataset.n);
              if (thumbDone.has(canvas)) continue;
              thumbDone.add(canvas);
              thumbObserver?.unobserve(card);
              Promise.resolve(render.call(decoder, n, canvas))
                .then(() => {
                  canvas.hidden = false;
                  (card.querySelector('.card__plate') as HTMLElement).hidden = true;
                })
                .catch(() => {
                  // A thumbnail that will not decode leaves its plate showing.
                  thumbDone.delete(canvas);
                });
            }
          },
          { rootMargin: '320px' },
        )
      : null;

  for (const img of catalog.images) {
    const card = h('button', 'card');
    card.type = 'button';
    card.dataset.n = String(img.n);
    card.innerHTML = `
      <span class="card__well">
        <span class="card__plate">${String(img.n).padStart(3, '0')}</span>
        <canvas class="card__canvas" hidden width="320" height="240"></canvas>
        ${img.color ? '<span class="card__tag">RGB</span>' : ''}
      </span>
      <span class="card__foot">
        <span class="card__n">№ ${String(img.n).padStart(3, '0')}</span>
        <span class="card__title"></span>
      </span>`;
    (card.querySelector('.card__title') as HTMLElement).textContent = img.title;
    card.setAttribute('aria-label', `Image ${img.n}: ${img.title}. Open in the Lab.`);
    card.addEventListener('click', () => open(img.n));
    grid.append(card);
    cardByImage.set(img.n, card);
  }

  /**
   * Observing while the panel is still `display: none` records every card as
   * off-screen and never revisits it, so the cards are only handed to the
   * observer once the gallery has actually been laid out.
   */
  let thumbsArmed = false;
  function armThumbnails(): void {
    if (thumbsArmed || !decoder.renderThumbnail || !thumbObserver) return;
    thumbsArmed = true;
    for (const card of cardByImage.values()) thumbObserver.observe(card);
  }

  // ======================================================================
  // DIAGNOSTICS
  // ======================================================================

  const diagSheet = h('div', 'sheet');
  diagSheet.innerHTML = `
    <div class="sheet__head">
      <h1 class="sheet__title">Diagnostics</h1>
      <p class="sheet__note">everything the timebase recovery<br />learned about this frame</p>
    </div>
    <div class="diag">
      <div class="card-panel">
        <div class="card-panel__head"><span class="eng">Timebase</span></div>
        <div class="card-panel__body"><dl class="readouts" data-role="timebase"></dl></div>
      </div>
      <div class="card-panel">
        <div class="card-panel__head"><span class="eng">Frame</span></div>
        <div class="card-panel__body"><dl class="readouts" data-role="frame"></dl></div>
      </div>
      <div class="card-panel">
        <div class="card-panel__head"><span class="eng">Wow &amp; flutter</span></div>
        <div class="card-panel__body">
          <canvas class="plot" data-role="residuals"></canvas>
          <p class="plot__cap">Per-trace timing residual: how far each line fell from a constant
          line rate. This is the master tape breathing, not our measurement noise.</p>
        </div>
      </div>
      <div class="card-panel">
        <div class="card-panel__head"><span class="eng">Matched filter</span></div>
        <div class="card-panel__body">
          <canvas class="plot" data-role="template"></canvas>
          <p class="plot__cap">Several hundred lines folded together at the fitted period. Picture
          content averages away; everything locked to the line clock survives.</p>
        </div>
      </div>
    </div>`;
  panelEls.get('diagnostics')!.append(diagSheet);

  const dlTimebase = diagSheet.querySelector('[data-role="timebase"]') as HTMLElement;
  const dlFrame = diagSheet.querySelector('[data-role="frame"]') as HTMLElement;
  const plotResiduals = diagSheet.querySelector('[data-role="residuals"]') as HTMLCanvasElement;
  const plotTemplate = diagSheet.querySelector('[data-role="template"]') as HTMLCanvasElement;

  function rows(target: HTMLElement, items: [string, string][]): void {
    target.innerHTML = items
      .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
      .join('');
  }

  function paintDiagnostics(): void {
    const p = diag.period;
    const inUse = slots[editing].timebase.period;
    rows(dlTimebase, [
      ['Period measured', p ? `${p.toFixed(4)} smp` : '—'],
      ['Period in use', `${inUse.toFixed(4)} smp`],
      ['Period at 384 kHz', p ? `${(p * 4).toFixed(3)} smp` : '—'],
      ['Frame time', p ? `${((p * 512) / sampleRate).toFixed(4)} s` : '—'],
      ['Traces located', diag.tracesLocated != null ? String(diag.tracesLocated) : '—'],
      ['Jitter RMS', diag.jitterRms != null ? `${diag.jitterRms.toFixed(3)} smp` : '—'],
      [
        'Measurement noise',
        diag.measurementNoise != null ? `${diag.measurementNoise.toFixed(3)} smp` : '—',
      ],
      ['Porch drift', diag.porchDrift != null ? diag.porchDrift.toExponential(2) : '—'],
      [
        'Tau',
        diag.tauSamples != null
          ? diag.tauSamples > 0
            ? `${(diag.tauSamples / 1000).toFixed(2)}k smp`
            : 'not fitted'
          : '—',
      ],
      [
        'Levels',
        diag.levels ? `${diag.levels[0].toExponential(2)} … ${diag.levels[1].toExponential(2)}` : '—',
      ],
      ['Decode time', diag.decodeMs != null ? `${diag.decodeMs.toFixed(0)} ms` : '—'],
    ]);

    const frames = (currentImage?.frames ?? []).map((id) => frameById.get(id)).filter(Boolean) as CatalogFrame[];
    const first = frames[0];
    rows(dlFrame, [
      ['Image', currentImage ? `№ ${currentImage.n}` : '—'],
      ['Title', currentImage?.title ?? '—'],
      ['Frames', frames.length ? frames.map((f) => f.id).join(' · ') : '—'],
      ['Channel', first ? (first.channel === 0 ? 'left' : 'right') : '—'],
      ['Start sample', first ? first.startSample.toLocaleString('en-US') : '—'],
      ['Timecode in master', first ? clockText(first.startSample / masterRate) : '—'],
      ['Period guess', first ? `${first.periodGuess.toFixed(3)} smp` : '—'],
      ['Lead-in', first ? `${first.leadIn} smp` : '—'],
      ['Pre-normalisation peak', first ? first.peak.toFixed(4) : '—'],
      ['Orientation', first ? `${first.orientation} × 90°` : '—'],
    ]);

    drawResiduals();
    drawTemplate();
  }

  function fitCanvas(c: HTMLCanvasElement): CanvasRenderingContext2D | null {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(c.clientWidth * dpr));
    const hh = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== hh) {
      c.width = w;
      c.height = hh;
    }
    const ctx = c.getContext('2d');
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, c.clientWidth, c.clientHeight);
    return ctx;
  }

  function emptyPlot(ctx: CanvasRenderingContext2D, w: number, hh: number, msg: string): void {
    ctx.fillStyle = '#4a4433';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(msg, w / 2, hh / 2);
  }

  function drawResiduals(): void {
    const ctx = fitCanvas(plotResiduals);
    if (!ctx) return;
    const w = plotResiduals.clientWidth;
    const hh = plotResiduals.clientHeight;
    const r = toArray(diag.residuals);
    if (!r) {
      emptyPlot(ctx, w, hh, 'awaiting decode');
      return;
    }
    let peak = 1e-6;
    for (let i = 0; i < r.length; i++) peak = Math.max(peak, Math.abs(r[i]));
    const mid = hh / 2;
    const sy = (hh / 2 - 8) / peak;

    ctx.strokeStyle = '#2a2e35';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(w, mid);
    ctx.stroke();

    if (diag.jitterRms) {
      ctx.strokeStyle = 'rgba(200,160,82,.45)';
      ctx.setLineDash([3, 3]);
      for (const s of [-1, 1]) {
        ctx.beginPath();
        ctx.moveTo(0, mid - s * diag.jitterRms * sy);
        ctx.lineTo(w, mid - s * diag.jitterRms * sy);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    ctx.strokeStyle = '#78e0ce';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < r.length; i++) {
      const x = (i / Math.max(1, r.length - 1)) * w;
      const y = mid - r[i] * sy;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#7d6739';
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`±${peak.toFixed(2)} smp`, 6, 12);
    ctx.textAlign = 'right';
    ctx.fillText(`${r.length} traces`, w - 6, hh - 6);
  }

  function drawTemplate(): void {
    const ctx = fitCanvas(plotTemplate);
    if (!ctx) return;
    const w = plotTemplate.clientWidth;
    const hh = plotTemplate.clientHeight;
    const t = toArray(diag.template);
    if (!t) {
      emptyPlot(ctx, w, hh, 'awaiting decode');
      return;
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < t.length; i++) {
      lo = Math.min(lo, t[i]);
      hi = Math.max(hi, t[i]);
    }
    const pad = (hi - lo) * 0.12 || 1e-6;
    lo -= pad;
    hi += pad;
    const y = (v: number): number => hh - ((v - lo) / (hi - lo)) * hh;

    ctx.strokeStyle = '#78e0ce';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < t.length; i++) {
      const x = (i / Math.max(1, t.length - 1)) * w;
      if (i === 0) ctx.moveTo(x, y(t[i]));
      else ctx.lineTo(x, y(t[i]));
    }
    ctx.stroke();

    if (diag.templateOrigin != null) {
      const x = (diag.templateOrigin / Math.max(1, t.length - 1)) * w;
      ctx.strokeStyle = '#c8a052';
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, hh);
      ctx.stroke();
      ctx.fillStyle = '#c8a052';
      ctx.font = '9px ui-monospace, monospace';
      ctx.textAlign = x > w / 2 ? 'right' : 'left';
      ctx.fillText('trace start', x + (x > w / 2 ? -5 : 5), 12);
    }
  }

  // ======================================================================
  // THE GAP
  // ======================================================================

  // The whole essay is one arithmetic: a per-line error times 512 lines.
  const coverPeriodAtAsset = COVER_PERIOD_384K / 4;
  const gapDrift = COVER_PERIOD_384K - NOMINAL_PERIOD_384K; // 2.6 samples at 384 kHz
  const gapSlip384 = gapDrift * 512;
  const gapSlipPixels =
    (gapSlip384 / ((TRACE_REGIONS.pictureEnd - TRACE_REGIONS.pictureStart) * NOMINAL_PERIOD_384K)) *
    384;

  const gapSheet = h('div', 'sheet sheet--essay');
  gapSheet.innerHTML = `
    <div class="gap__hero">
      <span class="gap__eyebrow">The Gap</span>
      <div class="gap__figure">${gapDrift.toFixed(1)}</div>
      <div class="gap__figure-unit">samples per line</div>
      <p class="gap__caption">The difference between the line the record specifies and the line
      that is actually on this digitisation. Everything hard about reading these pictures lives
      inside it.</p>
    </div>
    <div class="prose">
      <p class="lede">The cover of the record tells you how to build this decoder. It gives the
      line time as 11,845,632 periods of the hydrogen hyperfine transition — 8.34 milliseconds —
      because hydrogen is the one clock you can assume a stranger already owns.</p>

      <p>At 384 kHz, 8.34 ms is <span class="num">3200</span> samples. Fold this recording at 3200
      and the picture tears itself apart. Fit the period from the signal instead and it comes out
      at <span class="num">${NOMINAL_PERIOD_384K}</span>. The gap is 2.6 samples per line: about
      0.08 per cent, the sort of error a tape machine accumulates without anyone noticing.</p>

      <p>Two and a half samples is nothing until you multiply it by 512. Over one frame the error
      compounds to <strong>${Math.round(gapSlip384).toLocaleString('en-US')} samples</strong>,
      which is ${Math.round(gapSlipPixels)} pixels of a 384-pixel column — more than half the
      picture. The last line of the frame would be read from somewhere near the middle of the
      previous one.</p>

      <p>You can do this yourself. The dial in the Lab is the line period, and the button below
      sets it to the value the cover specifies rather than the value the signal has.</p>
      <button type="button" class="pullbtn" data-act="detune">
        Set the dial to <span class="num">${coverPeriodAtAsset.toFixed(3)}</span> and watch it tear
      </button>

      <div class="rule"></div>

      <p>The gap is not a mistake in the record. It is the residue of everything the signal passed
      through on the way here: a Colorado Video encoder writing to analogue tape in 1977, a lacquer
      cut from that tape, a copper mother, a gold-plated pressing, a stylus, and finally an
      analogue-to-digital converter running very slightly fast. Each stage kept the picture and
      shifted the clock.</p>

      <p>The first modern decoders papered over it. Ron Barry's 2017 program found each line by
      taking the tallest sample in a 190-sample window, which latches onto whichever sync feature
      happens to peak on that line — so his traces came out alternately 3100 and 3300 samples
      apart, and he corrected them with a hardcoded <em>±12 samples on even traces, changing at
      trace 164</em>. It worked. It also meant the timing was an assumption rather than a
      measurement.</p>

      <p>This decoder measures it. It folds several hundred lines together at the nominal period,
      which averages the picture away and leaves a high-signal template of the whole sync region;
      cross-correlates every line against that template and interpolates the correlation peak to a
      fraction of a sample; then fits position against line index with a robust straight line. The
      slope of that fit is the true line period. The intercept is the frame phase. And the
      residuals — plotted in Diagnostics — are the wow and flutter of a tape machine that stopped
      turning half a century ago.</p>

      <p>There is a second gap in the catalog. The record carries
      <strong>${catalog.frames.length} frames</strong>, but only
      <strong>${catalog.images.length} pictures</strong>: twenty of the photographs were sent three
      times over, as red, green and blue separations, to be recombined by whoever finds them. The
      arithmetic only closes if you already know which frames belong together — which is a thing
      the record tells you in diagrams, and a thing we take on trust from a table.</p>

      <p>And the last gap is the one the whole object is addressed to. Voyager 1 will not pass near
      another star for roughly forty thousand years. The pictures are stored as an analogue
      waveform because analogue degrades gracefully, and as a negative because a negative survives
      a scratch better than a positive does. Whoever plays it will have to do what you just did:
      guess a period, watch the picture tear, and tune until it stands up.</p>
    </div>`;
  panelEls.get('gap')!.append(gapSheet);

  (gapSheet.querySelector('[data-act="detune"]') as HTMLButtonElement).addEventListener(
    'click',
    () => {
      show('lab');
      const tb = slots[editing].timebase;
      tb.period = Math.min(knobMax, Math.max(knobMin, coverPeriodAtAsset));
      tb.locked = false;
      knob.set(tb.period, false);
      decoder.setTimebase(tb, editing);
      updateShearContext();
      explainer.show(
        'Line period',
        'This is the period the record cover specifies. The signal on this digitisation runs slightly faster, so every line starts a little late — and the error compounds down the frame. Press Snap to lock to put it right.',
      );
    },
  );

  // ======================================================================
  // THE COVER
  // ======================================================================

  {
    const host = panelEls.get('cover')!;
    if (deps.mountCover) {
      const cleanup = deps.mountCover(host);
      if (typeof cleanup === 'function') disposers.push(cleanup);
    } else {
      const sheet = h('div', 'sheet');
      sheet.innerHTML = `
        <div class="sheet__head"><h1 class="sheet__title">The Cover</h1></div>
        <div class="prose">
          <p class="lede">The cover panel is not wired up in this build.</p>
          <p>Pass <code>mountCover</code> from <code>src/panels/cover.ts</code> into
          <code>init(root, deps)</code> and it will render here.</p>
        </div>`;
      host.append(sheet);
    }
  }

  // ======================================================================
  // Oscilloscope
  // ======================================================================

  /** Vertical scale, eased between frames so the trace does not jump about. */
  let scopeScale = 1;

  const SCOPE_T0 = TRACE_REGIONS.refStart - 0.02;
  const SCOPE_T1 = 1.015;

  const BANDS: { from: number; to: number; fill: string; label: string }[] = [
    {
      from: TRACE_REGIONS.refStart,
      to: TRACE_REGIONS.refEnd,
      fill: 'rgba(200,160,82,.10)',
      label: 'DC ref',
    },
    {
      from: TRACE_REGIONS.pictureStart,
      to: TRACE_REGIONS.pictureEnd,
      fill: 'rgba(120,224,206,.055)',
      label: 'Picture',
    },
    {
      from: TRACE_REGIONS.plateauStart,
      to: TRACE_REGIONS.plateauEnd,
      fill: 'rgba(200,160,82,.085)',
      label: 'Sync',
    },
    {
      from: TRACE_REGIONS.edgeStart,
      to: TRACE_REGIONS.edgeEnd,
      fill: 'rgba(229,98,60,.35)',
      label: 'Edge',
    },
    {
      from: TRACE_REGIONS.porchStart,
      to: TRACE_REGIONS.porchEnd,
      fill: 'rgba(230,227,220,.05)',
      label: 'Porch',
    },
  ];

  function drawScope(): void {
    const ctx = fitCanvas(scopeCanvas);
    if (!ctx) return;
    const w = scopeCanvas.clientWidth;
    const hh = scopeCanvas.clientHeight;
    const padTop = 16;
    const padBottom = 16;
    const plotH = Math.max(10, hh - padTop - padBottom);
    const mid = padTop + plotH / 2;

    const snap: TraceSnapshot | null = decoder.traceSnapshot?.() ?? null;
    const st = slots[editing].settings;
    const pictureStart = snap?.pictureStart ?? st.pictureStart;
    const pictureEnd = snap?.pictureEnd ?? st.pictureEnd;
    const bins = snap?.bins ?? st.height;
    const period = snap?.period ?? slots[editing].timebase.period;

    const x = (t: number): number => ((t - SCOPE_T0) / (SCOPE_T1 - SCOPE_T0)) * w;

    // --- bands ---------------------------------------------------------
    for (const b of BANDS) {
      const from = b.label === 'Picture' ? pictureStart : b.from;
      const to = b.label === 'Picture' ? pictureEnd : b.to;
      const x0 = x(from);
      const x1 = x(to);
      ctx.fillStyle = b.fill;
      ctx.fillRect(x0, padTop, Math.max(1, x1 - x0), plotH);
      ctx.strokeStyle = 'rgba(255,255,255,.06)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(x0) + 0.5, padTop);
      ctx.lineTo(Math.round(x0) + 0.5, padTop + plotH);
      ctx.stroke();
    }

    // --- pixel bin boundaries -------------------------------------------
    // 384 bins across ~500 px is denser than the screen, so we decimate and
    // say so, rather than drawing a grey wash and calling it a grid.
    const binW = (x(pictureEnd) - x(pictureStart)) / Math.max(1, bins);
    const every = Math.max(1, Math.ceil(5 / Math.max(binW, 0.001)));
    ctx.strokeStyle = 'rgba(200,160,82,.14)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= bins; i += every) {
      const px = Math.round(x(pictureStart + ((pictureEnd - pictureStart) * i) / bins)) + 0.5;
      ctx.moveTo(px, padTop + plotH * 0.62);
      ctx.lineTo(px, padTop + plotH);
    }
    ctx.stroke();

    // --- labels ----------------------------------------------------------
    // Bands range from 79% of the width (picture) to 0.3% (the falling edge),
    // so labels are laid out left to right and pushed along whenever the
    // previous one has already claimed the space. The edge gets a leader.
    ctx.font = '9px Futura, "Century Gothic", "Avenir Next", sans-serif';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    let claimed = 0;
    for (const b of BANDS) {
      const from = b.label === 'Picture' ? pictureStart : b.from;
      const to = b.label === 'Picture' ? pictureEnd : b.to;
      const x0 = x(from);
      const x1 = x(to);
      const tw = ctx.measureText(b.label.toUpperCase()).width;
      const narrow = x1 - x0 < tw + 10;
      let tx = narrow ? x1 + 6 : (x0 + x1) / 2 - tw / 2;
      if (tx < claimed) tx = claimed;
      // On a narrow screen the last labels genuinely do not fit. Dropping one
      // is honest; overlapping two so neither can be read is not.
      if (tx + tw > w) continue;
      ctx.fillStyle = narrow ? '#e5623c' : '#8a7444';
      ctx.fillText(b.label.toUpperCase(), tx, 3);
      if (narrow) {
        ctx.strokeStyle = '#e5623c';
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.moveTo(x1, padTop);
        ctx.lineTo(tx - 2, 10);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      claimed = tx + tw + 8;
    }

    // trace start marker
    const xz = Math.round(x(0)) + 0.5;
    ctx.strokeStyle = 'rgba(200,160,82,.7)';
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(xz, padTop);
    ctx.lineTo(xz, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    if (!snap || !snap.samples.length) {
      ctx.fillStyle = '#4a4433';
      ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('no signal — press play', w / 2, mid);
      scopeTrace.textContent = '—';
      scopePeriod.textContent = period.toFixed(3);
      scopeBins.textContent = `${bins}${every > 1 ? ` · every ${every}` : ''}`;
      return;
    }

    // --- waveform --------------------------------------------------------
    const { samples, origin } = snap;
    let peak = 1e-6;
    const i0 = Math.max(0, Math.floor(origin + SCOPE_T0 * period));
    const i1 = Math.min(samples.length, Math.ceil(origin + SCOPE_T1 * period));
    for (let i = i0; i < i1; i++) peak = Math.max(peak, Math.abs(samples[i]));
    // Ease towards the new scale so a loud line does not make the whole
    // display twitch.
    scopeScale += (peak - scopeScale) * 0.15;
    const sy = (plotH / 2 - 4) / Math.max(scopeScale, 1e-6);

    ctx.strokeStyle = 'rgba(255,255,255,.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, Math.round(mid) + 0.5);
    ctx.lineTo(w, Math.round(mid) + 0.5);
    ctx.stroke();

    if (snap.dcReference != null) {
      ctx.strokeStyle = 'rgba(200,160,82,.6)';
      ctx.setLineDash([4, 4]);
      const yref = mid - snap.dcReference * sy;
      ctx.beginPath();
      ctx.moveTo(0, yref);
      ctx.lineTo(w, yref);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Min/max envelope per screen column: at 96 kHz one trace is ~800 samples
    // across ~600 px, so this only matters at narrow widths, but it keeps the
    // trace honest instead of aliasing.
    ctx.strokeStyle = '#78e0ce';
    ctx.lineWidth = 1;
    ctx.shadowColor = 'rgba(120,224,206,.55)';
    ctx.shadowBlur = 5;
    ctx.beginPath();
    let started = false;
    for (let px = 0; px < w; px++) {
      const tA = SCOPE_T0 + ((px / w) * (SCOPE_T1 - SCOPE_T0));
      const tB = SCOPE_T0 + (((px + 1) / w) * (SCOPE_T1 - SCOPE_T0));
      const a = Math.max(0, Math.floor(origin + tA * period));
      const b = Math.min(samples.length, Math.max(a + 1, Math.ceil(origin + tB * period)));
      if (a >= samples.length) break;
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = a; i < b; i++) {
        const v = samples[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (lo === Infinity) continue;
      const yLo = mid - lo * sy;
      const yHi = mid - hi * sy;
      if (!started) {
        ctx.moveTo(px + 0.5, yHi);
        started = true;
      }
      ctx.lineTo(px + 0.5, yHi);
      ctx.lineTo(px + 0.5, yLo);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    scopeTrace.textContent = `${snap.index + 1} / ${st.traces}`;
    scopePeriod.textContent = period.toFixed(3);
    scopeBins.textContent = `${bins}${every > 1 ? ` · every ${every}` : ''}`;
  }

  // ======================================================================
  // Scan line
  // ======================================================================

  /**
   * Scale the picture to fill the viewport without distorting it, and size the
   * well to match so the overlays (scan line, wipe divider) land on the picture
   * rather than on the padding around it. The DSP has already applied `aspect`
   * to the canvas dimensions, so we only ever fit what we are given.
   */
  function fitImage(): void {
    const cw = imageCanvas.width;
    const ch = imageCanvas.height;
    if (!cw || !ch) return;
    const availW = viewStage.clientWidth;
    const availH = viewStage.clientHeight;
    if (availW <= 0 || availH <= 0) return;
    const scale = Math.min(availW / cw, availH / ch);
    mount.style.width = `${Math.round(cw * scale)}px`;
    mount.style.height = `${Math.round(ch * scale)}px`;
  }

  function paintScanline(): void {
    const st = slots[editing].settings;
    const p = st.traces > 0 ? Math.min(1, Math.max(0, traceIndex / st.traces)) : 0;
    const q = st.flipScan ? 1 - p : p;
    const rot = ((st.rotate % 4) + 4) % 4;
    // The scan sweeps along the trace axis, which the display rotation turns.
    const horizontal = rot === 1 || rot === 3;
    const pos = rot === 2 || rot === 3 ? 1 - q : q;
    if (horizontal) {
      scanline.style.cssText = `left:0;right:0;width:auto;top:${(pos * 100).toFixed(3)}%;bottom:auto;height:1px`;
    } else {
      scanline.style.cssText = `top:0;bottom:0;height:auto;left:${(pos * 100).toFixed(3)}%;width:1px`;
    }
    scanline.dataset.live = playing ? '1' : '0';
  }

  // ======================================================================
  // Transport painting
  // ======================================================================

  function paintClock(): void {
    clockNow.textContent = clockText(position);
    clockDur.textContent = clockText(duration);
    if (!scrubbing) {
      scrub.max = String(Math.max(0.001, duration));
      scrub.value = String(position);
    }
    const pct = duration > 0 ? (position / duration) * 100 : 0;
    scrub.style.setProperty('--pct', `${pct.toFixed(2)}%`);
  }

  function paintStatus(): void {
    lamp.dataset.status = status;
    lampText.textContent = STATUS_TEXT[status];
    playBtn.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
    playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    playBtn.dataset.live = playing ? '1' : '0';
  }

  // ======================================================================
  // Navigation
  // ======================================================================

  function show(id: PanelId): void {
    currentPanel = id;
    for (const p of PANELS) {
      const el = panelEls.get(p.id)!;
      const on = p.id === id;
      el.hidden = !on;
      tabs.get(p.id)!.setAttribute('aria-selected', on ? 'true' : 'false');
      tabs.get(p.id)!.tabIndex = on ? 0 : -1;
    }
    if (location.hash.slice(1).split(':')[0] !== id) {
      history.replaceState(null, '', `#${id}${id === 'lab' && currentImage ? `:${currentImage.n}` : ''}`);
    }
    if (id === 'diagnostics') paintDiagnostics();
    if (id === 'gallery') requestAnimationFrame(armThumbnails);
  }

  function open(n: number): void {
    const img = catalog.images.find((i) => i.n === n);
    if (!img) return;
    currentImage = img;

    slateNum.textContent = `№ ${String(img.n).padStart(3, '0')}${img.color ? ' · RGB triplet' : ''}`;
    slateTitle.textContent = img.title;
    const frames = img.frames.map((id) => frameById.get(id)).filter(Boolean) as CatalogFrame[];
    slateMeta.textContent = frames.length
      ? `${frames.map((f) => f.id).join(' · ')} · ${clockText(frames[0].startSample / masterRate)} into the master`
      : '';

    // Colour controls only mean something for a triplet.
    gColour.el.hidden = !img.color;
    gColour.note.textContent = img.color ? '' : 'mono';
    if (img.color) gColour.el.dataset.open = '1';

    // Barry's orientation column is the sensible starting rotation.
    const rot = frames[0]?.orientation ?? 0;
    slots.a.settings.rotate = rot;
    slots.b.settings.rotate = rot;

    for (const [num, card] of cardByImage) {
      card.dataset.current = num === n ? '1' : '0';
    }

    syncControls();
    pushSettings('a');
    pushSettings('b');
    // Opening an image normally means "show it to me". The one exception is the
    // entry point choosing a default at boot, which must not yank the visitor
    // off the panel their link asked for.
    if (firstOpen) firstOpen = false;
    else show('lab');
    void decoder.open(n);
    if (currentPanel === 'lab') history.replaceState(null, '', `#lab:${n}`);
  }

  // ======================================================================
  // Decoder events
  // ======================================================================

  disposers.push(
    decoder.subscribe((e: DecoderEvent) => {
      switch (e.type) {
        case 'status':
          status = e.status;
          playing = e.status === 'playing';
          paintStatus();
          if (e.status === 'error' && e.message) explainer.show('Fault', e.message);
          break;
        case 'time':
          position = e.seconds;
          duration = e.duration;
          traceIndex = e.trace;
          paintClock();
          break;
        case 'image':
          if (e.imageNumber !== currentImage?.n) open(e.imageNumber);
          break;
        case 'diagnostics': {
          diag = e.data;
          if (diag.period != null) {
            measured = diag.period;
            knob.setMeasured(measured);
            // A newly measured period is adopted only by slots still in lock.
            for (const s of ['a', 'b'] as SettingsSlot[]) {
              if (slots[s].timebase.locked) {
                slots[s].timebase.period = measured;
                decoder.setTimebase(slots[s].timebase, s);
              }
            }
            knob.set(slots[editing].timebase.period, slots[editing].timebase.locked);
            updateShearContext();
          }
          if (currentPanel === 'diagnostics') paintDiagnostics();
          break;
        }
      }
    }),
  );

  // ======================================================================
  // Keyboard
  // ======================================================================

  /**
   * True when the focused element handles its own keys. Buttons count: a
   * focused button already turns Space into a click, so letting the global
   * handler through as well would toggle playback twice and appear to do
   * nothing.
   */
  function ownsKeys(t: EventTarget | null): boolean {
    const el = t as HTMLElement | null;
    if (!el || !el.tagName) return false;
    if (el.isContentEditable) return true;
    if (/^(INPUT|TEXTAREA|SELECT|BUTTON|A)$/.test(el.tagName)) return true;
    const role = el.getAttribute('role');
    return role !== null && ['slider', 'switch', 'radio', 'tab', 'button'].includes(role);
  }

  const onKey = (e: KeyboardEvent): void => {
    if (ownsKeys(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (currentPanel !== 'lab') return;
    switch (e.key) {
      case ' ':
        e.preventDefault();
        playing ? decoder.pause() : decoder.play();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        decoder.seek(Math.max(0, position - (e.shiftKey ? 0.2 : 2)));
        break;
      case 'ArrowRight':
        e.preventDefault();
        decoder.seek(Math.min(duration, position + (e.shiftKey ? 0.2 : 2)));
        break;
      case 'l':
      case 'L':
        e.preventDefault();
        snapToLock();
        break;
      default:
        break;
    }
  };
  window.addEventListener('keydown', onKey);
  disposers.push(() => window.removeEventListener('keydown', onKey));

  // Back and forward move between panels and images. `show` and `open` write
  // the hash with replaceState, which does not fire this, so there is no loop.
  const onHash = (): void => {
    const [p, hn] = location.hash.slice(1).split(':');
    const n = Number(hn);
    if (Number.isFinite(n) && n > 0 && n !== currentImage?.n) open(n);
    if (PANELS.some((x) => x.id === p) && p !== currentPanel) show(p as PanelId);
  };
  window.addEventListener('hashchange', onHash);
  disposers.push(() => window.removeEventListener('hashchange', onHash));

  // ======================================================================
  // Frame loop and resize
  // ======================================================================

  let raf = 0;
  const tick = (): void => {
    raf = requestAnimationFrame(tick);
    if (currentPanel !== 'lab' || panelEls.get('lab')!.hidden) return;
    fitImage();
    drawScope();
    paintScanline();
  };
  raf = requestAnimationFrame(tick);
  disposers.push(() => cancelAnimationFrame(raf));

  const onResize = (): void => {
    if (currentPanel === 'diagnostics') {
      drawResiduals();
      drawTemplate();
    }
  };
  window.addEventListener('resize', onResize);
  disposers.push(() => window.removeEventListener('resize', onResize));

  // ======================================================================
  // Boot
  // ======================================================================

  decoder.attach({ image: imageCanvas });
  presetBar.setSlot('a');
  syncControls();
  pushSettings('a');
  pushSettings('b');
  decoder.setTimebase(slots.a.timebase, 'a');
  decoder.setTimebase(slots.b.timebase, 'b');
  decoder.setColor(slots.a.color, 'a');
  decoder.setColor(slots.b.color, 'b');
  setWipe(1);
  paintStatus();
  paintClock();
  paintDiagnostics();

  // A shared link like #lab:57 opens that image. The entry point should only
  // pick a default image when `current()` comes back null.
  {
    const [hashPanel, hashN] = location.hash.slice(1).split(':');
    const known = PANELS.some((p) => p.id === hashPanel);
    const n = Number(hashN);
    show(known ? (hashPanel as PanelId) : 'cover');
    if (Number.isFinite(n) && n > 0 && catalog.images.some((i) => i.n === n)) open(n);
  }

  return {
    show,
    open,
    current: () => currentImage?.n ?? null,
    state: (slot) => slots[slot],
    patch(patch, slot = editing) {
      Object.assign(slots[slot].settings, patch);
      if (slot === editing) syncControls();
      pushSettings(slot);
    },
    destroy() {
      for (const d of disposers) d();
      thumbObserver?.disconnect();
      decoder.dispose?.();
      root.textContent = '';
    },
  };
}
