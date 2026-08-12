/**
 * The control atoms: engraved faders, switches, segmented selects, and the one
 * rotary knob in the build.
 *
 * Two rules run through all of them:
 *
 *  - Every control is draggable *and* keyboard-operable against the same value
 *    model, so nothing is reachable only by mouse. Faders are role="slider",
 *    switches are role="switch", selects are radio groups with roving focus.
 *  - Every control reports its one-line explanation to a shared `Explainer` on
 *    hover and on focus. The shell renders that into a fixed legend window at
 *    the foot of the rack rather than a floating tooltip: a lab instrument
 *    tells you what a knob does in a window on the panel, not in a speech
 *    bubble, and a fixed target never occludes the thing you are dragging.
 */

/** Where one-line control explanations go. Implemented by the shell. */
export interface Explainer {
  show(name: string, text: string): void;
  clear(): void;
}

export type Curve = 'lin' | 'log';

export interface DialSpec {
  label: string;
  unit?: string;
  min: number;
  max: number;
  /** Quantum for a keyboard press and for drag rounding. */
  step: number;
  /** Quantum while shift is held. Defaults to step / 10. */
  fine?: number;
  value: number;
  /** Where the pipeline's default sits; drawn as a hairline on the track. */
  home?: number;
  describe: string;
  curve?: Curve;
  format?: (v: number) => string;
  onInput: (v: number) => void;
  explainer?: Explainer;
}

export interface DialHandle {
  el: HTMLElement;
  /** Update the displayed value without firing onInput. */
  set(value: number): void;
  value(): number;
  destroy(): void;
}

/** Drag travel, in CSS pixels, that covers the whole range. */
const DRAG_SPAN = 260;
const FINE_FACTOR = 8;

let uid = 0;
const nextId = () => `c${++uid}`;

// ---------------------------------------------------------------------------
// value <-> position mapping
// ---------------------------------------------------------------------------

/** Log curve with an offset so ranges that include zero still work. */
function logEps(min: number, max: number): number {
  return (max - min) / 400;
}

function toPos(v: number, s: { min: number; max: number; curve?: Curve }): number {
  if (s.max === s.min) return 0;
  if (s.curve === 'log') {
    const e = logEps(s.min, s.max);
    return Math.log((v - s.min + e) / e) / Math.log((s.max - s.min + e) / e);
  }
  return (v - s.min) / (s.max - s.min);
}

function fromPos(p: number, s: { min: number; max: number; curve?: Curve }): number {
  if (s.curve === 'log') {
    const e = logEps(s.min, s.max);
    return s.min + e * (Math.pow((s.max - s.min + e) / e, p) - 1);
  }
  return s.min + p * (s.max - s.min);
}

function quantize(v: number, step: number, min: number, max: number): number {
  if (step > 0) v = Math.round((v - min) / step) * step + min;
  // Steps like 0.005 accumulate binary dust; round to the step's own precision.
  const dp = decimals(step);
  v = Number(v.toFixed(dp));
  return Math.min(max, Math.max(min, v));
}

function decimals(step: number): number {
  if (step <= 0) return 6;
  const s = String(step);
  const dot = s.indexOf('.');
  if (dot < 0) return 0;
  return Math.min(8, s.length - dot - 1);
}

/** Default readout: as many decimals as the step implies, never more. */
export function autoFormat(step: number): (v: number) => string {
  const dp = decimals(step);
  return (v) => v.toFixed(dp);
}

// ---------------------------------------------------------------------------
// fader
// ---------------------------------------------------------------------------

export function createDial(spec: DialSpec): DialHandle {
  const fmt = spec.format ?? autoFormat(spec.step);
  const fine = spec.fine ?? spec.step / 10;
  let value = spec.value;

  const el = document.createElement('div');
  el.className = 'dial';
  el.tabIndex = 0;
  el.setAttribute('role', 'slider');
  el.setAttribute('aria-label', spec.label);
  el.setAttribute('aria-valuemin', String(spec.min));
  el.setAttribute('aria-valuemax', String(spec.max));

  const descId = nextId();
  el.setAttribute('aria-describedby', descId);

  el.innerHTML = `
    <span class="dial__label"></span>
    <span class="dial__value"><span class="dial__num"></span>${
      spec.unit ? `<span class="dial__unit"></span>` : ''
    }</span>
    <span class="dial__track"><span class="dial__fill"></span>${
      spec.home !== undefined ? '<span class="dial__home"></span>' : ''
    }</span>
    <span class="vh" id="${descId}"></span>`;

  const labelEl = el.querySelector('.dial__label') as HTMLElement;
  const numEl = el.querySelector('.dial__num') as HTMLElement;
  const unitEl = el.querySelector('.dial__unit') as HTMLElement | null;
  const fillEl = el.querySelector('.dial__fill') as HTMLElement;
  const homeEl = el.querySelector('.dial__home') as HTMLElement | null;
  const descEl = el.querySelector('.vh') as HTMLElement;

  labelEl.textContent = spec.label;
  if (unitEl) unitEl.textContent = spec.unit ?? '';
  descEl.textContent = spec.describe;
  if (homeEl && spec.home !== undefined) {
    homeEl.style.setProperty('--home', `${(toPos(spec.home, spec) * 100).toFixed(2)}%`);
  }

  function paint(): void {
    const text = fmt(value);
    numEl.textContent = text;
    fillEl.style.setProperty('--fill', `${(toPos(value, spec) * 100).toFixed(2)}%`);
    el.setAttribute('aria-valuenow', String(value));
    el.setAttribute('aria-valuetext', spec.unit ? `${text} ${spec.unit}` : text);
    if (spec.home !== undefined) {
      el.dataset.modified = Math.abs(value - spec.home) > spec.step / 2 ? '1' : '0';
    }
  }

  function commit(v: number): void {
    const next = Math.min(spec.max, Math.max(spec.min, v));
    if (next === value) return;
    value = next;
    paint();
    spec.onInput(value);
  }

  // --- drag -------------------------------------------------------------
  let pointer = -1;
  let startX = 0;
  let startValue = 0;

  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    pointer = e.pointerId;
    startX = e.clientX;
    startValue = value;
    el.dataset.dragging = '1';
    el.setPointerCapture(pointer);
    el.focus();
    e.preventDefault();
  });

  el.addEventListener('pointermove', (e) => {
    if (e.pointerId !== pointer) return;
    const travel = e.shiftKey ? DRAG_SPAN * FINE_FACTOR : DRAG_SPAN;
    const p = toPos(startValue, spec) + (e.clientX - startX) / travel;
    const raw = fromPos(Math.min(1, Math.max(0, p)), spec);
    commit(quantize(raw, e.shiftKey ? fine : spec.step, spec.min, spec.max));
  });

  const endDrag = (e: PointerEvent): void => {
    if (e.pointerId !== pointer) return;
    pointer = -1;
    el.dataset.dragging = '0';
  };
  el.addEventListener('pointerup', endDrag);
  el.addEventListener('pointercancel', endDrag);

  el.addEventListener(
    'wheel',
    (e) => {
      if (document.activeElement !== el && !el.matches(':hover')) return;
      e.preventDefault();
      const dir = e.deltaY > 0 || e.deltaX > 0 ? -1 : 1;
      commit(quantize(value + dir * (e.shiftKey ? fine : spec.step), fine, spec.min, spec.max));
    },
    { passive: false },
  );

  // Double-click returns the knob to where the pipeline would have left it.
  el.addEventListener('dblclick', () => {
    if (spec.home !== undefined) commit(spec.home);
  });

  // --- keyboard ---------------------------------------------------------
  el.addEventListener('keydown', (e) => {
    const q = e.shiftKey ? fine : spec.step;
    let next: number | null = null;
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        next = value + q;
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        next = value - q;
        break;
      case 'PageUp':
        next = value + q * 10;
        break;
      case 'PageDown':
        next = value - q * 10;
        break;
      case 'Home':
        next = spec.min;
        break;
      case 'End':
        next = spec.max;
        break;
      default:
        return;
    }
    e.preventDefault();
    commit(quantize(next, q, spec.min, spec.max));
  });

  // --- explanation ------------------------------------------------------
  const explain = (): void => spec.explainer?.show(spec.label, spec.describe);
  const unexplain = (): void => spec.explainer?.clear();
  el.addEventListener('pointerenter', explain);
  el.addEventListener('focus', explain);
  el.addEventListener('pointerleave', unexplain);
  el.addEventListener('blur', unexplain);

  paint();

  return {
    el,
    set(v) {
      value = Math.min(spec.max, Math.max(spec.min, v));
      paint();
    },
    value: () => value,
    destroy() {
      el.remove();
    },
  };
}

// ---------------------------------------------------------------------------
// switch
// ---------------------------------------------------------------------------

export interface ToggleSpec {
  label: string;
  value: boolean;
  describe: string;
  onInput: (v: boolean) => void;
  explainer?: Explainer;
}

export interface ToggleHandle {
  el: HTMLElement;
  set(value: boolean): void;
  value(): boolean;
}

export function createToggle(spec: ToggleSpec): ToggleHandle {
  let value = spec.value;

  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'switch';
  el.setAttribute('role', 'switch');

  const descId = nextId();
  el.setAttribute('aria-describedby', descId);
  el.innerHTML = `
    <span class="switch__label"></span>
    <span class="switch__body"><span class="switch__knob"></span></span>
    <span class="vh" id="${descId}"></span>`;

  (el.querySelector('.switch__label') as HTMLElement).textContent = spec.label;
  (el.querySelector('.vh') as HTMLElement).textContent = spec.describe;

  const paint = (): void => el.setAttribute('aria-checked', value ? 'true' : 'false');

  el.addEventListener('click', () => {
    value = !value;
    paint();
    spec.onInput(value);
  });

  const explain = (): void => spec.explainer?.show(spec.label, spec.describe);
  el.addEventListener('pointerenter', explain);
  el.addEventListener('focus', explain);
  el.addEventListener('pointerleave', () => spec.explainer?.clear());
  el.addEventListener('blur', () => spec.explainer?.clear());

  paint();
  return {
    el,
    set(v) {
      value = v;
      paint();
    },
    value: () => value,
  };
}

// ---------------------------------------------------------------------------
// segmented select
// ---------------------------------------------------------------------------

export interface SelectSpec<T extends string> {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  describe: string;
  onInput: (v: T) => void;
  explainer?: Explainer;
}

export interface SelectHandle<T extends string> {
  el: HTMLElement;
  set(value: T): void;
  value(): T;
}

export function createSelect<T extends string>(spec: SelectSpec<T>): SelectHandle<T> {
  let value = spec.value;

  const el = document.createElement('div');
  el.className = 'seg';

  const label = document.createElement('span');
  label.className = 'dial__label';
  label.textContent = spec.label;
  el.append(label);

  const row = document.createElement('div');
  row.className = 'seg__row';
  row.setAttribute('role', 'radiogroup');
  row.setAttribute('aria-label', spec.label);

  const descId = nextId();
  row.setAttribute('aria-describedby', descId);

  const buttons = spec.options.map((opt) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'seg__opt';
    b.setAttribute('role', 'radio');
    b.textContent = opt.label;
    b.addEventListener('click', () => pick(opt.value));
    row.append(b);
    return b;
  });

  const desc = document.createElement('span');
  desc.className = 'vh';
  desc.id = descId;
  desc.textContent = spec.describe;

  el.append(row, desc);

  function paint(): void {
    spec.options.forEach((opt, i) => {
      const on = opt.value === value;
      buttons[i].setAttribute('aria-checked', on ? 'true' : 'false');
      // Roving tabindex: the group is one tab stop, arrows move within it.
      buttons[i].tabIndex = on ? 0 : -1;
    });
  }

  function pick(v: T, focus = false): void {
    if (v === value) return;
    value = v;
    paint();
    if (focus) buttons[spec.options.findIndex((o) => o.value === v)].focus();
    spec.onInput(value);
  }

  row.addEventListener('keydown', (e) => {
    const dir = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    const i = spec.options.findIndex((o) => o.value === value);
    const n = spec.options.length;
    pick(spec.options[(i + dir + n) % n].value, true);
  });

  const explain = (): void => spec.explainer?.show(spec.label, spec.describe);
  row.addEventListener('pointerenter', explain);
  row.addEventListener('focusin', explain);
  row.addEventListener('pointerleave', () => spec.explainer?.clear());
  row.addEventListener('focusout', () => spec.explainer?.clear());

  paint();
  return {
    el,
    set(v) {
      value = v;
      paint();
    },
    value: () => value,
  };
}

// ---------------------------------------------------------------------------
// The hero: the line period knob
// ---------------------------------------------------------------------------

export interface KnobSpec {
  min: number;
  max: number;
  step: number;
  fine: number;
  value: number;
  /** The value the timebase recovery measured, if it has reported one yet. */
  measured: number | null;
  describe: string;
  onInput: (value: number) => void;
  /** Fired when the user asks to re-lock onto the measured period. */
  onSnap: () => void;
  explainer?: Explainer;
}

export interface KnobHandle {
  el: HTMLElement;
  /** Set the displayed period and lock state without firing onInput. */
  set(value: number, locked: boolean): void;
  /** Report a newly measured period; redraws the lock window. */
  setMeasured(period: number | null): void;
  /** Context for the live shear readout. */
  setContext(ctx: { traces: number; spanSamples: number; height: number }): void;
  value(): number;
  destroy(): void;
}

const SWEEP = 300; // degrees of usable rotation, centred on straight up
const R_TICK = 78;
const R_TICK_IN = 69;
const R_MAJOR_IN = 63;
const R_ARC = 84;

function polar(r: number, deg: number): [number, number] {
  const a = ((deg - 90) * Math.PI) / 180;
  return [r * Math.cos(a), r * Math.sin(a)];
}

function arc(r: number, from: number, to: number): string {
  const [x0, y0] = polar(r, from);
  const [x1, y1] = polar(r, to);
  const large = Math.abs(to - from) > 180 ? 1 : 0;
  const sweep = to > from ? 1 : 0;
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${large} ${sweep} ${x1.toFixed(
    2,
  )} ${y1.toFixed(2)}`;
}

export function createKnob(spec: KnobSpec): KnobHandle {
  let value = spec.value;
  let locked = true;
  let measured = spec.measured;
  let ctx = { traces: 512, spanSamples: 635, height: 384 };

  const angleOf = (v: number): number =>
    -SWEEP / 2 + SWEEP * Math.min(1, Math.max(0, (v - spec.min) / (spec.max - spec.min)));

  const wrap = document.createElement('div');
  wrap.className = 'hero';
  wrap.dataset.lock = '1';

  const descId = nextId();
  wrap.innerHTML = `
    <div class="hero__head">
      <span class="hero__name">Timebase</span>
      <span class="group__rule"></span>
    </div>
    <div class="hero__body">
      <div class="knob" tabindex="0" role="slider"
           aria-label="Line period"
           aria-valuemin="${spec.min}" aria-valuemax="${spec.max}"
           aria-describedby="${descId}" data-lock="1">
        <svg class="knob__scale" viewBox="-100 -100 200 200" aria-hidden="true">
          <g class="knob__ticks"></g>
          <path class="knob__rail" fill="none" stroke="#2a2e35" stroke-width="2"
                stroke-linecap="round"></path>
          <path class="knob__lockarc" fill="none" stroke="#2c5f58" stroke-width="3"
                stroke-linecap="round"></path>
          <path class="knob__drift" fill="none" stroke="#e5623c" stroke-width="3"
                stroke-linecap="round"></path>
          <line class="knob__lockmark" stroke="#78e0ce" stroke-width="1.6"
                stroke-linecap="round"></line>
        </svg>
        <div class="knob__cap"><span class="knob__index"></span></div>
      </div>
      <div class="hero__readout">
        <div class="hero__big"><span class="knob__num">—</span></div>
        <div class="hero__unit">samples per trace</div>
        <div class="hero__drift">
          <span>drift</span><span><b class="knob__drift-n">—</b> smp/tr</span>
          <span>shear</span><span><b class="knob__shear">—</b> px</span>
        </div>
        <div class="hero__lockrow">
          <span class="lockchip"><span class="knob__lockword">in lock</span></span>
          <button type="button" class="snap">Snap to lock</button>
        </div>
      </div>
    </div>
    <span class="vh" id="${descId}"></span>`;

  const knobEl = wrap.querySelector('.knob') as HTMLElement;
  const capEl = wrap.querySelector('.knob__cap') as HTMLElement;
  const ticksEl = wrap.querySelector('.knob__ticks') as SVGGElement;
  const railEl = wrap.querySelector('.knob__rail') as SVGPathElement;
  const lockArcEl = wrap.querySelector('.knob__lockarc') as SVGPathElement;
  const driftArcEl = wrap.querySelector('.knob__drift') as SVGPathElement;
  const lockMarkEl = wrap.querySelector('.knob__lockmark') as SVGLineElement;
  const numEl = wrap.querySelector('.knob__num') as HTMLElement;
  const driftEl = wrap.querySelector('.knob__drift-n') as HTMLElement;
  const shearEl = wrap.querySelector('.knob__shear') as HTMLElement;
  const lockWordEl = wrap.querySelector('.knob__lockword') as HTMLElement;
  const snapEl = wrap.querySelector('.snap') as HTMLButtonElement;
  (wrap.querySelector(`#${descId}`) as HTMLElement).textContent = spec.describe;

  // --- static scale -----------------------------------------------------
  {
    const N = 60;
    const parts: string[] = [];
    for (let i = 0; i <= N; i++) {
      const deg = -SWEEP / 2 + (SWEEP * i) / N;
      const major = i % 10 === 0;
      const [x0, y0] = polar(R_TICK, deg);
      const [x1, y1] = polar(major ? R_MAJOR_IN : R_TICK_IN, deg);
      parts.push(
        `<line x1="${x0.toFixed(2)}" y1="${y0.toFixed(2)}" x2="${x1.toFixed(2)}" y2="${y1.toFixed(
          2,
        )}" stroke="${major ? '#c8a052' : '#4a4433'}" stroke-width="${major ? 1.4 : 0.9}" />`,
      );
    }
    ticksEl.innerHTML = parts.join('');
    railEl.setAttribute('d', arc(R_ARC, -SWEEP / 2, SWEEP / 2));
  }

  // --- painting ---------------------------------------------------------

  function shearPixels(drift: number): number {
    // A constant period error accumulates linearly along the frame. By the last
    // trace the picture window has slipped by drift * traces samples, which is
    // (that / picture span) * height pixels of the column.
    return (Math.abs(drift) * ctx.traces * ctx.height) / Math.max(1, ctx.spanSamples);
  }

  function paint(): void {
    const deg = angleOf(value);
    capEl.style.setProperty('--angle', `${deg.toFixed(2)}deg`);
    numEl.textContent = value.toFixed(3);
    knobEl.setAttribute('aria-valuenow', String(value));

    if (measured === null) {
      driftEl.textContent = '—';
      shearEl.textContent = '—';
      lockArcEl.setAttribute('d', '');
      driftArcEl.setAttribute('d', '');
      lockMarkEl.setAttribute('x1', '0');
      lockMarkEl.setAttribute('y1', '0');
      lockMarkEl.setAttribute('x2', '0');
      lockMarkEl.setAttribute('y2', '0');
      knobEl.setAttribute('aria-valuetext', `${value.toFixed(3)} samples per trace`);
      return;
    }

    const drift = value - measured;
    const mDeg = angleOf(measured);
    driftEl.textContent = `${drift >= 0 ? '+' : '−'}${Math.abs(drift).toFixed(3)}`;
    shearEl.textContent = shearPixels(drift).toFixed(0);

    // The lock window: a narrow arc around the measured period, wide enough to
    // be visible, which is where the picture stands up straight.
    lockArcEl.setAttribute('d', arc(R_ARC, mDeg - 2.5, mDeg + 2.5));
    // The drift arc runs from the measured period to wherever you have dragged.
    driftArcEl.setAttribute('d', Math.abs(deg - mDeg) < 0.4 ? '' : arc(R_ARC, mDeg, deg));
    const [mx0, my0] = polar(R_ARC + 6, mDeg);
    const [mx1, my1] = polar(R_ARC + 13, mDeg);
    lockMarkEl.setAttribute('x1', mx0.toFixed(2));
    lockMarkEl.setAttribute('y1', my0.toFixed(2));
    lockMarkEl.setAttribute('x2', mx1.toFixed(2));
    lockMarkEl.setAttribute('y2', my1.toFixed(2));

    const inLock = Math.abs(drift) < spec.step / 2;
    const state = locked || inLock ? '1' : '0';
    wrap.dataset.lock = state;
    knobEl.dataset.lock = state;
    lockWordEl.textContent = locked ? 'in lock' : inLock ? 'at lock' : 'free run';
    snapEl.disabled = locked;
    knobEl.setAttribute(
      'aria-valuetext',
      `${value.toFixed(3)} samples per trace, ${
        locked ? 'locked to the measured period' : `${drift.toFixed(3)} from lock`
      }`,
    );
  }

  function commit(v: number): void {
    const next = Math.min(spec.max, Math.max(spec.min, v));
    if (next === value && !locked) return;
    value = next;
    locked = false;
    paint();
    spec.onInput(value);
  }

  // --- drag: vertical and horizontal both work, and they add ------------
  let pointer = -1;
  let startX = 0;
  let startY = 0;
  let startValue = 0;

  knobEl.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    pointer = e.pointerId;
    startX = e.clientX;
    startY = e.clientY;
    startValue = value;
    knobEl.dataset.dragging = '1';
    knobEl.setPointerCapture(pointer);
    knobEl.focus();
    e.preventDefault();
  });

  knobEl.addEventListener('pointermove', (e) => {
    if (e.pointerId !== pointer) return;
    // Up and right both increase. Combining the axes means the knob responds
    // to whatever direction the hand happens to move, which is what makes a
    // plugin knob feel good; pure angular tracking does not.
    const delta = e.clientX - startX - (e.clientY - startY);
    const travel = e.shiftKey ? DRAG_SPAN * FINE_FACTOR : DRAG_SPAN;
    const raw = startValue + (delta / travel) * (spec.max - spec.min);
    commit(quantize(raw, e.shiftKey ? spec.fine : spec.step, spec.min, spec.max));
  });

  const end = (e: PointerEvent): void => {
    if (e.pointerId !== pointer) return;
    pointer = -1;
    knobEl.dataset.dragging = '0';
  };
  knobEl.addEventListener('pointerup', end);
  knobEl.addEventListener('pointercancel', end);

  knobEl.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const dir = e.deltaY > 0 ? -1 : 1;
      commit(
        quantize(value + dir * (e.shiftKey ? spec.fine : spec.step), spec.fine, spec.min, spec.max),
      );
    },
    { passive: false },
  );

  knobEl.addEventListener('dblclick', () => spec.onSnap());

  knobEl.addEventListener('keydown', (e) => {
    const q = e.shiftKey ? spec.fine : spec.step;
    let next: number | null = null;
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        next = value + q;
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        next = value - q;
        break;
      case 'PageUp':
        next = value + q * 20;
        break;
      case 'PageDown':
        next = value - q * 20;
        break;
      case 'Home':
      case 'End':
        e.preventDefault();
        spec.onSnap();
        return;
      default:
        return;
    }
    e.preventDefault();
    commit(quantize(next, q, spec.min, spec.max));
  });

  snapEl.addEventListener('click', () => spec.onSnap());

  const explain = (): void => spec.explainer?.show('Line period', spec.describe);
  knobEl.addEventListener('pointerenter', explain);
  knobEl.addEventListener('focus', explain);
  knobEl.addEventListener('pointerleave', () => spec.explainer?.clear());
  knobEl.addEventListener('blur', () => spec.explainer?.clear());

  paint();

  return {
    el: wrap,
    set(v, isLocked) {
      value = Math.min(spec.max, Math.max(spec.min, v));
      locked = isLocked;
      paint();
    },
    setMeasured(period) {
      measured = period;
      paint();
    },
    setContext(next) {
      ctx = next;
      paint();
    },
    value: () => value,
    destroy() {
      wrap.remove();
    },
  };
}
